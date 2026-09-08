"""Immutable next-session forecasts and later result settlement."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .collector import DailyMarketCollector, MarketDataset
from .market_calendar import NYSETradingCalendar
from .model import LeadingLambdaClassifier
from .signals import REQUIRED_SYMBOLS, build_leading_features, build_training_set


SCHEMA_VERSION = "market-forward-v2"
FORWARD_REQUIRED_CLOSE = tuple(
    dict.fromkeys(
        (*REQUIRED_SYMBOLS, "DIA", "IWM", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLRE", "XLU", "XLV")
    )
)


@dataclass(frozen=True)
class FrozenMarketSignal:
    schema_version: str
    generated_at_utc: str
    signal_session: str
    target_session: str
    target: str
    action: str
    predicted_class: int
    confidence: float
    edge: float
    probabilities: dict[str, float]
    training_first_date: str
    training_last_date: str
    training_rows: int
    feature_count: int
    excluded_feature_count: int
    imputed_feature_count: int
    lambda_reg: float
    variance_target: float
    neutral_band: float
    input_sha256: str
    status: str = "PENDING"


def _utc_iso(value: pd.Timestamp | None = None) -> str:
    stamp = value if value is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def _signal_session(dataset: MarketDataset) -> pd.Timestamp:
    missing_columns = sorted(set(FORWARD_REQUIRED_CLOSE) - set(dataset.close.columns))
    if missing_columns:
        raise ValueError(f"forward signal is missing required close columns: {missing_columns}")
    required = dataset.close.loc[:, list(FORWARD_REQUIRED_CLOSE)]
    valid = required.notna().all(axis=1)
    if not valid.any():
        raise ValueError("no completed row contains the complete forward market universe")
    return pd.Timestamp(required.index[valid][-1])


def _input_hash(dataset: MarketDataset, signal_session: pd.Timestamp) -> str:
    payload = pd.concat(
        {
            "close": dataset.close.loc[:signal_session],
            "volume": dataset.volume.loc[:signal_session],
        },
        axis=1,
    ).to_csv(index_label="date", float_format="%.12g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_forward_signals(
    dataset: MarketDataset,
    calendar: NYSETradingCalendar,
    generated_at_utc: pd.Timestamp | None = None,
    targets: tuple[str, ...] = ("SPY", "QQQ"),
    neutral_band: float = 0.001,
) -> list[FrozenMarketSignal]:
    """Fit only on known outcomes and predict the session after the latest input row."""
    signal_session = _signal_session(dataset)
    close = dataset.close.loc[:signal_session]
    volume = dataset.volume.reindex(close.index)
    features = build_leading_features(close, volume)
    latest = features.loc[signal_session].replace([np.inf, -np.inf], np.nan)
    past_features = features.loc[features.index < signal_session]
    past_medians = past_features.median(axis=0, skipna=True)
    # A series that reappears today after a long provider gap must not drag the
    # complete-case training cutoff months backwards. Keep only features with
    # strong coverage in the most recent year, using information available now.
    recent_coverage = past_features.tail(252).notna().mean(axis=0)
    usable = past_medians.index[
        past_medians.notna() & recent_coverage.ge(0.95)
    ]
    latest = latest.loc[usable]
    imputed = latest.isna()
    latest = latest.fillna(past_medians.loc[usable])
    if latest.isna().any():
        raise ValueError("latest signal row cannot be completed using prior-only medians")

    target_session = calendar.next_session(signal_session.date())
    generated = _utc_iso(generated_at_utc)
    digest = _input_hash(dataset, signal_session)
    records: list[FrozenMarketSignal] = []
    for target in targets:
        X, y, _ = build_training_set(features.loc[:, usable], close[target], neutral_band=neutral_band)
        if X.empty or X.index.max() >= signal_session:
            raise ValueError("training data unexpectedly includes the pending signal session")
        prior_target_sessions = close[target].dropna().index
        prior_target_sessions = prior_target_sessions[prior_target_sessions < signal_session]
        if len(prior_target_sessions) == 0 or X.index.max() != prior_target_sessions[-1]:
            raise ValueError(
                f"{target}: recent training cutoff {X.index.max()} does not reach the prior target session"
            )
        model = LeadingLambdaClassifier(
            lambda_reg=0.10,
            variance_target=0.90,
            no_trade_threshold=0.45,
            min_samples=60,
        ).fit(X, y)
        prediction = model.predict_one(latest)
        records.append(
            FrozenMarketSignal(
                schema_version=SCHEMA_VERSION,
                generated_at_utc=generated,
                signal_session=signal_session.date().isoformat(),
                target_session=target_session.isoformat(),
                target=target,
                action=prediction.action,
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
                edge=prediction.edge,
                probabilities={str(key): value for key, value in prediction.probabilities.items()},
                training_first_date=X.index.min().date().isoformat(),
                training_last_date=X.index.max().date().isoformat(),
                training_rows=len(X),
                feature_count=len(usable),
                excluded_feature_count=len(features.columns) - len(usable),
                imputed_feature_count=int(imputed.sum()),
                lambda_reg=0.10,
                variance_target=0.90,
                neutral_band=neutral_band,
                input_sha256=digest,
            )
        )
    return records


def freeze_signals(records: list[FrozenMarketSignal], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen signal: {destination}")
    document = {"schema_version": SCHEMA_VERSION, "signals": [asdict(record) for record in records]}
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def carry_forward_same_session(
    previous_path: str | Path,
    dataset: MarketDataset,
    output_path: str | Path,
) -> Path | None:
    """Preserve the first v2 forecast when the same input session runs again."""
    source = Path(previous_path)
    if not source.exists():
        return None
    document = json.loads(source.read_text(encoding="utf-8"))
    signals = document.get("signals", [])
    current_session = _signal_session(dataset).date().isoformat()
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or not signals
        or any(signal.get("schema_version") != SCHEMA_VERSION for signal in signals)
        or any(signal.get("signal_session") != current_session for signal in signals)
    ):
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite carried signal: {destination}")
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def settle_frozen_signals(
    frozen_path: str | Path,
    dataset: MarketDataset,
    output_path: str | Path,
) -> Path | None:
    """Settle an earlier immutable forecast only when its target close is available."""
    source = Path(frozen_path)
    if not source.exists():
        return None
    document = json.loads(source.read_text(encoding="utf-8"))
    settlements: list[dict[str, object]] = []
    for signal in document["signals"]:
        target_day = pd.Timestamp(signal["target_session"])
        signal_day = pd.Timestamp(signal["signal_session"])
        target = signal["target"]
        if target_day not in dataset.close.index:
            continue
        previous_close = float(dataset.close.loc[signal_day, target])
        target_close = float(dataset.close.loc[target_day, target])
        if not np.isfinite(previous_close) or not np.isfinite(target_close):
            continue
        actual_return = target_close / previous_close - 1.0
        band = float(signal["neutral_band"])
        actual_class = 1 if actual_return > band else -1 if actual_return < -band else 0
        predicted_class = int(signal["predicted_class"])
        settlements.append(
            {
                **signal,
                "status": "SETTLED",
                "actual_return": actual_return,
                "actual_class": actual_class,
                "direction_correct": predicted_class == actual_class,
                "strategy_return_before_cost": predicted_class * actual_return,
            }
        )
    if not settlements:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite settlement: {destination}")
    destination.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "settlements": settlements}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def load_dataset(directory: str | Path) -> MarketDataset:
    source = Path(directory)
    close = pd.read_csv(source / "daily_close.csv", index_col="date", parse_dates=True)
    volume = pd.read_csv(source / "daily_volume.csv", index_col="date", parse_dates=True)
    return MarketDataset(close=close.sort_index(), volume=volume.reindex(close.index).sort_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze next-session market signals")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--input-dir", default=None, help="reuse an already collected raw CSV directory")
    parser.add_argument("--output", default="artifacts/validation/forward_signal.json")
    parser.add_argument("--previous", default=None)
    parser.add_argument("--settlement-output", default="artifacts/validation/settled_previous_signal.json")
    parser.add_argument("--exceptional-closures", default="config/exceptional_nyse_closures.json")
    args = parser.parse_args()
    calendar = NYSETradingCalendar(exceptional_closures=args.exceptional_closures)
    if args.input_dir:
        dataset = load_dataset(args.input_dir)
    else:
        completed = calendar.last_completed_session()
        dataset = DailyMarketCollector().collect(args.start, completed.end_exclusive.isoformat())
    path = None
    if args.previous:
        path = carry_forward_same_session(args.previous, dataset, args.output)
    if path is None:
        records = generate_forward_signals(dataset, calendar)
        path = freeze_signals(records, args.output)
    print(path.read_text(encoding="utf-8"))
    if args.previous:
        settled = settle_frozen_signals(args.previous, dataset, args.settlement_output)
        print(f"settlement: {settled or 'pending'}")


if __name__ == "__main__":
    main()
