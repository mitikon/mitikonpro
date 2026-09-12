"""Immutable next-session forecasts and later result settlement."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .collector import DailyMarketCollector, MarketDataset
from .market_calendar import NYSETradingCalendar
from .model import LeadingLambdaClassifier
from .signals import REQUIRED_SYMBOLS, build_leading_features, build_training_set
from .rsi import RSI_FEATURE_VERSION, RSI_PERIODS


SCHEMA_VERSION = "market-forward-v6"
TRADE_SELECTION_RULE = "maximum_absolute_predicted_return_v1"
EXTREME_SELECTION_RULE = "predicted_return_extremes_v1"
TARGET_METADATA: dict[str, tuple[str, str]] = {
    "SPY": ("市場ETF", "S&P 500"),
    "QQQ": ("市場ETF", "NASDAQ 100"),
    "DIA": ("市場ETF", "Dow Jones"),
    "RSP": ("市場ETF", "S&P 500均等加重"),
    "IWM": ("市場ETF", "Russell 2000"),
    "SMH": ("テーマETF", "半導体"),
    "HYG": ("信用ETF", "ハイイールド債"),
    "LQD": ("信用ETF", "投資適格社債"),
    "XLB": ("セクターETF", "素材"),
    "XLC": ("セクターETF", "コミュニケーション"),
    "XLE": ("セクターETF", "エネルギー"),
    "XLF": ("セクターETF", "金融"),
    "XLI": ("セクターETF", "資本財"),
    "XLK": ("セクターETF", "情報技術"),
    "XLP": ("セクターETF", "生活必需品"),
    "XLRE": ("セクターETF", "不動産"),
    "XLU": ("セクターETF", "公益"),
    "XLV": ("セクターETF", "ヘルスケア"),
    "XLY": ("セクターETF", "一般消費財"),
}
FORWARD_TARGETS = tuple(TARGET_METADATA)
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
    target_category: str
    target_name: str
    action: str
    predicted_class: int
    predicted_return: float
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
    rsi_feature_version: str
    rsi_periods: tuple[int, ...]
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
    targets: tuple[str, ...] = FORWARD_TARGETS,
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
        X, y, next_returns = build_training_set(
            features.loc[:, usable], close[target], neutral_band=neutral_band
        )
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
        class_mean_returns = next_returns.groupby(y).mean().to_dict()
        predicted_return = float(
            sum(
                probability * float(class_mean_returns.get(cls, 0.0))
                for cls, probability in prediction.probabilities.items()
            )
        )
        category, target_name = TARGET_METADATA.get(target, ("ETF", target))
        records.append(
            FrozenMarketSignal(
                schema_version=SCHEMA_VERSION,
                generated_at_utc=generated,
                signal_session=signal_session.date().isoformat(),
                target_session=target_session.isoformat(),
                target=target,
                target_category=category,
                target_name=target_name,
                action=prediction.action,
                predicted_class=prediction.predicted_class,
                predicted_return=predicted_return,
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
                rsi_feature_version=RSI_FEATURE_VERSION,
                rsi_periods=RSI_PERIODS,
                input_sha256=digest,
            )
        )
    return records


def select_primary_trade(
    signals: list[FrozenMarketSignal] | list[dict[str, object]],
) -> dict[str, object]:
    """Select exactly one trade without using any result information."""
    if not signals:
        raise ValueError("at least one frozen signal is required for trade selection")
    values = [asdict(signal) if isinstance(signal, FrozenMarketSignal) else signal for signal in signals]
    selected = max(values, key=lambda signal: abs(float(signal["predicted_return"])))
    predicted_return = float(selected["predicted_return"])
    position = 1 if predicted_return >= 0.0 else -1
    return {
        "selection_rule": TRADE_SELECTION_RULE,
        "signal_session": selected["signal_session"],
        "target_session": selected["target_session"],
        "target": selected["target"],
        "target_category": selected["target_category"],
        "target_name": selected["target_name"],
        "action": "LONG" if position == 1 else "SHORT",
        "position": position,
        "predicted_return": predicted_return,
        "confidence": float(selected["confidence"]),
        "edge": float(selected["edge"]),
        "input_sha256": selected["input_sha256"],
        "status": "PENDING",
    }


def select_extreme_forecasts(
    signals: list[FrozenMarketSignal] | list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Freeze one maximum-upside and one maximum-downside ETF forecast."""
    if not signals:
        raise ValueError("at least one frozen signal is required for extreme selection")
    values = [asdict(signal) if isinstance(signal, FrozenMarketSignal) else signal for signal in signals]

    def selected(signal: dict[str, object], side: str) -> dict[str, object]:
        predicted_return = float(signal["predicted_return"])
        return {
            "selection_rule": EXTREME_SELECTION_RULE,
            "side": side,
            "signal_session": signal["signal_session"],
            "target_session": signal["target_session"],
            "target": signal["target"],
            "target_category": signal["target_category"],
            "target_name": signal["target_name"],
            "predicted_return": predicted_return,
            "direction_signal_present": predicted_return > 0.0 if side == "UPSIDE" else predicted_return < 0.0,
            "confidence": float(signal["confidence"]),
            "edge": float(signal["edge"]),
            "input_sha256": signal["input_sha256"],
            "status": "PENDING",
        }

    return {
        "upside": selected(max(values, key=lambda signal: float(signal["predicted_return"])), "UPSIDE"),
        "downside": selected(min(values, key=lambda signal: float(signal["predicted_return"])), "DOWNSIDE"),
    }


def freeze_signals(records: list[FrozenMarketSignal], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen signal: {destination}")
    document = {
        "schema_version": SCHEMA_VERSION,
        "primary_trade": select_primary_trade(records),
        "extreme_forecasts": select_extreme_forecasts(records),
        "signals": [asdict(record) for record in records],
    }
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def carry_forward_same_session(
    previous_path: str | Path,
    dataset: MarketDataset,
    output_path: str | Path,
) -> Path | None:
    """Preserve the first compatible forecast when the same input session runs again."""
    source = Path(previous_path)
    if not source.exists():
        return None
    document = json.loads(source.read_text(encoding="utf-8"))
    signals = document.get("signals", [])
    current_session = _signal_session(dataset).date().isoformat()
    compatible_versions = {"market-forward-v4", "market-forward-v5", SCHEMA_VERSION}
    if (
        document.get("schema_version") not in compatible_versions
        or not signals
        or any(signal.get("schema_version") not in compatible_versions for signal in signals)
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
        predicted_return = signal.get("predicted_return")
        signed_error = (
            actual_return - float(predicted_return) if predicted_return is not None else None
        )
        settlements.append(
            {
                **signal,
                "status": "SETTLED",
                "actual_return": actual_return,
                "actual_class": actual_class,
                "direction_correct": predicted_class == actual_class,
                "strategy_return_before_cost": predicted_class * actual_return,
                "return_error": signed_error,
                "absolute_divergence_pp": abs(signed_error) * 100.0 if signed_error is not None else None,
            }
        )
    if not settlements:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite settlement: {destination}")
    primary = document.get("primary_trade") or select_primary_trade(document["signals"])
    selected_result = next(
        (row for row in settlements if row["target"] == primary["target"]), None
    )
    if selected_result is None:
        return None
    position = int(primary["position"])
    actual_return = float(selected_result["actual_return"])
    spy_result = next((row for row in settlements if row["target"] == "SPY"), None)
    spy_return = float(spy_result["actual_return"]) if spy_result else None
    gross_return = position * actual_return
    primary_settlement = {
        **primary,
        "status": "SETTLED",
        "actual_return": actual_return,
        "actual_class": int(selected_result["actual_class"]),
        "direction_correct": (
            actual_return > float(selected_result["neutral_band"])
            if position == 1
            else actual_return < -float(selected_result["neutral_band"])
        ),
        "gross_return_before_cost": gross_return,
        "spy_return": spy_return,
        "excess_return_vs_spy_before_cost": (
            gross_return - spy_return if spy_return is not None else None
        ),
        "absolute_divergence_pp": abs(
            actual_return - float(primary["predicted_return"])
        ) * 100.0,
    }
    frozen_extremes = document.get("extreme_forecasts") or select_extreme_forecasts(
        document["signals"]
    )

    def settle_extreme(side: str) -> dict[str, object]:
        forecast = frozen_extremes[side]
        reverse = side == "upside"
        ranked = sorted(
            settlements, key=lambda row: float(row["actual_return"]), reverse=reverse
        )
        selected_row = next(row for row in settlements if row["target"] == forecast["target"])
        actual_extreme = ranked[0]
        predicted_return = float(forecast["predicted_return"])
        actual_return = float(selected_row["actual_return"])
        expected_class = 1 if side == "upside" else -1
        return {
            **forecast,
            "status": "SETTLED",
            "expected_class": expected_class,
            "selected_actual_return": actual_return,
            "selected_actual_class": int(selected_row["actual_class"]),
            "direction_correct": actual_return > 0.0 if side == "upside" else actual_return < 0.0,
            "selected_actual_rank": next(
                rank for rank, row in enumerate(ranked, start=1) if row["target"] == forecast["target"]
            ),
            "actual_extreme_target": actual_extreme["target"],
            "actual_extreme_name": actual_extreme["target_name"],
            "actual_extreme_return": float(actual_extreme["actual_return"]),
            "exact_target_hit": forecast["target"] == actual_extreme["target"],
            "absolute_divergence_pp": abs(actual_return - predicted_return) * 100.0,
        }

    extreme_settlements = {
        "upside": settle_extreme("upside"),
        "downside": settle_extreme("downside"),
    }
    destination.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "primary_trade": primary_settlement,
                "extreme_forecasts": extreme_settlements,
                "settlements": settlements,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


REPORT_COLUMNS = (
    "signal_session", "target_session", "target", "target_category", "target_name",
    "action", "predicted_class", "predicted_return", "confidence", "edge",
    "actual_return", "actual_class", "direction_correct",
    "return_error", "absolute_divergence_pp", "strategy_return_before_cost",
    "input_sha256",
)


def _rate(frame: pd.DataFrame, column: str = "direction_correct") -> float | None:
    if frame.empty:
        return None
    values = frame[column]
    if values.dtype == bool:
        return float(values.mean())
    normalized = values.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"true", "1"}
    )
    return float(normalized.mean())


def _history_paths(
    value: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def carry_forward_histories(
    previous_paths: str | Path | Sequence[str | Path] | None,
    output_path: str | Path,
    *,
    deduplicate_by: list[str],
    sort_by: list[str],
) -> Path | None:
    """Carry cumulative CSV history through runs that have no new settlement."""
    frames = [
        pd.read_csv(path)
        for path in _history_paths(previous_paths)
        if path.exists()
    ]
    if not frames:
        return None
    history = pd.concat(frames, ignore_index=True, sort=False)
    history = history.drop_duplicates(subset=deduplicate_by, keep="first")
    history = history.sort_values(sort_by, kind="stable")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(destination, index=False, float_format="%.10g")
    return destination


def write_signal_result_report(
    settlement_path: str | Path,
    report_path: str | Path,
    rows_path: str | Path,
    history_path: str | Path,
    previous_history_path: str | Path | Sequence[str | Path] | None = None,
    trade_history_path: str | Path | None = None,
    previous_trade_history_path: str | Path | Sequence[str | Path] | None = None,
) -> tuple[Path, Path, Path]:
    """Create a separate daily/cumulative audit report from settled frozen signals."""
    settlement = json.loads(Path(settlement_path).read_text(encoding="utf-8"))
    rows = pd.DataFrame(settlement["settlements"])
    for column in REPORT_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    rows = rows.loc[:, REPORT_COLUMNS]
    rows_destination = Path(rows_path)
    rows_destination.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(rows_destination, index=False, float_format="%.10g")

    history = rows.copy()
    previous_histories = [
        pd.read_csv(path)
        for path in _history_paths(previous_history_path)
        if path.exists()
    ]
    if previous_histories:
        history = pd.concat([*previous_histories, rows], ignore_index=True, sort=False)
    history = history.drop_duplicates(subset=["signal_session", "target"], keep="first")
    history = history.sort_values(["signal_session", "target"], kind="stable")
    history_destination = Path(history_path)
    history_destination.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_destination, index=False, float_format="%.10g")

    sector_rows = rows[rows["target_category"] == "セクターETF"]
    other_rows = rows[rows["target_category"] != "セクターETF"]
    divergence = pd.to_numeric(rows["absolute_divergence_pp"], errors="coerce").dropna()
    cumulative_divergence = pd.to_numeric(
        history["absolute_divergence_pp"], errors="coerce"
    ).dropna()
    cumulative_sectors = history[history["target_category"] == "セクターETF"]
    cumulative_other = history[history["target_category"] != "セクターETF"]
    primary_trade = settlement["primary_trade"]

    def derive_extremes(group: pd.DataFrame) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        for side, ascending in (("upside", False), ("downside", True)):
            predicted = group.sort_values("predicted_return", ascending=ascending).iloc[0]
            actual_ranked = group.sort_values("actual_return", ascending=ascending)
            actual = actual_ranked.iloc[0]
            results[side] = {
                "target": predicted["target"],
                "target_name": predicted["target_name"],
                "predicted_return": float(predicted["predicted_return"]),
                "selected_actual_return": float(predicted["actual_return"]),
                "selected_actual_rank": int(
                    list(actual_ranked["target"]).index(predicted["target"]) + 1
                ),
                "actual_extreme_target": actual["target"],
                "actual_extreme_name": actual["target_name"],
                "actual_extreme_return": float(actual["actual_return"]),
                "direction_correct": (
                    float(predicted["actual_return"]) > 0.0
                    if side == "upside"
                    else float(predicted["actual_return"]) < 0.0
                ),
                "exact_target_hit": predicted["target"] == actual["target"],
                "direction_signal_present": (
                    float(predicted["predicted_return"]) > 0.0
                    if side == "upside"
                    else float(predicted["predicted_return"]) < 0.0
                ),
                "absolute_divergence_pp": abs(
                    float(predicted["actual_return"]) - float(predicted["predicted_return"])
                ) * 100.0,
            }
        return results

    daily_extremes = settlement.get("extreme_forecasts") or derive_extremes(rows)
    historical_extremes = [derive_extremes(group) for _, group in history.groupby("signal_session")]

    def derive_trade(group: pd.DataFrame) -> dict[str, object]:
        selected = group.loc[
            pd.to_numeric(group["predicted_return"], errors="coerce").abs().idxmax()
        ]
        predicted_return = float(selected["predicted_return"])
        position = 1 if predicted_return >= 0.0 else -1
        actual_return = float(selected["actual_return"])
        spy_rows = group[group["target"] == "SPY"]
        spy_return = float(spy_rows.iloc[0]["actual_return"]) if not spy_rows.empty else np.nan
        gross_return = position * actual_return
        return {
            "selection_rule": TRADE_SELECTION_RULE,
            "signal_session": selected["signal_session"],
            "target_session": selected["target_session"],
            "target": selected["target"],
            "target_category": selected["target_category"],
            "target_name": selected["target_name"],
            "action": "LONG" if position == 1 else "SHORT",
            "position": position,
            "predicted_return": predicted_return,
            "confidence": float(selected["confidence"]),
            "edge": float(selected["edge"]),
            "input_sha256": selected["input_sha256"],
            "status": "SETTLED",
            "actual_return": actual_return,
            "actual_class": int(selected["actual_class"]),
            "direction_correct": position == int(selected["actual_class"]),
            "gross_return_before_cost": gross_return,
            "spy_return": spy_return,
            "excess_return_vs_spy_before_cost": gross_return - spy_return,
            "absolute_divergence_pp": abs(actual_return - predicted_return) * 100.0,
        }

    # Reconstruct earlier selections from frozen prediction rows using the same
    # outcome-blind rule, so the official loop starts on 2026-09-09 rather than
    # only after this report format was introduced.
    derived_trades = [derive_trade(group) for _, group in history.groupby("signal_session")]
    trade_history = pd.DataFrame(derived_trades)
    previous_trade_histories = [
        pd.read_csv(path)
        for path in _history_paths(previous_trade_history_path)
        if path.exists()
    ]
    if previous_trade_histories:
        trade_history = pd.concat(
            [*previous_trade_histories, trade_history], ignore_index=True, sort=False
        )
    trade_history = trade_history.drop_duplicates(
        subset=["signal_session"], keep="first"
    ).sort_values(["signal_session", "target"], kind="stable")
    if trade_history_path:
        trade_destination = Path(trade_history_path)
        trade_destination.parent.mkdir(parents=True, exist_ok=True)
        trade_history.to_csv(trade_destination, index=False, float_format="%.10g")

    trade_correct = _rate(trade_history)
    cumulative_gross = pd.to_numeric(
        trade_history["gross_return_before_cost"], errors="coerce"
    ).fillna(0.0)
    cumulative_spy = pd.to_numeric(trade_history["spy_return"], errors="coerce").fillna(0.0)
    compounded_trade = float((1.0 + cumulative_gross).prod() - 1.0)
    compounded_spy = float((1.0 + cumulative_spy).prod() - 1.0)

    def records(frame: pd.DataFrame, count: int = 5) -> list[dict[str, object]]:
        return json.loads(frame.head(count).to_json(orient="records", force_ascii=False))

    risers = rows.sort_values("actual_return", ascending=False)
    fallers = rows.sort_values("actual_return", ascending=True)
    misses = rows.sort_values("absolute_divergence_pp", ascending=False, na_position="last")
    report = {
        "schema_version": "signal-result-report-v2",
        "definition": {
            "direction_correct": "predicted_classとactual_classの一致",
            "absolute_divergence_pp": "abs(actual_return - predicted_return) * 100（％ポイント）",
            "no_lookahead": "予測値は凍結済みforward_signalから取得し、結果で再計算しない",
            "extreme_target_hit": "予測上昇1位・下落1位のETF銘柄が実績1位と一致",
            "extreme_direction_correct": "上昇候補は実績騰落率が正、下落候補は実績騰落率が負",
        },
        "daily": {
            "signal_session": str(rows["signal_session"].iloc[0]),
            "target_session": str(rows["target_session"].iloc[0]),
            "settled_targets": int(len(rows)),
            "direction_accuracy": _rate(rows),
            "sector_direction_accuracy": _rate(sector_rows),
            "other_etf_direction_accuracy": _rate(other_rows),
            "mean_absolute_divergence_pp": float(divergence.mean()) if len(divergence) else None,
            "median_absolute_divergence_pp": float(divergence.median()) if len(divergence) else None,
            "primary_trade": primary_trade,
            "extreme_forecasts": daily_extremes,
            "results": records(rows, len(rows)),
            "largest_risers": records(risers),
            "largest_fallers": records(fallers),
            "largest_forecast_misses": records(misses),
        },
        "cumulative": {
            "settled_predictions": int(len(history)),
            "settled_sessions": int(history["signal_session"].nunique()),
            "direction_accuracy": _rate(history),
            "sector_direction_accuracy": _rate(cumulative_sectors),
            "other_etf_direction_accuracy": _rate(cumulative_other),
            "mean_absolute_divergence_pp": (
                float(cumulative_divergence.mean()) if len(cumulative_divergence) else None
            ),
            "median_absolute_divergence_pp": (
                float(cumulative_divergence.median()) if len(cumulative_divergence) else None
            ),
            "primary_trade_count": int(len(trade_history)),
            "primary_trade_direction_accuracy": trade_correct,
            "primary_trade_win_rate_before_cost": float(cumulative_gross.gt(0.0).mean()),
            "primary_trade_compounded_return_before_cost": compounded_trade,
            "spy_compounded_return": compounded_spy,
            "primary_trade_excess_vs_spy_before_cost": compounded_trade - compounded_spy,
            "upside_top1_hit_rate": float(
                np.mean([value["upside"]["exact_target_hit"] for value in historical_extremes])
            ),
            "downside_top1_hit_rate": float(
                np.mean([value["downside"]["exact_target_hit"] for value in historical_extremes])
            ),
            "upside_direction_accuracy": float(
                np.mean([value["upside"]["direction_correct"] for value in historical_extremes])
            ),
            "downside_direction_accuracy": float(
                np.mean([value["downside"]["direction_correct"] for value in historical_extremes])
            ),
        },
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_destination, rows_destination, history_destination


def write_signal_result_markdown(report_path: str | Path, output_path: str | Path) -> Path:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    daily = report["daily"]
    cumulative = report["cumulative"]
    trade = daily["primary_trade"]
    extremes = daily["extreme_forecasts"]

    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.2f}%"

    def movers(title: str, values: list[dict[str, object]]) -> list[str]:
        lines = [f"## {title}", "", "|順位|ETF|区分|実績騰落率|予測騰落率|方向正誤|乖離（pp）|", "|---:|---|---|---:|---:|---|---:|"]
        for rank, value in enumerate(values, start=1):
            predicted = value.get("predicted_return")
            divergence_value = value.get("absolute_divergence_pp")
            lines.append(
                f"|{rank}|{value['target']} {value.get('target_name', '')}|{value.get('target_category', '')}|"
                f"{float(value['actual_return']) * 100:.2f}%|"
                f"{'—' if predicted is None else f'{float(predicted) * 100:.2f}%'}|"
                f"{'正解' if value.get('direction_correct') else '不正解'}|"
                f"{'—' if divergence_value is None else f'{float(divergence_value):.2f}'}|"
            )
        return lines

    lines = [
        "# 先行シグナル予測λ 正誤・乖離分析レポート",
        "",
        "## 主検証：上昇1位・下落1位の事前選出",
        "",
        "|区分|事前選出ETF|予測騰落率|実績騰落率|実績順位|実績1位ETF|銘柄的中|方向正誤|",
        "|---|---|---:|---:|---:|---|---|---|",
        *[
            f"|{'上昇1位' if side == 'upside' else '下落1位'}|{value['target']} {value.get('target_name', '')}|"
            f"{float(value['predicted_return']) * 100:.2f}%|{float(value['selected_actual_return']) * 100:.2f}%|"
            f"{value['selected_actual_rank']}位|{value['actual_extreme_target']} {value.get('actual_extreme_name', '')}|"
            f"{'的中' if value['exact_target_hit'] else '不的中'}|{'正解' if value['direction_correct'] else '不正解'}|"
            for side, value in extremes.items()
        ],
        "",
        f"- 累積上昇1位銘柄的中率: {percent(cumulative['upside_top1_hit_rate'])}",
        f"- 累積下落1位銘柄的中率: {percent(cumulative['downside_top1_hit_rate'])}",
        "",
        "## 主判定：前日に選定した単独トレード",
        "",
        f"- 選定ETF: {trade['target']} {trade.get('target_name', '')}",
        f"- 売買方向: {trade['action']}",
        f"- 予測騰落率: {float(trade['predicted_return']) * 100:.2f}%",
        f"- 実績騰落率: {float(trade['actual_return']) * 100:.2f}%",
        f"- 方向判定: {'正解' if trade['direction_correct'] else '不正解'}",
        f"- 仮想損益（コスト前）: {float(trade['gross_return_before_cost']) * 100:.2f}%",
        f"- SPY騰落率: {float(trade['spy_return']) * 100:.2f}%",
        f"- SPY超過成績（コスト前）: {float(trade['excess_return_vs_spy_before_cost']) * 100:.2f}%",
        f"- 予測乖離: {float(trade['absolute_divergence_pp']):.2f} pp",
        "",
        "## 19 ETF補助診断",
        "",
        f"- 予測基準日: {daily['signal_session']}",
        f"- 結果対象日: {daily['target_session']}",
        f"- 確定銘柄数: {daily['settled_targets']}",
        f"- 全体方向正解率: {percent(daily['direction_accuracy'])}",
        f"- セクターETF方向正解率: {percent(daily['sector_direction_accuracy'])}",
        f"- その他ETF方向正解率: {percent(daily['other_etf_direction_accuracy'])}",
        f"- 平均絶対乖離: {daily['mean_absolute_divergence_pp']:.2f} pp" if daily["mean_absolute_divergence_pp"] is not None else "- 平均絶対乖離: —",
        f"- 累積方向正解率: {percent(cumulative['direction_accuracy'])}（{cumulative['settled_predictions']}件）",
        f"- 単独トレード累積損益（コスト前）: {float(cumulative['primary_trade_compounded_return_before_cost']) * 100:.2f}%（{cumulative['primary_trade_count']}回）",
        f"- 単独トレード勝率（コスト前）: {percent(cumulative['primary_trade_win_rate_before_cost'])}",
        "",
        "乖離は `abs(実績騰落率 - 予測騰落率)` の％ポイント差です。予測値は発信時点の凍結値で、結果取得後に変更しません。",
        "",
    ]
    lines += movers("実績上昇上位", daily["largest_risers"])
    lines += [""] + movers("実績下落上位", daily["largest_fallers"])
    lines += [""] + movers("予測乖離上位", daily["largest_forecast_misses"])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--report-output", default="artifacts/validation/signal_result_report.json")
    parser.add_argument("--report-markdown-output", default="artifacts/validation/signal_result_report.md")
    parser.add_argument("--report-rows-output", default="artifacts/validation/signal_result_rows.csv")
    parser.add_argument("--history-output", default="artifacts/validation/signal_result_history.csv")
    parser.add_argument("--previous-history", action="append", default=[])
    parser.add_argument("--trade-history-output", default="artifacts/validation/selected_trade_history.csv")
    parser.add_argument("--previous-trade-history", action="append", default=[])
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
        if settled:
            report, _, _ = write_signal_result_report(
                settled,
                args.report_output,
                args.report_rows_output,
                args.history_output,
                args.previous_history,
                args.trade_history_output,
                args.previous_trade_history,
            )
            markdown = write_signal_result_markdown(report, args.report_markdown_output)
            print(f"result report: {report}, {markdown}")
        else:
            carried_history = carry_forward_histories(
                args.previous_history,
                args.history_output,
                deduplicate_by=["signal_session", "target"],
                sort_by=["signal_session", "target"],
            )
            carried_trades = carry_forward_histories(
                args.previous_trade_history,
                args.trade_history_output,
                deduplicate_by=["signal_session"],
                sort_by=["signal_session", "target"],
            )
            print(
                "cumulative history: "
                f"{carried_history or 'none'}, {carried_trades or 'none'}"
            )


if __name__ == "__main__":
    main()
