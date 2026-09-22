"""Immutable next-session forecasts and later result settlement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from maintenance_rsi import ExternalDataGuard

from .collector import DailyMarketCollector, MarketDataset
from .error_classification import summarize_error_classification
from .market_calendar import NYSETradingCalendar
from .model import LeadingLambdaClassifier
from .signals import REQUIRED_SYMBOLS, build_leading_features, build_training_set
from .relative_strength_feature import RELATIVE_STRENGTH_FEATURE_VERSION, RELATIVE_STRENGTH_PERIODS
from .recursive_self_improvement import canonicalize_parameter_keys
from .recursive_runtime import (
    PARALLEL_CANDIDATES,
    bootstrap_state,
    ensure_candidate_slots,
    evaluate_and_rotate_candidates,
    load_state,
    register_frozen_trials,
    settle_pending_trials,
    write_state,
)


SCHEMA_VERSION = "market-forward-v8"
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

DEFAULT_MODEL_PARAMETERS: dict[str, object] = {
    "relative_strength_periods": [5, 7, 14, 21],
    "relative_strength_feature_set": ["level", "velocity3", "cross50", "extreme_state"],
    "relative_strength_feature_weight": 1.0,
    "feature_lags": 5,
    # 2026-09-22 calibration_diagnostics on real SPY/QQQ walk-forward data showed
    # 0.001 leaves the no-trade class at ~9-11% of sessions, forcing a near-binary
    # up/down call almost every day. 0.005 raises it to ~46%, a materially
    # healthier three-way split; see docs/PAPER_PCA_SUB_SPEC.md history and the
    # calibration-diagnostics workflow artifacts for the measured trade-off.
    "neutral_band": 0.005,
    "no_trade_threshold": 0.45,
    # Softens the distance-to-probability softmax in LeadingLambdaClassifier.
    # 1.0 = unchanged. The same 2026-09-22 diagnostic found the 90-100% stated
    # confidence bin (~75% of all predictions) realizing only ~45-49% accuracy
    # (calibration gap ~0.5). Kept at 1.0 until an empirical grid search (via
    # the calibration-diagnostics workflow, which alone has real market data
    # access) selects a value that measurably closes that gap.
    "confidence_temperature": 1.0,
}


def normalize_model_parameters(
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    # A parameters mapping loaded from a pre-rename artifact may still use the
    # old rsi_* keys; canonicalize before merging so a learned value is never
    # silently dropped in favor of the default.
    supplied = canonicalize_parameter_keys(parameters or {})
    values = {**DEFAULT_MODEL_PARAMETERS, **supplied}
    values["relative_strength_periods"] = [int(value) for value in values["relative_strength_periods"]]
    values["relative_strength_feature_set"] = [str(value) for value in values["relative_strength_feature_set"]]
    values["relative_strength_feature_weight"] = float(values["relative_strength_feature_weight"])
    values["feature_lags"] = int(values["feature_lags"])
    values["neutral_band"] = float(values["neutral_band"])
    values["no_trade_threshold"] = float(values["no_trade_threshold"])
    values["confidence_temperature"] = float(values["confidence_temperature"])
    if not 0.0 <= values["neutral_band"] <= 0.02:
        raise ValueError("neutral_band must be in [0, 0.02]")
    if not 0.0 <= values["no_trade_threshold"] <= 1.0:
        raise ValueError("no_trade_threshold must be in [0, 1]")
    if not 0.0 < values["confidence_temperature"] <= 10.0:
        raise ValueError("confidence_temperature must be in (0, 10]")
    return values


def model_parameters_digest(parameters: dict[str, object]) -> str:
    payload = json.dumps(
        normalize_model_parameters(parameters),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    confidence_temperature: float
    relative_strength_feature_version: str
    relative_strength_periods: tuple[int, ...]
    input_sha256: str
    model_generation: int
    model_config_sha256: str
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
    neutral_band: float | None = None,
    model_parameters: dict[str, object] | None = None,
    model_generation: int = 0,
) -> list[FrozenMarketSignal]:
    """Fit only on known outcomes and predict the session after the latest input row."""
    parameters = normalize_model_parameters(model_parameters)
    if neutral_band is not None:
        parameters["neutral_band"] = float(neutral_band)
    neutral_band = float(parameters["neutral_band"])
    config_digest = model_parameters_digest(parameters)
    signal_session = _signal_session(dataset)
    close = dataset.close.loc[:signal_session]
    volume = dataset.volume.reindex(close.index)
    features = build_leading_features(
        close,
        volume,
        feature_lags=int(parameters["feature_lags"]),
        relative_strength_periods=tuple(parameters["relative_strength_periods"]),
        relative_strength_feature_set=tuple(parameters["relative_strength_feature_set"]),
        relative_strength_feature_weight=float(parameters["relative_strength_feature_weight"]),
    )
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
            no_trade_threshold=float(parameters["no_trade_threshold"]),
            min_samples=60,
            feature_family_weights={"rs": float(parameters["relative_strength_feature_weight"])},
            confidence_temperature=float(parameters["confidence_temperature"]),
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
                confidence_temperature=float(parameters["confidence_temperature"]),
                relative_strength_feature_version=RELATIVE_STRENGTH_FEATURE_VERSION,
                relative_strength_periods=tuple(parameters["relative_strength_periods"]),
                input_sha256=digest,
                model_generation=int(model_generation),
                model_config_sha256=config_digest,
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
            "neutral_band": float(signal.get("neutral_band", 0.0)),
            "status": "PENDING",
        }

    return {
        "upside": selected(max(values, key=lambda signal: float(signal["predicted_return"])), "UPSIDE"),
        "downside": selected(min(values, key=lambda signal: float(signal["predicted_return"])), "DOWNSIDE"),
    }


def freeze_signals(records: list[FrozenMarketSignal], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        "primary_trade": select_primary_trade(records),
        "extreme_forecasts": select_extreme_forecasts(records),
        "signals": [asdict(record) for record in records],
    }
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, ensure_ascii=False, indent=2))
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen signal: {destination}") from None
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
    compatible_versions = {
        "market-forward-v4",
        "market-forward-v5",
        "market-forward-v6",
        "market-forward-v7",
        SCHEMA_VERSION,
    }
    if (
        document.get("schema_version") not in compatible_versions
        or not signals
        or any(signal.get("schema_version") not in compatible_versions for signal in signals)
        or any(signal.get("signal_session") != current_session for signal in signals)
    ):
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(source.read_text(encoding="utf-8"))
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite carried signal: {destination}") from None
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
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "primary_trade": primary_settlement,
            "extreme_forecasts": extreme_settlements,
            "settlements": settlements,
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite settlement: {destination}") from None
    return destination


REPORT_COLUMNS = (
    "signal_session", "target_session", "target", "target_category", "target_name",
    "action", "predicted_class", "predicted_return", "confidence", "edge",
    "actual_return", "actual_class", "direction_correct",
    "return_error", "absolute_divergence_pp", "strategy_return_before_cost",
    "input_sha256", "neutral_band", "imputed_feature_count",
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
                "neutral_band": float(predicted["neutral_band"]),
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
    daily_error_classification = summarize_error_classification(
        rows.to_dict("records"), [daily_extremes]
    )
    cumulative_error_classification = summarize_error_classification(
        history.to_dict("records"), historical_extremes
    )
    report = {
        "schema_version": "signal-result-report-v2",
        "definition": {
            "direction_correct": "predicted_classとactual_classの一致",
            "absolute_divergence_pp": "abs(actual_return - predicted_return) * 100（％ポイント）",
            "no_lookahead": "予測値は凍結済みforward_signalから取得し、結果で再計算しない",
            "extreme_target_hit": "予測上昇1位・下落1位のETF銘柄が実績1位と一致",
            "extreme_direction_correct": "上昇候補は実績騰落率が正、下落候補は実績騰落率が負",
            "error_classification": (
                "外れの分類: extraction_miss(抽出漏れ)/overestimation(過大評価)/"
                "final_exclusion(最終除外)/missing_input(入力欠損)/market_noise(市場ノイズ)。"
                "分類は候補提案の根拠記録のみに用い、個々の外れを直ちに恒久ルールへは反映しない"
            ),
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
            "error_classification": daily_error_classification,
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
            "error_classification": cumulative_error_classification,
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


def _inspect_external_csv(path: Path) -> None:
    """Reject or quarantine an external CSV before any parser sees it."""
    guard = ExternalDataGuard()
    inspection = guard.inspect(path)
    if not inspection.accepted:
        quarantined = guard.quarantine(path, path.parent / "quarantine")
        raise RuntimeError(
            f"external data guard rejected {path}: {list(inspection.reasons)}; "
            f"quarantined at {quarantined}"
        )


def load_dataset(directory: str | Path) -> MarketDataset:
    source = Path(directory)
    close_path = source / "daily_close.csv"
    volume_path = source / "daily_volume.csv"
    _inspect_external_csv(close_path)
    _inspect_external_csv(volume_path)
    close = pd.read_csv(close_path, index_col="date", parse_dates=True)
    volume = pd.read_csv(volume_path, index_col="date", parse_dates=True)
    return MarketDataset(close=close.sort_index(), volume=volume.reindex(close.index).sort_index())


def _slot_path(base: str | Path, slot_index: int) -> Path:
    """Derive the Nth parallel candidate's file path from a base path.

    Slot 0 keeps the exact base filename (preserving the pre-parallel
    artifact name for continuity with existing workflow/history tooling);
    later slots insert a 1-based suffix before the extension, e.g.
    "..._2.json", "..._3.json".
    """
    base = Path(base)
    if slot_index == 0:
        return base
    return base.with_name(f"{base.stem}_{slot_index + 1}{base.suffix}")


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
    parser.add_argument("--previous-rsi-state", default=None)
    parser.add_argument(
        "--previous-candidate-signal",
        action="append",
        default=[],
        help="repeatable; one per parallel candidate slot, in slot order",
    )
    parser.add_argument(
        "--rsi-state-output",
        default="artifacts/validation/recursive_rsi_state.json",
    )
    parser.add_argument(
        "--candidate-output",
        default="artifacts/validation/recursive_rsi_candidate_signal.json",
        help="base path for slot 1; later parallel slots derive _2/_3-suffixed paths",
    )
    parser.add_argument(
        "--candidate-settlement-output",
        default="artifacts/validation/settled_recursive_rsi_candidate.json",
    )
    parser.add_argument("--source-commit", default=os.environ.get("GITHUB_SHA", "0" * 40))
    parser.add_argument("--exceptional-closures", default="config/exceptional_nyse_closures.json")
    args = parser.parse_args()
    calendar = NYSETradingCalendar(exceptional_closures=args.exceptional_closures)
    if args.input_dir:
        dataset = load_dataset(args.input_dir)
    else:
        completed = calendar.last_completed_session()
        dataset = DailyMarketCollector().collect(args.start, completed.end_exclusive.isoformat())
    state = (
        load_state(args.previous_rsi_state)
        if args.previous_rsi_state
        else bootstrap_state(DEFAULT_MODEL_PARAMETERS, args.source_commit)
    )
    generated_at = pd.Timestamp.now(tz="UTC")
    # A state just migrated from the single-candidate schema carries only one
    # slot; top it up to the full parallel pool before anything else touches it.
    ensure_candidate_slots(
        state, args.source_commit, generated_at.to_pydatetime(), parallel_candidates=PARALLEL_CANDIDATES
    )
    candidate_outputs = [_slot_path(args.candidate_output, i) for i in range(len(state["candidate_slots"]))]
    candidate_settlement_outputs = [
        _slot_path(args.candidate_settlement_output, i) for i in range(len(state["candidate_slots"]))
    ]

    path = None
    if args.previous:
        path = carry_forward_same_session(args.previous, dataset, args.output)
    if path is not None:
        state_path = Path(args.rsi_state_output)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        for candidate_output in candidate_outputs:
            candidate_output.parent.mkdir(parents=True, exist_ok=True)
        if any(output.exists() for output in candidate_outputs) or state_path.exists():
            raise FileExistsError("refusing to overwrite same-session RSI artifacts")
        if args.previous_rsi_state and args.previous_candidate_signal:
            if len(args.previous_candidate_signal) > len(candidate_outputs):
                raise ValueError(
                    f"got {len(args.previous_candidate_signal)} --previous-candidate-signal values "
                    f"but only {len(candidate_outputs)} parallel slots exist"
                )
            for source, destination in zip(args.previous_candidate_signal, candidate_outputs):
                shutil.copyfile(source, destination)
            # A slot padded on top of a state migrated from fewer historical
            # slots has no prior candidate forecast to carry forward; give it
            # the same baseline placeholder as the fully-legacy branch below.
            for destination in candidate_outputs[len(args.previous_candidate_signal):]:
                shutil.copyfile(path, destination)
            shutil.copyfile(args.previous_rsi_state, state_path)
        elif not args.previous_rsi_state and not args.previous_candidate_signal:
            # Legacy migration on an already-frozen session must not compare a
            # newly generated challenger against an earlier baseline whose
            # provider history may since have been revised.  Carry the exact
            # baseline as a non-evaluated placeholder for every parallel slot
            # and begin real candidate trials only after the next completed
            # market session.
            for candidate_output in candidate_outputs:
                shutil.copyfile(path, candidate_output)
            state["migration_status"] = "WAITING_FOR_NEXT_COMPLETED_SESSION"
            write_state(state, state_path)
        else:
            raise ValueError("recursive RSI state and candidate signals must be restored together")
        print(path.read_text(encoding="utf-8"))
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
            "same-session RSI artifacts preserved; cumulative history: "
            f"{carried_history or 'none'}, {carried_trades or 'none'}"
        )
        return

    settled = None
    candidate_settled_paths: list[Path | None] = []
    if args.previous:
        settled = settle_frozen_signals(args.previous, dataset, args.settlement_output)
        print(f"settlement: {settled or 'pending'}")
        if args.previous_rsi_state:
            if not args.previous_candidate_signal:
                raise ValueError("prior recursive RSI state requires its frozen candidate signals")
            if len(args.previous_candidate_signal) > len(state["candidate_slots"]):
                raise ValueError(
                    f"got {len(args.previous_candidate_signal)} --previous-candidate-signal values "
                    f"but only {len(state['candidate_slots'])} parallel slots exist"
                )
            provided = len(args.previous_candidate_signal)
            # A state just topped up from fewer historical slots (e.g. right
            # after migrating from the single-candidate schema) has no prior
            # forecast for its newly padded slots; settle only the slots that
            # actually have history instead of treating the shorter list as
            # an error.
            real_candidate_settled = [
                settle_frozen_signals(source, dataset, destination)
                for source, destination in zip(
                    args.previous_candidate_signal, candidate_settlement_outputs[:provided]
                )
            ]
            if any(bool(settled) != bool(candidate_settled) for candidate_settled in real_candidate_settled):
                raise RuntimeError("baseline and recursive RSI candidates did not settle together")
            padding = len(state["candidate_slots"]) - provided
            candidate_settled_paths = real_candidate_settled + [None] * padding
            if settled and all(real_candidate_settled):
                settle_pending_trials(
                    state,
                    previous_baseline_signal=args.previous,
                    previous_candidate_signals=list(args.previous_candidate_signal) + [None] * padding,
                    baseline_settlement=settled,
                    candidate_settlements=candidate_settled_paths,
                )
                evaluate_and_rotate_candidates(state, args.source_commit)

    if any(slot.get("pending_trial") is not None for slot in state["candidate_slots"]):
        # register_frozen_trials() below would refuse a second pending trial
        # anyway, but it only fails *after* freeze_signals() has already
        # written this session's immutable forecast files. Fail here instead,
        # before any new artifact is frozen, so a stalled settlement never
        # leaves a forward_signal.json orphaned from its recursive RSI state.
        raise RuntimeError(
            "recursive RSI trial from the previous run has not settled yet; "
            "refusing to freeze a new forecast until it does"
        )

    active = state["active_model"]
    records = generate_forward_signals(
        dataset,
        calendar,
        generated_at_utc=generated_at,
        model_parameters=dict(active["parameters"]),
        model_generation=int(active["generation"]),
    )
    path = freeze_signals(records, args.output)
    candidate_paths = []
    for slot, candidate_output in zip(state["candidate_slots"], candidate_outputs):
        candidate = slot["candidate"]
        candidate_records = generate_forward_signals(
            dataset,
            calendar,
            generated_at_utc=generated_at,
            model_parameters=dict(candidate["parameters"]),
            model_generation=int(candidate["generation"]),
        )
        candidate_paths.append(freeze_signals(candidate_records, candidate_output))
    register_frozen_trials(state, path, candidate_paths)
    state_path = write_state(state, args.rsi_state_output, args.previous_rsi_state)
    print(path.read_text(encoding="utf-8"))
    print(f"recursive RSI: {state_path}, candidates: {candidate_paths}")
    if args.previous:
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
