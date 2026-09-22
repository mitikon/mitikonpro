"""Read-only calibration and class-balance diagnostic for the leading-lambda model.

This module itself changes no prediction logic - it only measures the
existing, already-shipped ``LeadingLambdaClassifier`` / ``walk_forward_validate``
/ ``build_training_set`` pipeline using their public API. It was built to
answer two questions raised by the 2026-09-22 walk-forward results (SPY/QQQ
direction accuracy below 50% while per-day confidence sits at 60-99%, and
``trade_coverage`` pinned at 1.0 for every symbol):

1. How rare is the neutral ("no trade") label at a given ``neutral_band``,
   across several candidate band widths? (``class_balance``,
   ``DEFAULT_NEUTRAL_BAND_CANDIDATES``)
2. Is the model's stated confidence calibrated - i.e. among predictions
   where it claims e.g. 90-100% confidence, does it actually win 90-100%
   of the time? (``calibration_bins``) And, since
   ``LeadingLambdaClassifier.confidence_temperature`` can soften that
   softmax, which temperature actually closes the gap on real data?
   (``calibration_by_temperature``, ``rescale_probabilities``)

The evidence this module produced motivated two production changes on
2026-09-22 (see ``forward.DEFAULT_MODEL_PARAMETERS``): raising
``neutral_band`` from 0.001 to 0.005, and adding the (currently
untouched, temperature=1.0) ``confidence_temperature`` knob. This module
stays read-only and diagnostic; it is invoked explicitly via
``leading-lambda-calibration-diagnostics`` and never writes model
parameters itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .collector import DailyMarketCollector
from .market_calendar import NYSETradingCalendar
from .signals import build_leading_features, build_training_set
from .validation import walk_forward_validate


DEFAULT_NEUTRAL_BAND_CANDIDATES = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015)
# A narrower subset actually refit and walk-forward validated: class_balance
# above is a free label re-count with no refit, but a genuine walk-forward
# comparison of trading-relevant metrics (direction_accuracy, drawdown,
# win rate) needs a full retrain-and-predict loop per band, so this list is
# kept short to bound workflow run time.
DEFAULT_NEUTRAL_BAND_METRIC_CANDIDATES = (0.001, 0.003, 0.005, 0.0075, 0.01)
DEFAULT_TEMPERATURE_CANDIDATES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 30.0)


def class_balance(next_returns: pd.Series, neutral_band: float) -> dict[str, dict[str, float | int]]:
    """Label share for +1/0/-1 at a given neutral_band, without refitting anything."""
    labels = pd.Series(
        np.select(
            [next_returns > neutral_band, next_returns < -neutral_band],
            [1, -1],
            default=0,
        ),
        index=next_returns.index,
    )
    total = len(labels)
    counts = labels.value_counts()
    return {
        str(cls): {"count": int(counts.get(cls, 0)), "share": float(counts.get(cls, 0)) / total}
        for cls in (1, 0, -1)
    }


def rescale_probabilities(row: pd.Series, temperature: float) -> tuple[int, float]:
    """Recompute (predicted_class, confidence) at another softmax temperature.

    Uses only the already-recorded prob_-1/prob_0/prob_1 columns (fit at
    temperature 1.0), so no refit or predict_one call is needed. Valid because
    softmax(logit / T) depends only on the pairwise logit differences, which
    are fully recoverable from any softmax output via log(p_i) - log(p_peak);
    see LeadingLambdaClassifier.predict_one for the T=1 computation this
    mirrors. Temperature scaling is monotonic, so the argmax (predicted_class)
    never changes with T - only the reported confidence does. This does not
    replay the no_trade_threshold gate in predict_one, which operates on the
    post-temperature confidence during actual prediction.
    """
    probs = {-1: float(row["prob_neg1"]), 0: float(row["prob_0"]), 1: float(row["prob_1"])}
    logs = {cls: np.log(max(p, 1e-15)) for cls, p in probs.items()}
    peak = max(logs.values())
    weights = {cls: np.exp((value - peak) / temperature) for cls, value in logs.items()}
    total = sum(weights.values())
    rescaled = {cls: weight / total for cls, weight in weights.items()}
    predicted_class = max(rescaled, key=rescaled.get)
    return predicted_class, rescaled[predicted_class]


def calibration_by_temperature(
    predictions: pd.DataFrame,
    temperature_candidates: tuple[float, ...] = DEFAULT_TEMPERATURE_CANDIDATES,
    n_bins: int = 10,
) -> dict[str, dict[str, object]]:
    """For each candidate temperature, report the overall calibration gap.

    ``overall_gap`` is mean(stated confidence) - mean(realized accuracy)
    across every prediction (not just the top bin), a single-number summary
    of over/under-confidence to compare across temperatures. Requires the
    prob_neg1/prob_0/prob_1 columns walk_forward_validate now records.
    """
    results: dict[str, dict[str, object]] = {}
    for temperature in temperature_candidates:
        rescaled = predictions.apply(
            lambda row: rescale_probabilities(row, temperature), axis=1, result_type="expand"
        )
        rescaled.columns = ["predicted_class", "confidence"]
        frame = predictions[["actual_class"]].join(rescaled)
        correct = frame["predicted_class"] == frame["actual_class"]
        results[str(temperature)] = {
            "mean_stated_confidence": float(frame["confidence"].mean()),
            "realized_accuracy": float(correct.mean()),
            "overall_calibration_gap": float(frame["confidence"].mean() - correct.mean()),
            "bins": calibration_bins(frame, n_bins=n_bins),
        }
    return results


def calibration_bins(predictions: pd.DataFrame, n_bins: int = 10) -> list[dict[str, float | int | str]]:
    """Reliability-diagram data: does stated confidence match realized accuracy?"""
    frame = predictions.copy()
    frame["correct"] = frame["predicted_class"] == frame["actual_class"]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(frame["confidence"], bins=edges, include_lowest=True)
    rows: list[dict[str, float | int | str]] = []
    for interval, group in frame.groupby("bin", observed=True):
        if len(group) == 0:
            continue
        rows.append(
            {
                "confidence_range": f"{interval.left:.2f}-{interval.right:.2f}",
                "n": int(len(group)),
                "mean_stated_confidence": float(group["confidence"].mean()),
                "realized_accuracy": float(group["correct"].mean()),
                "calibration_gap": float(group["confidence"].mean() - group["correct"].mean()),
            }
        )
    return rows


def walk_forward_metrics_by_neutral_band(
    features: pd.DataFrame,
    close_target: pd.Series,
    *,
    train_size: int = 504,
    neutral_band_candidates: tuple[float, ...] = DEFAULT_NEUTRAL_BAND_METRIC_CANDIDATES,
) -> dict[str, dict[str, float]]:
    """Refit and walk-forward each candidate band, not just recount labels.

    class_balance() alone cannot say whether a wider band actually helps: it
    only recounts labels without refitting. A wider band also changes the
    training class priors the model fits on, which can move
    direction_accuracy in either direction. This runs the real
    production config (lambda_reg=0.10, variance_target=0.90, min_samples=60,
    confidence_temperature=1.0) once per candidate band and reports the same
    metrics walk_forward_validate always has (direction_accuracy,
    annualized_return, max_drawdown, trade_win_rate, trade_coverage).
    """
    metrics_by_band: dict[str, dict[str, float]] = {}
    for band in neutral_band_candidates:
        X, y, returns = build_training_set(features, close_target, neutral_band=band)
        result = walk_forward_validate(
            X, y, returns, train_size=train_size, test_size=21, lambda_reg=0.10, variance_target=0.90, min_samples=60
        )
        metrics_by_band[str(band)] = dict(result.metrics)
    return metrics_by_band


def diagnose_target(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    target: str,
    *,
    train_size: int = 504,
    neutral_band_candidates: tuple[float, ...] = DEFAULT_NEUTRAL_BAND_CANDIDATES,
    neutral_band_metric_candidates: tuple[float, ...] = DEFAULT_NEUTRAL_BAND_METRIC_CANDIDATES,
    temperature_candidates: tuple[float, ...] = DEFAULT_TEMPERATURE_CANDIDATES,
) -> dict[str, object]:
    features = build_leading_features(close, volume)
    _, _, next_returns = build_training_set(features, close[target])

    balance_by_band = {
        str(band): class_balance(next_returns, band) for band in neutral_band_candidates
    }
    walk_forward_by_band = walk_forward_metrics_by_neutral_band(
        features, close[target], train_size=train_size, neutral_band_candidates=neutral_band_metric_candidates
    )

    # Run the walk-forward with the current production default (see
    # forward.DEFAULT_MODEL_PARAMETERS: neutral_band=0.005, lambda_reg=0.10,
    # variance_target=0.90, no_trade_threshold=0.45, min_samples=60) - same
    # config the daily pipeline uses, at confidence_temperature=1.0 (untouched)
    # so calibration_by_temperature below can grid-search from a known baseline.
    X, y, returns = build_training_set(features, close[target])
    result = walk_forward_validate(
        X,
        y,
        returns,
        train_size=train_size,
        test_size=21,
        lambda_reg=0.10,
        variance_target=0.90,
        min_samples=60,
    )
    calibration = calibration_bins(result.predictions)
    calibration_temperature_sweep = calibration_by_temperature(result.predictions, temperature_candidates)
    return {
        "target": target,
        "rows": len(result.predictions),
        "class_balance_by_neutral_band": balance_by_band,
        "walk_forward_metrics_by_neutral_band": walk_forward_by_band,
        "calibration": calibration,
        "calibration_by_temperature": calibration_temperature_sweep,
        "overall_trade_coverage": float((result.predictions["action"] != "NO_TRADE").mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only class-balance and confidence-calibration diagnostic (changes no prediction logic)"
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end-exclusive", default="auto")
    parser.add_argument("--targets", default="SPY,QQQ")
    parser.add_argument("--output", default="artifacts/calibration")
    parser.add_argument("--exceptional-closures", default="config/exceptional_nyse_closures.json")
    args = parser.parse_args()

    if args.end_exclusive == "auto":
        completed = NYSETradingCalendar(exceptional_closures=args.exceptional_closures).last_completed_session()
        end_exclusive = completed.end_exclusive.isoformat()
        print(f"last completed XNYS session: {completed.session_date} (close {completed.close_utc})")
    else:
        end_exclusive = args.end_exclusive

    dataset = DailyMarketCollector().collect(args.start, end_exclusive)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for target in args.targets.split(","):
        report = diagnose_target(dataset.close, dataset.volume, target)
        (output / f"{target.lower()}_calibration.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
