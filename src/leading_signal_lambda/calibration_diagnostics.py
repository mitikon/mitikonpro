"""Read-only calibration and class-balance diagnostic for the leading-lambda model.

This module changes no prediction logic. It only measures the existing,
already-shipped ``LeadingLambdaClassifier`` / ``walk_forward_validate`` /
``build_training_set`` pipeline using their public API, to answer two
questions raised by the 2026-09-22 walk-forward results (SPY/QQQ direction
accuracy below 50% while per-day confidence sits at 60-99%, and
``trade_coverage`` pinned at 1.0 for every symbol):

1. How rare is the neutral ("no trade") label at the current
   ``neutral_band=0.001``, for several candidate band widths?
2. Is the model's stated confidence calibrated - i.e. among predictions
   where it claims e.g. 90-100% confidence, does it actually win 90-100%
   of the time?

Nothing here is wired into the daily production pipeline; it is invoked
explicitly via ``leading-lambda-calibration-diagnostics``.
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


def diagnose_target(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    target: str,
    *,
    train_size: int = 504,
    neutral_band_candidates: tuple[float, ...] = DEFAULT_NEUTRAL_BAND_CANDIDATES,
) -> dict[str, object]:
    features = build_leading_features(close, volume)
    _, _, next_returns = build_training_set(features, close[target])

    balance_by_band = {
        str(band): class_balance(next_returns, band) for band in neutral_band_candidates
    }

    # Run the walk-forward with the production default (neutral_band=0.001,
    # lambda_reg=0.10, variance_target=0.90, no_trade_threshold=0.45,
    # min_samples=60) - same config the daily pipeline uses.
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
    return {
        "target": target,
        "rows": len(result.predictions),
        "class_balance_by_neutral_band": balance_by_band,
        "calibration": calibration,
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
