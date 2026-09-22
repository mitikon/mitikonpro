"""Read-only test of whether the leading-lambda model has real directional edge.

Stage 2 of the 2026-09-22 diagnostic work (see calibration_diagnostics.py for
Stage 1: class balance and confidence calibration). Stage 1 found direction
accuracy at neutral_band=0.005 sitting close to the uniform 3-class chance
rate (~33%). This module asks the harder question directly: is the model's
walk-forward accuracy actually, statistically, better than a trivial
baseline a computer could produce with no market model at all - and is that
edge real once the serial correlation in daily market data is accounted for?

Two baselines are computed walk-forward-consistently (no lookahead):

- ``majority_class_baseline``: predict the most frequent class in the
  training window, for every session in the following test window. The
  laziest baseline that still adapts to a changing class distribution.
- ``persistence_baseline``: predict yesterday's already-settled actual
  class. A classic "no model beats naive persistence" check.

Naive significance testing (e.g. a binomial test assuming i.i.d. Bernoulli
trials) overstates confidence for daily market data, whose correctness flags
are serially correlated (a volatile regime makes many consecutive sessions
easy or hard together). This module instead uses a moving block bootstrap
(block_size matching the walk-forward test_size) to build a resampling
distribution that respects that correlation, and reports a percentile
bootstrap confidence interval and one-sided p-value from it.

This module changes no prediction logic; it is invoked explicitly via
``leading-lambda-edge-diagnostics``.
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


DEFAULT_BLOCK_SIZE = 21
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_SEED = 0


def majority_class_baseline(y: pd.Series, train_size: int, test_size: int) -> pd.Series:
    """The training window's most frequent class, applied to the following test window.

    Mirrors validation.walk_forward_validate's own expanding-window loop so
    the baseline is evaluated on exactly the same sessions with exactly the
    same no-lookahead discipline: only labels strictly before ``start`` are
    used to pick the majority class for the block starting at ``start``.
    """
    predictions: dict[object, int] = {}
    for start in range(train_size, len(y), test_size):
        stop = min(start + test_size, len(y))
        majority = int(y.iloc[:start].mode().iloc[0])
        for position in range(start, stop):
            predictions[y.index[position]] = majority
    return pd.Series(predictions, name="majority_class_baseline")


def persistence_baseline(y: pd.Series, train_size: int) -> pd.Series:
    """Predict the previous session's already-settled actual class.

    Causal by construction (uses only y.shift(1)); train_size only trims the
    series to the same test region walk_forward_validate covers, so it lines
    up with the model's predictions for a fair comparison.
    """
    return y.shift(1).iloc[train_size:].rename("persistence_baseline")


def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Moving block bootstrap distribution of the mean of ``values``.

    Resamples contiguous blocks (not single points) with replacement so
    within-block serial correlation is preserved in each bootstrap draw,
    unlike an i.i.d. bootstrap or a plain binomial confidence interval.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < block_size:
        raise ValueError("not enough rows for the requested block size")
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(0, n - block_size + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        resampled = np.concatenate([values[s : s + block_size] for s in chosen])[:n]
        means[i] = resampled.mean()
    return {
        "observed_mean": float(values.mean()),
        "bootstrap_mean": float(means.mean()),
        "ci_lower_2.5": float(np.percentile(means, 2.5)),
        "ci_upper_97.5": float(np.percentile(means, 97.5)),
    }


def one_sample_edge_test(
    values: np.ndarray,
    baseline_rate: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Percentile-bootstrap test of H0: true mean(values) <= baseline_rate.

    ``p_value_not_above_baseline`` is the fraction of block-bootstrap
    resamples of the *observed* series whose mean falls at or below
    ``baseline_rate``. A small value means the observed advantage over the
    baseline is unlikely to be a fluke of this particular test window.
    """
    ci = block_bootstrap_ci(values, block_size, n_bootstrap, seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(0, n - block_size + 1)
    rng = np.random.default_rng(seed)
    at_or_below = 0
    for _ in range(n_bootstrap):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        resampled = np.concatenate([values[s : s + block_size] for s in chosen])[:n]
        if resampled.mean() <= baseline_rate:
            at_or_below += 1
    return {**ci, "baseline_rate": float(baseline_rate), "p_value_not_above_baseline": at_or_below / n_bootstrap}


def paired_edge_test(
    model_correct: np.ndarray,
    baseline_correct: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Paired one_sample_edge_test on (model_correct - baseline_correct).

    Pairing controls for days that are simply easy or hard for everyone
    (both the model and the baseline face the same market on the same day),
    which is more powerful than comparing the two accuracies independently.
    """
    diff = np.asarray(model_correct, dtype=float) - np.asarray(baseline_correct, dtype=float)
    return one_sample_edge_test(diff, 0.0, block_size, n_bootstrap, seed)


def per_class_report(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Precision/recall/support per class, to catch a model that is just
    always predicting one class and coasting on that class's prevalence.
    """
    report: dict[str, dict[str, float]] = {}
    for cls in sorted(predictions["actual_class"].unique()):
        predicted_positive = predictions["predicted_class"] == cls
        actual_positive = predictions["actual_class"] == cls
        true_positive = int((predicted_positive & actual_positive).sum())
        support = int(actual_positive.sum())
        predicted_count = int(predicted_positive.sum())
        report[str(cls)] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / support if support else 0.0,
        }
    return report


def evaluate_edge_significance(
    predictions: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    test_size: int = 21,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    majority = majority_class_baseline(y, train_size, test_size).reindex(predictions.index)
    persistence = persistence_baseline(y, train_size).reindex(predictions.index)

    model_correct = (predictions["predicted_class"] == predictions["actual_class"]).to_numpy()
    majority_correct = (majority == predictions["actual_class"]).to_numpy()
    persistence_correct = (persistence == predictions["actual_class"]).to_numpy()
    uniform_random_rate = 1.0 / len(y.unique())

    return {
        "n": len(predictions),
        "model_accuracy": float(model_correct.mean()),
        "majority_class_baseline_accuracy": float(majority_correct.mean()),
        "persistence_baseline_accuracy": float(persistence_correct.mean()),
        "uniform_random_baseline_rate": uniform_random_rate,
        "per_class": per_class_report(predictions),
        "model_accuracy_block_bootstrap_ci": block_bootstrap_ci(
            model_correct.astype(float), block_size, n_bootstrap, seed
        ),
        "model_vs_uniform_random": one_sample_edge_test(
            model_correct.astype(float), uniform_random_rate, block_size, n_bootstrap, seed
        ),
        "model_vs_majority_class_baseline": paired_edge_test(
            model_correct, majority_correct, block_size, n_bootstrap, seed
        ),
        "model_vs_persistence_baseline": paired_edge_test(
            model_correct, persistence_correct, block_size, n_bootstrap, seed + 1
        ),
    }


def diagnose_edge_significance(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    target: str,
    *,
    train_size: int = 504,
    test_size: int = 21,
    neutral_band: float = 0.005,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    features = build_leading_features(close, volume)
    X, y, returns = build_training_set(features, close[target], neutral_band=neutral_band)
    result = walk_forward_validate(
        X, y, returns, train_size=train_size, test_size=test_size, lambda_reg=0.10, variance_target=0.90, min_samples=60
    )
    significance = evaluate_edge_significance(
        result.predictions, y, train_size, test_size, block_size, n_bootstrap, seed
    )
    return {
        "target": target,
        "neutral_band": neutral_band,
        "walk_forward_metrics": dict(result.metrics),
        "edge_significance": significance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only test of statistical edge significance against naive baselines (changes no prediction logic)"
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end-exclusive", default="auto")
    parser.add_argument("--targets", default="SPY,QQQ")
    parser.add_argument("--neutral-band", type=float, default=0.005)
    parser.add_argument("--output", default="artifacts/edge")
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
        report = diagnose_edge_significance(dataset.close, dataset.volume, target, neutral_band=args.neutral_band)
        (output / f"{target.lower()}_edge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
