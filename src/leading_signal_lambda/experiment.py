from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .collector import DailyMarketCollector
from .signals import build_leading_features, build_training_set
from .validation import walk_forward_validate


@dataclass(frozen=True)
class TargetReport:
    target: str
    rows: int
    first_date: str
    last_date: str
    strategy_annualized_return: float
    benchmark_annualized_return: float
    annualized_excess_return: float
    strategy_max_drawdown: float
    benchmark_max_drawdown: float
    direction_accuracy: float
    trade_win_rate: float
    trade_coverage: float
    ending_equity: float
    benchmark_ending_equity: float


def _benchmark_metrics(returns: pd.Series) -> tuple[float, float, float]:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(equity) / 252.0, 1.0 / 252.0)
    annualized = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    return annualized, float(drawdown.min()), float(equity.iloc[-1])


def validate_target(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    target: str,
    output: Path,
    train_size: int = 504,
    transaction_cost_bps: float = 5.0,
) -> TargetReport:
    features = build_leading_features(close, volume)
    X, y, next_returns = build_training_set(features, close[target])
    result = walk_forward_validate(
        X,
        y,
        next_returns,
        train_size=train_size,
        test_size=21,
        transaction_cost_bps=transaction_cost_bps,
        lambda_reg=0.10,
        variance_target=0.90,
        min_samples=60,
    )
    result.predictions.to_csv(output / f"{target.lower()}_frozen_predictions.csv", index_label="date")
    benchmark = next_returns.reindex(result.predictions.index)
    benchmark_annualized, benchmark_drawdown, benchmark_equity = _benchmark_metrics(benchmark)
    metrics = result.metrics
    return TargetReport(
        target=target,
        rows=len(result.predictions),
        first_date=str(result.predictions.index.min().date()),
        last_date=str(result.predictions.index.max().date()),
        strategy_annualized_return=metrics["annualized_return"],
        benchmark_annualized_return=benchmark_annualized,
        annualized_excess_return=metrics["annualized_return"] - benchmark_annualized,
        strategy_max_drawdown=metrics["max_drawdown"],
        benchmark_max_drawdown=benchmark_drawdown,
        direction_accuracy=metrics["direction_accuracy"],
        trade_win_rate=metrics["trade_win_rate"],
        trade_coverage=metrics["trade_coverage"],
        ending_equity=metrics["ending_equity"],
        benchmark_ending_equity=benchmark_equity,
    )


def run_experiment(start: str, end_exclusive: str, output_dir: str | Path) -> list[TargetReport]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = DailyMarketCollector().collect(start, end_exclusive)
    dataset.save_csv(output / "raw")

    quality = {
        "rows": len(dataset.close),
        "first_date": str(dataset.close.index.min().date()),
        "last_date": str(dataset.close.index.max().date()),
        "missing_fraction": {
            column: float(dataset.close[column].isna().mean()) for column in dataset.close.columns
        },
        "duplicate_dates": int(dataset.close.index.duplicated().sum()),
        "lambda_reg": 0.10,
        "pca_variance_target": 0.90,
    }
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    reports = [validate_target(dataset.close, dataset.volume, target, output) for target in ("SPY", "QQQ")]
    (output / "validation_summary.json").write_text(
        json.dumps([asdict(report) for report in reports], indent=2), encoding="utf-8"
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe walk-forward validation")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end-exclusive", default=date.today().isoformat())
    parser.add_argument("--output", default="artifacts/validation")
    args = parser.parse_args()
    reports = run_experiment(args.start, args.end_exclusive, args.output)
    for report in reports:
        print(json.dumps(asdict(report), ensure_ascii=False))


if __name__ == "__main__":
    main()
