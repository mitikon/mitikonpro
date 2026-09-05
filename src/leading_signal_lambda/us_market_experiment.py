"""米国市場PCA SUBを実データで検証するコマンド。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .collector import DailyMarketCollector, MarketDataset
from .market_calendar import NYSETradingCalendar
from .us_market_pca_sub import prepare_us_market_returns
from .us_market_validation import USMarketBacktestResult, run_us_market_backtest


def validate_us_market_dataset(
    dataset: MarketDataset,
    output_dir: str | Path,
    *,
    prior_end: str = "2021-12-31",
    evaluation_start: str = "2022-01-01",
    calibration_observations: int = 60,
    signal_threshold: float = 0.25,
    transaction_cost_bps: float = 5.0,
) -> USMarketBacktestResult:
    """事前期間と評価期間を分離し、結果と診断情報を保存する。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    returns = prepare_us_market_returns(dataset.close)
    prior_end_ts = pd.Timestamp(prior_end)
    evaluation_start_ts = pd.Timestamp(evaluation_start)
    if prior_end_ts >= evaluation_start_ts:
        raise ValueError("prior_end must be earlier than evaluation_start")

    prior = returns.loc[returns.index <= prior_end_ts]
    evaluation = returns.loc[returns.index >= evaluation_start_ts]
    if prior.empty or evaluation.empty:
        raise ValueError("prior or evaluation period has no complete observations")

    diagnostics = {
        "prior_first_date": str(prior.index.min().date()),
        "prior_last_date": str(prior.index.max().date()),
        "prior_rows": len(prior),
        "evaluation_first_date": str(evaluation.index.min().date()),
        "evaluation_last_date": str(evaluation.index.max().date()),
        "evaluation_rows": len(evaluation),
        "paper_lambda_prior": 0.90,
        "paper_window": 60,
        "paper_components": 3,
        "calibration_observations": calibration_observations,
        "signal_threshold": signal_threshold,
        "transaction_cost_bps": transaction_cost_bps,
    }
    (output / "us_pca_sub_data_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    result = run_us_market_backtest(
        evaluation,
        prior,
        calibration_observations=calibration_observations,
        signal_threshold=signal_threshold,
        transaction_cost_bps=transaction_cost_bps,
    )
    result.save_frozen(output / "us_pca_sub_frozen_predictions.csv")
    (output / "us_pca_sub_validation_summary.json").write_text(
        json.dumps(result.metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the US-market paper PCA SUB leading signal"
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end-exclusive", default="auto")
    parser.add_argument("--prior-end", default="2021-12-31")
    parser.add_argument("--evaluation-start", default="2022-01-01")
    parser.add_argument("--output", default="artifacts/us_pca_sub_validation")
    parser.add_argument(
        "--exceptional-closures", default="config/exceptional_nyse_closures.json"
    )
    args = parser.parse_args()

    if args.end_exclusive == "auto":
        completed = NYSETradingCalendar(
            exceptional_closures=args.exceptional_closures
        ).last_completed_session()
        end_exclusive = completed.end_exclusive.isoformat()
    else:
        end_exclusive = args.end_exclusive

    dataset = DailyMarketCollector().collect(args.start, end_exclusive)
    result = validate_us_market_dataset(
        dataset,
        args.output,
        prior_end=args.prior_end,
        evaluation_start=args.evaluation_start,
    )
    print(json.dumps(result.metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
