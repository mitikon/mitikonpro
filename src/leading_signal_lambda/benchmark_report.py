"""Read-only dividend-adjusted buy-and-hold benchmark CAGR.

The 2026-09-22 core/satellite strategy discussion set the primary target as
"guarantee the same annualized return SP500 delivered over its own trailing
5-year window" (rather than any prediction-model return). This module
answers that question directly from real market data - no walk-forward, no
model, no prediction - so the target itself is measured from the same
provider (DailyMarketCollector, Adj Close = dividends reinvested) the rest
of this repository already trusts, rather than from anyone's memory of
what the index "roughly" did.

Invoked explicitly via ``leading-lambda-benchmark``; changes no prediction
or trading logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .collector import DailyMarketCollector


def buy_and_hold_annualized_return(close: pd.Series, start: str, end_inclusive: str) -> dict[str, object]:
    """Simple dividend-adjusted buy-and-hold CAGR over [start, end_inclusive].

    Uses whatever price series is passed in (DailyMarketCollector's Adj
    Close column reinvests dividends, the conventional basis for quoting an
    index's "annualized return"). Bounds to the first/last *observed*
    sessions inside the window, since start/end may fall on non-trading
    days.
    """
    series = close.dropna()
    window = series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end_inclusive))]
    if len(window) < 2:
        raise ValueError(f"not enough observed sessions between {start} and {end_inclusive}")
    start_date, end_date = window.index[0], window.index[-1]
    start_price, end_price = float(window.iloc[0]), float(window.iloc[-1])
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        raise ValueError("window must span a positive number of days")
    return {
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "start_price": start_price,
        "end_price": end_price,
        "years": years,
        "total_return": end_price / start_price - 1.0,
        "annualized_return": (end_price / start_price) ** (1.0 / years) - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only dividend-adjusted buy-and-hold benchmark CAGR (no model or trading logic)"
    )
    parser.add_argument("--targets", default="SPY")
    parser.add_argument("--start", required=True, help="inclusive YYYY-MM-DD")
    parser.add_argument("--end-inclusive", required=True, help="inclusive YYYY-MM-DD")
    parser.add_argument(
        "--collector-start",
        default=None,
        help="data fetch start passed to DailyMarketCollector; defaults to 30 days before --start",
    )
    parser.add_argument("--output", default="artifacts/benchmark")
    args = parser.parse_args()

    collector_start = args.collector_start or (pd.Timestamp(args.start) - pd.Timedelta(days=30)).date().isoformat()
    end_exclusive = (pd.Timestamp(args.end_inclusive) + pd.Timedelta(days=1)).date().isoformat()
    dataset = DailyMarketCollector().collect(collector_start, end_exclusive)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for target in args.targets.split(","):
        report = buy_and_hold_annualized_return(dataset.close[target], args.start, args.end_inclusive)
        report["target"] = target
        (output / f"{target.lower()}_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
