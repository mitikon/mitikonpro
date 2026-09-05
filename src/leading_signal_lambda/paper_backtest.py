"""論文準拠PCA SUBのウォークフォワード検証。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .paper_pca_sub import (
    JAPAN_SECTORS,
    PAPER_WINDOW,
    US_SECTORS,
    PaperPcaSubModel,
    quantile_long_short_weights,
)


@dataclass(frozen=True)
class PaperBacktestResult:
    daily: pd.DataFrame
    frozen_scores: pd.DataFrame
    metrics: dict[str, float]


def _maximum_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def run_paper_backtest(
    combined_close_returns: pd.DataFrame,
    japan_open_to_close: pd.DataFrame,
    full_prior_returns: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> PaperBacktestResult:
    """米国t日終値情報で日本t+1日寄付から引けを売買する。

    すべて同一の共通営業日インデックスに整列済みであることを要求する。
    長期事前標本は評価開始より前に終了し、推定期間と重複させない。
    """
    columns = list(US_SECTORS + JAPAN_SECTORS)
    if list(combined_close_returns.columns) != columns:
        raise ValueError(f"combined_close_returns columns must be: {columns}")
    if list(japan_open_to_close.columns) != list(JAPAN_SECTORS):
        raise ValueError(f"japan_open_to_close columns must be: {list(JAPAN_SECTORS)}")
    if list(full_prior_returns.columns) != columns:
        raise ValueError(f"full_prior_returns columns must be: {columns}")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    joint_index = combined_close_returns.index.intersection(japan_open_to_close.index)
    close_returns = combined_close_returns.loc[joint_index, columns].dropna()
    target = japan_open_to_close.loc[close_returns.index, list(JAPAN_SECTORS)]
    valid = ~target.isna().any(axis=1)
    close_returns = close_returns.loc[valid]
    target = target.loc[valid]
    if len(close_returns) <= PAPER_WINDOW + 1:
        raise ValueError("not enough aligned observations for a paper backtest")
    if full_prior_returns.isna().any().any():
        raise ValueError("full_prior_returns must contain no missing values")

    first_signal_date = close_returns.index[PAPER_WINDOW]
    if full_prior_returns.index.max() >= first_signal_date:
        raise ValueError("full prior period must end before the first evaluation signal")

    records: list[dict[str, object]] = []
    score_records: list[pd.Series] = []
    previous_weights = pd.Series(0.0, index=JAPAN_SECTORS)
    cost_rate = transaction_cost_bps / 10_000.0

    for position in range(PAPER_WINDOW, len(close_returns) - 1):
        rolling = close_returns.iloc[position - PAPER_WINDOW : position]
        signal_date = close_returns.index[position]
        trade_date = close_returns.index[position + 1]

        model = PaperPcaSubModel().fit(rolling, full_prior_returns)
        signal = model.predict(close_returns.iloc[position].loc[list(US_SECTORS)])
        scores = signal.japan_standardized_prediction.rename(signal_date)
        weights = quantile_long_short_weights(scores)

        turnover = float((weights - previous_weights).abs().sum())
        gross_return = float(weights @ target.iloc[position + 1])
        cost = turnover * cost_rate
        net_return = gross_return - cost
        records.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "net_return": net_return,
            }
        )
        score_records.append(scores)
        previous_weights = weights

    daily = pd.DataFrame(records).set_index("trade_date")
    frozen_scores = pd.DataFrame(score_records)
    frozen_scores.index.name = "signal_date"
    returns = daily["net_return"]
    annualized_return = float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)
    annualized_risk = float(returns.std(ddof=0) * np.sqrt(252.0))
    metrics = {
        "annualized_return": annualized_return,
        "annualized_risk": annualized_risk,
        "return_risk_ratio": (
            annualized_return / annualized_risk if annualized_risk > 0 else float("nan")
        ),
        "maximum_drawdown": _maximum_drawdown(returns),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "observations": float(len(daily)),
    }
    return PaperBacktestResult(daily=daily, frozen_scores=frozen_scores, metrics=metrics)
