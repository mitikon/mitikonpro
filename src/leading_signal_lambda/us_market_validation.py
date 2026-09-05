"""米国市場先行シグナル予測λの時系列検証。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_pca_sub import PAPER_WINDOW
from .us_market_pca_sub import (
    US_LEADING_SECTORS,
    US_MARKET_TARGETS,
    US_MODEL_COLUMNS,
    USMarketPcaSubModel,
)


@dataclass(frozen=True)
class USMarketBacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float]]

    def save_frozen(self, path: str | Path) -> Path:
        """予測記録を新規保存し、既存ファイルの上書きを拒否する。"""
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"frozen prediction already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(destination)
        return destination


def _past_correlation(predicted: list[float], realized: list[float], minimum: int) -> float:
    """現在の実績を含めず、過去に固定済みの予測だけでλを推定する。"""
    if len(predicted) < minimum:
        return float("nan")
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(realized, dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))


def _maximum_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def run_us_market_backtest(
    returns: pd.DataFrame,
    full_prior_returns: pd.DataFrame,
    *,
    calibration_observations: int = 60,
    signal_threshold: float = 0.25,
    transaction_cost_bps: float = 5.0,
) -> USMarketBacktestResult:
    """米国セクターt日からSPY・QQQのt+1日方向と収益性を検証する。"""
    expected = list(US_MODEL_COLUMNS)
    if list(returns.columns) != expected or list(full_prior_returns.columns) != expected:
        raise ValueError(f"returns and prior columns must be: {expected}")
    if returns.isna().any().any() or full_prior_returns.isna().any().any():
        raise ValueError("returns and prior must contain no missing values")
    if len(returns) <= PAPER_WINDOW + 1:
        raise ValueError("not enough observations for a US-market backtest")
    if calibration_observations < 2:
        raise ValueError("calibration_observations must be at least two")
    if signal_threshold < 0 or transaction_cost_bps < 0:
        raise ValueError("threshold and transaction cost must be non-negative")
    first_signal_date = returns.index[PAPER_WINDOW]
    if full_prior_returns.index.max() >= first_signal_date:
        raise ValueError("full prior period must end before the first evaluation signal")

    history_prediction = {target: [] for target in US_MARKET_TARGETS}
    history_realized = {target: [] for target in US_MARKET_TARGETS}
    previous_action = {target: 0 for target in US_MARKET_TARGETS}
    records: list[dict[str, object]] = []
    cost_rate = transaction_cost_bps / 10_000.0

    for position in range(PAPER_WINDOW, len(returns) - 1):
        rolling = returns.iloc[position - PAPER_WINDOW : position]
        signal_date = returns.index[position]
        trade_date = returns.index[position + 1]
        model = USMarketPcaSubModel().fit(rolling, full_prior_returns)
        signal = model.predict(
            returns.iloc[position].loc[list(US_LEADING_SECTORS)]
        )

        for target in US_MARKET_TARGETS:
            raw_score = float(signal.target_standardized_prediction[target])
            realized_return = float(returns.iloc[position + 1][target])
            realized_z = float(
                (realized_return - model.mean_[target]) / model.std_[target]
            )
            signal_lambda = _past_correlation(
                history_prediction[target],
                history_realized[target],
                calibration_observations,
            )
            effective_score = (
                raw_score * max(signal_lambda, 0.0)
                if np.isfinite(signal_lambda)
                else 0.0
            )
            action = 0
            if (
                np.isfinite(signal_lambda)
                and signal_lambda > 0.0
                and abs(raw_score) >= signal_threshold
            ):
                action = 1 if raw_score > 0 else -1

            turnover = abs(action - previous_action[target])
            cost = turnover * cost_rate
            strategy_return = action * realized_return - cost
            records.append(
                {
                    "signal_date": signal_date,
                    "trade_date": trade_date,
                    "target": target,
                    "raw_score": raw_score,
                    "signal_lambda": signal_lambda,
                    "effective_score": effective_score,
                    "action": action,
                    "realized_return": realized_return,
                    "realized_z": realized_z,
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "strategy_return": strategy_return,
                }
            )
            history_prediction[target].append(raw_score)
            history_realized[target].append(realized_z)
            previous_action[target] = action

    predictions = pd.DataFrame(records).set_index(["signal_date", "target"])
    metrics: dict[str, dict[str, float]] = {}
    for target in US_MARKET_TARGETS:
        frame = predictions.xs(target, level="target")
        strategy = frame["strategy_return"]
        benchmark = frame["realized_return"]
        strategy_equity = (1.0 + strategy).cumprod()
        benchmark_equity = (1.0 + benchmark).cumprod()
        years = len(frame) / 252.0
        annualized = float(strategy_equity.iloc[-1] ** (1.0 / years) - 1.0)
        benchmark_annualized = float(benchmark_equity.iloc[-1] ** (1.0 / years) - 1.0)
        traded = frame["action"] != 0
        lambda_history = frame["signal_lambda"].dropna()
        metrics[target] = {
            "annualized_return": annualized,
            "benchmark_annualized_return": benchmark_annualized,
            "annualized_excess_return": annualized - benchmark_annualized,
            "maximum_drawdown": _maximum_drawdown(strategy),
            "ending_equity": float(strategy_equity.iloc[-1]),
            "benchmark_ending_equity": float(benchmark_equity.iloc[-1]),
            "direction_accuracy": (
                float(
                    (
                        np.sign(frame.loc[traded, "realized_return"])
                        == frame.loc[traded, "action"]
                    ).mean()
                )
                if traded.any()
                else 0.0
            ),
            "trade_coverage": float(traded.mean()),
            "last_signal_lambda": (
                float(lambda_history.iloc[-1]) if not lambda_history.empty else 0.0
            ),
            "observations": float(len(frame)),
        }
    return USMarketBacktestResult(predictions=predictions, metrics=metrics)
