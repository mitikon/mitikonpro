"""最大歪みETFの選択と買い・持越し・売り状態管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_pca_sub import PAPER_WINDOW
from .sector_rotation import (
    SECTOR_ETFS,
    LaggedSectorPcaSubModel,
    make_lagged_sector_pairs,
)

BUY = "BUY"
HOLD = "HOLD"
TAKE_PROFIT = "TAKE_PROFIT"
EXIT_RISK = "EXIT_RISK"
EXIT_TIME = "EXIT_TIME"
EXIT_SIGNAL = "EXIT_SIGNAL"
STAY = "STAY"


@dataclass(frozen=True)
class PositionState:
    symbol: str
    entry_price: float
    entry_date: pd.Timestamp
    days_held: int = 0


@dataclass(frozen=True)
class PositionDecision:
    action: str
    reason: str


def decide_position_action(
    position: PositionState | None,
    *,
    top_symbol: str,
    top_score: float,
    top_lambda: float,
    held_score: float = 0.0,
    held_lambda: float = 0.0,
    unrealized_return: float = 0.0,
    entry_threshold: float = 0.25,
    take_profit: float = 0.10,
    stop_loss: float = 0.05,
    max_holding_days: int = 15,
) -> PositionDecision:
    """予測と保有状態から、翌営業日寄付の操作を決める。"""
    if position is None:
        if top_lambda > 0.0 and top_score >= entry_threshold:
            return PositionDecision(BUY, f"largest positive distortion: {top_symbol}")
        return PositionDecision(STAY, "no positive validated edge")
    if unrealized_return >= take_profit:
        return PositionDecision(TAKE_PROFIT, "profit target reached")
    if unrealized_return <= -stop_loss:
        return PositionDecision(EXIT_RISK, "loss limit reached")
    if position.days_held >= max_holding_days:
        return PositionDecision(EXIT_TIME, "maximum holding period reached")
    if held_lambda <= 0.0 or held_score <= 0.0:
        action = TAKE_PROFIT if unrealized_return > 0.0 else EXIT_SIGNAL
        return PositionDecision(action, "held ETF signal lost")
    return PositionDecision(HOLD, "held ETF retains positive edge")


@dataclass(frozen=True)
class SectorRotationBacktestResult:
    decisions: pd.DataFrame
    rankings: pd.DataFrame
    metrics: dict[str, float]

    def save_frozen(self, directory: str | Path) -> tuple[Path, Path, Path]:
        destination = Path(directory)
        paths = (
            destination / "sector_rotation_decisions.csv",
            destination / "sector_rotation_rankings.csv",
            destination / "sector_rotation_metrics.json",
        )
        if any(path.exists() for path in paths):
            raise FileExistsError("frozen sector-rotation result already exists")
        destination.mkdir(parents=True, exist_ok=True)
        self.decisions.to_csv(paths[0], index_label="signal_date")
        self.rankings.to_csv(paths[1])
        import json

        paths[2].write_text(
            json.dumps(self.metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        return paths


def _past_correlation(predicted: list[float], realized: list[float], minimum: int) -> float:
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


def run_sector_rotation_backtest(
    sector_returns: pd.DataFrame,
    prior_sector_returns: pd.DataFrame,
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,
    calibration_observations: int = 60,
    entry_threshold: float = 0.25,
    take_profit: float = 0.10,
    stop_loss: float = 0.05,
    max_holding_days: int = 15,
    transaction_cost_bps: float = 5.0,
) -> SectorRotationBacktestResult:
    """日次確定値のみで翌営業日寄付の売買状態を検証する。"""
    expected = list(SECTOR_ETFS)
    for name, frame in (
        ("sector_returns", sector_returns),
        ("prior_sector_returns", prior_sector_returns),
        ("open_prices", open_prices),
        ("close_prices", close_prices),
    ):
        if list(frame.columns) != expected:
            raise ValueError(f"{name} columns must be: {expected}")
    if len(sector_returns) <= PAPER_WINDOW + 2:
        raise ValueError("not enough observations for sector rotation")
    if prior_sector_returns.index.max() >= sector_returns.index[PAPER_WINDOW + 1]:
        raise ValueError("prior period must end before the first sector signal")
    if calibration_observations < 2 or max_holding_days < 1:
        raise ValueError("invalid calibration or holding period")
    if min(entry_threshold, take_profit, stop_loss, transaction_cost_bps) < 0:
        raise ValueError("trade thresholds and cost must be non-negative")
    if sector_returns.isna().any().any() or prior_sector_returns.isna().any().any():
        raise ValueError("return histories must be complete")

    dates = sector_returns.index
    opens = open_prices.reindex(dates).loc[:, expected]
    closes = close_prices.reindex(dates).loc[:, expected]
    if opens.isna().any().any() or closes.isna().any().any():
        raise ValueError("open and close prices must cover every evaluation date")
    prior_pairs = make_lagged_sector_pairs(prior_sector_returns)
    history_prediction = {symbol: [] for symbol in SECTOR_ETFS}
    history_realized = {symbol: [] for symbol in SECTOR_ETFS}
    cost_rate = transaction_cost_bps / 10_000.0

    position: PositionState | None = None
    last_mark_price: float | None = None
    decision_records: list[dict[str, object]] = []
    ranking_records: list[dict[str, object]] = []

    for current in range(PAPER_WINDOW + 1, len(sector_returns) - 1):
        history = sector_returns.iloc[current - PAPER_WINDOW - 1 : current]
        rolling_pairs = make_lagged_sector_pairs(history)
        signal_date = dates[current]
        trade_date = dates[current + 1]
        model = LaggedSectorPcaSubModel().fit(rolling_pairs, prior_pairs)
        signal = model.predict(sector_returns.iloc[current])

        lambdas: dict[str, float] = {}
        effective: dict[str, float] = {}
        for symbol in SECTOR_ETFS:
            value = _past_correlation(
                history_prediction[symbol],
                history_realized[symbol],
                calibration_observations,
            )
            lambdas[symbol] = value
            raw = float(signal.next_standardized_prediction[symbol])
            effective[symbol] = raw * max(value, 0.0) if np.isfinite(value) else 0.0

        ranked = sorted(
            SECTOR_ETFS,
            key=lambda symbol: (effective[symbol], signal.next_standardized_prediction[symbol]),
            reverse=True,
        )
        top_symbol = ranked[0]
        for rank, symbol in enumerate(ranked, start=1):
            ranking_records.append(
                {
                    "signal_date": signal_date,
                    "symbol": symbol,
                    "rank": rank,
                    "raw_score": float(signal.next_standardized_prediction[symbol]),
                    "signal_lambda": lambdas[symbol],
                    "effective_score": effective[symbol],
                    "current_distortion": float(signal.current_distortion[symbol]),
                }
            )

        held_before = position.symbol if position else None
        holding_return = 0.0
        unrealized = 0.0
        held_score = 0.0
        held_lambda = 0.0
        if position is not None:
            next_open = float(opens.loc[trade_date, position.symbol])
            if last_mark_price is None:
                raise RuntimeError("held position is missing its mark price")
            holding_return = next_open / last_mark_price - 1.0
            unrealized = float(closes.loc[signal_date, position.symbol]) / position.entry_price - 1.0
            held_score = float(signal.next_standardized_prediction[position.symbol])
            held_lambda = lambdas[position.symbol]
            position = PositionState(
                position.symbol, position.entry_price, position.entry_date, position.days_held + 1
            )

        decision = decide_position_action(
            position,
            top_symbol=top_symbol,
            top_score=float(signal.next_standardized_prediction[top_symbol]),
            top_lambda=lambdas[top_symbol],
            held_score=held_score,
            held_lambda=held_lambda,
            unrealized_return=unrealized,
            entry_threshold=entry_threshold,
            take_profit=take_profit,
            stop_loss=stop_loss,
            max_holding_days=max_holding_days,
        )
        cost = 0.0
        realized_trade_return = float("nan")
        if decision.action == BUY:
            entry_price = float(opens.loc[trade_date, top_symbol])
            position = PositionState(top_symbol, entry_price, trade_date, 0)
            last_mark_price = entry_price
            cost = cost_rate
        elif decision.action in {TAKE_PROFIT, EXIT_RISK, EXIT_TIME, EXIT_SIGNAL}:
            if position is None:
                raise RuntimeError("sell decision requires a position")
            exit_price = float(opens.loc[trade_date, position.symbol])
            realized_trade_return = exit_price / position.entry_price - 1.0 - 2.0 * cost_rate
            cost = cost_rate
            position = None
            last_mark_price = None
        elif position is not None:
            last_mark_price = float(opens.loc[trade_date, position.symbol])

        strategy_return = holding_return - cost
        decision_records.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "action": decision.action,
                "reason": decision.reason,
                "top_candidate": top_symbol,
                "top_raw_score": float(signal.next_standardized_prediction[top_symbol]),
                "top_signal_lambda": lambdas[top_symbol],
                "top_effective_score": effective[top_symbol],
                "held_before": held_before,
                "position_after": position.symbol if position else None,
                "unrealized_return": unrealized,
                "realized_trade_return": realized_trade_return,
                "strategy_return": strategy_return,
            }
        )

        for symbol in SECTOR_ETFS:
            actual_return = float(sector_returns.iloc[current + 1][symbol])
            target_name = f"next:{symbol}"
            realized_z = (actual_return - model.mean_[target_name]) / model.std_[target_name]
            history_prediction[symbol].append(
                float(signal.next_standardized_prediction[symbol])
            )
            history_realized[symbol].append(float(realized_z))

    decisions = pd.DataFrame(decision_records).set_index("signal_date")
    rankings = pd.DataFrame(ranking_records).set_index(["signal_date", "symbol"])
    strategy = decisions["strategy_return"]
    if benchmark_returns is None:
        benchmark = sector_returns.mean(axis=1).reindex(decisions["trade_date"].to_list())
        benchmark.index = decisions.index
    else:
        benchmark = benchmark_returns.reindex(decisions["trade_date"].to_list())
        benchmark.index = decisions.index
    if benchmark.isna().any():
        raise ValueError("benchmark must cover every trade date")
    strategy_equity = (1.0 + strategy).cumprod()
    benchmark_equity = (1.0 + benchmark).cumprod()
    years = len(strategy) / 252.0
    closed = decisions["realized_trade_return"].dropna()
    annualized = float(strategy_equity.iloc[-1] ** (1.0 / years) - 1.0)
    benchmark_annualized = float(benchmark_equity.iloc[-1] ** (1.0 / years) - 1.0)
    metrics = {
        "annualized_return": annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "annualized_excess_return": annualized - benchmark_annualized,
        "maximum_drawdown": _maximum_drawdown(strategy),
        "ending_equity": float(strategy_equity.iloc[-1]),
        "benchmark_ending_equity": float(benchmark_equity.iloc[-1]),
        "completed_trades": float(len(closed)),
        "profitable_trade_rate": float((closed > 0).mean()) if len(closed) else 0.0,
        "buy_signals": float((decisions["action"] == BUY).sum()),
        "hold_signals": float((decisions["action"] == HOLD).sum()),
        "take_profit_signals": float((decisions["action"] == TAKE_PROFIT).sum()),
        "risk_exit_signals": float(
            decisions["action"].isin({EXIT_RISK, EXIT_TIME, EXIT_SIGNAL}).sum()
        ),
        "observations": float(len(decisions)),
    }
    return SectorRotationBacktestResult(decisions, rankings, metrics)
