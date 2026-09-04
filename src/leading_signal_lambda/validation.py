from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import LeadingLambdaClassifier


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    next_returns: pd.Series,
    train_size: int = 252,
    test_size: int = 21,
    transaction_cost_bps: float = 5.0,
    **model_kwargs: float,
) -> WalkForwardResult:
    """拡大型ウォークフォワード検証。予測は各時点で固定され、後から書き換えない。"""
    common = X.index.intersection(y.index).intersection(next_returns.index)
    X, y, next_returns = X.loc[common], y.loc[common], next_returns.loc[common]
    if len(X) <= train_size:
        raise ValueError("not enough rows for the requested walk-forward split")

    records: list[dict[str, float | int | str | pd.Timestamp]] = []
    for start in range(train_size, len(X), test_size):
        stop = min(start + test_size, len(X))
        model = LeadingLambdaClassifier(**model_kwargs).fit(X.iloc[:start], y.iloc[:start])
        for position in range(start, stop):
            prediction = model.predict_one(X.iloc[position])
            records.append(
                {
                    "date": X.index[position],
                    "actual_class": int(y.iloc[position]),
                    "actual_return": float(next_returns.iloc[position]),
                    "predicted_class": prediction.predicted_class,
                    "action": prediction.action,
                    "confidence": prediction.confidence,
                    "edge": prediction.edge,
                }
            )
    frame = pd.DataFrame(records).set_index("date")
    position = frame["predicted_class"].astype(float)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * transaction_cost_bps / 10_000.0
    frame["strategy_return"] = position * frame["actual_return"] - cost
    traded = position != 0
    equity = (1.0 + frame["strategy_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(frame) / 252.0, 1.0 / 252.0)
    annualized = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    metrics = {
        "annualized_return": annualized,
        "max_drawdown": float(drawdown.min()),
        "trade_win_rate": float((frame.loc[traded, "strategy_return"] > 0).mean()) if traded.any() else 0.0,
        "trade_coverage": float(traded.mean()),
        "direction_accuracy": float((frame["predicted_class"] == frame["actual_class"]).mean()),
        "ending_equity": float(equity.iloc[-1]),
    }
    return WalkForwardResult(frame, metrics)

