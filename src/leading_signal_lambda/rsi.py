"""Leakage-safe RSI features for the daily self-learning loop."""

from __future__ import annotations

import numpy as np
import pandas as pd


RSI_PERIODS = (5, 7, 14, 21)
RSI_FEATURE_VERSION = "rsi-self-learning-v1"


def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    """Return Wilder RSI using only observations available at each timestamp."""
    if period < 2:
        raise ValueError("RSI period must be at least 2")
    observed = pd.to_numeric(close, errors="coerce").dropna()
    delta = observed.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)
    rsi = rsi.mask((average_gain > 0.0) & (average_loss == 0.0), 100.0)
    return rsi.reindex(close.index)


def _signed_cross(values: pd.Series, threshold: float) -> pd.Series:
    previous = values.shift(1)
    return pd.Series(
        np.select(
            [(values >= threshold) & (previous < threshold),
             (values <= threshold) & (previous > threshold)],
            [1.0, -1.0],
            default=0.0,
        ),
        index=values.index,
        dtype=float,
    ).where(values.notna() & previous.notna())


def _extreme_state(values: pd.Series) -> pd.Series:
    """Encode 30/70 persistence and exits without treating them as trade rules."""
    previous = values.shift(1)
    state = pd.Series(
        np.select(
            [
                (values >= 70.0) & (previous >= 70.0),
                (values <= 30.0) & (previous <= 30.0),
                (values < 70.0) & (previous >= 70.0),
                (values > 30.0) & (previous <= 30.0),
                values >= 70.0,
                values <= 30.0,
            ],
            [2.0, -2.0, -1.0, 1.0, 0.5, -0.5],
            default=0.0,
        ),
        index=values.index,
        dtype=float,
    )
    return state.where(values.notna() & previous.notna())


def build_rsi_features(
    close: pd.DataFrame,
    symbols: list[str] | tuple[str, ...],
    periods: tuple[int, ...] = RSI_PERIODS,
) -> pd.DataFrame:
    """Build RSI states that the existing PCA/λ model learns from outcomes.

    Threshold events are inputs only. They never directly create LONG/SHORT
    decisions, and each series is calculated on its own observed sessions.
    """
    features: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol not in close:
            continue
        for period in periods:
            rsi = calculate_rsi(close[symbol], period)
            prefix = f"rsi{period}_{symbol}"
            features[f"{prefix}_level"] = (rsi - 50.0) / 50.0
            features[f"{prefix}_velocity3"] = rsi.diff(3) / 100.0
            features[f"{prefix}_cross50"] = _signed_cross(rsi, 50.0)
            features[f"{prefix}_extreme_state"] = _extreme_state(rsi)
    return pd.DataFrame(features, index=close.index).replace([np.inf, -np.inf], np.nan)
