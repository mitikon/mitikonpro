"""Leakage-safe relative-strength-index feature family for the daily learning loop.

This module computes the classic technical indicator known as the Relative
Strength Index. It is deliberately named without the bare "RSI" abbreviation:
in this repository "RSI" is reserved for Recursive Self-Improvement (see
``recursive_self_improvement.py``). The indicator here is only ever an input
feature to the PCA/lambda model, never a fixed 70/30 trading rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RELATIVE_STRENGTH_PERIODS = (5, 7, 14, 21)
RELATIVE_STRENGTH_FEATURE_VERSION = "relative-strength-feature-v1"


def calculate_relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    """Return Wilder's Relative Strength Index using only observations available at each timestamp."""
    if period < 2:
        raise ValueError("relative strength index period must be at least 2")
    observed = pd.to_numeric(close, errors="coerce").dropna()
    delta = observed.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    index = 100.0 - 100.0 / (1.0 + relative_strength)
    index = index.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)
    index = index.mask((average_gain > 0.0) & (average_loss == 0.0), 100.0)
    return index.reindex(close.index)


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


def build_relative_strength_features(
    close: pd.DataFrame,
    symbols: list[str] | tuple[str, ...],
    periods: tuple[int, ...] = RELATIVE_STRENGTH_PERIODS,
) -> pd.DataFrame:
    """Build relative-strength-index states that the PCA/lambda model learns from outcomes.

    Threshold events are inputs only. They never directly create LONG/SHORT
    decisions, and each series is calculated on its own observed sessions.
    """
    features: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol not in close:
            continue
        for period in periods:
            index = calculate_relative_strength_index(close[symbol], period)
            prefix = f"rs{period}_{symbol}"
            features[f"{prefix}_level"] = (index - 50.0) / 50.0
            features[f"{prefix}_velocity3"] = index.diff(3) / 100.0
            features[f"{prefix}_cross50"] = _signed_cross(index, 50.0)
            features[f"{prefix}_extreme_state"] = _extreme_state(index)
    return pd.DataFrame(features, index=close.index).replace([np.inf, -np.inf], np.nan)
