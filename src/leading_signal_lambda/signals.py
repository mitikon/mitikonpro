from __future__ import annotations

import numpy as np
import pandas as pd

from .relative_strength_feature import RELATIVE_STRENGTH_PERIODS, build_relative_strength_features


REQUIRED_SYMBOLS = ("SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP")
MIN_SERIES_OBSERVATIONS = 60
MAX_SERIES_STALENESS_ROWS = 5


def _has_usable_history(series: pd.Series, *, positive_only: bool = False) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.gt(0) if positive_only else numeric.notna()
    positions = np.flatnonzero(valid.to_numpy())
    if len(positions) < MIN_SERIES_OBSERVATIONS:
        return False
    return len(series) - 1 - int(positions[-1]) <= MAX_SERIES_STALENESS_ROWS


RELATIVE_STRENGTH_FEATURE_SET = ("level", "velocity3", "cross50", "extreme_state")


def build_leading_features(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    *,
    feature_lags: int = 5,
    relative_strength_periods: tuple[int, ...] = RELATIVE_STRENGTH_PERIODS,
    relative_strength_feature_set: tuple[str, ...] = RELATIVE_STRENGTH_FEATURE_SET,
    relative_strength_feature_weight: float = 1.0,
) -> pd.DataFrame:
    """日次終値から1～5日ラグと主要な市場内部乖離を作る。

    入力は日付昇順を前提とし、未来方向への補完（backfill）は一切行わない。
    """
    if not 1 <= int(feature_lags) <= 10:
        raise ValueError("feature_lags must be in [1, 10]")
    periods = tuple(int(period) for period in relative_strength_periods)
    if not periods or any(period < 2 or period > 60 for period in periods):
        raise ValueError("relative_strength_periods must contain values in [2, 60]")
    feature_set = tuple(str(value) for value in relative_strength_feature_set)
    unknown_features = set(feature_set) - set(RELATIVE_STRENGTH_FEATURE_SET)
    if not feature_set or unknown_features:
        raise ValueError(f"unsupported relative strength feature set: {sorted(unknown_features)}")
    if not 0.0 <= float(relative_strength_feature_weight) <= 3.0:
        raise ValueError("relative_strength_feature_weight must be in [0, 3]")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted in ascending time order")
    missing = set(REQUIRED_SYMBOLS) - set(close.columns)
    if missing:
        raise ValueError(f"missing required symbols: {sorted(missing)}")
    numeric_close = close.apply(pd.to_numeric, errors="coerce")
    unusable_required = sorted(
        symbol for symbol in REQUIRED_SYMBOLS if not _has_usable_history(numeric_close[symbol])
    )
    if unusable_required:
        raise ValueError(f"required symbols have insufficient close coverage: {unusable_required}")
    usable_close = sorted(
        symbol for symbol in numeric_close.columns if _has_usable_history(numeric_close[symbol])
    )
    numeric_close = numeric_close[usable_close]
    # The provider returns the union of calendars (NYSE, FX, futures, indices).
    # Calculate each symbol's return on its own observed sessions so that a US
    # holiday row created by FX does not make the next NYSE return disappear.
    returns = pd.DataFrame(
        {
            symbol: numeric_close[symbol]
            .dropna()
            .pct_change(fill_method=None)
            .reindex(numeric_close.index)
            for symbol in numeric_close.columns
        },
        index=numeric_close.index,
    )
    features: dict[str, pd.Series] = {}
    for symbol in numeric_close.columns:
        observed_returns = returns[symbol].dropna()
        for lag in range(1, int(feature_lags) + 1):
            # Lag by that market's observed sessions, not by union-calendar rows.
            features[f"ret_{symbol}_lag{lag}"] = observed_returns.shift(lag - 1).reindex(
                numeric_close.index
            )

    features["spread_smh_qqq"] = returns["SMH"] - returns["QQQ"]
    features["spread_rsp_spy"] = returns["RSP"] - returns["SPY"]
    features["spread_hyg_lqd"] = returns["HYG"] - returns["LQD"]
    features["spread_xly_xlp"] = returns["XLY"] - returns["XLP"]

    # Relative Strength Index is an observed feature family, never a fixed 70/30
    # trading rule. The daily fit/settlement loop relearns its usefulness from
    # next-session outcomes. This is unrelated to RSI (Recursive Self-Improvement).
    relative_strength_features = build_relative_strength_features(
        numeric_close, tuple(numeric_close.columns), periods=periods
    )
    selected_columns = [
        column
        for column in relative_strength_features.columns
        if any(column.endswith(f"_{feature}") for feature in feature_set)
    ]
    features.update(
        {
            column: relative_strength_features[column] * float(relative_strength_feature_weight)
            for column in selected_columns
        }
    )

    if {"VIX9D", "VIX3M"}.issubset(numeric_close.columns):
        features["vix_term_spread"] = numeric_close["VIX9D"] / numeric_close["VIX3M"] - 1.0
    if volume is not None:
        aligned = volume.reindex(close.index)
        for symbol in sorted(set(aligned.columns) & set(numeric_close.columns)):
            # Yahoo Finance returns zero or missing volume for indices, rates and FX.
            # Treat those as "volume unavailable", not as a numeric signal.  A single
            # unusable series must never invalidate every row in the training set.
            observed = pd.to_numeric(aligned[symbol], errors="coerce").where(lambda value: value > 0)
            if not _has_usable_history(observed, positive_only=True):
                continue
            # Compute the rolling baseline on that market's own observed sessions.
            # Otherwise one cross-market holiday poisons the following 20 rows.
            baseline = observed.dropna().rolling(20, min_periods=20).mean().reindex(close.index)
            features[f"volume_ratio_{symbol}"] = observed / baseline - 1.0
    return pd.DataFrame(features, index=close.index).replace([np.inf, -np.inf], np.nan)


def build_training_set(
    features: pd.DataFrame,
    target_close: pd.Series,
    neutral_band: float = 0.001,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """当日までの特徴と翌営業日リターンを整列し、リーク無しの教師データを作る。"""
    observed_target = target_close.astype(float).dropna()
    # Map each session to the return of the next observed target session. Union
    # calendar rows (FX-only days and US holidays) must not break the label.
    next_return = (
        observed_target.pct_change(fill_method=None).shift(-1).reindex(target_close.index)
    )
    labels = pd.Series(
        np.select([next_return > neutral_band, next_return < -neutral_band], [1, -1], default=0),
        index=next_return.index,
        dtype=int,
        name="target_class",
    )
    valid = features.notna().all(axis=1) & next_return.notna()
    return features.loc[valid], labels.loc[valid], next_return.loc[valid].rename("next_return")
