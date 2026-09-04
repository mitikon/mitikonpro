from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SYMBOLS = ("SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP")


def build_leading_features(close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """日次終値から1～5日ラグと主要な市場内部乖離を作る。

    入力は日付昇順を前提とし、未来方向への補完（backfill）は一切行わない。
    """
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted in ascending time order")
    missing = set(REQUIRED_SYMBOLS) - set(close.columns)
    if missing:
        raise ValueError(f"missing required symbols: {sorted(missing)}")
    returns = close.astype(float).pct_change(fill_method=None)
    features: dict[str, pd.Series] = {}
    for symbol in close.columns:
        for lag in range(1, 6):
            features[f"ret_{symbol}_lag{lag}"] = returns[symbol].shift(lag - 1)

    features["spread_smh_qqq"] = returns["SMH"] - returns["QQQ"]
    features["spread_rsp_spy"] = returns["RSP"] - returns["SPY"]
    features["spread_hyg_lqd"] = returns["HYG"] - returns["LQD"]
    features["spread_xly_xlp"] = returns["XLY"] - returns["XLP"]

    if {"VIX9D", "VIX3M"}.issubset(close.columns):
        features["vix_term_spread"] = close["VIX9D"] / close["VIX3M"] - 1.0
    if volume is not None:
        aligned = volume.reindex(close.index)
        for symbol in sorted(set(aligned.columns) & set(close.columns)):
            baseline = aligned[symbol].rolling(20, min_periods=20).mean()
            features[f"volume_ratio_{symbol}"] = aligned[symbol] / baseline - 1.0
    return pd.DataFrame(features, index=close.index).replace([np.inf, -np.inf], np.nan)


def build_training_set(
    features: pd.DataFrame,
    target_close: pd.Series,
    neutral_band: float = 0.001,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """当日までの特徴と翌営業日リターンを整列し、リーク無しの教師データを作る。"""
    next_return = target_close.astype(float).pct_change(fill_method=None).shift(-1)
    labels = pd.Series(
        np.select([next_return > neutral_band, next_return < -neutral_band], [1, -1], default=0),
        index=next_return.index,
        dtype=int,
        name="target_class",
    )
    valid = features.notna().all(axis=1) & next_return.notna()
    return features.loc[valid], labels.loc[valid], next_return.loc[valid].rename("next_return")

