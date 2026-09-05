"""SMH・IWM・セクター出来高による独立した先行信号候補。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sector_rotation import SECTOR_ETFS

INTERNAL_PRICE_SYMBOLS = ("SMH", "IWM", "SPY", "XLK")
INTERNAL_WINDOW = 252
RIDGE_ALPHA = 10.0


def build_market_internal_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """当日引け後に確定する市場内部・出来高特徴量を作る。

    出来高は当日値を過去20営業日の平均と比較する。基準平均には当日を
    含めず、未来方向の補完も行わない。
    """
    required_close = set(INTERNAL_PRICE_SYMBOLS) | set(SECTOR_ETFS)
    missing_close = sorted(required_close - set(close.columns))
    missing_volume = sorted(set(SECTOR_ETFS) - set(volume.columns))
    if missing_close:
        raise ValueError(f"missing internal close series: {missing_close}")
    if missing_volume:
        raise ValueError(f"missing sector volume series: {missing_volume}")
    if not close.index.is_monotonic_increasing or not volume.index.is_monotonic_increasing:
        raise ValueError("close and volume must be sorted in ascending time order")

    numeric_close = close.apply(pd.to_numeric, errors="coerce")
    returns = numeric_close.pct_change(fill_method=None)
    features: dict[str, pd.Series] = {
        "return_smh": returns["SMH"],
        "return_iwm": returns["IWM"],
        "spread_smh_xlk": returns["SMH"] - returns["XLK"],
        "spread_iwm_spy": returns["IWM"] - returns["SPY"],
    }

    volume_shocks: list[pd.Series] = []
    aligned_volume = volume.reindex(close.index)
    for symbol in SECTOR_ETFS:
        observed = pd.to_numeric(aligned_volume[symbol], errors="coerce").where(
            lambda values: values > 0
        )
        log_volume = np.log(observed)
        past_baseline = (
            log_volume.dropna().shift(1).rolling(20, min_periods=20).mean().reindex(close.index)
        )
        shock = (log_volume - past_baseline).rename(f"volume_shock_{symbol}")
        features[shock.name] = shock
        volume_shocks.append(shock)
    features["sector_volume_breadth"] = pd.concat(volume_shocks, axis=1).mean(
        axis=1, skipna=False
    )
    return pd.DataFrame(features, index=close.index).replace([np.inf, -np.inf], np.nan)


class MarketInternalRidgeModel:
    """市場内部特徴から翌日11セクターを予測する固定リッジ回帰。"""

    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)

    def fit(self, features: pd.DataFrame, next_returns: pd.DataFrame) -> "MarketInternalRidgeModel":
        if len(features) != len(next_returns) or len(features) < 2:
            raise ValueError("features and next_returns need equal non-trivial rows")
        if list(next_returns.columns) != list(SECTOR_ETFS):
            raise ValueError(f"next return columns must be: {list(SECTOR_ETFS)}")
        if features.isna().any().any() or next_returns.isna().any().any():
            raise ValueError("ridge training data must be complete")
        self.feature_columns_ = list(features.columns)
        self.x_mean_ = features.mean(axis=0)
        self.x_std_ = features.std(axis=0, ddof=0).replace(0.0, 1.0)
        self.y_mean_ = next_returns.mean(axis=0)
        self.y_std_ = next_returns.std(axis=0, ddof=0).replace(0.0, 1.0)
        x = ((features - self.x_mean_) / self.x_std_).to_numpy(dtype=float)
        y = ((next_returns - self.y_mean_) / self.y_std_).to_numpy(dtype=float)
        penalty = np.eye(x.shape[1]) * self.alpha
        self.coefficients_ = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict(self, row: pd.Series) -> pd.Series:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("fit must be called before predict")
        if list(row.index) != self.feature_columns_:
            raise ValueError("prediction feature order differs from training")
        if row.isna().any():
            raise ValueError("prediction features must be complete")
        x = ((row.astype(float) - self.x_mean_) / self.x_std_).to_numpy()
        return pd.Series(x @ self.coefficients_, index=SECTOR_ETFS, name="internal_score")
