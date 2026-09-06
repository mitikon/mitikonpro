"""論文準拠PCA SUBを米国市場内リード・ラグへ移植した基準モデル。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .paper_pca_sub import (
    PAPER_COMPONENTS,
    PAPER_LAMBDA_PRIOR,
    PAPER_WINDOW,
    US_CYCLICAL,
    US_DEFENSIVE,
    US_SECTORS,
    _orthonormalize,
    build_prior_correlation,
    regularize_correlation,
    standardize_window,
)

US_LEADING_SECTORS = US_SECTORS
US_MARKET_TARGETS = ("SPY", "QQQ")
US_MODEL_COLUMNS = US_LEADING_SECTORS + US_MARKET_TARGETS


def prepare_us_market_returns(close: pd.DataFrame) -> pd.DataFrame:
    """確定終値から米国モデル用Close-to-Closeリターンを作る。"""
    missing = sorted(set(US_MODEL_COLUMNS) - set(close.columns))
    if missing:
        raise ValueError(f"missing US PCA SUB close series: {missing}")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted in ascending time order")
    numeric = close.loc[:, list(US_MODEL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return numeric.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()


def build_us_market_prior_basis() -> np.ndarray:
    """論文の3方向を米国セクター→指数の2ブロックへ対応させる。"""
    index = {name: position for position, name in enumerate(US_MODEL_COLUMNS)}
    n_assets = len(US_MODEL_COLUMNS)

    global_factor = np.ones(n_assets, dtype=float)
    block_spread = np.ones(n_assets, dtype=float)
    block_spread[len(US_LEADING_SECTORS) :] = -1.0

    cyclical_defensive = np.zeros(n_assets, dtype=float)
    for name in US_CYCLICAL:
        cyclical_defensive[index[name]] = 1.0
    for name in US_DEFENSIVE:
        cyclical_defensive[index[name]] = -1.0
    return _orthonormalize((global_factor, block_spread, cyclical_defensive))


@dataclass(frozen=True)
class USMarketSignal:
    leader_standardized: pd.Series
    factor: pd.Series
    target_standardized_prediction: pd.Series
    transmission: pd.DataFrame


class USMarketPcaSubModel:
    """米国11セクターから翌日のSPY・QQQを予測するPCA SUB。"""

    def __init__(self, lambda_prior: float = PAPER_LAMBDA_PRIOR) -> None:
        if not 0.0 <= lambda_prior <= 1.0:
            raise ValueError("lambda_prior must be between 0 and 1")
        self.lambda_prior = float(lambda_prior)
        self.columns = US_MODEL_COLUMNS

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        if list(frame.columns) != list(US_MODEL_COLUMNS):
            raise ValueError(f"expected columns in this order: {list(US_MODEL_COLUMNS)}")
        if frame.empty or frame.isna().any().any():
            raise ValueError("returns must be non-empty and contain no missing values")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("returns must contain only finite values")
        return frame.astype(float)

    def fit(
        self,
        rolling_returns: pd.DataFrame,
        full_prior_returns: pd.DataFrame,
    ) -> "USMarketPcaSubModel":
        rolling = self._validate(rolling_returns)
        prior = self._validate(full_prior_returns)
        if len(rolling) != PAPER_WINDOW:
            raise ValueError(f"rolling_returns must contain exactly {PAPER_WINDOW} rows")

        standardized, self.mean_, self.std_ = standardize_window(rolling)
        values = standardized.to_numpy()
        self.rolling_correlation_ = values.T @ values / len(values)
        self.prior_basis_ = build_us_market_prior_basis()
        self.prior_correlation_ = build_prior_correlation(prior, self.prior_basis_)
        self.regularized_correlation_ = regularize_correlation(
            self.rolling_correlation_, self.prior_correlation_, self.lambda_prior
        )

        eigenvalues, eigenvectors = np.linalg.eigh(self.regularized_correlation_)
        order = np.argsort(eigenvalues)[::-1][:PAPER_COMPONENTS]
        self.eigenvalues_ = eigenvalues[order]
        self.components_ = eigenvectors[:, order]
        split = len(US_LEADING_SECTORS)
        self.leader_loadings_ = self.components_[:split, :]
        self.target_loadings_ = self.components_[split:, :]
        self.transmission_ = self.target_loadings_ @ self.leader_loadings_.T
        return self

    def predict(self, current_leader_return: pd.Series) -> USMarketSignal:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before predict")
        if list(current_leader_return.index) != list(US_LEADING_SECTORS):
            raise ValueError(
                f"current_leader_return must use this order: {list(US_LEADING_SECTORS)}"
            )
        if current_leader_return.isna().any():
            raise ValueError("current_leader_return must contain no missing values")

        mean = self.mean_.loc[list(US_LEADING_SECTORS)]
        std = self.std_.loc[list(US_LEADING_SECTORS)]
        standardized = (current_leader_return.astype(float) - mean) / std
        factors = self.leader_loadings_.T @ standardized.to_numpy()
        prediction = self.target_loadings_ @ factors
        return USMarketSignal(
            leader_standardized=pd.Series(
                standardized, index=US_LEADING_SECTORS, name="z_leader"
            ),
            factor=pd.Series(
                factors,
                index=[f"PC{position}" for position in range(1, PAPER_COMPONENTS + 1)],
                name="factor",
            ),
            target_standardized_prediction=pd.Series(
                prediction, index=US_MARKET_TARGETS, name="z_hat_target"
            ),
            transmission=pd.DataFrame(
                self.transmission_,
                index=US_MARKET_TARGETS,
                columns=US_LEADING_SECTORS,
            ),
        )
