"""論文準拠の部分空間正則化付きPCA（PCA SUB）。

中川慧ほか（2026）の式(8)〜(21)に対応する計算中核。
米国11業種の当日リターンから、翌営業日の日本17業種を予測する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

PAPER_LAMBDA_PRIOR = 0.90
PAPER_COMPONENTS = 3
PAPER_WINDOW = 60
PAPER_QUANTILE = 0.30

US_SECTORS = (
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
)
JAPAN_SECTORS = tuple(f"{ticker}.T" for ticker in range(1617, 1634))

US_CYCLICAL = ("XLB", "XLE", "XLF", "XLRE")
US_DEFENSIVE = ("XLK", "XLP", "XLU", "XLV")
JAPAN_CYCLICAL = ("1618.T", "1625.T", "1629.T", "1631.T")
JAPAN_DEFENSIVE = ("1617.T", "1621.T", "1627.T", "1630.T")


def _orthonormalize(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Gram-Schmidt法で列ベクトルを直交正規化する。"""
    basis: list[np.ndarray] = []
    for original in vectors:
        vector = np.asarray(original, dtype=float).copy()
        for existing in basis:
            vector -= existing * float(existing @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("prior exposure vectors must be linearly independent")
        basis.append(vector / norm)
    return np.column_stack(basis)


def build_prior_basis(
    us_columns: Sequence[str] = US_SECTORS,
    japan_columns: Sequence[str] = JAPAN_SECTORS,
) -> np.ndarray:
    """式(8)の3本の事前エクスポージャーを構築する。"""
    columns = tuple(us_columns) + tuple(japan_columns)
    index = {name: position for position, name in enumerate(columns)}
    n_assets = len(columns)

    global_factor = np.ones(n_assets, dtype=float)

    country_spread = np.empty(n_assets, dtype=float)
    country_spread[: len(us_columns)] = 1.0
    country_spread[len(us_columns) :] = -1.0

    cyclical_defensive = np.zeros(n_assets, dtype=float)
    for name in (*US_CYCLICAL, *JAPAN_CYCLICAL):
        if name in index:
            cyclical_defensive[index[name]] = 1.0
    for name in (*US_DEFENSIVE, *JAPAN_DEFENSIVE):
        if name in index:
            cyclical_defensive[index[name]] = -1.0

    return _orthonormalize(
        (global_factor, country_spread, cyclical_defensive)
    )


def _validate_returns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    expected = list(columns)
    if list(frame.columns) != expected:
        raise ValueError(f"expected columns in this order: {expected}")
    if frame.empty or frame.isna().any().any():
        raise ValueError("returns must be non-empty and contain no missing values")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    return frame.astype(float)


def standardize_window(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """式(4)に従い、窓内の平均と母標準偏差で標準化する。"""
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0)
    if (std <= 0).any():
        raise ValueError("all assets need non-zero variance in the rolling window")
    return (frame - mean) / std, mean, std


def correlation_from_returns(frame: pd.DataFrame) -> np.ndarray:
    standardized, _, _ = standardize_window(frame)
    n_rows = len(standardized)
    return standardized.to_numpy().T @ standardized.to_numpy() / n_rows


def build_prior_correlation(
    full_prior_returns: pd.DataFrame,
    prior_basis: np.ndarray,
) -> np.ndarray:
    """式(9)〜(12): 長期標本から事前相関行列C0を構築する。"""
    c_full = correlation_from_returns(full_prior_returns)
    d0 = np.diag(np.diag(prior_basis.T @ c_full @ prior_basis))
    c0_raw = prior_basis @ d0 @ prior_basis.T
    scale = np.sqrt(np.diag(c0_raw))
    if (scale <= 0).any():
        raise ValueError("prior covariance has a non-positive diagonal")
    c0 = c0_raw / np.outer(scale, scale)
    return (c0 + c0.T) / 2.0


def regularize_correlation(
    rolling_correlation: np.ndarray,
    prior_correlation: np.ndarray,
    lambda_prior: float = PAPER_LAMBDA_PRIOR,
) -> np.ndarray:
    """式(13): (1-λ)Ct + λC0。論文設定はλ=0.9。"""
    if not 0.0 <= lambda_prior <= 1.0:
        raise ValueError("lambda_prior must be between 0 and 1")
    if rolling_correlation.shape != prior_correlation.shape:
        raise ValueError("rolling and prior correlations must have the same shape")
    combined = (
        (1.0 - lambda_prior) * rolling_correlation
        + lambda_prior * prior_correlation
    )
    return (combined + combined.T) / 2.0


@dataclass(frozen=True)
class PaperSignal:
    """米国因子と翌日日本業種の標準化予測値。"""

    us_standardized: pd.Series
    factor: pd.Series
    japan_standardized_prediction: pd.Series
    transmission: pd.DataFrame


class PaperPcaSubModel:
    """論文の固定仕様 L=60, K=3 によるPCA SUBモデル。"""

    def __init__(self, lambda_prior: float = PAPER_LAMBDA_PRIOR) -> None:
        if not 0.0 <= lambda_prior <= 1.0:
            raise ValueError("lambda_prior must be between 0 and 1")
        self.lambda_prior = float(lambda_prior)
        self.columns = US_SECTORS + JAPAN_SECTORS

    def fit(
        self,
        rolling_returns: pd.DataFrame,
        full_prior_returns: pd.DataFrame,
    ) -> "PaperPcaSubModel":
        rolling = _validate_returns(rolling_returns, self.columns)
        prior = _validate_returns(full_prior_returns, self.columns)
        if len(rolling) != PAPER_WINDOW:
            raise ValueError(f"rolling_returns must contain exactly {PAPER_WINDOW} rows")

        standardized, mean, std = standardize_window(rolling)
        self.mean_ = mean
        self.std_ = std
        self.rolling_correlation_ = (
            standardized.to_numpy().T @ standardized.to_numpy() / len(standardized)
        )
        self.prior_basis_ = build_prior_basis()
        self.prior_correlation_ = build_prior_correlation(prior, self.prior_basis_)
        self.regularized_correlation_ = regularize_correlation(
            self.rolling_correlation_, self.prior_correlation_, self.lambda_prior
        )

        eigenvalues, eigenvectors = np.linalg.eigh(self.regularized_correlation_)
        order = np.argsort(eigenvalues)[::-1][:PAPER_COMPONENTS]
        self.eigenvalues_ = eigenvalues[order]
        self.components_ = eigenvectors[:, order]
        self.us_loadings_ = self.components_[: len(US_SECTORS), :]
        self.japan_loadings_ = self.components_[len(US_SECTORS) :, :]
        self.transmission_ = self.japan_loadings_ @ self.us_loadings_.T
        return self

    def predict(self, current_us_return: pd.Series) -> PaperSignal:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before predict")
        if list(current_us_return.index) != list(US_SECTORS):
            raise ValueError(f"current_us_return must use this order: {list(US_SECTORS)}")
        if current_us_return.isna().any():
            raise ValueError("current_us_return must contain no missing values")

        us_mean = self.mean_.loc[list(US_SECTORS)]
        us_std = self.std_.loc[list(US_SECTORS)]
        standardized = (current_us_return.astype(float) - us_mean) / us_std
        factor_values = self.us_loadings_.T @ standardized.to_numpy()
        prediction = self.japan_loadings_ @ factor_values

        return PaperSignal(
            us_standardized=pd.Series(standardized, index=US_SECTORS, name="z_us"),
            factor=pd.Series(
                factor_values,
                index=[f"PC{position}" for position in range(1, PAPER_COMPONENTS + 1)],
                name="factor",
            ),
            japan_standardized_prediction=pd.Series(
                prediction, index=JAPAN_SECTORS, name="z_hat_japan"
            ),
            transmission=pd.DataFrame(
                self.transmission_, index=JAPAN_SECTORS, columns=US_SECTORS
            ),
        )


def quantile_long_short_weights(
    scores: pd.Series,
    quantile: float = PAPER_QUANTILE,
) -> pd.Series:
    """式(20)〜(21): 上位・下位qを等ウェイトでロング・ショートする。"""
    if list(scores.index) != list(JAPAN_SECTORS):
        raise ValueError(f"scores must use this order: {list(JAPAN_SECTORS)}")
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    count = int(np.floor(len(scores) * quantile))
    if count < 1:
        raise ValueError("quantile selects no assets")

    ordered = scores.sort_values(kind="mergesort")
    weights = pd.Series(0.0, index=scores.index, name="weight")
    weights.loc[ordered.index[:count]] = -1.0 / count
    weights.loc[ordered.index[-count:]] = 1.0 / count
    return weights
