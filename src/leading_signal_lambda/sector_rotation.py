"""米国11セクターの今日→翌日を直接学習するラグ付きPCA SUB。"""

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

SECTOR_ETFS = US_SECTORS
LEAD_COLUMNS = tuple(f"lead:{symbol}" for symbol in SECTOR_ETFS)
NEXT_COLUMNS = tuple(f"next:{symbol}" for symbol in SECTOR_ETFS)
LAGGED_COLUMNS = LEAD_COLUMNS + NEXT_COLUMNS


def make_lagged_sector_pairs(sector_returns: pd.DataFrame) -> pd.DataFrame:
    """r_tとr_(t+1)のペアを作り、行の日付には結果判明日を付ける。"""
    if list(sector_returns.columns) != list(SECTOR_ETFS):
        raise ValueError(f"sector return columns must be: {list(SECTOR_ETFS)}")
    if not sector_returns.index.is_monotonic_increasing:
        raise ValueError("sector returns must be sorted in ascending time order")
    if len(sector_returns) < 2 or sector_returns.isna().any().any():
        raise ValueError("at least two complete sector-return rows are required")
    lead = sector_returns.iloc[:-1].to_numpy(dtype=float)
    following = sector_returns.iloc[1:].to_numpy(dtype=float)
    return pd.DataFrame(
        np.column_stack((lead, following)),
        index=sector_returns.index[1:],
        columns=LAGGED_COLUMNS,
    )


def build_sector_rotation_prior_basis() -> np.ndarray:
    """グローバル、今日/翌日差、景気循環/防御の3事前方向。"""
    n_sectors = len(SECTOR_ETFS)
    global_factor = np.ones(n_sectors * 2, dtype=float)
    lead_next_spread = np.r_[np.ones(n_sectors), -np.ones(n_sectors)]
    sector_style = np.zeros(n_sectors, dtype=float)
    position = {symbol: index for index, symbol in enumerate(SECTOR_ETFS)}
    for symbol in US_CYCLICAL:
        sector_style[position[symbol]] = 1.0
    for symbol in US_DEFENSIVE:
        sector_style[position[symbol]] = -1.0
    duplicated_style = np.r_[sector_style, sector_style]
    return _orthonormalize((global_factor, lead_next_spread, duplicated_style))


@dataclass(frozen=True)
class SectorRotationSignal:
    current_standardized: pd.Series
    current_distortion: pd.Series
    next_standardized_prediction: pd.Series
    transmission: pd.DataFrame


class LaggedSectorPcaSubModel:
    """当日11セクターから翌日11セクターを予測する論文派生モデル。"""

    def __init__(self, lambda_prior: float = PAPER_LAMBDA_PRIOR) -> None:
        if not 0.0 <= lambda_prior <= 1.0:
            raise ValueError("lambda_prior must be between 0 and 1")
        self.lambda_prior = float(lambda_prior)

    @staticmethod
    def _validate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
        if list(frame.columns) != list(LAGGED_COLUMNS):
            raise ValueError(f"lagged pair columns must be: {list(LAGGED_COLUMNS)}")
        if frame.empty or frame.isna().any().any():
            raise ValueError("lagged pairs must be complete")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("lagged pairs must contain only finite values")
        return frame.astype(float)

    def fit(
        self,
        rolling_pairs: pd.DataFrame,
        prior_pairs: pd.DataFrame,
    ) -> "LaggedSectorPcaSubModel":
        rolling = self._validate_pairs(rolling_pairs)
        prior = self._validate_pairs(prior_pairs)
        if len(rolling) != PAPER_WINDOW:
            raise ValueError(f"rolling_pairs must contain exactly {PAPER_WINDOW} rows")

        standardized, self.mean_, self.std_ = standardize_window(rolling)
        z = standardized.to_numpy()
        self.rolling_correlation_ = z.T @ z / len(z)
        self.prior_basis_ = build_sector_rotation_prior_basis()
        self.prior_correlation_ = build_prior_correlation(prior, self.prior_basis_)
        self.regularized_correlation_ = regularize_correlation(
            self.rolling_correlation_, self.prior_correlation_, self.lambda_prior
        )
        eigenvalues, eigenvectors = np.linalg.eigh(self.regularized_correlation_)
        order = np.argsort(eigenvalues)[::-1][:PAPER_COMPONENTS]
        self.eigenvalues_ = eigenvalues[order]
        self.components_ = eigenvectors[:, order]
        split = len(SECTOR_ETFS)
        self.lead_loadings_ = self.components_[:split, :]
        self.next_loadings_ = self.components_[split:, :]
        self.transmission_ = self.next_loadings_ @ self.lead_loadings_.T
        return self

    def predict(self, current_sector_returns: pd.Series) -> SectorRotationSignal:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before predict")
        if list(current_sector_returns.index) != list(SECTOR_ETFS):
            raise ValueError(f"current sector order must be: {list(SECTOR_ETFS)}")
        if current_sector_returns.isna().any():
            raise ValueError("current sector returns must be complete")
        lead_mean = self.mean_.loc[list(LEAD_COLUMNS)].copy()
        lead_mean.index = SECTOR_ETFS
        lead_std = self.std_.loc[list(LEAD_COLUMNS)].copy()
        lead_std.index = SECTOR_ETFS
        current_z = (current_sector_returns.astype(float) - lead_mean) / lead_std
        factors = self.lead_loadings_.T @ current_z.to_numpy()
        reconstruction = self.lead_loadings_ @ factors
        prediction = self.next_loadings_ @ factors
        return SectorRotationSignal(
            current_standardized=pd.Series(current_z, index=SECTOR_ETFS),
            current_distortion=pd.Series(
                current_z.to_numpy() - reconstruction,
                index=SECTOR_ETFS,
                name="current_distortion",
            ),
            next_standardized_prediction=pd.Series(
                prediction, index=SECTOR_ETFS, name="next_prediction"
            ),
            transmission=pd.DataFrame(
                self.transmission_, index=SECTOR_ETFS, columns=SECTOR_ETFS
            ),
        )
