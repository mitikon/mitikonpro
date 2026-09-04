from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CLASSES = np.array([-1, 0, 1], dtype=int)


@dataclass(frozen=True)
class Prediction:
    predicted_class: int
    action: str
    confidence: float
    edge: float
    probabilities: dict[int, float]


class LeadingLambdaClassifier:
    """PCA部分空間上の正則化されたクラス別距離で翌日方向を判定する。

    lambda_reg=0.10 は共分散行列の縮約率、variance_target=0.90 は
    PCAで保持する累積説明分散率であり、役割を混同しない。
    """

    def __init__(
        self,
        lambda_reg: float = 0.10,
        variance_target: float = 0.90,
        no_trade_threshold: float = 0.45,
        min_samples: int = 60,
    ) -> None:
        if not 0.0 <= lambda_reg <= 1.0:
            raise ValueError("lambda_reg must be in [0, 1]")
        if not 0.0 < variance_target <= 1.0:
            raise ValueError("variance_target must be in (0, 1]")
        self.lambda_reg = lambda_reg
        self.variance_target = variance_target
        self.no_trade_threshold = no_trade_threshold
        self.min_samples = min_samples

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LeadingLambdaClassifier":
        X, y = self._clean_xy(X, y)
        if len(X) < self.min_samples:
            raise ValueError(f"at least {self.min_samples} aligned rows are required")
        observed = np.sort(y.unique())
        if len(observed) < 2:
            raise ValueError("training data must contain at least two classes")

        self.columns_ = list(X.columns)
        self.mean_ = X.mean(axis=0).to_numpy(dtype=float)
        scale = X.std(axis=0, ddof=0).to_numpy(dtype=float)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        Z = (X.to_numpy(dtype=float) - self.mean_) / self.scale_

        _, singular, vt = np.linalg.svd(Z, full_matrices=False)
        variances = singular**2
        ratios = variances / max(variances.sum(), 1e-12)
        k = int(np.searchsorted(np.cumsum(ratios), self.variance_target) + 1)
        self.components_ = vt[:k].T
        scores = Z @ self.components_

        self.class_stats_: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for cls in observed:
            points = scores[y.to_numpy() == cls]
            center = points.mean(axis=0)
            if len(points) > 1:
                covariance = np.atleast_2d(np.cov(points, rowvar=False, ddof=1))
            else:
                covariance = np.eye(k)
            if covariance.shape != (k, k):
                covariance = np.eye(k) * float(np.squeeze(covariance))
            target = np.eye(k) * (np.trace(covariance) / max(k, 1))
            regularized = (1.0 - self.lambda_reg) * covariance + self.lambda_reg * target
            regularized += np.eye(k) * 1e-8
            self.class_stats_[int(cls)] = (center, np.linalg.pinv(regularized), len(points) / len(scores))
        return self

    def predict_one(self, row: pd.Series) -> Prediction:
        missing = set(self.columns_) - set(row.index)
        if missing:
            raise ValueError(f"missing features: {sorted(missing)}")
        values = row[self.columns_].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("prediction row contains missing or infinite values")
        score = ((values - self.mean_) / self.scale_) @ self.components_

        logits: dict[int, float] = {}
        for cls, (center, precision, prior) in self.class_stats_.items():
            delta = score - center
            distance = float(delta @ precision @ delta)
            logits[cls] = -0.5 * distance + np.log(max(prior, 1e-12))
        peak = max(logits.values())
        weights = {cls: np.exp(value - peak) for cls, value in logits.items()}
        total = sum(weights.values())
        probabilities = {cls: float(weight / total) for cls, weight in weights.items()}
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        predicted_class, confidence = ranked[0]
        edge = confidence - (ranked[1][1] if len(ranked) > 1 else 0.0)
        if confidence < self.no_trade_threshold:
            predicted_class = 0
        action = {-1: "SHORT", 0: "NO_TRADE", 1: "LONG"}[int(predicted_class)]
        return Prediction(int(predicted_class), action, confidence, edge, probabilities)

    @staticmethod
    def _clean_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if not X.index.equals(y.index):
            y = y.reindex(X.index)
        frame = X.replace([np.inf, -np.inf], np.nan).copy()
        valid = frame.notna().all(axis=1) & y.notna()
        return frame.loc[valid].astype(float), y.loc[valid].astype(int)

