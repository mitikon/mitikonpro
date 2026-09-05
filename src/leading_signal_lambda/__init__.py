"""部分空間正則化PCAと先行シグナル予測λ。"""

from .model import LeadingLambdaClassifier, Prediction
from .signals import build_leading_features, build_training_set
from .validation import WalkForwardResult, walk_forward_validate
from .collector import DailyMarketCollector, MarketDataset, DEFAULT_UNIVERSE
from .paper_backtest import PaperBacktestResult, run_paper_backtest
from .paper_pca_sub import PaperPcaSubModel, PaperSignal
from .us_market_pca_sub import USMarketPcaSubModel, USMarketSignal, prepare_us_market_returns
from .us_market_validation import USMarketBacktestResult, run_us_market_backtest
from .sector_rotation import LaggedSectorPcaSubModel, SectorRotationSignal
from .sector_rotation_validation import (
    PositionState,
    SectorRotationBacktestResult,
    decide_position_action,
    run_sector_rotation_backtest,
)

__all__ = [
    "LeadingLambdaClassifier",
    "Prediction",
    "WalkForwardResult",
    "build_leading_features",
    "build_training_set",
    "walk_forward_validate",
    "DailyMarketCollector",
    "MarketDataset",
    "DEFAULT_UNIVERSE",
    "PaperPcaSubModel",
    "PaperSignal",
    "PaperBacktestResult",
    "run_paper_backtest",
    "USMarketPcaSubModel",
    "USMarketSignal",
    "prepare_us_market_returns",
    "USMarketBacktestResult",
    "run_us_market_backtest",
    "LaggedSectorPcaSubModel",
    "SectorRotationSignal",
    "PositionState",
    "SectorRotationBacktestResult",
    "decide_position_action",
    "run_sector_rotation_backtest",
]
