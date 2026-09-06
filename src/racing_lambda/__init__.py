"""Auditable horse-racing prediction components."""

from .evaluation import EvaluationReport, evaluate_prediction
from .freeze import freeze_prediction, load_frozen_prediction
from .layer2_live_input import (
    MonthlyConditionStats,
    PastRun,
    RaceDayHorseInput,
    body_weight_fit,
    build_statistical_inputs,
    normalized_market_probabilities,
    pace_position_score,
    recent_form_score,
)
from .monthly_stats_db import (
    AggregateEvidence,
    HorseMonthlyEvidence,
    MonthlyBuildResult,
    MonthlySnapshot,
    build_monthly_condition_stats,
    build_snapshot_stats,
)
from .odds import OddsDistortion, rank_odds_distortion
from .schema import (
    ComponentWeights,
    HorseEntry,
    LeadingSignalPolicy,
    OfficialResult,
    PredictionRow,
    RaceContext,
)
from .scoring import build_prediction
from .simple_leading_signal_v02 import (
    BugType,
    Going,
    SimpleHorseFeatures,
    SimpleLeadingSignalLambdaV02,
    SimplePredictionOutput,
    SimpleRaceContext,
    SimpleScoreBreakdown,
)
from .validation_2026_09_06 import (
    RecordedFrozenPrediction,
    RecordedRaceResult,
    ThreeRaceValidationReport,
    VALIDATION_RECORDS_2026_09_06,
    validate_record,
    validation_summary_2026_09_06,
)

__all__ = [
    "AggregateEvidence",
    "BugType",
    "ComponentWeights",
    "EvaluationReport",
    "Going",
    "HorseEntry",
    "HorseMonthlyEvidence",
    "LeadingSignalPolicy",
    "MonthlyBuildResult",
    "MonthlyConditionStats",
    "MonthlySnapshot",
    "OddsDistortion",
    "OfficialResult",
    "PastRun",
    "PredictionRow",
    "RaceContext",
    "RaceDayHorseInput",
    "RecordedFrozenPrediction",
    "RecordedRaceResult",
    "SimpleHorseFeatures",
    "SimpleLeadingSignalLambdaV02",
    "SimplePredictionOutput",
    "SimpleRaceContext",
    "SimpleScoreBreakdown",
    "ThreeRaceValidationReport",
    "VALIDATION_RECORDS_2026_09_06",
    "body_weight_fit",
    "build_monthly_condition_stats",
    "build_prediction",
    "build_snapshot_stats",
    "build_statistical_inputs",
    "evaluate_prediction",
    "freeze_prediction",
    "load_frozen_prediction",
    "normalized_market_probabilities",
    "pace_position_score",
    "rank_odds_distortion",
    "recent_form_score",
    "validate_record",
    "validation_summary_2026_09_06",
]
