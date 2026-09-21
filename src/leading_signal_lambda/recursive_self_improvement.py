"""Controlled Recursive Self-Improvement (RSI) for the market predictor.

This is intentionally unrelated to Relative Strength Index.  A candidate is
sealed before its evaluation sessions exist, then judged only on later frozen
forecasts and outcomes.  The module can propose promotion, but cannot edit
source, merge a branch, or place a trade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


RECURSIVE_SELF_IMPROVEMENT_VERSION = "market-recursive-self-improvement-v1"
FIXED_LAMBDA_REG = 0.10
FIXED_VARIANCE_TARGET = 0.90
FIXED_MIN_SAMPLES = 60
ALLOWED_CANDIDATE_PARAMETERS = frozenset(
    {
        "rsi_periods",
        "rsi_feature_set",
        "rsi_feature_weight",
        "feature_lags",
        "neutral_band",
        "no_trade_threshold",
    }
)


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _digest(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return normalized


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parameter_manifest_digest(parameters: Mapping[str, object]) -> str:
    return sha256(_canonical(dict(parameters))).hexdigest()


@dataclass(frozen=True)
class MarketRsiCandidate:
    candidate_id: str
    parent_version: str
    generation: int
    created_at: datetime
    source_commit: str
    parameter_manifest_sha256: str
    parameters: Mapping[str, object]
    parent_report_sha256: str | None = None
    lambda_reg: float = FIXED_LAMBDA_REG
    variance_target: float = FIXED_VARIANCE_TARGET
    min_samples: int = FIXED_MIN_SAMPLES

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.parent_version.strip():
            raise ValueError("candidate_id and parent_version are required")
        if self.generation < 1:
            raise ValueError("generation must be positive")
        _utc(self.created_at, "created_at")
        _digest(self.parameter_manifest_sha256, "parameter_manifest_sha256")
        if self.parent_report_sha256 is not None:
            _digest(self.parent_report_sha256, "parent_report_sha256")
        if (self.generation == 1) != (self.parent_report_sha256 is None):
            raise ValueError("generation 1 has no parent report; later generations require one")
        if len(self.source_commit) != 40 or any(char not in "0123456789abcdef" for char in self.source_commit.lower()):
            raise ValueError("source_commit must be a full Git commit SHA")
        unknown = set(self.parameters) - ALLOWED_CANDIDATE_PARAMETERS
        if unknown:
            raise ValueError(f"candidate attempts non-allow-listed changes: {sorted(unknown)}")
        if not self.parameters:
            raise ValueError("candidate must change at least one allow-listed parameter")
        if self.parameter_manifest_sha256 != parameter_manifest_digest(self.parameters):
            raise ValueError("parameter_manifest_sha256 does not match candidate parameters")
        if self.lambda_reg != FIXED_LAMBDA_REG or self.variance_target != FIXED_VARIANCE_TARGET:
            raise ValueError("fixed PCA/lambda invariants cannot be changed by recursive RSI")
        if self.min_samples != FIXED_MIN_SAMPLES:
            raise ValueError("fixed minimum training sample count cannot be changed")

    def sealed_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "created_at": _utc(self.created_at, "created_at").isoformat(),
            "parameters": dict(self.parameters),
        }


def candidate_manifest_digest(candidate: MarketRsiCandidate) -> str:
    return sha256(_canonical(candidate.sealed_payload())).hexdigest()


@dataclass(frozen=True)
class MarketFrozenTrial:
    """Pre-outcome attestation binding a candidate to two frozen forecasts."""

    candidate_id: str
    session: str
    registered_at: datetime
    prediction_frozen_at: datetime
    baseline_prediction_sha256: str
    candidate_prediction_sha256: str
    candidate_manifest_sha256: str
    input_sha256: str

    def __post_init__(self) -> None:
        registered = _utc(self.registered_at, "registered_at")
        frozen = _utc(self.prediction_frozen_at, "prediction_frozen_at")
        if registered > frozen:
            raise ValueError("trial must be registered no later than prediction freeze")
        date.fromisoformat(self.session)
        _digest(self.baseline_prediction_sha256, "baseline_prediction_sha256")
        _digest(self.candidate_prediction_sha256, "candidate_prediction_sha256")
        _digest(self.candidate_manifest_sha256, "candidate_manifest_sha256")
        _digest(self.input_sha256, "input_sha256")

    def sealed_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "registered_at": _utc(self.registered_at, "registered_at").isoformat(),
            "prediction_frozen_at": _utc(self.prediction_frozen_at, "prediction_frozen_at").isoformat(),
        }


def trial_manifest_digest(trial: MarketFrozenTrial) -> str:
    return sha256(_canonical(trial.sealed_payload())).hexdigest()


@dataclass(frozen=True)
class MarketFutureEvaluation:
    candidate_id: str
    session: str
    prediction_frozen_at: datetime
    outcome_known_at: datetime
    frozen_prediction_sha256: str
    candidate_prediction_sha256: str
    candidate_manifest_sha256: str
    trial_manifest_sha256: str
    baseline_loss: float
    candidate_loss: float
    baseline_upside_hit: bool
    candidate_upside_hit: bool
    baseline_downside_hit: bool
    candidate_downside_hit: bool
    baseline_net_return: float
    candidate_net_return: float
    baseline_drawdown: float
    candidate_drawdown: float

    def __post_init__(self) -> None:
        frozen = _utc(self.prediction_frozen_at, "prediction_frozen_at")
        known = _utc(self.outcome_known_at, "outcome_known_at")
        if frozen >= known:
            raise ValueError("prediction must be frozen before the outcome is known")
        date.fromisoformat(self.session)
        _digest(self.frozen_prediction_sha256, "frozen_prediction_sha256")
        _digest(self.candidate_prediction_sha256, "candidate_prediction_sha256")
        _digest(self.candidate_manifest_sha256, "candidate_manifest_sha256")
        _digest(self.trial_manifest_sha256, "trial_manifest_sha256")
        numeric = (
            self.baseline_loss,
            self.candidate_loss,
            self.baseline_net_return,
            self.candidate_net_return,
            self.baseline_drawdown,
            self.candidate_drawdown,
        )
        if any(not isinstance(value, (int, float)) or not isfinite(float(value)) for value in numeric):
            raise ValueError("evaluation metrics must be finite numbers")
        if self.baseline_loss < 0 or self.candidate_loss < 0:
            raise ValueError("loss cannot be negative")
        if self.baseline_drawdown < 0 or self.candidate_drawdown < 0:
            raise ValueError("drawdown must be a non-negative magnitude")


@dataclass(frozen=True)
class MarketPromotionReport:
    status: str
    candidate_id: str
    generation: int
    evaluated_sessions: int
    baseline_mean_loss: float
    candidate_mean_loss: float
    loss_improvement: float
    baseline_upside_hits: int
    candidate_upside_hits: int
    baseline_downside_hits: int
    candidate_downside_hits: int
    baseline_mean_net_return: float
    candidate_mean_net_return: float
    baseline_worst_drawdown: float
    candidate_worst_drawdown: float
    gates: Mapping[str, bool]
    report_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "gates": dict(self.gates),
            "autonomous_source_edits": False,
            "autonomous_main_merge": False,
            "autonomous_parameter_promotion": True,
            "trading_authority": False,
            "human_approval_required": True,
            "human_approval_required_for_source_changes": True,
        }


class MarketRecursiveImprovementGate:
    """Evaluate one sealed child generation on genuinely later sessions."""

    def __init__(self, *, min_future_sessions: int = 20, min_loss_improvement: float = 0.001) -> None:
        if min_future_sessions < 20 or min_loss_improvement < 0:
            raise ValueError("unsafe recursive-improvement gate configuration")
        self.min_future_sessions = int(min_future_sessions)
        self.min_loss_improvement = float(min_loss_improvement)

    def evaluate(
        self,
        candidate: MarketRsiCandidate,
        observations: Sequence[MarketFutureEvaluation],
        trials: Sequence[MarketFrozenTrial],
    ) -> MarketPromotionReport:
        rows = sorted(observations, key=lambda item: item.session)
        if len(rows) < self.min_future_sessions:
            raise ValueError("insufficient future sessions for recursive RSI evaluation")
        if len({row.session for row in rows}) != len(rows):
            raise ValueError("duplicate evaluation sessions are prohibited")
        trials_by_session = {trial.session: trial for trial in trials}
        if len(trials_by_session) != len(trials) or set(trials_by_session) != {row.session for row in rows}:
            raise ValueError("each evaluation requires exactly one pre-outcome frozen trial")
        created = _utc(candidate.created_at, "created_at")
        sealed_candidate_hash = candidate_manifest_digest(candidate)
        for row in rows:
            trial = trials_by_session[row.session]
            if row.candidate_id != candidate.candidate_id:
                raise ValueError("evaluation candidate_id mismatch")
            if trial.candidate_id != candidate.candidate_id:
                raise ValueError("trial candidate_id mismatch")
            if _utc(trial.registered_at, "registered_at") <= created:
                raise ValueError("candidate must be sealed before every registered trial")
            if _utc(row.prediction_frozen_at, "prediction_frozen_at") != _utc(
                trial.prediction_frozen_at, "trial prediction_frozen_at"
            ):
                raise ValueError("evaluation timestamp does not match frozen trial")
            if _utc(row.prediction_frozen_at, "prediction_frozen_at") <= created:
                raise ValueError("candidate must be sealed before every evaluated prediction")
            if row.candidate_manifest_sha256 != sealed_candidate_hash:
                raise ValueError("evaluation is not bound to the sealed candidate manifest")
            if trial.candidate_manifest_sha256 != sealed_candidate_hash:
                raise ValueError("trial is not bound to the sealed candidate manifest")
            if (
                row.frozen_prediction_sha256 != trial.baseline_prediction_sha256
                or row.candidate_prediction_sha256 != trial.candidate_prediction_sha256
                or row.trial_manifest_sha256 != trial_manifest_digest(trial)
            ):
                raise ValueError("settled evaluation does not match pre-outcome frozen trial")

        baseline_loss = mean(row.baseline_loss for row in rows)
        candidate_loss = mean(row.candidate_loss for row in rows)
        baseline_up = sum(row.baseline_upside_hit for row in rows)
        candidate_up = sum(row.candidate_upside_hit for row in rows)
        baseline_down = sum(row.baseline_downside_hit for row in rows)
        candidate_down = sum(row.candidate_downside_hit for row in rows)
        baseline_return = mean(row.baseline_net_return for row in rows)
        candidate_return = mean(row.candidate_net_return for row in rows)
        baseline_dd = max(row.baseline_drawdown for row in rows)
        candidate_dd = max(row.candidate_drawdown for row in rows)
        improvement = baseline_loss - candidate_loss
        gates = {
            "sealed_before_future_predictions": True,
            "pre_outcome_trial_attested": True,
            "minimum_future_sessions": True,
            "loss_improved": improvement >= self.min_loss_improvement,
            "upside_selection_not_worse": candidate_up >= baseline_up,
            "downside_selection_not_worse": candidate_down >= baseline_down,
            "net_return_not_worse": candidate_return >= baseline_return,
            "drawdown_not_worse": candidate_dd <= baseline_dd,
            "fixed_core_invariants": True,
        }
        status = "PROMOTION_PROPOSED" if all(gates.values()) else "REJECTED"
        unsigned = {
            "version": RECURSIVE_SELF_IMPROVEMENT_VERSION,
            "status": status,
            "candidate_id": candidate.candidate_id,
            "generation": candidate.generation,
            "evaluated_sessions": len(rows),
            "baseline_mean_loss": baseline_loss,
            "candidate_mean_loss": candidate_loss,
            "loss_improvement": improvement,
            "baseline_upside_hits": baseline_up,
            "candidate_upside_hits": candidate_up,
            "baseline_downside_hits": baseline_down,
            "candidate_downside_hits": candidate_down,
            "baseline_mean_net_return": baseline_return,
            "candidate_mean_net_return": candidate_return,
            "baseline_worst_drawdown": baseline_dd,
            "candidate_worst_drawdown": candidate_dd,
            "gates": gates,
            "candidate_manifest_sha256": sealed_candidate_hash,
        }
        report_hash = sha256(_canonical(unsigned)).hexdigest()
        return MarketPromotionReport(
            status=status,
            candidate_id=candidate.candidate_id,
            generation=candidate.generation,
            evaluated_sessions=len(rows),
            baseline_mean_loss=baseline_loss,
            candidate_mean_loss=candidate_loss,
            loss_improvement=improvement,
            baseline_upside_hits=baseline_up,
            candidate_upside_hits=candidate_up,
            baseline_downside_hits=baseline_down,
            candidate_downside_hits=candidate_down,
            baseline_mean_net_return=baseline_return,
            candidate_mean_net_return=candidate_return,
            baseline_worst_drawdown=baseline_dd,
            candidate_worst_drawdown=candidate_dd,
            gates=gates,
            report_sha256=report_hash,
        )


def validate_successor(previous: MarketPromotionReport, candidate: MarketRsiCandidate) -> None:
    if previous.status != "PROMOTION_PROPOSED":
        raise ValueError("a rejected generation cannot become the recursive parent")
    if candidate.generation != previous.generation + 1:
        raise ValueError("recursive candidate generation is not sequential")
    if candidate.parent_version != previous.candidate_id:
        raise ValueError("recursive candidate parent_version mismatch")
    if candidate.parent_report_sha256 != previous.report_sha256:
        raise ValueError("recursive candidate is not chained to the prior report")


def freeze_candidate(candidate: MarketRsiCandidate, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = candidate.sealed_payload()
    payload["candidate_manifest_sha256"] = candidate_manifest_digest(candidate)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def freeze_trial(trial: MarketFrozenTrial, path: str | Path) -> Path:
    """Write the trial once before the outcome; Git should timestamp this artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = trial.sealed_payload()
    payload["trial_manifest_sha256"] = trial_manifest_digest(trial)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def freeze_promotion_report(report: MarketPromotionReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
