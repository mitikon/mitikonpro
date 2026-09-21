"""Permanent, outcome-blind runtime loop for market recursive self-improvement.

The runtime promotes only bounded model configuration.  It never edits source,
merges Git branches, or places trades.  Every challenger forecast is frozen
before its target session and can only be evaluated against a later close.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .recursive_self_improvement import (
    MarketFutureEvaluation,
    MarketFrozenTrial,
    MarketRecursiveImprovementGate,
    MarketRsiCandidate,
    candidate_manifest_digest,
    parameter_manifest_digest,
    trial_manifest_digest,
)


RUNTIME_SCHEMA_VERSION = "market-recursive-runtime-v1"
DEFAULT_TRANSACTION_COST = 0.0005


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_digest(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("recursive RSI timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _candidate_from_payload(payload: Mapping[str, object]) -> MarketRsiCandidate:
    return MarketRsiCandidate(
        candidate_id=str(payload["candidate_id"]),
        parent_version=str(payload["parent_version"]),
        generation=int(payload["generation"]),
        created_at=_parse_time(str(payload["created_at"])),
        source_commit=str(payload["source_commit"]),
        parameter_manifest_sha256=str(payload["parameter_manifest_sha256"]),
        parameters=dict(payload["parameters"]),
        parent_report_sha256=(
            str(payload["parent_report_sha256"])
            if payload.get("parent_report_sha256") is not None
            else None
        ),
        lambda_reg=float(payload.get("lambda_reg", 0.10)),
        variance_target=float(payload.get("variance_target", 0.90)),
        min_samples=int(payload.get("min_samples", 60)),
    )


def _trial_from_payload(payload: Mapping[str, object]) -> MarketFrozenTrial:
    return MarketFrozenTrial(
        candidate_id=str(payload["candidate_id"]),
        session=str(payload["session"]),
        registered_at=_parse_time(str(payload["registered_at"])),
        prediction_frozen_at=_parse_time(str(payload["prediction_frozen_at"])),
        baseline_prediction_sha256=str(payload["baseline_prediction_sha256"]),
        candidate_prediction_sha256=str(payload["candidate_prediction_sha256"]),
        candidate_manifest_sha256=str(payload["candidate_manifest_sha256"]),
        input_sha256=str(payload["input_sha256"]),
    )


def _evaluation_from_payload(payload: Mapping[str, object]) -> MarketFutureEvaluation:
    return MarketFutureEvaluation(
        candidate_id=str(payload["candidate_id"]),
        session=str(payload["session"]),
        prediction_frozen_at=_parse_time(str(payload["prediction_frozen_at"])),
        outcome_known_at=_parse_time(str(payload["outcome_known_at"])),
        frozen_prediction_sha256=str(payload["frozen_prediction_sha256"]),
        candidate_prediction_sha256=str(payload["candidate_prediction_sha256"]),
        candidate_manifest_sha256=str(payload["candidate_manifest_sha256"]),
        trial_manifest_sha256=str(payload["trial_manifest_sha256"]),
        baseline_loss=float(payload["baseline_loss"]),
        candidate_loss=float(payload["candidate_loss"]),
        baseline_upside_hit=bool(payload["baseline_upside_hit"]),
        candidate_upside_hit=bool(payload["candidate_upside_hit"]),
        baseline_downside_hit=bool(payload["baseline_downside_hit"]),
        candidate_downside_hit=bool(payload["candidate_downside_hit"]),
        baseline_net_return=float(payload["baseline_net_return"]),
        candidate_net_return=float(payload["candidate_net_return"]),
        baseline_drawdown=float(payload["baseline_drawdown"]),
        candidate_drawdown=float(payload["candidate_drawdown"]),
    )


def _source_commit(value: str | None) -> str:
    commit = (value or "0" * 40).lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("source commit must be a full Git SHA")
    return commit


MUTATION_SCHEDULE: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("rsi_feature_weight", (1.0, 0.75, 1.25, 1.50, 0.50, 2.00)),
    ("feature_lags", (5, 3, 7, 10, 2)),
    ("neutral_band", (0.001, 0.0005, 0.0015, 0.0020, 0.0)),
    ("no_trade_threshold", (0.45, 0.40, 0.50, 0.55, 0.35)),
    (
        "rsi_periods",
        ([5, 7, 14, 21], [5, 14, 21], [7, 14, 28], [3, 7, 14, 21], [5, 10, 20, 40]),
    ),
    (
        "rsi_feature_set",
        (
            ["level", "velocity3", "cross50", "extreme_state"],
            ["level", "velocity3", "cross50"],
            ["level", "velocity3", "extreme_state"],
            ["velocity3", "cross50", "extreme_state"],
            ["level", "cross50", "extreme_state"],
        ),
    ),
)


def _next_parameters(active: Mapping[str, object], attempt: int) -> dict[str, object]:
    search_size = 1
    for _, choices in MUTATION_SCHEDULE:
        search_size *= len(choices)
    for offset in range(search_size):
        cursor = (attempt + offset) % search_size
        parameters = deepcopy(dict(active))
        for name, choices in MUTATION_SCHEDULE:
            parameters[name] = deepcopy(choices[cursor % len(choices)])
            cursor //= len(choices)
        if parameters != dict(active):
            return parameters
    raise RuntimeError("recursive RSI search space contains no alternative configuration")


def _new_candidate(
    state: Mapping[str, object], source_commit: str, created_at: datetime
) -> MarketRsiCandidate:
    active = dict(state["active_model"])
    attempt = int(state.get("attempt", 0)) + 1
    generation = int(active["generation"]) + 1
    parameters = _next_parameters(dict(active["parameters"]), attempt)
    return MarketRsiCandidate(
        candidate_id=f"market-rsi-g{generation}-a{attempt}",
        parent_version=str(active["model_id"]),
        generation=generation,
        created_at=created_at,
        source_commit=_source_commit(source_commit),
        parameter_manifest_sha256=parameter_manifest_digest(parameters),
        parameters=parameters,
        parent_report_sha256=(
            str(active["promotion_report_sha256"])
            if generation > 1
            else None
        ),
    )


def bootstrap_state(
    default_parameters: Mapping[str, object],
    source_commit: str,
    created_at: datetime | None = None,
) -> dict[str, object]:
    now = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "active_model": {
            "model_id": "market-forward-v7-baseline",
            "generation": 0,
            "parameters": deepcopy(dict(default_parameters)),
            "promotion_report_sha256": None,
        },
        "attempt": 0,
        "candidate": None,
        "candidate_manifest_sha256": None,
        "trials": [],
        "evaluations": [],
        "pending_trial": None,
        "completed_candidates": [],
        "previous_state_sha256": None,
        "updated_at": now.isoformat(),
    }
    candidate = _new_candidate(state, source_commit, now - timedelta(microseconds=1))
    state["attempt"] = 1
    state["candidate"] = candidate.sealed_payload()
    state["candidate_manifest_sha256"] = candidate_manifest_digest(candidate)
    return state


def load_state(path: str | Path) -> dict[str, object]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported recursive RSI runtime state")
    expected = state.pop("state_sha256", None)
    actual = sha256(_canonical(state)).hexdigest()
    state["state_sha256"] = expected
    if expected != actual:
        raise ValueError("recursive RSI state hash mismatch")
    _candidate_from_payload(state["candidate"])
    return state


def write_state(
    state: Mapping[str, object], path: str | Path, previous_path: str | Path | None = None
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite recursive RSI state: {destination}")
    payload = deepcopy(dict(state))
    payload.pop("state_sha256", None)
    payload["previous_state_sha256"] = (
        _file_digest(previous_path) if previous_path and Path(previous_path).exists() else None
    )
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["state_sha256"] = sha256(_canonical(payload)).hexdigest()
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _settlement_metrics(path: str | Path) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    settlements = document["settlements"]
    return_error = sum(abs(float(row["return_error"])) for row in settlements) / len(settlements)
    classification_error = 1.0 - (
        sum(bool(row.get("direction_correct", False)) for row in settlements)
        / len(settlements)
    )
    # The small classification term makes the NO_TRADE threshold learnable
    # without allowing it to dominate next-session return calibration.
    loss = return_error + 0.001 * classification_error
    trade = document["primary_trade"]
    net_return = float(trade["gross_return_before_cost"]) - DEFAULT_TRANSACTION_COST
    return {
        "loss": loss,
        "upside_hit": bool(document["extreme_forecasts"]["upside"]["exact_target_hit"]),
        "downside_hit": bool(document["extreme_forecasts"]["downside"]["exact_target_hit"]),
        "net_return": net_return,
        "drawdown": max(0.0, -net_return),
    }


def settle_pending_trial(
    state: dict[str, object],
    *,
    previous_baseline_signal: str | Path,
    previous_candidate_signal: str | Path,
    baseline_settlement: str | Path,
    candidate_settlement: str | Path,
    outcome_known_at: datetime | None = None,
) -> bool:
    pending_payload = state.get("pending_trial")
    if not pending_payload:
        return False
    trial = _trial_from_payload(pending_payload)
    if _file_digest(previous_baseline_signal) != trial.baseline_prediction_sha256:
        raise ValueError("baseline forecast changed after recursive RSI freeze")
    if _file_digest(previous_candidate_signal) != trial.candidate_prediction_sha256:
        raise ValueError("candidate forecast changed after recursive RSI freeze")
    baseline = _settlement_metrics(baseline_settlement)
    candidate = _settlement_metrics(candidate_settlement)
    known = (outcome_known_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    evaluation = MarketFutureEvaluation(
        candidate_id=trial.candidate_id,
        session=trial.session,
        prediction_frozen_at=trial.prediction_frozen_at,
        outcome_known_at=known,
        frozen_prediction_sha256=trial.baseline_prediction_sha256,
        candidate_prediction_sha256=trial.candidate_prediction_sha256,
        candidate_manifest_sha256=trial.candidate_manifest_sha256,
        trial_manifest_sha256=trial_manifest_digest(trial),
        baseline_loss=float(baseline["loss"]),
        candidate_loss=float(candidate["loss"]),
        baseline_upside_hit=bool(baseline["upside_hit"]),
        candidate_upside_hit=bool(candidate["upside_hit"]),
        baseline_downside_hit=bool(baseline["downside_hit"]),
        candidate_downside_hit=bool(candidate["downside_hit"]),
        baseline_net_return=float(baseline["net_return"]),
        candidate_net_return=float(candidate["net_return"]),
        baseline_drawdown=float(baseline["drawdown"]),
        candidate_drawdown=float(candidate["drawdown"]),
    )
    state.setdefault("evaluations", []).append(
        {
            **evaluation.__dict__,
            "prediction_frozen_at": evaluation.prediction_frozen_at.isoformat(),
            "outcome_known_at": evaluation.outcome_known_at.isoformat(),
        }
    )
    state["pending_trial"] = None
    return True


def evaluate_and_rotate_candidate(
    state: dict[str, object],
    source_commit: str,
    now: datetime | None = None,
    *,
    min_future_sessions: int = 20,
) -> dict[str, object] | None:
    if len(state.get("evaluations", [])) < min_future_sessions:
        return None
    candidate = _candidate_from_payload(state["candidate"])
    trials = [_trial_from_payload(value) for value in state["trials"]]
    evaluations = [_evaluation_from_payload(value) for value in state["evaluations"]]
    report = MarketRecursiveImprovementGate(
        min_future_sessions=min_future_sessions
    ).evaluate(candidate, evaluations, trials)
    report_payload = report.to_dict()
    promoted = report.status == "PROMOTION_PROPOSED"
    if promoted:
        state["active_model"] = {
            "model_id": candidate.candidate_id,
            "generation": candidate.generation,
            "parameters": deepcopy(dict(candidate.parameters)),
            "promotion_report_sha256": report.report_sha256,
        }
    state.setdefault("completed_candidates", []).append(
        {
            **report_payload,
            "parameter_promotion_applied": promoted,
            "completed_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        }
    )
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    next_candidate = _new_candidate(state, source_commit, created - timedelta(microseconds=1))
    state["attempt"] = int(state["attempt"]) + 1
    state["candidate"] = next_candidate.sealed_payload()
    state["candidate_manifest_sha256"] = candidate_manifest_digest(next_candidate)
    state["trials"] = []
    state["evaluations"] = []
    state["pending_trial"] = None
    return report_payload


def register_frozen_trial(
    state: dict[str, object],
    baseline_signal_path: str | Path,
    candidate_signal_path: str | Path,
) -> MarketFrozenTrial:
    if state.get("pending_trial") is not None:
        raise ValueError("a recursive RSI trial is already awaiting its outcome")
    baseline = json.loads(Path(baseline_signal_path).read_text(encoding="utf-8"))
    challenger = json.loads(Path(candidate_signal_path).read_text(encoding="utf-8"))
    first = baseline["signals"][0]
    other = challenger["signals"][0]
    if (
        first["signal_session"] != other["signal_session"]
        or first["target_session"] != other["target_session"]
        or first["input_sha256"] != other["input_sha256"]
    ):
        raise ValueError("baseline and candidate forecasts are not aligned")
    frozen_at = max(
        _parse_time(str(first["generated_at_utc"])),
        _parse_time(str(other["generated_at_utc"])),
    )
    candidate = _candidate_from_payload(state["candidate"])
    trial = MarketFrozenTrial(
        candidate_id=candidate.candidate_id,
        session=str(first["target_session"]),
        registered_at=frozen_at,
        prediction_frozen_at=frozen_at,
        baseline_prediction_sha256=_file_digest(baseline_signal_path),
        candidate_prediction_sha256=_file_digest(candidate_signal_path),
        candidate_manifest_sha256=candidate_manifest_digest(candidate),
        input_sha256=str(first["input_sha256"]),
    )
    payload = {
        **trial.__dict__,
        "registered_at": trial.registered_at.isoformat(),
        "prediction_frozen_at": trial.prediction_frozen_at.isoformat(),
        "trial_manifest_sha256": trial_manifest_digest(trial),
    }
    state.setdefault("trials", []).append(payload)
    state["pending_trial"] = payload
    return trial
