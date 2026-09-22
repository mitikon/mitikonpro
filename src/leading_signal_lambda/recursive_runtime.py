"""Permanent, outcome-blind runtime loop for market recursive self-improvement.

The runtime promotes only bounded model configuration.  It never edits source,
merges Git branches, or places trades.  Every challenger forecast is frozen
before its target session and can only be evaluated against a later close.

Several candidates are explored in parallel (``PARALLEL_CANDIDATES``
independent slots), each sealed against the same baseline and the same
sessions, so a doomed candidate no longer has to occupy the only search slot
for its full evaluation window before a different mutation gets a turn.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from .recursive_self_improvement import (
    MarketFutureEvaluation,
    MarketFrozenTrial,
    MarketRecursiveImprovementGate,
    MarketRsiCandidate,
    candidate_manifest_digest,
    canonicalize_parameter_keys,
    parameter_manifest_digest,
    sequential_loss_improvement_test,
    trial_manifest_digest,
    validate_successor,
)


RUNTIME_SCHEMA_VERSION = "market-recursive-runtime-v2"
LEGACY_RUNTIME_SCHEMA_VERSION = "market-recursive-runtime-v1"
DEFAULT_TRANSACTION_COST = 0.0005
PARALLEL_CANDIDATES = 3


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
    ("relative_strength_feature_weight", (1.0, 0.75, 1.25, 1.50, 0.50, 2.00)),
    ("feature_lags", (5, 3, 7, 10, 2)),
    ("neutral_band", (0.001, 0.0005, 0.0015, 0.0020, 0.0)),
    ("no_trade_threshold", (0.45, 0.40, 0.50, 0.55, 0.35)),
    (
        "relative_strength_periods",
        ([5, 7, 14, 21], [5, 14, 21], [7, 14, 28], [3, 7, 14, 21], [5, 10, 20, 40]),
    ),
    (
        "relative_strength_feature_set",
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
    # A not-yet-promoted active configuration inherited from a pre-rename
    # artifact may still carry old rsi_* keys; canonicalize once here so
    # every newly minted candidate uses only the current parameter names.
    base = canonicalize_parameter_keys(active)
    search_size = 1
    for _, choices in MUTATION_SCHEDULE:
        search_size *= len(choices)
    for offset in range(search_size):
        cursor = (attempt + offset) % search_size
        parameters = deepcopy(base)
        for name, choices in MUTATION_SCHEDULE:
            parameters[name] = deepcopy(choices[cursor % len(choices)])
            cursor //= len(choices)
        if parameters != base:
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


def _new_slot(candidate: MarketRsiCandidate, attempt: int) -> dict[str, object]:
    return {
        "attempt": attempt,
        "candidate": candidate.sealed_payload(),
        "candidate_manifest_sha256": candidate_manifest_digest(candidate),
        "trials": [],
        "evaluations": [],
        "pending_trial": None,
    }


def _seed_slot(state: dict[str, object], source_commit: str, created_at: datetime) -> dict[str, object]:
    candidate = _new_candidate(state, source_commit, created_at)
    state["attempt"] = int(state.get("attempt", 0)) + 1
    return _new_slot(candidate, state["attempt"])


def ensure_candidate_slots(
    state: dict[str, object],
    source_commit: str,
    now: datetime | None = None,
    *,
    parallel_candidates: int = PARALLEL_CANDIDATES,
) -> None:
    """Top up the candidate pool to ``parallel_candidates`` concurrent slots.

    A fresh bootstrap already seeds the full pool; this matters for a state
    just migrated from the single-candidate schema (one carried-over slot)
    and, defensively, for any state short of the target parallelism.
    """
    slots = state.setdefault("candidate_slots", [])
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    while len(slots) < parallel_candidates:
        slots.append(_seed_slot(state, source_commit, created - timedelta(microseconds=1)))


def bootstrap_state(
    default_parameters: Mapping[str, object],
    source_commit: str,
    created_at: datetime | None = None,
    *,
    parallel_candidates: int = PARALLEL_CANDIDATES,
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
        "candidate_slots": [],
        "completed_candidates": [],
        "previous_state_sha256": None,
        "updated_at": now.isoformat(),
    }
    ensure_candidate_slots(state, source_commit, now, parallel_candidates=parallel_candidates)
    return state


def _migrate_v1_slot(legacy: Mapping[str, object]) -> dict[str, object]:
    return {
        "attempt": int(legacy.get("attempt", 0)),
        "candidate": legacy["candidate"],
        "candidate_manifest_sha256": legacy["candidate_manifest_sha256"],
        "trials": list(legacy.get("trials", [])),
        "evaluations": list(legacy.get("evaluations", [])),
        "pending_trial": legacy.get("pending_trial"),
    }


def load_state(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version == LEGACY_RUNTIME_SCHEMA_VERSION:
        legacy = dict(raw)
        expected = legacy.pop("state_sha256", None)
        actual = sha256(_canonical(legacy)).hexdigest()
        if expected != actual:
            raise ValueError("recursive RSI state hash mismatch")
        _candidate_from_payload(legacy["candidate"])
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "active_model": legacy["active_model"],
            "attempt": legacy.get("attempt", 0),
            "candidate_slots": [_migrate_v1_slot(legacy)],
            "completed_candidates": list(legacy.get("completed_candidates", [])),
            "previous_state_sha256": legacy.get("previous_state_sha256"),
            "updated_at": legacy.get("updated_at"),
            "migrated_from": LEGACY_RUNTIME_SCHEMA_VERSION,
        }
    if version != RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported recursive RSI runtime state")
    state = dict(raw)
    expected = state.pop("state_sha256", None)
    actual = sha256(_canonical(state)).hexdigest()
    state["state_sha256"] = expected
    if expected != actual:
        raise ValueError("recursive RSI state hash mismatch")
    for slot in state["candidate_slots"]:
        _candidate_from_payload(slot["candidate"])
    return state


def write_state(
    state: Mapping[str, object], path: str | Path, previous_path: str | Path | None = None
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(dict(state))
    payload.pop("state_sha256", None)
    payload["previous_state_sha256"] = (
        _file_digest(previous_path) if previous_path and Path(previous_path).exists() else None
    )
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["state_sha256"] = sha256(_canonical(payload)).hexdigest()
    try:
        # Exclusive create (not check-then-write) so two concurrent writers
        # targeting the same path cannot both pass a staleness check and
        # have the second silently clobber the first's hash-chained state.
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite recursive RSI state: {destination}") from exc
    return destination


def _settlement_metrics(path: str | Path) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    settlements = document["settlements"]
    if not settlements:
        raise ValueError(f"settlement document has no settled rows: {path}")
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


def settle_pending_trials(
    state: dict[str, object],
    *,
    previous_baseline_signal: str | Path,
    previous_candidate_signals: Sequence[str | Path | None],
    baseline_settlement: str | Path,
    candidate_settlements: Sequence[str | Path | None],
    outcome_known_at: datetime | None = None,
) -> list[bool]:
    """Settle whichever slots currently have a pending trial.

    ``previous_candidate_signals``/``candidate_settlements`` must have one
    entry per slot, but entries for a slot with no pending trial (freshly
    seeded, e.g. a slot padded on top of a state migrated from fewer
    historical slots) are never read and may be ``None``.
    """
    slots = state.get("candidate_slots", [])
    if len(previous_candidate_signals) != len(slots) or len(candidate_settlements) != len(slots):
        raise ValueError(
            "candidate signal/settlement counts must match the number of parallel slots"
        )
    pending_indexes = [index for index, slot in enumerate(slots) if slot.get("pending_trial") is not None]
    if not pending_indexes:
        return [False] * len(slots)

    baseline_digest = _file_digest(previous_baseline_signal)
    baseline_metrics = _settlement_metrics(baseline_settlement)
    known = (outcome_known_at or datetime.now(timezone.utc)).astimezone(timezone.utc)

    built: list[tuple[dict[str, object], MarketFutureEvaluation]] = []
    for index in pending_indexes:
        slot = slots[index]
        trial = _trial_from_payload(slot["pending_trial"])
        if baseline_digest != trial.baseline_prediction_sha256:
            raise ValueError("baseline forecast changed after recursive RSI freeze")
        if _file_digest(previous_candidate_signals[index]) != trial.candidate_prediction_sha256:
            raise ValueError("candidate forecast changed after recursive RSI freeze")
        candidate_metrics = _settlement_metrics(candidate_settlements[index])
        evaluation = MarketFutureEvaluation(
            candidate_id=trial.candidate_id,
            session=trial.session,
            prediction_frozen_at=trial.prediction_frozen_at,
            outcome_known_at=known,
            frozen_prediction_sha256=trial.baseline_prediction_sha256,
            candidate_prediction_sha256=trial.candidate_prediction_sha256,
            candidate_manifest_sha256=trial.candidate_manifest_sha256,
            trial_manifest_sha256=trial_manifest_digest(trial),
            baseline_loss=float(baseline_metrics["loss"]),
            candidate_loss=float(candidate_metrics["loss"]),
            baseline_upside_hit=bool(baseline_metrics["upside_hit"]),
            candidate_upside_hit=bool(candidate_metrics["upside_hit"]),
            baseline_downside_hit=bool(baseline_metrics["downside_hit"]),
            candidate_downside_hit=bool(candidate_metrics["downside_hit"]),
            baseline_net_return=float(baseline_metrics["net_return"]),
            candidate_net_return=float(candidate_metrics["net_return"]),
            baseline_drawdown=float(baseline_metrics["drawdown"]),
            candidate_drawdown=float(candidate_metrics["drawdown"]),
        )
        built.append((slot, evaluation))

    for slot, evaluation in built:
        slot.setdefault("evaluations", []).append(
            {
                **evaluation.__dict__,
                "prediction_frozen_at": evaluation.prediction_frozen_at.isoformat(),
                "outcome_known_at": evaluation.outcome_known_at.isoformat(),
            }
        )
        slot["pending_trial"] = None

    results = [False] * len(slots)
    for index in pending_indexes:
        results[index] = True
    return results


def evaluate_and_rotate_candidates(
    state: dict[str, object],
    source_commit: str,
    now: datetime | None = None,
    *,
    min_future_sessions: int = 20,
    min_early_rejection_sessions: int = 5,
    false_promotion_rate: float = 0.05,
    false_rejection_rate: float = 0.10,
) -> list[dict[str, object]]:
    """Evaluate every concluded parallel slot and reseed its replacement.

    At most one promotion is applied to ``active_model`` per call: when
    several slots conclude PROMOTION_PROPOSED in the same cycle, the one
    with the largest loss improvement is applied; the others are recorded
    honestly as PROMOTION_PROPOSED but ``parameter_promotion_applied=False``,
    since only one configuration can be "the" production model at a time.
    """
    slots = state.get("candidate_slots", [])
    conclusions: list[dict[str, object]] = []
    for index, slot in enumerate(slots):
        evaluations_payload = slot.get("evaluations", [])
        sessions_so_far = len(evaluations_payload)
        if sessions_so_far < min_early_rejection_sessions:
            continue
        evaluations = [_evaluation_from_payload(value) for value in evaluations_payload]

        # A Wald SPRT verdict is valid evidence at any sample size, so a
        # candidate that is *already* statistically conclusively worse than
        # baseline can be abandoned now instead of burning the rest of the
        # min_future_sessions window on a doomed candidate. It also means a
        # candidate must not be force-decided just because it reached
        # min_future_sessions: CONTINUE is a real verdict (the boundary
        # simply has not been crossed yet), so it keeps accumulating
        # sessions past the window exactly as it does before the window,
        # instead of collapsing an inconclusive result into REJECTED.
        evidence = sequential_loss_improvement_test(
            [row.baseline_loss for row in evaluations],
            [row.candidate_loss for row in evaluations],
            alpha=false_promotion_rate,
            beta=false_rejection_rate,
        )
        if evidence.decision == "CONTINUE":
            continue

        if sessions_so_far < min_future_sessions:
            # Never promotes on a partial window: only an early REJECT
            # short-circuits here.
            if evidence.decision != "REJECT":
                continue
            candidate = _candidate_from_payload(slot["candidate"])
            conclusions.append(
                {
                    "index": index,
                    "entry": {
                        "status": "EARLY_REJECTED",
                        "candidate_id": candidate.candidate_id,
                        "generation": candidate.generation,
                        "evaluated_sessions": sessions_so_far,
                        "mean_loss_improvement": evidence.mean_improvement,
                        "log_likelihood_ratio": evidence.log_likelihood_ratio,
                    },
                    "candidate": None,
                    "report": None,
                }
            )
            continue

        candidate = _candidate_from_payload(slot["candidate"])
        trials = [_trial_from_payload(value) for value in slot.get("trials", [])]
        report = MarketRecursiveImprovementGate(
            min_future_sessions=min_future_sessions,
            false_promotion_rate=false_promotion_rate,
            false_rejection_rate=false_rejection_rate,
        ).evaluate(candidate, evaluations, trials)
        promotable = report.status == "PROMOTION_PROPOSED"
        conclusions.append(
            {
                "index": index,
                "entry": report.to_dict(),
                "candidate": candidate if promotable else None,
                "report": report if promotable else None,
            }
        )

    if not conclusions:
        return []

    promotable_conclusions = [item for item in conclusions if item["report"] is not None]
    winner = (
        max(promotable_conclusions, key=lambda item: item["report"].loss_improvement)
        if promotable_conclusions
        else None
    )
    if winner is not None:
        winning_candidate = winner["candidate"]
        winning_report = winner["report"]
        state["active_model"] = {
            "model_id": winning_candidate.candidate_id,
            "generation": winning_candidate.generation,
            "parameters": deepcopy(dict(winning_candidate.parameters)),
            "promotion_report_sha256": winning_report.report_sha256,
        }

    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    results: list[dict[str, object]] = []
    for conclusion in conclusions:
        applied = winner is not None and conclusion["index"] == winner["index"]
        entry = {
            **conclusion["entry"],
            "parameter_promotion_applied": applied,
            "completed_at": created.isoformat(),
        }
        state.setdefault("completed_candidates", []).append(entry)
        next_candidate = _new_candidate(state, source_commit, created - timedelta(microseconds=1))
        if applied:
            # Belt-and-suspenders: confirm the freshly promoted model_id/
            # generation actually chains to this report before it becomes
            # the recursive parent.
            validate_successor(conclusion["report"], next_candidate)
        state["attempt"] = int(state["attempt"]) + 1
        state["candidate_slots"][conclusion["index"]] = _new_slot(next_candidate, state["attempt"])
        results.append(entry)
    return results


def register_frozen_trials(
    state: dict[str, object],
    baseline_signal_path: str | Path,
    candidate_signal_paths: Sequence[str | Path],
) -> list[MarketFrozenTrial]:
    slots = state.get("candidate_slots", [])
    if len(candidate_signal_paths) != len(slots):
        raise ValueError(
            f"expected {len(slots)} candidate signals for {len(slots)} parallel slots, "
            f"got {len(candidate_signal_paths)}"
        )
    for slot in slots:
        if slot.get("pending_trial") is not None:
            raise ValueError("a recursive RSI trial is already awaiting its outcome")

    baseline = json.loads(Path(baseline_signal_path).read_text(encoding="utf-8"))
    if not baseline.get("signals"):
        raise ValueError(f"baseline signal document has no frozen signals: {baseline_signal_path}")
    first = baseline["signals"][0]
    baseline_digest = _file_digest(baseline_signal_path)

    built: list[tuple[dict[str, object], MarketFrozenTrial, dict[str, object]]] = []
    for slot, candidate_signal_path in zip(slots, candidate_signal_paths):
        challenger = json.loads(Path(candidate_signal_path).read_text(encoding="utf-8"))
        if not challenger.get("signals"):
            raise ValueError(f"candidate signal document has no frozen signals: {candidate_signal_path}")
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
        candidate = _candidate_from_payload(slot["candidate"])
        trial = MarketFrozenTrial(
            candidate_id=candidate.candidate_id,
            session=str(first["target_session"]),
            registered_at=frozen_at,
            prediction_frozen_at=frozen_at,
            baseline_prediction_sha256=baseline_digest,
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
        built.append((slot, trial, payload))

    for slot, _trial, payload in built:
        slot.setdefault("trials", []).append(payload)
        slot["pending_trial"] = payload
    return [trial for _, trial, _ in built]
