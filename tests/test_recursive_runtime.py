from datetime import datetime, timedelta, timezone
import json

import pytest

from leading_signal_lambda.forward import DEFAULT_MODEL_PARAMETERS, normalize_model_parameters
from leading_signal_lambda.recursive_self_improvement import (
    MarketRsiCandidate,
    parameter_manifest_digest,
)
from leading_signal_lambda.recursive_runtime import (
    LEGACY_RUNTIME_SCHEMA_VERSION,
    PARALLEL_CANDIDATES,
    _new_candidate,
    _next_parameters,
    bootstrap_state,
    candidate_manifest_digest,
    ensure_candidate_slots,
    evaluate_and_rotate_candidates,
    load_state,
    register_frozen_trials,
    settle_pending_trials,
    write_state,
)


UTC = timezone.utc
COMMIT = "a" * 40


def _signal(path, frozen, session, predicted_return):
    payload = {
        "schema_version": "market-forward-v7",
        "primary_trade": {"target": "SPY"},
        "extreme_forecasts": {},
        "signals": [
            {
                "signal_session": (session - timedelta(days=1)).date().isoformat(),
                "target_session": session.date().isoformat(),
                "target": "SPY",
                "generated_at_utc": frozen.isoformat(),
                "input_sha256": "b" * 64,
                "predicted_return": predicted_return,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _settlement(path, loss, net_return, up_hit, down_hit):
    payload = {
        "settlements": [{"return_error": loss}],
        "primary_trade": {"gross_return_before_cost": net_return + 0.0005},
        "extreme_forecasts": {
            "upside": {"exact_target_hit": up_hit},
            "downside": {"exact_target_hit": down_hit},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _register_and_settle_round(
    state, tmp_path, index, created, *, baseline_loss, candidate_losses, hits=False
):
    """Register + settle one day's trial across every parallel slot."""
    frozen = created + timedelta(days=index + 1)
    session = frozen + timedelta(days=1)
    baseline_signal = _signal(tmp_path / f"baseline-{index}.json", frozen, session, 0.01)
    candidate_signals = [
        _signal(tmp_path / f"candidate-{slot}-{index}.json", frozen, session, 0.02)
        for slot in range(len(state["candidate_slots"]))
    ]
    register_frozen_trials(state, baseline_signal, candidate_signals)
    baseline_settlement = _settlement(
        tmp_path / f"baseline-settlement-{index}.json", baseline_loss, 0.001, hits, hits
    )
    candidate_settlements = [
        _settlement(
            tmp_path / f"candidate-settlement-{slot}-{index}.json",
            candidate_losses[slot],
            0.003,
            hits,
            hits,
        )
        for slot in range(len(state["candidate_slots"]))
    ]
    settled = settle_pending_trials(
        state,
        previous_baseline_signal=baseline_signal,
        previous_candidate_signals=candidate_signals,
        baseline_settlement=baseline_settlement,
        candidate_settlements=candidate_settlements,
        outcome_known_at=frozen + timedelta(days=2),
    )
    assert settled == [True] * len(state["candidate_slots"])


def test_state_is_hash_chained_and_tampering_is_rejected(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    first = write_state(state, tmp_path / "state.json")
    loaded = load_state(first)
    assert loaded["active_model"]["generation"] == 0
    assert len(loaded["candidate_slots"]) == PARALLEL_CANDIDATES
    assert all(slot["candidate"]["generation"] == 1 for slot in loaded["candidate_slots"])

    payload = json.loads(first.read_text())
    payload["active_model"]["parameters"]["feature_lags"] = 10
    first.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash mismatch"):
        load_state(first)


def test_write_state_refuses_to_overwrite_via_atomic_create(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    destination = tmp_path / "state.json"
    write_state(state, destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite recursive RSI state"):
        write_state(state, destination)
    # The refused second write must not have touched the first write's bytes.
    assert load_state(destination)["active_model"]["generation"] == 0


def _legacy_v1_state(created):
    legacy_state = {
        "schema_version": LEGACY_RUNTIME_SCHEMA_VERSION,
        "active_model": {
            "model_id": "market-forward-v7-baseline",
            "generation": 0,
            "parameters": dict(DEFAULT_MODEL_PARAMETERS),
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
        "updated_at": created.isoformat(),
    }
    candidate = _new_candidate(legacy_state, COMMIT, created - timedelta(microseconds=1))
    legacy_state["attempt"] = 1
    legacy_state["candidate"] = candidate.sealed_payload()
    legacy_state["candidate_manifest_sha256"] = candidate_manifest_digest(candidate)
    return legacy_state, candidate


def test_pre_rename_rsi_parameter_keys_still_load_and_resolve(tmp_path):
    """Reproduces production state frozen before the rsi_* -> relative_strength_* rename.

    Historical state and candidates are hash-locked and must load exactly as
    written, without their parameter keys being rewritten in place.
    """
    created = datetime(2026, 9, 15, tzinfo=UTC)
    old_style_parameters = {
        "rsi_periods": [5, 7, 14, 21],
        "rsi_feature_set": ["level", "velocity3", "cross50", "extreme_state"],
        "rsi_feature_weight": 1.0,
        "feature_lags": 5,
        "neutral_band": 0.001,
        "no_trade_threshold": 0.45,
    }
    old_style_candidate_parameters = {**old_style_parameters, "rsi_feature_weight": 1.25}
    candidate = MarketRsiCandidate(
        candidate_id="market-rsi-g1-a1",
        parent_version="market-forward-v7-baseline",
        generation=1,
        created_at=created - timedelta(microseconds=1),
        source_commit=COMMIT,
        parameter_manifest_sha256=parameter_manifest_digest(old_style_candidate_parameters),
        parameters=old_style_candidate_parameters,
    )
    legacy_state = {
        "schema_version": LEGACY_RUNTIME_SCHEMA_VERSION,
        "active_model": {
            "model_id": "market-forward-v7-baseline",
            "generation": 0,
            "parameters": dict(old_style_parameters),
            "promotion_report_sha256": None,
        },
        "attempt": 1,
        "candidate": candidate.sealed_payload(),
        "candidate_manifest_sha256": candidate_manifest_digest(candidate),
        "trials": [],
        "evaluations": [],
        "pending_trial": None,
        "completed_candidates": [],
        "previous_state_sha256": None,
        "updated_at": created.isoformat(),
    }
    legacy_path = write_state(legacy_state, tmp_path / "pre_rename_state.json")

    loaded = load_state(legacy_path)

    assert loaded["active_model"]["parameters"] == old_style_parameters
    assert loaded["candidate_slots"][0]["candidate"]["parameters"] == old_style_candidate_parameters
    # The old spellings must still resolve to the right values, not silently
    # fall back to defaults because the new keys were absent.
    resolved = normalize_model_parameters(loaded["active_model"]["parameters"])
    assert resolved["relative_strength_periods"] == [5, 7, 14, 21]
    assert resolved["relative_strength_feature_weight"] == 1.0
    resolved_candidate = normalize_model_parameters(
        loaded["candidate_slots"][0]["candidate"]["parameters"]
    )
    assert resolved_candidate["relative_strength_feature_weight"] == 1.25
    # Any newly minted candidate must use only the current parameter names.
    next_parameters = _next_parameters(loaded["active_model"]["parameters"], attempt=1)
    assert "rsi_feature_weight" not in next_parameters
    assert "relative_strength_feature_weight" in next_parameters
    # This production state predates confidence_temperature entirely (no key
    # at all, not even an old spelling). normalize_model_parameters must fill
    # it in from the default rather than raise a KeyError, and a freshly
    # minted candidate must always carry an explicit value for it.
    assert "confidence_temperature" not in old_style_parameters
    assert resolved["confidence_temperature"] == 1.0
    assert "confidence_temperature" in next_parameters


def test_legacy_v1_state_migrates_and_tops_up_to_parallel_slots(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    legacy_state, candidate = _legacy_v1_state(created)
    legacy_path = write_state(legacy_state, tmp_path / "legacy_state.json")

    loaded = load_state(legacy_path)
    assert loaded["schema_version"] == "market-recursive-runtime-v2"
    assert loaded["migrated_from"] == LEGACY_RUNTIME_SCHEMA_VERSION
    assert len(loaded["candidate_slots"]) == 1
    assert loaded["candidate_slots"][0]["candidate"]["candidate_id"] == candidate.candidate_id

    ensure_candidate_slots(loaded, COMMIT, created, parallel_candidates=PARALLEL_CANDIDATES)
    assert len(loaded["candidate_slots"]) == PARALLEL_CANDIDATES
    # The carried-over slot stays first and is untouched by the top-up.
    assert loaded["candidate_slots"][0]["candidate"]["candidate_id"] == candidate.candidate_id
    new_ids = {slot["candidate"]["candidate_id"] for slot in loaded["candidate_slots"][1:]}
    assert candidate.candidate_id not in new_ids
    assert len(new_ids) == PARALLEL_CANDIDATES - 1


def test_legacy_v1_state_tampering_is_rejected(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    legacy_state, _candidate = _legacy_v1_state(created)
    legacy_path = write_state(legacy_state, tmp_path / "legacy_state.json")

    payload = json.loads(legacy_path.read_text())
    payload["candidate"]["parameters"]["relative_strength_feature_weight"] = 999.0
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash mismatch"):
        load_state(legacy_path)


def test_parallel_candidates_pick_the_strongest_promotion_and_reseed_every_slot(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    assert len(state["candidate_slots"]) == PARALLEL_CANDIDATES
    starting_candidate_ids = [slot["candidate"]["candidate_id"] for slot in state["candidate_slots"]]
    # Baseline loss is 0.020; slot 1 has the largest consistent improvement
    # (0.015) and must win even though it is not the first slot.
    candidate_losses = [0.015, 0.005, 0.018]
    promoted_parameters = dict(state["candidate_slots"][1]["candidate"]["parameters"])

    for index in range(20):
        _register_and_settle_round(
            state, tmp_path, index, created,
            baseline_loss=0.020, candidate_losses=candidate_losses, hits=True,
        )

    results = evaluate_and_rotate_candidates(state, COMMIT, created + timedelta(days=30))
    assert len(results) == 3
    applied = [entry for entry in results if entry["parameter_promotion_applied"]]
    assert len(applied) == 1
    assert applied[0]["candidate_id"] == starting_candidate_ids[1]
    # Every concluded slot's own report is honest, even the ones not applied.
    statuses = {entry["candidate_id"]: entry["status"] for entry in results}
    assert statuses[starting_candidate_ids[0]] == "PROMOTION_PROPOSED"
    assert statuses[starting_candidate_ids[1]] == "PROMOTION_PROPOSED"
    assert statuses[starting_candidate_ids[2]] == "PROMOTION_PROPOSED"

    assert state["active_model"]["generation"] == 1
    assert state["active_model"]["parameters"] == promoted_parameters
    assert len(state["candidate_slots"]) == PARALLEL_CANDIDATES
    for slot in state["candidate_slots"]:
        assert slot["candidate"]["candidate_id"] not in starting_candidate_ids
        assert slot["candidate"]["generation"] == 2
        assert slot["trials"] == []
        assert slot["evaluations"] == []
        assert slot["pending_trial"] is None


def test_all_clearly_worse_candidates_are_abandoned_before_the_full_window(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    starting_active_model = dict(state["active_model"])
    starting_candidate_ids = [slot["candidate"]["candidate_id"] for slot in state["candidate_slots"]]

    for index in range(5):
        _register_and_settle_round(
            state, tmp_path, index, created,
            baseline_loss=0.010, candidate_losses=[0.030, 0.030, 0.030],
        )
        if index < 4:
            # Fewer than min_early_rejection_sessions (5): still accumulating.
            frozen = created + timedelta(days=index + 1)
            assert evaluate_and_rotate_candidates(state, COMMIT, frozen) == []

    results = evaluate_and_rotate_candidates(state, COMMIT, created + timedelta(days=10))
    assert len(results) == 3
    assert all(entry["status"] == "EARLY_REJECTED" for entry in results)
    assert all(entry["evaluated_sessions"] == 5 for entry in results)
    assert all(not entry["parameter_promotion_applied"] for entry in results)
    # A consistently worse candidate never gets promoted on a partial window.
    assert state["active_model"] == starting_active_model
    assert len(state["candidate_slots"]) == PARALLEL_CANDIDATES
    for slot in state["candidate_slots"]:
        assert slot["candidate"]["candidate_id"] not in starting_candidate_ids
        assert slot["candidate"]["generation"] == 1
        assert slot["trials"] == []
        assert slot["evaluations"] == []
        assert slot["pending_trial"] is None


def test_ambiguous_candidates_keep_accumulating_past_the_early_floor(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)

    # Noisy, mostly-positive-but-inconsistent per-session advantage: neither
    # a confident PROMOTE nor a confident REJECT under the SPRT boundaries.
    per_session_losses = [0.010, 0.028, 0.014, 0.024, 0.011, 0.023]
    for index in range(6):
        loss = per_session_losses[index]
        _register_and_settle_round(
            state, tmp_path, index, created,
            baseline_loss=0.020, candidate_losses=[loss, loss, loss],
        )

    # Still well short of min_future_sessions (20) and not a statistically
    # conclusive early REJECT either: the loop must keep waiting, never
    # forcing a decision on a partial, ambiguous window.
    assert evaluate_and_rotate_candidates(state, COMMIT, created + timedelta(days=10)) == []
    assert all(len(slot["evaluations"]) == 6 for slot in state["candidate_slots"])


def test_ambiguous_candidates_keep_accumulating_past_min_future_sessions(tmp_path):
    """A CONTINUE verdict at/after the 20-session window must not collapse to REJECTED.

    Regression test: evaluate_and_rotate_candidates used to call the full
    promotion gate unconditionally once a slot reached min_future_sessions,
    even when the Wald SPRT verdict was still CONTINUE (inconclusive). That
    turned gates["loss_improved"] False and forced status="REJECTED",
    discarding 20 sessions of genuinely unresolved evidence and reseeding
    the slot from scratch instead of letting the SPRT keep accumulating,
    exactly as it is allowed to before the window.
    """
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    starting_candidate_ids = [slot["candidate"]["candidate_id"] for slot in state["candidate_slots"]]

    # Same noisy, mostly-positive-but-inconsistent pattern used below the
    # early floor, just cycled long enough to cross min_future_sessions (20)
    # while remaining strictly between the SPRT's REJECT/PROMOTE boundaries.
    per_session_losses = [0.010, 0.028, 0.014, 0.024, 0.011, 0.023]
    for index in range(20):
        loss = per_session_losses[index % len(per_session_losses)]
        _register_and_settle_round(
            state, tmp_path, index, created,
            baseline_loss=0.020, candidate_losses=[loss, loss, loss],
        )

    results = evaluate_and_rotate_candidates(state, COMMIT, created + timedelta(days=30))
    assert results == []
    for slot in state["candidate_slots"]:
        assert slot["candidate"]["candidate_id"] in starting_candidate_ids
        assert len(slot["evaluations"]) == 20
        assert slot["pending_trial"] is None


def test_settle_pending_trials_rejects_settlement_with_no_rows(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    frozen = created + timedelta(days=1)
    baseline = _signal(tmp_path / "baseline.json", frozen, frozen + timedelta(days=1), 0.01)
    candidates = [
        _signal(tmp_path / f"candidate-{slot}.json", frozen, frozen + timedelta(days=1), 0.02)
        for slot in range(PARALLEL_CANDIDATES)
    ]
    register_frozen_trials(state, baseline, candidates)

    empty_settlement = tmp_path / "empty-settlement.json"
    empty_settlement.write_text(
        json.dumps(
            {
                "settlements": [],
                "primary_trade": {"gross_return_before_cost": 0.0},
                "extreme_forecasts": {
                    "upside": {"exact_target_hit": False},
                    "downside": {"exact_target_hit": False},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no settled rows"):
        settle_pending_trials(
            state,
            previous_baseline_signal=baseline,
            previous_candidate_signals=candidates,
            baseline_settlement=empty_settlement,
            candidate_settlements=[empty_settlement] * PARALLEL_CANDIDATES,
            outcome_known_at=frozen + timedelta(days=2),
        )


def test_register_frozen_trials_rejects_document_with_no_signals(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"signals": []}), encoding="utf-8")
    candidates = []
    for slot in range(PARALLEL_CANDIDATES):
        candidate_path = tmp_path / f"candidate-{slot}.json"
        candidate_path.write_text(json.dumps({"signals": []}), encoding="utf-8")
        candidates.append(candidate_path)
    with pytest.raises(ValueError, match="no frozen signals"):
        register_frozen_trials(state, baseline, candidates)


def test_register_frozen_trials_rejects_slot_count_mismatch(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    baseline = _signal(tmp_path / "baseline.json", created, created + timedelta(days=1), 0.01)
    only_one_candidate = [_signal(tmp_path / "candidate.json", created, created + timedelta(days=1), 0.02)]
    with pytest.raises(ValueError, match="expected 3 candidate signals"):
        register_frozen_trials(state, baseline, only_one_candidate)


def test_frozen_forecast_tampering_blocks_learning(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    frozen = created + timedelta(days=1)
    baseline = _signal(tmp_path / "baseline.json", frozen, frozen + timedelta(days=1), 0.01)
    candidates = [
        _signal(tmp_path / f"candidate-{slot}.json", frozen, frozen + timedelta(days=1), 0.02)
        for slot in range(PARALLEL_CANDIDATES)
    ]
    register_frozen_trials(state, baseline, candidates)
    baseline.write_text(baseline.read_text() + "\n", encoding="utf-8")
    baseline_settlement = _settlement(tmp_path / "baseline-settlement.json", 0.02, 0.0, False, False)
    candidate_settlement = _settlement(tmp_path / "candidate-settlement.json", 0.01, 0.0, False, False)
    with pytest.raises(ValueError, match="changed after"):
        settle_pending_trials(
            state,
            previous_baseline_signal=baseline,
            previous_candidate_signals=candidates,
            baseline_settlement=baseline_settlement,
            candidate_settlements=[candidate_settlement] * PARALLEL_CANDIDATES,
            outcome_known_at=frozen + timedelta(days=2),
        )
