from datetime import datetime, timedelta, timezone
import json

import pytest

from leading_signal_lambda.forward import DEFAULT_MODEL_PARAMETERS
from leading_signal_lambda.recursive_runtime import (
    bootstrap_state,
    evaluate_and_rotate_candidate,
    load_state,
    register_frozen_trial,
    settle_pending_trial,
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


def test_state_is_hash_chained_and_tampering_is_rejected(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    first = write_state(state, tmp_path / "state.json")
    loaded = load_state(first)
    assert loaded["active_model"]["generation"] == 0
    assert loaded["candidate"]["generation"] == 1

    payload = json.loads(first.read_text())
    payload["active_model"]["parameters"]["feature_lags"] = 10
    first.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash mismatch"):
        load_state(first)


def test_twenty_future_only_trials_promote_and_start_next_generation(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    promoted_parameters = dict(state["candidate"]["parameters"])

    for index in range(20):
        frozen = created + timedelta(days=index + 1)
        session = frozen + timedelta(days=1)
        baseline_signal = _signal(
            tmp_path / f"baseline-{index}.json", frozen, session, 0.01
        )
        candidate_signal = _signal(
            tmp_path / f"candidate-{index}.json", frozen, session, 0.02
        )
        register_frozen_trial(state, baseline_signal, candidate_signal)
        baseline_settlement = _settlement(
            tmp_path / f"baseline-settlement-{index}.json", 0.020, 0.001, index < 5, index < 5
        )
        candidate_settlement = _settlement(
            tmp_path / f"candidate-settlement-{index}.json", 0.010, 0.003, index < 8, index < 8
        )
        settled = settle_pending_trial(
            state,
            previous_baseline_signal=baseline_signal,
            previous_candidate_signal=candidate_signal,
            baseline_settlement=baseline_settlement,
            candidate_settlement=candidate_settlement,
            outcome_known_at=frozen + timedelta(days=2),
        )
        assert settled

    report = evaluate_and_rotate_candidate(
        state, COMMIT, created + timedelta(days=30)
    )
    assert report["status"] == "PROMOTION_PROPOSED"
    assert state["active_model"]["generation"] == 1
    assert state["active_model"]["parameters"] == promoted_parameters
    assert state["candidate"]["generation"] == 2
    assert state["trials"] == []
    assert state["evaluations"] == []
    assert state["completed_candidates"][-1]["parameter_promotion_applied"] is True


def test_frozen_forecast_tampering_blocks_learning(tmp_path):
    created = datetime(2026, 9, 15, tzinfo=UTC)
    state = bootstrap_state(DEFAULT_MODEL_PARAMETERS, COMMIT, created)
    frozen = created + timedelta(days=1)
    baseline = _signal(tmp_path / "baseline.json", frozen, frozen + timedelta(days=1), 0.01)
    candidate = _signal(tmp_path / "candidate.json", frozen, frozen + timedelta(days=1), 0.02)
    register_frozen_trial(state, baseline, candidate)
    baseline.write_text(baseline.read_text() + "\n", encoding="utf-8")
    baseline_settlement = _settlement(tmp_path / "baseline-settlement.json", 0.02, 0.0, False, False)
    candidate_settlement = _settlement(tmp_path / "candidate-settlement.json", 0.01, 0.0, False, False)
    with pytest.raises(ValueError, match="changed after"):
        settle_pending_trial(
            state,
            previous_baseline_signal=baseline,
            previous_candidate_signal=candidate,
            baseline_settlement=baseline_settlement,
            candidate_settlement=candidate_settlement,
            outcome_known_at=frozen + timedelta(days=2),
        )

