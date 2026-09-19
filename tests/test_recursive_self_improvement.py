from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from leading_signal_lambda import (
    MarketFutureEvaluation,
    MarketFrozenTrial,
    MarketRecursiveImprovementGate,
    MarketRsiCandidate,
    candidate_manifest_digest,
    freeze_candidate,
    freeze_promotion_report,
    freeze_trial,
    parameter_manifest_digest,
    trial_manifest_digest,
    validate_successor,
)


UTC = timezone.utc
CREATED = datetime(2026, 9, 14, 0, 0, tzinfo=UTC)
COMMIT = "a" * 40


def candidate(**changes):
    values = {
        "candidate_id": "market-rsi-g1",
        "parent_version": "market-forward-v6",
        "generation": 1,
        "created_at": CREATED,
        "source_commit": COMMIT,
        "parameters": {"relative_strength_feature_weight": 0.20},
    }
    values.update(changes)
    values.setdefault("parameter_manifest_sha256", parameter_manifest_digest(values["parameters"]))
    return MarketRsiCandidate(**values)


def trial_rows():
    sealed = candidate_manifest_digest(candidate())
    rows = []
    for index in range(20):
        frozen = CREATED + timedelta(days=index + 1)
        rows.append(MarketFrozenTrial(
            candidate_id="market-rsi-g1",
            session=f"2026-10-{index + 1:02d}",
            registered_at=frozen - timedelta(hours=1),
            prediction_frozen_at=frozen,
            baseline_prediction_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            candidate_prediction_sha256=hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
            candidate_manifest_sha256=sealed,
            input_sha256=hashlib.sha256(f"input-{index}".encode()).hexdigest(),
        ))
    return rows


def observations(*, better=True):
    rows = []
    sealed_candidate_hash = candidate_manifest_digest(candidate())
    for index, trial in enumerate(trial_rows()):
        frozen = trial.prediction_frozen_at
        rows.append(
            MarketFutureEvaluation(
                candidate_id="market-rsi-g1",
                session=f"2026-10-{index + 1:02d}",
                prediction_frozen_at=frozen,
                outcome_known_at=frozen + timedelta(hours=24),
                frozen_prediction_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                candidate_prediction_sha256=hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
                candidate_manifest_sha256=sealed_candidate_hash,
                trial_manifest_sha256=trial_manifest_digest(trial),
                baseline_loss=0.20,
                candidate_loss=0.15 if better else 0.25,
                baseline_upside_hit=index < 6,
                candidate_upside_hit=index < 8,
                baseline_downside_hit=index < 5,
                candidate_downside_hit=index < 7,
                baseline_net_return=0.001,
                candidate_net_return=0.002 if better else 0.0,
                baseline_drawdown=0.10,
                candidate_drawdown=0.08 if better else 0.12,
            )
        )
    return rows


def test_future_only_candidate_can_propose_promotion():
    report = MarketRecursiveImprovementGate().evaluate(candidate(), observations(), trial_rows())
    assert report.status == "PROMOTION_PROPOSED"
    assert all(report.gates.values())
    assert len(report.report_sha256) == 64
    assert report.to_dict()["human_approval_required"] is True
    assert report.to_dict()["trading_authority"] is False


def test_worse_candidate_is_rejected():
    report = MarketRecursiveImprovementGate().evaluate(candidate(), observations(better=False), trial_rows())
    assert report.status == "REJECTED"
    assert not report.gates["loss_improved"]


def test_post_hoc_candidate_is_blocked():
    rows = observations()
    late = candidate(created_at=rows[0].prediction_frozen_at)
    with pytest.raises(ValueError, match="sealed before"):
        MarketRecursiveImprovementGate().evaluate(late, rows, trial_rows())


def test_fixed_core_and_unknown_parameters_cannot_be_changed():
    with pytest.raises(ValueError, match="fixed PCA"):
        candidate(lambda_reg=0.20)
    with pytest.raises(ValueError, match="non-allow-listed"):
        candidate(parameters={"execute_trade": True})


def test_candidate_and_report_are_write_once(tmp_path):
    manifest = freeze_candidate(candidate(), tmp_path / "candidate.json")
    report = MarketRecursiveImprovementGate().evaluate(candidate(), observations(), trial_rows())
    proposal = freeze_promotion_report(report, tmp_path / "proposal.json")
    frozen_trial = freeze_trial(trial_rows()[0], tmp_path / "trial.json")
    assert manifest.exists() and proposal.exists() and frozen_trial.exists()
    with pytest.raises(FileExistsError):
        freeze_candidate(candidate(), manifest)
    with pytest.raises(FileExistsError):
        freeze_promotion_report(report, proposal)
    with pytest.raises(FileExistsError):
        freeze_trial(trial_rows()[0], frozen_trial)


def test_minimum_future_sessions_is_enforced():
    with pytest.raises(ValueError, match="insufficient future sessions"):
        MarketRecursiveImprovementGate().evaluate(candidate(), observations()[:19], trial_rows()[:19])


def test_evaluation_cannot_be_rebound_after_outcome():
    rows = observations()
    row = rows[0]
    rows[0] = MarketFutureEvaluation(
        **{**row.__dict__, "candidate_prediction_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="does not match pre-outcome"):
        MarketRecursiveImprovementGate().evaluate(candidate(), rows, trial_rows())


def test_next_generation_must_chain_to_promoted_report():
    report = MarketRecursiveImprovementGate().evaluate(candidate(), observations(), trial_rows())
    params = {"relative_strength_feature_weight": 0.25}
    child = candidate(
        candidate_id="market-rsi-g2",
        parent_version=report.candidate_id,
        generation=2,
        parent_report_sha256=report.report_sha256,
        parameters=params,
        parameter_manifest_sha256=parameter_manifest_digest(params),
    )
    validate_successor(report, child)
    with pytest.raises(ValueError, match="chained"):
        validate_successor(report, candidate(
            candidate_id="bad-g2", parent_version=report.candidate_id, generation=2,
            parent_report_sha256="b" * 64,
        ))
