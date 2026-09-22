import json

import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.paper_pca_promotion import (
    KNOWN_LIMITATIONS,
    MIN_PROMOTION_SESSIONS,
    evaluate_promotion,
    freeze_daily_forecast,
    initial_state,
    settle_daily_forecast,
)
from leading_signal_lambda.paper_pca_sub import JAPAN_SECTORS, PAPER_WINDOW, US_SECTORS


COLUMNS = list(US_SECTORS + JAPAN_SECTORS)


def sample_returns(rows: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.008, size=(rows, 3))
    loadings = rng.normal(size=(3, len(COLUMNS)))
    noise = rng.normal(0.0, 0.003, size=(rows, len(COLUMNS)))
    values = common @ loadings * 0.25 + noise
    return pd.DataFrame(values, index=pd.bdate_range(start, periods=rows), columns=COLUMNS)


def test_known_limitations_flow_from_state_into_every_report():
    assert "XLC" in KNOWN_LIMITATIONS[0]
    state = initial_state()
    assert state["known_limitations"] == list(KNOWN_LIMITATIONS)
    assert state["production_pca_attached"] is False
    assert state["automatic_trade_execution"] is False


def test_freeze_is_write_once_and_settle_never_mutates_it(tmp_path):
    rolling = sample_returns(PAPER_WINDOW, "2015-01-05", 1)
    prior = sample_returns(500, "2010-01-04", 2)
    current_us = sample_returns(1, "2016-01-04", 3).iloc[0].loc[list(US_SECTORS)]
    baseline_scores = pd.Series(0.0, index=JAPAN_SECTORS)

    frozen = freeze_daily_forecast(
        rolling, prior, current_us, baseline_scores,
        signal_session="2016-01-04", target_session="2016-01-05",
        output=tmp_path / "frozen.json",
    )
    before = frozen.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_daily_forecast(
            rolling, prior, current_us, baseline_scores,
            signal_session="2016-01-04", target_session="2016-01-05",
            output=frozen,
        )
    assert frozen.read_bytes() == before

    actual = sample_returns(1, "2016-01-05", 4).iloc[0].loc[list(JAPAN_SECTORS)]
    settled = settle_daily_forecast(frozen, actual, tmp_path / "settled.json")
    assert frozen.read_bytes() == before  # settling must never touch the frozen artifact

    frozen_payload = json.loads(frozen.read_text())
    candidate_weights = pd.Series(frozen_payload["candidate_weights"])[list(JAPAN_SECTORS)]
    baseline_weights = pd.Series(frozen_payload["baseline_weights"])[list(JAPAN_SECTORS)]

    payload = json.loads(settled.read_text())
    assert payload["status"] == "SETTLED_WITHOUT_FORECAST_MUTATION"
    assert payload["candidate_net_return"] == pytest.approx(float(candidate_weights @ actual))
    assert payload["baseline_net_return"] == pytest.approx(float(baseline_weights @ actual))
    assert payload["candidate_loss"] == pytest.approx(-payload["candidate_net_return"])
    assert payload["baseline_loss"] == pytest.approx(-payload["baseline_net_return"])

    with pytest.raises(FileExistsError):
        settle_daily_forecast(frozen, actual, settled)


def _settlement(path, target_session, *, candidate_loss, baseline_loss):
    payload = {
        "schema_version": "paper-pca-promotion-v1",
        "signal_session": f"signal-{target_session}",
        "target_session": target_session,
        "forecast_sha256": "a" * 64,
        "status": "SETTLED_WITHOUT_FORECAST_MUTATION",
        "candidate_net_return": -candidate_loss,
        "baseline_net_return": -baseline_loss,
        "candidate_loss": candidate_loss,
        "baseline_loss": baseline_loss,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_evaluation_waits_below_the_early_rejection_floor(tmp_path):
    settlements = [
        _settlement(tmp_path / f"s{i}.json", f"2026-01-{i+1:02d}", candidate_loss=0.030, baseline_loss=0.010)
        for i in range(4)
    ]
    state, report = evaluate_promotion(initial_state(), settlements)
    assert report["status"] == "CONTINUE"
    assert report["evaluated_sessions"] == 4
    assert state["generation"] == 0


def test_a_consistently_worse_candidate_is_rejected_before_the_full_window(tmp_path):
    settlements = [
        _settlement(tmp_path / f"s{i}.json", f"2026-01-{i+1:02d}", candidate_loss=0.030, baseline_loss=0.010)
        for i in range(5)
    ]
    state, report = evaluate_promotion(initial_state(), settlements)
    assert report["status"] == "EARLY_REJECTED"
    assert report["evaluated_sessions"] == 5
    assert state["generation"] == 0
    assert state["production_pca_attached"] is False


def test_a_consistently_better_candidate_is_promoted_at_the_floor(tmp_path):
    settlements = [
        _settlement(tmp_path / f"s{i}.json", f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", candidate_loss=0.005, baseline_loss=0.020)
        for i in range(MIN_PROMOTION_SESSIONS)
    ]
    state, report = evaluate_promotion(initial_state(), settlements)
    assert report["status"] == "PROMOTION_PROPOSED"
    assert report["evaluated_sessions"] == MIN_PROMOTION_SESSIONS
    assert state["generation"] == 1
    # Promotion is a proposal for a human, never an automatic live attachment.
    assert report["production_pca_attached"] is False
    assert report["human_approval_required"] is True
    assert report["trading_authority"] is False


def test_an_ambiguous_candidate_keeps_waiting_past_the_floor_instead_of_forcing_rejected(tmp_path):
    """Same regression as recursive_runtime.py: a CONTINUE verdict at/after
    the promotion floor must stay CONTINUE, never collapse into REJECTED.
    """
    per_session_losses = [0.010, 0.028, 0.014, 0.024, 0.011, 0.023]
    settlements = [
        _settlement(
            tmp_path / f"s{i}.json",
            f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            candidate_loss=per_session_losses[i % len(per_session_losses)],
            baseline_loss=0.020,
        )
        for i in range(MIN_PROMOTION_SESSIONS)
    ]
    state, report = evaluate_promotion(initial_state(), settlements)
    assert report["status"] == "CONTINUE"
    assert report["evaluated_sessions"] == MIN_PROMOTION_SESSIONS
    assert state["generation"] == 0


def test_evaluate_promotion_deduplicates_repeated_target_sessions(tmp_path):
    duplicate = [
        _settlement(tmp_path / "dup-a.json", "2026-02-01", candidate_loss=0.005, baseline_loss=0.020),
        _settlement(tmp_path / "dup-b.json", "2026-02-01", candidate_loss=0.005, baseline_loss=0.020),
    ]
    state, report = evaluate_promotion(initial_state(), duplicate)
    assert report["evaluated_sessions"] == 1
    assert state["generation"] == 0
