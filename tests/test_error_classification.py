from datetime import datetime, timezone

import pytest

from leading_signal_lambda import (
    EXTRACTION_MISS,
    FINAL_EXCLUSION,
    MARKET_NOISE,
    MISSING_INPUT,
    MIN_RATIONALE_SESSIONS,
    OVERESTIMATION,
    CandidateRationale,
    classify_extreme_selection,
    classify_target_settlement,
    freeze_candidate_rationale,
    summarize_error_classification,
)


UTC = timezone.utc


def _target_row(**changes):
    values = {
        "direction_correct": False,
        "actual_return": 0.01,
        "predicted_return": -0.01,
        "neutral_band": 0.001,
        "imputed_feature_count": 0,
    }
    values.update(changes)
    return values


def test_correct_direction_is_not_classified():
    assert classify_target_settlement(_target_row(direction_correct=True)) is None


def test_missing_input_takes_priority():
    row = _target_row(imputed_feature_count=2)
    assert classify_target_settlement(row) == MISSING_INPUT


def test_small_actual_move_is_market_noise():
    row = _target_row(actual_return=0.0005, neutral_band=0.001)
    assert classify_target_settlement(row) == MARKET_NOISE


def test_overestimated_same_direction_magnitude():
    row = _target_row(actual_return=0.005, predicted_return=0.02, direction_correct=False)
    assert classify_target_settlement(row) == OVERESTIMATION


def test_wrong_direction_without_noise_or_overestimation_is_extraction_miss():
    row = _target_row(actual_return=0.01, predicted_return=-0.011, neutral_band=0.001)
    assert classify_target_settlement(row) == EXTRACTION_MISS


def test_old_history_row_missing_new_columns_does_not_crash():
    row = {
        "direction_correct": "False",
        "actual_return": 0.01,
        "predicted_return": -0.01,
    }
    assert classify_target_settlement(row) == EXTRACTION_MISS


def _extreme(**changes):
    values = {
        "exact_target_hit": False,
        "actual_extreme_return": 0.02,
        "direction_signal_present": True,
        "selected_actual_rank": 2,
        "predicted_return": 0.03,
        "selected_actual_return": 0.005,
    }
    values.update(changes)
    return values


def test_exact_hit_is_not_classified():
    assert classify_extreme_selection(_extreme(exact_target_hit=True)) is None


def test_no_directional_signal_is_extraction_miss():
    assert classify_extreme_selection(_extreme(direction_signal_present=False)) == EXTRACTION_MISS


def test_near_top_rank_is_final_exclusion():
    assert classify_extreme_selection(_extreme(selected_actual_rank=2)) == FINAL_EXCLUSION


def test_far_rank_with_large_prediction_is_overestimation():
    row = _extreme(selected_actual_rank=10, predicted_return=0.05, selected_actual_return=0.005)
    assert classify_extreme_selection(row) == OVERESTIMATION


def test_negligible_actual_extreme_move_is_market_noise():
    row = _extreme(actual_extreme_return=0.0001, market_noise_band=0.0)
    assert classify_extreme_selection(row, market_noise_band=0.001) == MARKET_NOISE


def test_summarize_error_classification_counts_targets_and_extremes():
    rows = [_target_row(imputed_feature_count=1), _target_row(direction_correct=True)]
    sessions = [{"upside": _extreme(), "downside": _extreme(exact_target_hit=True)}]
    counts = summarize_error_classification(rows, sessions)
    assert counts[MISSING_INPUT] == 1
    assert counts[FINAL_EXCLUSION] == 1
    assert sum(counts.values()) == 2


def _rationale(**changes):
    values = {
        "rationale_id": "rationale-1",
        "created_at": datetime(2026, 9, 14, tzinfo=UTC),
        "evaluated_signal_sessions": tuple(f"2026-09-{day:02d}" for day in range(1, 1 + MIN_RATIONALE_SESSIONS)),
        "error_counts": {MISSING_INPUT: MIN_RATIONALE_SESSIONS},
        "narrative": "欠損入力による抽出漏れが連続したため、代替特徴量の重みを調整する候補を提案する。",
        "source_report_sha256": "a" * 64,
    }
    values.update(changes)
    return CandidateRationale(**values)


def test_rationale_requires_minimum_sessions_to_avoid_single_outlier_overfitting():
    with pytest.raises(ValueError, match="single outlier"):
        _rationale(evaluated_signal_sessions=("2026-09-01",), error_counts={MISSING_INPUT: 1})


def test_rationale_rejects_unexplained_narrative():
    with pytest.raises(ValueError, match="narrative"):
        _rationale(narrative="   ")


def test_rationale_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown error categories"):
        _rationale(error_counts={"unexplained_bucket": 5})


def test_rationale_and_freeze_are_write_once(tmp_path):
    rationale = _rationale()
    path = freeze_candidate_rationale(rationale, tmp_path / "rationale.json")
    assert path.exists()
    with pytest.raises(FileExistsError):
        freeze_candidate_rationale(rationale, path)
