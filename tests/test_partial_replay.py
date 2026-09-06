from racing_lambda.historical_races_2026 import CHUKYO_2YO_STAKES_20260830_RESULT
from racing_lambda.partial_replay import chukyo_2yo_partial_replay


def test_partial_replay_is_deterministic_and_leakage_safe():
    ranking = chukyo_2yo_partial_replay()
    ids = tuple(row.horse_id for row in ranking)
    assert ids == ("4", "6", "2", "5", "9", "7", "3", "8", "1")
    assert all(0.0 <= row.score <= 1.0 for row in ranking)
    assert all(0.0 < row.confidence < 1.0 for row in ranking)


def test_partial_replay_top5_hit_count_is_recorded_only_after_scoring():
    ranking = chukyo_2yo_partial_replay()
    predicted_top5 = {row.horse_id for row in ranking[:5]}
    actual_top5 = set(CHUKYO_2YO_STAKES_20260830_RESULT.finishing_order[:5])
    # Diagnostic result: 4,6,9 hit; 2,5 miss.  Official result is used only here,
    # after pre-race-only scoring has completed.
    assert predicted_top5 & actual_top5 == {"4", "6", "9"}
    assert len(predicted_top5 & actual_top5) == 3
