from racing_lambda.chukyo_performance_detail import (
    chukyo_performance_detail_comparison,
    chukyo_pre_race_performance_scores,
)


def test_performance_detail_scores_are_pre_race_and_bounded():
    scores = chukyo_pre_race_performance_scores()
    assert set(scores) == {str(i) for i in range(1, 10)}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    # Pre-race clocks identify 9 as the strongest performance-detail signal;
    # horse 3 is also materially stronger than front-running 5.
    assert scores["9"] > scores["3"] > scores["5"]


def test_performance_detail_comparison_is_deterministic():
    result = chukyo_performance_detail_comparison()
    assert result.ranking == ("9", "2", "7", "4", "6", "3", "5", "8", "1")
    assert result.top5_hits == 4
    assert result.ranks_of_interest["3"] == 6
    assert result.ranks_of_interest["9"] == 1
