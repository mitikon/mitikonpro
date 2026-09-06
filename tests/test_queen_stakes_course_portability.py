from racing_lambda.queen_stakes_course_portability import queen_course_portability


def test_frozen_prediction_baseline_is_three_of_five():
    result = queen_course_portability()
    assert result.frozen_top5_hits == 3


def test_verified_exact_course_signals_are_positive_for_7_and_11():
    result = queen_course_portability()
    assert result.recovered_course_scores["7"] > 0.5
    assert result.recovered_course_scores["11"] > 0.5
    assert result.official_top5_with_positive_exact_course_signal == ("7", "11")
