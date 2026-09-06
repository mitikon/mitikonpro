from racing_lambda.ibis_course_aggregate_judgement import ibis_course_aggregate_judgement


def test_official_top5_course_signal_is_pre_race_supported_without_forcing_ranking():
    result = ibis_course_aggregate_judgement()
    assert result.positive_official_top5 == ("6", "11", "4", "16")
    assert result.positive_official_top5_count == 4
    assert result.full_ranking_recalc_allowed is False


def test_course_ranking_is_deterministic():
    result = ibis_course_aggregate_judgement()
    assert result.ranking[:9] == ("1", "10", "8", "9", "13", "4", "6", "11", "16")
