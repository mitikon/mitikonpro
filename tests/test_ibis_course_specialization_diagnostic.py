from racing_lambda.ibis_course_specialization_diagnostic import ibis_course_specialization_diagnostic


def test_ibis_course_diagnostic_uses_only_recovered_horses():
    result = ibis_course_specialization_diagnostic()
    assert result.recovered_horses == ("1", "2", "4", "6", "8", "10", "11", "12", "13", "16")
    assert 0.58 < result.recovery_coverage < 0.60


def test_recovered_official_top5_all_have_positive_exact_course_signal():
    result = ibis_course_specialization_diagnostic()
    # 6, 11, 4 and 16 had independently recovered exact-course evidence.
    assert set(result.official_top5_with_positive_course_signal) == {"4", "6", "11", "16"}


def test_repeated_exact_course_wins_are_strong_but_shrunk():
    result = ibis_course_specialization_diagnostic()
    assert result.course_scores["8"] > result.course_scores["6"] > 0.5
    assert result.course_scores["8"] < 1.0


def test_poor_exact_course_run_does_not_create_false_positive():
    result = ibis_course_specialization_diagnostic()
    assert result.course_scores["12"] < 0.5
