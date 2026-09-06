from racing_lambda.ibis_course_complete_recalc import ibis_complete_course_recalc


def test_all_17_horses_have_course_scores():
    result = ibis_complete_course_recalc()
    assert result.complete_coverage
    assert set(result.course_scores) == {str(i) for i in range(1, 18)}


def test_known_positive_and_negative_course_signals_are_preserved():
    result = ibis_complete_course_recalc()
    for horse_id in ("4", "6", "8", "9", "10", "11", "13", "16"):
        assert result.course_scores[horse_id] > 0.5
    for horse_id in ("2", "7", "12", "14"):
        assert result.course_scores[horse_id] < 0.5


def test_no_experience_stays_neutral():
    result = ibis_complete_course_recalc()
    for horse_id in ("3", "5", "15", "17"):
        assert result.course_scores[horse_id] == 0.5


def test_full_total_rerank_refuses_fake_ordinal_score_conversion():
    result = ibis_complete_course_recalc()
    assert not result.total_rerank_ready
    assert "ordinal" in result.blocker.lower()
