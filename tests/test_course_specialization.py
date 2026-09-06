from racing_lambda.course_specialization import CourseRunEvidence, course_specialization_score


def test_exact_same_course_good_run_beats_nearby_distance():
    exact = course_specialization_score([
        CourseRunEvidence(True, True, 0, True, 2, 18),
    ])
    near = course_specialization_score([
        CourseRunEvidence(True, True, 200, True, 2, 18),
    ])
    assert exact > near > 0.5


def test_surface_mismatch_cannot_create_positive_course_signal():
    score = course_specialization_score([
        CourseRunEvidence(True, False, 0, True, 1, 18),
    ])
    assert score == 0.5


def test_sparse_exact_win_is_shrunk_below_full_strength():
    score = course_specialization_score([
        CourseRunEvidence(True, True, 0, True, 1, 18),
    ])
    assert 0.5 < score < 1.0


def test_repeated_same_course_evidence_strengthens_signal():
    one = course_specialization_score([
        CourseRunEvidence(True, True, 0, True, 2, 18),
    ])
    repeated = course_specialization_score([
        CourseRunEvidence(True, True, 0, True, 2, 18),
        CourseRunEvidence(True, True, 0, True, 3, 18),
        CourseRunEvidence(True, True, 0, True, 1, 18),
    ])
    assert repeated > one
