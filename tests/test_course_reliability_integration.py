from racing_lambda.course_signal_integration import split_performance_course_weight
from racing_lambda.course_specialization import (
    CourseRunEvidence,
    course_specialization_estimate,
)


def _run(*, same_venue=True, same_surface=True, delta=0, same_layout=True, finish=1, field=14):
    return CourseRunEvidence(
        same_venue=same_venue,
        same_surface=same_surface,
        distance_delta_m=delta,
        same_layout=same_layout,
        finish=finish,
        field_size=field,
    )


def test_no_course_evidence_has_zero_reliability():
    estimate = course_specialization_estimate(())
    assert estimate.score == 0.5
    assert estimate.reliability == 0.0
    assert estimate.effective_evidence == 0.0


def test_single_exact_run_is_positive_but_not_full_reliability():
    estimate = course_specialization_estimate((_run(finish=1),))
    assert estimate.score > 0.5
    assert 0.0 < estimate.reliability < 1.0


def test_repeated_exact_runs_raise_reliability():
    one = course_specialization_estimate((_run(finish=2),))
    three = course_specialization_estimate((_run(finish=2), _run(finish=3), _run(finish=1)))
    assert three.reliability > one.reliability


def test_missing_course_history_preserves_performance_budget():
    performance = 0.72
    value = split_performance_course_weight(
        performance,
        0.5,
        course_reliability=0.0,
    )
    assert value == 0.12 * performance


def test_sparse_course_signal_cannot_use_full_40_percent_share():
    estimate = course_specialization_estimate((_run(finish=1),))
    sparse = split_performance_course_weight(
        0.60,
        estimate.score,
        course_reliability=estimate.reliability,
    )
    full = split_performance_course_weight(0.60, estimate.score, course_reliability=1.0)
    assert sparse != full
    assert abs(sparse - 0.12 * 0.60) < abs(full - 0.12 * 0.60)
