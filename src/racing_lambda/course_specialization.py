"""Course-specialization signal for layer-2 racing lambda.

The signal is deliberately separate from generic clock/performance detail.
It rewards evidence on the *target course family* and shrinks sparse samples
toward neutral, preventing one exceptional run from becoming a full-strength
course claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CourseRunEvidence:
    same_venue: bool
    same_surface: bool
    distance_delta_m: int
    same_layout: bool
    finish: int
    field_size: int


@dataclass(frozen=True)
class CourseSpecializationEstimate:
    score: float
    reliability: float
    effective_evidence: float


def _finish_quality(finish: int, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (finish - 1) / (field_size - 1)))


def _match_strength(
    run: CourseRunEvidence,
    *,
    exact_distance_tolerance_m: int,
    near_distance_tolerance_m: int,
) -> float:
    if not run.same_surface:
        return 0.0
    delta = abs(run.distance_delta_m)
    if delta <= exact_distance_tolerance_m:
        distance_fit = 1.0
    elif delta <= near_distance_tolerance_m:
        distance_fit = 0.65
    else:
        distance_fit = 0.20
    layout_fit = 1.0 if run.same_layout else 0.35
    venue_fit = 1.0 if run.same_venue else 0.55
    return distance_fit * (0.55 * layout_fit + 0.45 * venue_fit)


def course_specialization_estimate(
    runs: Iterable[CourseRunEvidence],
    *,
    exact_distance_tolerance_m: int = 0,
    near_distance_tolerance_m: int = 200,
    prior_strength: float = 2.0,
) -> CourseSpecializationEstimate:
    """Return score plus evidence reliability without result-fitted tuning.

    Reliability is derived only from the amount/quality of structurally matching
    pre-race evidence.  It approaches 1 as effective evidence grows and is 0
    when no transferable evidence exists.  This lets the integrator avoid
    allocating the full course-weight budget to one sparse run.
    """
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")
    weighted_sum = 0.0
    evidence = 0.0
    for run in runs:
        match = _match_strength(
            run,
            exact_distance_tolerance_m=exact_distance_tolerance_m,
            near_distance_tolerance_m=near_distance_tolerance_m,
        )
        if match == 0.0:
            continue
        weighted_sum += match * _finish_quality(run.finish, run.field_size)
        evidence += match
    if evidence == 0.0:
        return CourseSpecializationEstimate(score=0.5, reliability=0.0, effective_evidence=0.0)
    score = (weighted_sum + prior_strength * 0.5) / (evidence + prior_strength)
    reliability = evidence / (evidence + prior_strength)
    return CourseSpecializationEstimate(
        score=score,
        reliability=reliability,
        effective_evidence=evidence,
    )


def course_specialization_score(
    runs: Iterable[CourseRunEvidence],
    *,
    exact_distance_tolerance_m: int = 0,
    near_distance_tolerance_m: int = 200,
    prior_strength: float = 2.0,
) -> float:
    """Backward-compatible 0..1 target-course specialization score."""
    return course_specialization_estimate(
        runs,
        exact_distance_tolerance_m=exact_distance_tolerance_m,
        near_distance_tolerance_m=near_distance_tolerance_m,
        prior_strength=prior_strength,
    ).score
