"""Course-specialization signal for layer-2 racing lambda.

The signal is deliberately separate from generic clock/performance detail.
It rewards evidence on the *target course family* and shrinks sparse samples
toward neutral, preventing one exceptional run from becoming a full-strength
course claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable


@dataclass(frozen=True)
class CourseRunEvidence:
    same_venue: bool
    same_surface: bool
    distance_delta_m: int
    same_layout: bool
    finish: int
    field_size: int


def _finish_quality(finish: int, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (finish - 1) / (field_size - 1)))


def course_specialization_score(
    runs: Iterable[CourseRunEvidence],
    *,
    exact_distance_tolerance_m: int = 0,
    near_distance_tolerance_m: int = 200,
    prior_strength: float = 2.0,
) -> float:
    """Return a 0..1 target-course specialization score.

    Match strength is structural and fixed before result evaluation:
    same surface is mandatory for positive transfer; same layout and venue are
    strongest; exact distance outranks nearby distance. Evidence is weighted by
    finish quality and then empirical-Bayes shrunk toward neutral 0.5.
    """
    weighted_sum = 0.0
    evidence = 0.0
    for run in runs:
        if not run.same_surface:
            continue
        delta = abs(run.distance_delta_m)
        if delta <= exact_distance_tolerance_m:
            distance_fit = 1.0
        elif delta <= near_distance_tolerance_m:
            distance_fit = 0.65
        else:
            distance_fit = 0.20
        layout_fit = 1.0 if run.same_layout else 0.35
        venue_fit = 1.0 if run.same_venue else 0.55
        match = distance_fit * (0.55 * layout_fit + 0.45 * venue_fit)
        quality = _finish_quality(run.finish, run.field_size)
        weighted_sum += match * quality
        evidence += match
    if evidence == 0.0:
        return 0.5
    return (weighted_sum + prior_strength * 0.5) / (evidence + prior_strength)
