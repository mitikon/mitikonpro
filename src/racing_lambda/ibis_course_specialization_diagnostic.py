"""Pre-race course-specialization diagnostic for the 2026 Ibis Summer Dash.

Only exact Niigata turf-straight 1000m runs that were independently recovered
before the target race are included. Missing horse histories remain unknown and
are not treated as proof of no experience.
"""
from __future__ import annotations

from dataclasses import dataclass

from .course_specialization import CourseRunEvidence, course_specialization_score
from .ibis_external_validation import IBIS_OFFICIAL_TOP5


@dataclass(frozen=True)
class IbisCourseDiagnostic:
    course_scores: dict[str, float]
    recovered_horses: tuple[str, ...]
    official_top5_with_positive_course_signal: tuple[str, ...]
    recovery_coverage: float


# Recovered pre-race exact-course evidence only.
# finish, field size. All entries are Niigata turf straight 1000m.
IBIS_EXACT_COURSE_RUNS: dict[str, tuple[tuple[int, int], ...]] = {
    "1": ((3, 18),),          # 2025 Ibis SD
    "2": ((5, 18),),          # 2025 Ibis SD
    "4": ((3, 16),),          # 2026 Shumpu S
    "6": ((1, 18),),          # 2025 Ibis SD
    "8": ((5, 16), (1, 16), (1, 18)),
    "10": ((1, 16),),         # 2026 Idaten S
    "11": ((2, 16),),         # 2026 Shumpu S
    "12": ((9, 16),),         # 2026 Idaten S
    "13": ((2, 18),),         # 2025 Ibis SD
    "16": ((2, 16),),         # 2026 Idaten S
}


def ibis_course_specialization_diagnostic() -> IbisCourseDiagnostic:
    scores: dict[str, float] = {}
    for horse_id, runs in IBIS_EXACT_COURSE_RUNS.items():
        evidence = tuple(
            CourseRunEvidence(
                same_venue=True,
                same_surface=True,
                distance_delta_m=0,
                same_layout=True,
                finish=finish,
                field_size=field_size,
            )
            for finish, field_size in runs
        )
        scores[horse_id] = course_specialization_score(evidence)

    # Positive is defined structurally relative to neutral 0.5; official result
    # is consulted only after scores have been produced.
    positive_top5 = tuple(
        horse_id for horse_id in IBIS_OFFICIAL_TOP5
        if scores.get(horse_id, 0.5) > 0.5
    )
    return IbisCourseDiagnostic(
        course_scores=scores,
        recovered_horses=tuple(sorted(scores, key=int)),
        official_top5_with_positive_course_signal=positive_top5,
        recovery_coverage=len(scores) / 17.0,
    )
