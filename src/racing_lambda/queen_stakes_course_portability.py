"""Partial portability check for course specialization on 2026 Queen Stakes.

This check intentionally uses only independently recovered pre-race exact-course
records.  It is not a full replay and must not be used to fabricate missing
continuous base scores.
"""
from __future__ import annotations

from dataclasses import dataclass

from .course_specialization import CourseRunEvidence, course_specialization_score


QUEEN_OFFICIAL_TOP5 = ("7", "11", "14", "9", "3")
QUEEN_FROZEN_TOP5 = ("9", "11", "7", "10", "2")

# Exact Sapporo turf 1800m evidence independently recovered from races before
# 2026-08-02.  Only horses with verified exact-course evidence are included.
QUEEN_EXACT_COURSE_RUNS: dict[str, tuple[tuple[int, int], ...]] = {
    "7": ((2, 14),),   # 2025 Queen Stakes, Sapporo turf 1800m
    "11": ((1, 14),),  # 2024 Queen Stakes, Sapporo turf 1800m
}


@dataclass(frozen=True)
class QueenCoursePortabilityResult:
    frozen_top5_hits: int
    recovered_course_scores: dict[str, float]
    official_top5_with_positive_exact_course_signal: tuple[str, ...]
    verdict: str


def queen_course_portability() -> QueenCoursePortabilityResult:
    scores: dict[str, float] = {}
    for horse_id, runs in QUEEN_EXACT_COURSE_RUNS.items():
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

    positive = tuple(
        horse_id for horse_id in QUEEN_OFFICIAL_TOP5
        if scores.get(horse_id, 0.5) > 0.5
    )
    frozen_hits = len(set(QUEEN_FROZEN_TOP5) & set(QUEEN_OFFICIAL_TOP5))
    return QueenCoursePortabilityResult(
        frozen_top5_hits=frozen_hits,
        recovered_course_scores=scores,
        official_top5_with_positive_exact_course_signal=positive,
        verdict=(
            "Exact-course specialization is directionally useful on a normal oval course, "
            "but available verified evidence only covers horses 7 and 11. Both are official "
            "top-five finishers and both receive positive signals. This supports portability "
            "without proving that course specialization alone improves the frozen 3/5 ranking. "
            "A full all-horse pre-race reconstruction is still required before total-rank claims."
        ),
    )
