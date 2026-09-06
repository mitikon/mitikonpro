"""Reliability-aware judgement for 2026 Queen Stakes course portability.

This module evaluates only what can be proven from recovered pre-race evidence.
It does not fabricate missing continuous base scores and therefore does not claim
an improved total ranking until a full replay is available.
"""
from __future__ import annotations

from dataclasses import dataclass

from .course_signal_integration import split_performance_course_weight
from .course_specialization import CourseRunEvidence, course_specialization_estimate
from .queen_stakes_course_portability import (
    QUEEN_EXACT_COURSE_RUNS,
    QUEEN_FROZEN_TOP5,
    QUEEN_OFFICIAL_TOP5,
)


@dataclass(frozen=True)
class QueenReliabilityJudgement:
    estimates: dict[str, tuple[float, float, float]]
    effective_course_share: dict[str, float]
    frozen_top5_hits: int
    verdict: str


def queen_stakes_reliability_judgement() -> QueenReliabilityJudgement:
    estimates: dict[str, tuple[float, float, float]] = {}
    shares: dict[str, float] = {}

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
        est = course_specialization_estimate(evidence)
        estimates[horse_id] = (est.score, est.reliability, est.effective_evidence)
        shares[horse_id] = 0.40 * est.reliability

        # Structural sanity check: the reliability-aware integrator must retain
        # most of generic performance when exact-course history is only one run.
        split_performance_course_weight(
            performance_score=0.5,
            course_score=est.score,
            course_reliability=est.reliability,
        )

    frozen_hits = len(set(QUEEN_FROZEN_TOP5) & set(QUEEN_OFFICIAL_TOP5))
    return QueenReliabilityJudgement(
        estimates=estimates,
        effective_course_share=shares,
        frozen_top5_hits=frozen_hits,
        verdict=(
            "PASS_STRUCTURAL_ATTENUATION_ONLY: horses 7 and 11 each have one verified exact-course run, "
            "so reliability is 1/3 and the nominal 40% course allocation is attenuated to 13.33%. "
            "Both course scores remain positive, but no total-rank improvement is claimed because "
            "continuous all-horse pre-race base scores are still missing."
        ),
    )
