"""Leakage-safe course-fit judgement for the 2026 Ibis Summer Dash.

Uses only pre-race JRA turf-1000m aggregate records. In JRA, turf 1000m maps to
Niigata's straight course, so the aggregate is a direct proxy for target-layout
experience. This module judges the *course-signal hypothesis* only; it does not
invent the missing continuous base scores of the earlier bridge replay.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CourseAggregate:
    wins: int
    seconds: int
    thirds: int
    others: int

    @property
    def starts(self) -> int:
        return self.wins + self.seconds + self.thirds + self.others

    @property
    def top3(self) -> int:
        return self.wins + self.seconds + self.thirds


IBIS_TURF_1000_PRE_RACE: dict[str, CourseAggregate] = {
    "1": CourseAggregate(0,1,1,0),
    "2": CourseAggregate(0,0,0,2),
    "3": CourseAggregate(0,0,0,0),
    "4": CourseAggregate(0,0,1,0),
    "5": CourseAggregate(0,0,0,0),
    "6": CourseAggregate(1,0,0,0),
    "7": CourseAggregate(0,0,0,2),
    "8": CourseAggregate(3,1,2,3),
    "9": CourseAggregate(1,0,2,1),
    "10": CourseAggregate(3,0,2,2),
    "11": CourseAggregate(0,1,0,0),
    "12": CourseAggregate(3,1,1,9),
    "13": CourseAggregate(1,1,1,1),
    "14": CourseAggregate(1,0,0,4),
    "15": CourseAggregate(0,0,0,0),
    "16": CourseAggregate(0,1,0,0),
    "17": CourseAggregate(0,0,0,0),
}

IBIS_OFFICIAL_TOP5 = ("6", "11", "4", "17", "16")


def aggregate_course_fit(record: CourseAggregate, *, prior_rate: float = 0.30, prior_strength: float = 2.0) -> float:
    """Empirical-Bayes top3 fit, shrunk toward neutral 0.5.

    Zero starts means no evidence and stays neutral. Constants are fixed for the
    diagnostic and are not searched against the target result.
    """
    if record.starts == 0:
        return 0.5
    posterior = (record.top3 + prior_strength * prior_rate) / (record.starts + prior_strength)
    centered = 0.5 + 0.5 * (posterior - prior_rate) / (1.0 - prior_rate)
    return max(0.0, min(1.0, centered))


@dataclass(frozen=True)
class IbisCourseJudgement:
    ranking: tuple[str, ...]
    scores: dict[str, float]
    positive_official_top5: tuple[str, ...]
    positive_official_top5_count: int
    conclusion: str
    full_ranking_recalc_allowed: bool


def ibis_course_aggregate_judgement() -> IbisCourseJudgement:
    scores = {horse_id: aggregate_course_fit(record) for horse_id, record in IBIS_TURF_1000_PRE_RACE.items()}
    ranking = tuple(sorted(scores, key=lambda h: (-scores[h], int(h))))
    positive = tuple(h for h in IBIS_OFFICIAL_TOP5 if scores[h] > 0.5)
    return IbisCourseJudgement(
        ranking=ranking,
        scores=scores,
        positive_official_top5=positive,
        positive_official_top5_count=len(positive),
        conclusion=(
            "Course specialization is structurally supported: four of the official top five had positive "
            "pre-race turf-1000 evidence. Horse 17 had no pre-race turf-1000 sample and therefore remains neutral. "
            "This supports adding an independent course signal, but does not prove a Top5-ranking improvement."
        ),
        full_ranking_recalc_allowed=False,
    )
