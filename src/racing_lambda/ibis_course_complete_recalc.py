"""Complete exact-course aggregate evidence for 2026 Ibis Summer Dash.

Source: pre-race JRA-hosted distance records as exposed by SportsNavi.  For JRA
racing, turf 1000m records in this field represent the Niigata straight 1000m
course family relevant to the target race.  This module deliberately keeps the
course signal independent from the official result.

A full total-score reranking is *not* fabricated here because the first bridge
calculation persisted only its ranking, not the underlying continuous per-horse
scores.  The module therefore computes the complete 17-horse course signal and
requires continuous base scores before total-score fusion.
"""
from __future__ import annotations

from dataclasses import dataclass


# wins, seconds, thirds, others at JRA turf 1000m before 2026-08-02.
IBIS_TURF1000_RECORD: dict[str, tuple[int, int, int, int]] = {
    "1": (0, 1, 1, 0),
    "2": (0, 0, 0, 2),
    "3": (0, 0, 0, 0),
    "4": (0, 0, 1, 0),
    "5": (0, 0, 0, 0),
    "6": (1, 0, 0, 0),
    "7": (0, 0, 0, 2),
    "8": (3, 1, 2, 3),
    "9": (1, 0, 2, 1),
    "10": (3, 0, 2, 2),
    "11": (0, 1, 0, 0),
    "12": (3, 1, 1, 9),
    "13": (1, 1, 1, 1),
    "14": (1, 0, 0, 4),
    "15": (0, 0, 0, 0),
    "16": (0, 1, 0, 0),
    "17": (0, 0, 0, 0),
}


@dataclass(frozen=True)
class IbisCourseRecalc:
    course_scores: dict[str, float]
    course_ranking: tuple[str, ...]
    positive_signal_horses: tuple[str, ...]
    complete_coverage: bool
    total_rerank_ready: bool
    blocker: str


def aggregate_course_score(record: tuple[int, int, int, int], prior_strength: float = 2.0) -> float:
    """Empirical-Bayes top3 score shrunk to neutral 0.5 for sparse evidence."""
    wins, seconds, thirds, others = record
    starts = wins + seconds + thirds + others
    if starts == 0:
        return 0.5
    top3 = wins + seconds + thirds
    return (top3 + prior_strength * 0.5) / (starts + prior_strength)


def ibis_complete_course_recalc(base_continuous_scores: dict[str, float] | None = None) -> IbisCourseRecalc:
    scores = {horse_id: aggregate_course_score(record) for horse_id, record in IBIS_TURF1000_RECORD.items()}
    ranking = tuple(sorted(scores, key=lambda i: (-scores[i], int(i))))
    positive = tuple(i for i in ranking if scores[i] > 0.5)
    ids = {str(i) for i in range(1, 18)}
    complete = set(scores) == ids

    if base_continuous_scores is None:
        ready = False
        blocker = (
            "The original external bridge persisted only an ordinal ranking, not continuous base scores. "
            "Ordinal ranks must not be converted into fake score distances for fusion. Recover/recompute the "
            "pre-race base continuous scores first, then split the existing 0.12 performance budget."
        )
    else:
        if set(base_continuous_scores) != ids:
            raise ValueError("base_continuous_scores must contain all 17 horses")
        ready = True
        blocker = ""

    return IbisCourseRecalc(
        course_scores=scores,
        course_ranking=ranking,
        positive_signal_horses=positive,
        complete_coverage=complete,
        total_rerank_ready=ready,
        blocker=blocker,
    )
