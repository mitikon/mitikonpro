"""Pre-race performance-detail diagnostic for the Chukyo 2yo replay.

The feature uses only historical race clocks and final-section times recorded
before the target race. The official target result is consulted only after the
ranking has been produced. No weight search is performed against the result.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .chukyo_2yo_recovered_runs import CHUKYO_2YO_RECOVERED_RUNS, layer2_past_runs_for
from .chukyo_variant_comparison import _aggregate_fit
from .historical_evidence_2026 import (
    CHUKYO_2YO_JOCKEY_STATS,
    CHUKYO_2YO_PAST_RUNS,
    CHUKYO_2YO_TRAINER_STATS,
)
from .historical_races_2026 import CHUKYO_2YO_STAKES_20260830, CHUKYO_2YO_STAKES_20260830_RESULT
from .layer2_live_input import recent_form_score


@dataclass(frozen=True)
class PerformanceDetailComparison:
    ranking: tuple[str, ...]
    top5_hits: int
    top3_hits: int
    performance_scores: dict[str, float]
    ranks_of_interest: dict[str, int]


def _time_seconds(value: str) -> float:
    minute, seconds = value.split(":", 1)
    return int(minute) * 60.0 + float(seconds)


def _percentile_high(value: float, population: list[float]) -> float:
    if len(population) < 2:
        return 0.5
    return float((sum(item <= value for item in population) - 1) / (len(population) - 1))


def _percentile_low(value: float, population: list[float]) -> float:
    if len(population) < 2:
        return 0.5
    return float((sum(item >= value for item in population) - 1) / (len(population) - 1))


def chukyo_pre_race_performance_scores() -> dict[str, float]:
    """Build recency-weighted clock/closing scores from pre-race turf runs.

    Total-race speed is normalized within the same distance among the recovered
    field history. Final-section time is normalized the same way. Dirt runs are
    not treated as transferable turf performance and therefore receive neutral
    0.5 rather than a positive speed credit.
    """
    speed_by_distance: dict[int, list[float]] = {}
    close_by_distance: dict[int, list[float]] = {}

    for horse_id, runs in CHUKYO_2YO_RECOVERED_RUNS.items():
        recorded = CHUKYO_2YO_PAST_RUNS[horse_id]
        if len(runs) != len(recorded):
            raise ValueError(f"recovered/recorded run count mismatch for horse {horse_id}")
        for run, rec in zip(runs, recorded):
            if run.surface != "芝" or rec.time is None or rec.final_section is None:
                continue
            speed_by_distance.setdefault(run.distance_m, []).append(run.distance_m / _time_seconds(rec.time))
            close_by_distance.setdefault(run.distance_m, []).append(float(rec.final_section))

    scores: dict[str, float] = {}
    for horse_id, runs in CHUKYO_2YO_RECOVERED_RUNS.items():
        recorded = CHUKYO_2YO_PAST_RUNS[horse_id]
        values: list[float] = []
        for run, rec in zip(runs, recorded):
            if run.surface != "芝" or rec.time is None or rec.final_section is None:
                values.append(0.5)
                continue
            speed = run.distance_m / _time_seconds(rec.time)
            speed_score = _percentile_high(speed, speed_by_distance[run.distance_m])
            close_score = _percentile_low(float(rec.final_section), close_by_distance[run.distance_m])
            # Fixed before result evaluation: slightly more weight on overall
            # race speed than closing time. This is not target-result tuned.
            values.append(0.55 * speed_score + 0.45 * close_score)
        weights = np.asarray([0.72**i for i in range(len(values))], dtype=float)
        weights /= weights.sum()
        scores[horse_id] = float(np.clip(np.dot(weights, np.asarray(values)), 0.0, 1.0))
    return scores


def chukyo_performance_detail_comparison() -> PerformanceDetailComparison:
    """Add independent performance detail to the best prior structural variant.

    Starting point is the previously identified position-neutralized + market-gap
    removed structure. The retired market-gap weight (0.12) is reassigned to an
    independent pre-race performance-detail signal; no weights are searched.
    """
    ids = [horse.horse_id for horse in CHUKYO_2YO_STAKES_20260830.horses]
    recent = {i: recent_form_score(layer2_past_runs_for(i)) for i in ids}
    jockey = {
        i: _aggregate_fit(
            CHUKYO_2YO_JOCKEY_STATS[i].starts,
            CHUKYO_2YO_JOCKEY_STATS[i].wins,
            CHUKYO_2YO_JOCKEY_STATS[i].seconds,
            CHUKYO_2YO_JOCKEY_STATS[i].thirds,
        ) for i in ids
    }
    trainer = {
        i: _aggregate_fit(
            CHUKYO_2YO_TRAINER_STATS[i].starts,
            CHUKYO_2YO_TRAINER_STATS[i].wins,
            CHUKYO_2YO_TRAINER_STATS[i].seconds,
            CHUKYO_2YO_TRAINER_STATS[i].thirds,
        ) for i in ids
    }
    performance = chukyo_pre_race_performance_scores()

    scores: dict[str, float] = {}
    evidence_coverage = 0.66 / 0.92
    for i in ids:
        # pace is neutralized at 0.5 and market-gap is replaced by independent
        # performance detail at the original 0.12 weight.
        raw = (
            0.22 * recent[i]
            + 0.14 * 0.5
            + 0.10 * jockey[i]
            + 0.08 * trainer[i]
            + 0.12 * performance[i]
        ) / 0.66
        run_coverage = min(len(layer2_past_runs_for(i)) / 5.0, 1.0)
        confidence = evidence_coverage * (0.5 + 0.5 * run_coverage) * 0.90
        scores[i] = 0.5 + confidence * (raw - 0.5)

    ranking = tuple(sorted(ids, key=lambda i: (-scores[i], int(i))))
    actual = CHUKYO_2YO_STAKES_20260830_RESULT.finishing_order
    actual_top5, actual_top3 = set(actual[:5]), set(actual[:3])
    interest = ("2", "3", "4", "5", "6", "7", "9")
    return PerformanceDetailComparison(
        ranking=ranking,
        top5_hits=len(set(ranking[:5]) & actual_top5),
        top3_hits=len(set(ranking[:3]) & actual_top3),
        performance_scores=performance,
        ranks_of_interest={i: ranking.index(i) + 1 for i in interest},
    )
