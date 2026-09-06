"""One-at-a-time structural comparison for the Chukyo 2yo partial replay.

Each variant changes one structural assumption only.  Official results are used
only by the evaluation helper after all pre-race scores have been produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .chukyo_2yo_recovered_runs import CHUKYO_2YO_RECOVERED_RUNS, layer2_past_runs_for
from .historical_evidence_2026 import CHUKYO_2YO_JOCKEY_STATS, CHUKYO_2YO_TRAINER_STATS
from .historical_races_2026 import CHUKYO_2YO_STAKES_20260830, CHUKYO_2YO_STAKES_20260830_RESULT
from .layer2_live_input import pace_position_score, recent_form_score


@dataclass(frozen=True)
class VariantComparison:
    name: str
    ranking: tuple[str, ...]
    top5_hits: int
    ranks_of_interest: dict[str, int]


def _aggregate_fit(starts: int, wins: int, seconds: int, thirds: int) -> float:
    top3 = wins + seconds + thirds
    posterior = (top3 + 8.0 * 0.30) / (starts + 8.0)
    centered = 0.5 + 0.5 * (posterior - 0.30) / 0.70
    return float(np.clip(centered, 0.0, 1.0))


def _recent_no_saturation(horse_id: str) -> float:
    """Separate finish quality and suitability so bonuses cannot force 1.0."""
    runs = CHUKYO_2YO_RECOVERED_RUNS[horse_id]
    weights = np.asarray([0.72**i for i in range(len(runs))], dtype=float)
    weights /= weights.sum()
    values = []
    for run in runs:
        finish = 1.0 - (run.finish - 1) / (run.field_size - 1)
        fit = np.mean([
            run.surface == "芝",
            abs(run.distance_m - 1400) <= 200,
            run.going == "良",
        ])
        values.append(0.90 * finish + 0.10 * fit)
    return float(np.dot(weights, np.asarray(values, dtype=float)))


def _base_components():
    ids = [horse.horse_id for horse in CHUKYO_2YO_STAKES_20260830.horses]
    recent = {horse_id: recent_form_score(layer2_past_runs_for(horse_id)) for horse_id in ids}
    pace = {horse_id: pace_position_score(layer2_past_runs_for(horse_id)) for horse_id in ids}
    jockey = {}
    trainer = {}
    for horse_id in ids:
        js = CHUKYO_2YO_JOCKEY_STATS[horse_id]
        ts = CHUKYO_2YO_TRAINER_STATS[horse_id]
        jockey[horse_id] = _aggregate_fit(js.starts, js.wins, js.seconds, js.thirds)
        trainer[horse_id] = _aggregate_fit(ts.starts, ts.wins, ts.seconds, ts.thirds)
    raw_market = {horse.horse_id: 1.0 / horse.win_odds for horse in CHUKYO_2YO_STAKES_20260830.horses}
    total = sum(raw_market.values())
    market = {horse_id: value / total for horse_id, value in raw_market.items()}
    return ids, recent, pace, jockey, trainer, market


def _score_variant(name: str) -> tuple[str, ...]:
    ids, recent, pace, jockey, trainer, market = _base_components()

    if name == "recent_saturation_fix":
        recent = {horse_id: _recent_no_saturation(horse_id) for horse_id in ids}
    elif name == "surface_mismatch_gate":
        recent = {
            horse_id: recent[horse_id] * (0.75 if CHUKYO_2YO_RECOVERED_RUNS[horse_id][0].surface != "芝" else 1.0)
            for horse_id in ids
        }
    elif name == "neutralize_raw_position":
        pace = {horse_id: 0.5 for horse_id in ids}

    condition_weights = {"recent": 0.30, "pace": 0.15, "jockey": 0.10, "trainer": 0.07}
    cw_total = sum(condition_weights.values())
    strengths = {
        horse_id: (
            condition_weights["recent"] * recent[horse_id]
            + condition_weights["pace"] * pace[horse_id]
            + condition_weights["jockey"] * jockey[horse_id]
            + condition_weights["trainer"] * trainer[horse_id]
        ) / cw_total
        for horse_id in ids
    }
    strength_total = sum(strengths.values())
    fair_share = {horse_id: strengths[horse_id] / strength_total for horse_id in ids}
    market_gap = {
        horse_id: 1.0 / (1.0 + exp(-8.0 * (fair_share[horse_id] - market[horse_id])))
        for horse_id in ids
    }
    if name == "market_gap_removed":
        market_gap = {horse_id: 0.5 for horse_id in ids}
    elif name == "market_gap_positive_only":
        market_gap = {horse_id: max(0.5, value) for horse_id, value in market_gap.items()}

    score_weights = {"recent": 0.22, "pace": 0.14, "jockey": 0.10, "trainer": 0.08, "market": 0.12}
    sw_total = sum(score_weights.values())
    evidence_coverage = sw_total / 0.92
    scores = {}
    for horse_id in ids:
        raw = (
            score_weights["recent"] * recent[horse_id]
            + score_weights["pace"] * pace[horse_id]
            + score_weights["jockey"] * jockey[horse_id]
            + score_weights["trainer"] * trainer[horse_id]
            + score_weights["market"] * market_gap[horse_id]
        ) / sw_total
        run_count = len(layer2_past_runs_for(horse_id))
        denominator = 2.0 if name == "juvenile_completeness" else 5.0
        run_coverage = min(run_count / denominator, 1.0)
        confidence = evidence_coverage * (0.5 + 0.5 * run_coverage) * 0.90
        scores[horse_id] = 0.5 + confidence * (raw - 0.5)
    return tuple(sorted(ids, key=lambda horse_id: (-scores[horse_id], int(horse_id))))


def chukyo_one_at_a_time_comparison() -> tuple[VariantComparison, ...]:
    """Evaluate structural variants only after their pre-race rankings are frozen."""
    names = (
        "baseline",
        "recent_saturation_fix",
        "surface_mismatch_gate",
        "neutralize_raw_position",
        "market_gap_positive_only",
        "market_gap_removed",
        "juvenile_completeness",
    )
    actual_top5 = set(CHUKYO_2YO_STAKES_20260830_RESULT.finishing_order[:5])
    interest = ("2", "5", "7", "3", "9", "6", "4")
    rows = []
    for name in names:
        ranking = _score_variant(name)
        rows.append(
            VariantComparison(
                name=name,
                ranking=ranking,
                top5_hits=len(set(ranking[:5]) & actual_top5),
                ranks_of_interest={horse_id: ranking.index(horse_id) + 1 for horse_id in interest},
            )
        )
    return tuple(rows)
