"""Pairwise structural comparison for Chukyo 2yo partial replay.

This module combines only previously identified structural fixes. It does not
search weights against the official result. The result is consulted only after
each pre-race ranking has been produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
import numpy as np

from .chukyo_2yo_recovered_runs import CHUKYO_2YO_RECOVERED_RUNS, layer2_past_runs_for
from .chukyo_variant_comparison import _aggregate_fit, _recent_no_saturation
from .historical_evidence_2026 import CHUKYO_2YO_JOCKEY_STATS, CHUKYO_2YO_TRAINER_STATS
from .historical_races_2026 import CHUKYO_2YO_STAKES_20260830, CHUKYO_2YO_STAKES_20260830_RESULT
from .layer2_live_input import pace_position_score, recent_form_score


@dataclass(frozen=True)
class PairwiseComparison:
    name: str
    ranking: tuple[str, ...]
    top5_hits: int
    top3_hits: int
    ranks_of_interest: dict[str, int]


def _pair_score(fixes: frozenset[str]) -> tuple[str, ...]:
    ids = [h.horse_id for h in CHUKYO_2YO_STAKES_20260830.horses]
    recent = {i: recent_form_score(layer2_past_runs_for(i)) for i in ids}
    pace = {i: pace_position_score(layer2_past_runs_for(i)) for i in ids}
    jockey = {i: _aggregate_fit(CHUKYO_2YO_JOCKEY_STATS[i].starts, CHUKYO_2YO_JOCKEY_STATS[i].wins, CHUKYO_2YO_JOCKEY_STATS[i].seconds, CHUKYO_2YO_JOCKEY_STATS[i].thirds) for i in ids}
    trainer = {i: _aggregate_fit(CHUKYO_2YO_TRAINER_STATS[i].starts, CHUKYO_2YO_TRAINER_STATS[i].wins, CHUKYO_2YO_TRAINER_STATS[i].seconds, CHUKYO_2YO_TRAINER_STATS[i].thirds) for i in ids}

    if "recent_saturation_fix" in fixes:
        recent = {i: _recent_no_saturation(i) for i in ids}
    if "surface_mismatch_gate" in fixes:
        recent = {i: recent[i] * (0.75 if CHUKYO_2YO_RECOVERED_RUNS[i][0].surface != "芝" else 1.0) for i in ids}
    if "neutralize_raw_position" in fixes:
        pace = {i: 0.5 for i in ids}

    raw_market = {h.horse_id: 1.0 / h.win_odds for h in CHUKYO_2YO_STAKES_20260830.horses}
    mt = sum(raw_market.values())
    market = {i: raw_market[i] / mt for i in ids}
    strengths = {i: (0.30*recent[i] + 0.15*pace[i] + 0.10*jockey[i] + 0.07*trainer[i]) / 0.62 for i in ids}
    st = sum(strengths.values())
    fair = {i: strengths[i] / st for i in ids}
    gap = {i: 1.0 / (1.0 + exp(-8.0*(fair[i]-market[i]))) for i in ids}
    if "market_gap_removed" in fixes:
        gap = {i: 0.5 for i in ids}
    elif "market_gap_positive_only" in fixes:
        gap = {i: max(0.5, gap[i]) for i in ids}

    scores = {}
    evidence = 0.66 / 0.92
    for i in ids:
        raw = (0.22*recent[i] + 0.14*pace[i] + 0.10*jockey[i] + 0.08*trainer[i] + 0.12*gap[i]) / 0.66
        denom = 2.0 if "juvenile_completeness" in fixes else 5.0
        coverage = min(len(layer2_past_runs_for(i))/denom, 1.0)
        confidence = evidence * (0.5 + 0.5*coverage) * 0.90
        scores[i] = 0.5 + confidence*(raw-0.5)
    return tuple(sorted(ids, key=lambda i: (-scores[i], int(i))))


def chukyo_pairwise_comparison() -> tuple[PairwiseComparison, ...]:
    pairs = (
        ("surface+position", frozenset(("surface_mismatch_gate", "neutralize_raw_position"))),
        ("market_removed+juvenile", frozenset(("market_gap_removed", "juvenile_completeness"))),
        ("market_positive+juvenile", frozenset(("market_gap_positive_only", "juvenile_completeness"))),
        ("surface+market_removed", frozenset(("surface_mismatch_gate", "market_gap_removed"))),
        ("position+market_removed", frozenset(("neutralize_raw_position", "market_gap_removed"))),
        ("recent+surface", frozenset(("recent_saturation_fix", "surface_mismatch_gate"))),
    )
    actual = CHUKYO_2YO_STAKES_20260830_RESULT.finishing_order
    actual_top5, actual_top3 = set(actual[:5]), set(actual[:3])
    interest = ("2", "5", "7", "3", "9", "6", "4")
    out = []
    for name, fixes in pairs:
        ranking = _pair_score(fixes)
        out.append(PairwiseComparison(name, ranking, len(set(ranking[:5]) & actual_top5), len(set(ranking[:3]) & actual_top3), {i: ranking.index(i)+1 for i in interest}))
    return tuple(out)
