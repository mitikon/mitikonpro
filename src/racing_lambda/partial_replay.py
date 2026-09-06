"""Leakage-safe partial replay diagnostics for historical races.

This module is deliberately diagnostic, not a replacement for the production
layer-2 scorer.  When a historical monthly snapshot is not yet available, it
scores only confirmed pre-race components and renormalizes their weights.  No
missing monthly condition block is filled with 0.5 or any other invented value.

The purpose is to expose what the currently recovered evidence alone would have
produced, so that later full-snapshot replay can be compared against the same
frozen baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Mapping

import numpy as np

from .historical_evidence_2026 import CHUKYO_2YO_JOCKEY_STATS, CHUKYO_2YO_TRAINER_STATS
from .historical_races_2026 import CHUKYO_2YO_STAKES_20260830
from .chukyo_2yo_recovered_runs import layer2_past_runs_for
from .layer2_live_input import pace_position_score, recent_form_score


@dataclass(frozen=True)
class PartialReplayResult:
    horse_id: str
    score: float
    raw_score: float
    confidence: float
    market_value_gap: float
    recent_form: float
    pace_position_fit: float
    jockey_fit: float
    trainer_fit: float


def _aggregate_fit(starts: int, wins: int, seconds: int, thirds: int) -> float:
    """Sample-size-aware top-3 fit using the same prior as monthly_stats_db."""
    if starts <= 0:
        raise ValueError("starts must be positive")
    top3 = wins + seconds + thirds
    if top3 > starts:
        raise ValueError("top3 cannot exceed starts")
    prior_rate = 0.30
    prior_strength = 8.0
    posterior = (top3 + prior_strength * prior_rate) / (starts + prior_strength)
    centered = 0.5 + 0.5 * (posterior - prior_rate) / max(prior_rate, 1.0 - prior_rate)
    return float(np.clip(centered, 0.0, 1.0))


def _market_probabilities() -> Mapping[str, float]:
    raw = {horse.horse_id: 1.0 / horse.win_odds for horse in CHUKYO_2YO_STAKES_20260830.horses}
    total = sum(raw.values())
    return {horse_id: value / total for horse_id, value in raw.items()}


def chukyo_2yo_partial_replay() -> list[PartialReplayResult]:
    """Replay confirmed Chukyo 2yo pre-race evidence without monthly backfill."""
    ids = [horse.horse_id for horse in CHUKYO_2YO_STAKES_20260830.horses]
    recent: dict[str, float] = {}
    pace: dict[str, float] = {}
    jockey: dict[str, float] = {}
    trainer: dict[str, float] = {}

    for horse_id in ids:
        runs = layer2_past_runs_for(horse_id)
        recent[horse_id] = recent_form_score(runs)
        pace[horse_id] = pace_position_score(runs)
        js = CHUKYO_2YO_JOCKEY_STATS[horse_id]
        ts = CHUKYO_2YO_TRAINER_STATS[horse_id]
        jockey[horse_id] = _aggregate_fit(js.starts, js.wins, js.seconds, js.thirds)
        trainer[horse_id] = _aggregate_fit(ts.starts, ts.wins, ts.seconds, ts.thirds)

    # Build a relative fair-share baseline from CONFIRMED components only.
    # Original condition-strength weights are renormalized over available terms.
    condition_weights = {
        "recent": 0.30,
        "pace": 0.15,
        "jockey": 0.10,
        "trainer": 0.07,
    }
    cw_total = sum(condition_weights.values())
    strengths: dict[str, float] = {}
    for horse_id in ids:
        strengths[horse_id] = (
            condition_weights["recent"] * recent[horse_id]
            + condition_weights["pace"] * pace[horse_id]
            + condition_weights["jockey"] * jockey[horse_id]
            + condition_weights["trainer"] * trainer[horse_id]
        ) / cw_total

    strength_total = sum(strengths.values())
    fair_share = {horse_id: strengths[horse_id] / strength_total for horse_id in ids}
    market = _market_probabilities()
    market_gap = {
        horse_id: 1.0 / (1.0 + exp(-8.0 * (fair_share[horse_id] - market[horse_id])))
        for horse_id in ids
    }

    # Production layer-2 weights restricted to evidence that is actually present.
    score_weights = {
        "recent": 0.22,
        "pace": 0.14,
        "jockey": 0.10,
        "trainer": 0.08,
        "market": 0.12,
    }
    sw_total = sum(score_weights.values())
    # Full non-body layer-2 weight sum is 0.92.  Coverage is therefore explicit.
    evidence_coverage = sw_total / 0.92

    results: list[PartialReplayResult] = []
    for horse_id in ids:
        raw = (
            score_weights["recent"] * recent[horse_id]
            + score_weights["pace"] * pace[horse_id]
            + score_weights["jockey"] * jockey[horse_id]
            + score_weights["trainer"] * trainer[horse_id]
            + score_weights["market"] * market_gap[horse_id]
        ) / sw_total
        run_count = len(layer2_past_runs_for(horse_id))
        run_coverage = min(run_count / 5.0, 1.0)
        # Missing monthly blocks and missing race-day body weight reduce confidence.
        confidence = evidence_coverage * (0.5 + 0.5 * run_coverage) * 0.90
        score = 0.5 + confidence * (raw - 0.5)
        results.append(
            PartialReplayResult(
                horse_id=horse_id,
                score=float(np.clip(score, 0.0, 1.0)),
                raw_score=float(np.clip(raw, 0.0, 1.0)),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                market_value_gap=market_gap[horse_id],
                recent_form=recent[horse_id],
                pace_position_fit=pace[horse_id],
                jockey_fit=jockey[horse_id],
                trainer_fit=trainer[horse_id],
            )
        )
    return sorted(results, key=lambda row: (-row.score, int(row.horse_id)))
