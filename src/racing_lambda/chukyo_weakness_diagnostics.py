"""Weakness diagnostics for the 2026 Chukyo 2yo partial replay.

This module does not tune weights to the official result. It decomposes the
already-frozen partial replay into its pre-race components and flags structural
failure modes that are visible from the model design itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from .partial_replay import chukyo_2yo_partial_replay


@dataclass(frozen=True)
class WeaknessFinding:
    code: str
    severity: str
    affected_horses: tuple[str, ...]
    explanation: str
    proposed_direction: str


def chukyo_2yo_component_table() -> tuple[dict[str, float | str], ...]:
    rows = []
    for row in chukyo_2yo_partial_replay():
        rows.append(
            {
                "horse_id": row.horse_id,
                "score": row.score,
                "raw_score": row.raw_score,
                "confidence": row.confidence,
                "recent_form": row.recent_form,
                "pace_position_fit": row.pace_position_fit,
                "jockey_fit": row.jockey_fit,
                "trainer_fit": row.trainer_fit,
                "market_value_gap": row.market_value_gap,
            }
        )
    return tuple(rows)


def chukyo_2yo_weakness_findings() -> tuple[WeaknessFinding, ...]:
    return (
        WeaknessFinding(
            code="RECENT_FORM_SATURATION",
            severity="critical",
            affected_horses=("5", "6", "7", "9"),
            explanation=(
                "Recent-form scoring caps at 1.0 after a finish score plus condition bonuses. "
                "Several one-win juveniles therefore become nearly indistinguishable even when "
                "race quality, surface, clock and closing performance differ materially."
            ),
            proposed_direction=(
                "Separate finish quality from suitability bonuses; prevent bonuses from saturating "
                "the base score; add race-time/closing-quality features only when pre-race comparable data exist."
            ),
        ),
        WeaknessFinding(
            code="SURFACE_MISMATCH_OVERVALUATION",
            severity="critical",
            affected_horses=("6",),
            explanation=(
                "A dirt debut win can still receive a perfect recent-form score because the strong finish "
                "plus distance/going bonuses hit the 1.0 cap. Surface mismatch is therefore not strong enough."
            ),
            proposed_direction=(
                "Apply a multiplicative surface-transfer penalty or a gated maximum when the target surface "
                "differs, instead of relying on a small additive same-surface bonus."
            ),
        ),
        WeaknessFinding(
            code="FRONT_POSITION_BIAS",
            severity="critical",
            affected_horses=("5", "6", "3"),
            explanation=(
                "pace_position_score rewards being near the front regardless of whether the expected race pace "
                "makes that position advantageous. This lifts front-running one-race winners and suppresses a "
                "horse such as 3 whose best prior run came from a deep position."
            ),
            proposed_direction=(
                "Replace raw forward-position preference with pace-conditioned position fit: expected pace x "
                "historical position x finishing efficiency."
            ),
        ),
        WeaknessFinding(
            code="MARKET_GAP_DOUBLE_COUNT",
            severity="critical",
            affected_horses=("2", "5", "6", "3", "9"),
            explanation=(
                "The fair-share baseline is built from recent form, pace, jockey and trainer, and those same "
                "components are then scored again directly. market_value_gap therefore reuses the same signal "
                "a second time and can amplify its errors."
            ),
            proposed_direction=(
                "Build market divergence from an independently calibrated probability model or orthogonalize "
                "the market-gap term against the components used to create fair share."
            ),
        ),
        WeaknessFinding(
            code="FAVORITE_PENALTY_ASYMMETRY",
            severity="high",
            affected_horses=("3", "9"),
            explanation=(
                "Strongly backed horses receive a low market_value_gap whenever market share exceeds the model's "
                "heuristic fair share. This treats popularity as possible overvaluation even when the market may "
                "be correctly recognizing strength."
            ),
            proposed_direction=(
                "Require independent evidence of overvaluation before penalizing a favorite; separate 'no bug' "
                "from 'negative bug' instead of forcing every market gap onto a symmetric 0..1 scale."
            ),
        ),
        WeaknessFinding(
            code="JUVENILE_COMPLETENESS_PENALTY",
            severity="medium",
            affected_horses=("1", "5", "6", "7", "8", "9"),
            explanation=(
                "run_count/5 is treated as completeness. For two-year-olds, one or two prior starts are normal, "
                "so the model can confuse age-appropriate history length with poor data quality."
            ),
            proposed_direction=(
                "Make expected history length conditional on age/season/race type and measure completeness against "
                "what should reasonably exist for that cohort."
            ),
        ),
        WeaknessFinding(
            code="UNUSED_PRE_RACE_PERFORMANCE_DETAIL",
            severity="high",
            affected_horses=("2", "3", "4", "5", "6", "7", "8", "9"),
            explanation=(
                "Recovered final-section times and race clocks exist in project history, but the current PastRun "
                "adapter uses final_section_rank only, which is absent here. Valuable pre-race performance detail "
                "is therefore discarded."
            ),
            proposed_direction=(
                "Introduce normalized speed/closing-efficiency features using only pre-race comparable races, with "
                "distance/surface/going normalization and no target-result information."
            ),
        ),
    )
