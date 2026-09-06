"""External reproduction check using the 2026 Ibis Summer Dash.

This is intentionally a diagnostic bridge, not a production scorer.  The race's
pre-race field, odds and last-five clocks were recoverable, while the exact
historical jockey/trainer aggregate blocks used by the Chukyo replay were not.
Those unavailable blocks are held neutral at 0.5 rather than reconstructed from
post-race information.

The purpose is to test whether the portable core identified at Chukyo
(position-neutral + market-gap removed + performance-detail) reproduces on a
second race without result-tuned weight search.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalValidationResult:
    race_id: str
    ranking: tuple[str, ...]
    official_top5: tuple[str, ...]
    top5_hits: int
    limitation: str
    finding: str


# Frozen output from the recovered pre-race-only bridge calculation.
# Inputs: last-five finish/field/surface/distance/going/clock/final-section,
# target=Niigata turf straight 1000m good, pace neutral, market-gap removed,
# missing historical jockey/trainer aggregate blocks held neutral (0.5), and
# performance-detail kept at the same 0.12 weight used in the Chukyo diagnostic.
IBIS_BRIDGE_RANKING = (
    "16", "1", "9", "6", "12", "2", "17", "8", "10",
    "4", "5", "15", "11", "13", "14", "7", "3",
)
IBIS_OFFICIAL_TOP5 = ("6", "11", "4", "17", "16")


def ibis_external_validation() -> ExternalValidationResult:
    hits = len(set(IBIS_BRIDGE_RANKING[:5]) & set(IBIS_OFFICIAL_TOP5))
    return ExternalValidationResult(
        race_id="2026-08-02-niigata-07",
        ranking=IBIS_BRIDGE_RANKING,
        official_top5=IBIS_OFFICIAL_TOP5,
        top5_hits=hits,
        limitation=(
            "Exact pre-race jockey/trainer aggregate blocks from the historical workflow are not yet recovered; "
            "they are neutralized, so this is a component portability test rather than a full layer-2 replay."
        ),
        finding=(
            "The portable Chukyo performance-detail core falls to 2/5.  The main structural warning is that "
            "normalizing clocks only within each historical distance does not express target-course specialization. "
            "For a unique straight 1000m race, exact course/distance fit must be represented independently rather "
            "than inferred from generic 1000/1200m speed. No result-fitted reweighting is applied here."
        ),
    )
