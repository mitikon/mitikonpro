"""External reproduction check using the 2026 Ibis Summer Dash.

This is intentionally a diagnostic bridge, not a production scorer.  The race's
pre-race field, odds and last-five clocks were recoverable, while the exact
historical jockey/trainer aggregate blocks used by the Chukyo replay were not.
Those unavailable blocks are held neutral at 0.5 rather than reconstructed from
post-race information.

The purpose is to test whether the portable core identified at Chukyo
(position-neutral + market-gap removed + performance-detail) reproduces on a
second race without result-tuned weight search.

Important: the original bridge persisted only the ordinal ranking, not every
horse's continuous pre-shrink score. Therefore later course-specialization work
must not fabricate numeric base scores from rank positions. Full combined
reranking is blocked until those continuous scores are rebuilt from pre-race
features.
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
            "Exact pre-race jockey/trainer aggregate blocks are not yet recovered and the original bridge did not "
            "persist continuous per-horse base scores. Rank positions must not be converted into fake scores for "
            "later feature fusion."
        ),
        finding=(
            "The portable Chukyo performance-detail core falls to 2/5. Course specialization is separately "
            "supported by pre-race turf-1000 evidence, but a combined Top5 improvement remains unproven until the "
            "continuous bridge scores are rebuilt from pre-race features."
        ),
    )
