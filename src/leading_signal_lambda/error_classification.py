"""Post-settlement failure classification for the market RSI (Recursive
Self-Improvement) candidate-generation loop.

A settled, outcome-known miss is classified into one of five categories so a
human can write an evidence-backed candidate rationale before proposing a new
generation. Classification never edits a frozen prediction or settlement,
never creates or applies a rule by itself, and a rationale log requires
evidence from several sessions so a single outlier can never justify a
candidate on its own (see ``MIN_RATIONALE_SESSIONS``). Only
``MarketRecursiveImprovementGate`` (in ``recursive_self_improvement.py``)
decides promotion, strictly on later, still-unknown sessions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


EXTRACTION_MISS = "extraction_miss"
"""抽出漏れ: no directional signal was present, or the true winner was never among the ranked candidates."""

OVERESTIMATION = "overestimation"
"""過大評価: the predicted move was far larger than what actually happened."""

FINAL_EXCLUSION = "final_exclusion"
"""最終除外: the true winner had signal and ranked near the top but the single-pick rule excluded it."""

MISSING_INPUT = "missing_input"
"""入力欠損: the prediction relied on one or more imputed (missing) input features."""

MARKET_NOISE = "market_noise"
"""市場ノイズ: the actual move was too small to be a meaningful directional test."""

ERROR_CATEGORIES = frozenset(
    {EXTRACTION_MISS, OVERESTIMATION, FINAL_EXCLUSION, MISSING_INPUT, MARKET_NOISE}
)

MIN_RATIONALE_SESSIONS = 5
DEFAULT_OVERESTIMATION_RATIO = 2.0
DEFAULT_FINAL_EXCLUSION_RANK = 3


def _digest(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return normalized


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _finite_float(value: object, default: float) -> float:
    """Tolerate columns absent from an older, pre-migration history CSV."""
    return default if _is_missing(value) else float(value)


def _as_bool(value: object) -> bool:
    """A CSV round trip can turn a boolean into the text "True"/"False"."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    if _is_missing(value):
        return False
    return bool(value)


def classify_target_settlement(
    row: Mapping[str, object],
    *,
    market_noise_band: float | None = None,
    overestimation_ratio: float = DEFAULT_OVERESTIMATION_RATIO,
) -> str | None:
    """Classify one settled per-target direction miss.

    Returns ``None`` when the direction call was correct: only misses are
    classified. ``market_noise_band`` defaults to the row's own
    ``neutral_band``, since a move inside that band is inherently low-signal.
    """
    if _as_bool(row.get("direction_correct")):
        return None
    if _finite_float(row.get("imputed_feature_count"), 0.0) > 0:
        return MISSING_INPUT
    band = (
        market_noise_band
        if market_noise_band is not None
        else _finite_float(row.get("neutral_band"), 0.0)
    )
    actual_return = float(row["actual_return"])
    predicted_return = float(row["predicted_return"])
    if abs(actual_return) <= band:
        return MARKET_NOISE
    if abs(predicted_return) >= overestimation_ratio * abs(actual_return) and predicted_return * actual_return >= 0:
        return OVERESTIMATION
    return EXTRACTION_MISS


def classify_extreme_selection(
    forecast_settlement: Mapping[str, object],
    *,
    market_noise_band: float | None = None,
    overestimation_ratio: float = DEFAULT_OVERESTIMATION_RATIO,
    final_exclusion_rank: int = DEFAULT_FINAL_EXCLUSION_RANK,
) -> str | None:
    """Classify a settled upside/downside extreme-ETF selection miss.

    Returns ``None`` when the predicted extreme ETF was the actual extreme
    (``exact_target_hit``). ``market_noise_band`` defaults to the
    settlement's own ``neutral_band``, matching classify_target_settlement,
    since a real market return is essentially never exactly zero: leaving
    the band at a hardcoded 0.0 would make MARKET_NOISE unreachable here.
    """
    if _as_bool(forecast_settlement.get("exact_target_hit")):
        return None
    band = (
        market_noise_band
        if market_noise_band is not None
        else _finite_float(forecast_settlement.get("neutral_band"), 0.0)
    )
    actual_extreme_return = float(forecast_settlement["actual_extreme_return"])
    if abs(actual_extreme_return) <= band:
        return MARKET_NOISE
    if not _as_bool(forecast_settlement.get("direction_signal_present")):
        return EXTRACTION_MISS
    rank = forecast_settlement.get("selected_actual_rank")
    if rank is not None and int(rank) <= final_exclusion_rank:
        return FINAL_EXCLUSION
    predicted_return = float(forecast_settlement["predicted_return"])
    selected_actual_return = float(forecast_settlement["selected_actual_return"])
    if abs(predicted_return) >= overestimation_ratio * max(abs(selected_actual_return), 1e-12):
        return OVERESTIMATION
    return EXTRACTION_MISS


def summarize_error_classification(
    target_rows: Sequence[Mapping[str, object]],
    extreme_settlement_sessions: Sequence[Mapping[str, Mapping[str, object]]] = (),
) -> dict[str, int]:
    """Aggregate per-target and extreme-selection misses into category counts."""
    counts = {category: 0 for category in sorted(ERROR_CATEGORIES)}
    for row in target_rows:
        category = classify_target_settlement(row)
        if category is not None:
            counts[category] += 1
    for session in extreme_settlement_sessions:
        for forecast in session.values():
            category = classify_extreme_selection(forecast)
            if category is not None:
                counts[category] += 1
    return counts


@dataclass(frozen=True)
class CandidateRationale:
    """Write-once, pre-candidate evidence log.

    Records which failure categories, over how many already-settled
    sessions, motivate proposing a new generation. This never becomes a rule
    by itself: a human reads it and, if warranted, hand-writes a
    ``MarketRsiCandidate`` whose allow-listed parameter change is later
    validated on genuinely future sessions by
    ``MarketRecursiveImprovementGate``.
    """

    rationale_id: str
    created_at: datetime
    evaluated_signal_sessions: tuple[str, ...]
    error_counts: Mapping[str, int]
    narrative: str
    source_report_sha256: str
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if not self.rationale_id.strip():
            raise ValueError("rationale_id is required")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        sessions = tuple(self.evaluated_signal_sessions)
        if len(set(sessions)) != len(sessions):
            raise ValueError("evaluated_signal_sessions must not contain duplicates")
        if len(sessions) < MIN_RATIONALE_SESSIONS:
            raise ValueError(
                f"a candidate rationale requires at least {MIN_RATIONALE_SESSIONS} evaluated "
                "sessions; a single outlier cannot justify a candidate"
            )
        unknown = set(self.error_counts) - ERROR_CATEGORIES
        if unknown:
            raise ValueError(f"unknown error categories: {sorted(unknown)}")
        if sum(self.error_counts.values()) <= 0:
            raise ValueError("candidate rationale requires at least one classified error")
        if not self.narrative.strip():
            raise ValueError("narrative rationale is required and must be explainable, not silent")
        _digest(self.source_report_sha256, "source_report_sha256")

    def sealed_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "evaluated_signal_sessions": list(self.evaluated_signal_sessions),
            "error_counts": dict(self.error_counts),
        }


def rationale_manifest_digest(rationale: CandidateRationale) -> str:
    return sha256(_canonical(rationale.sealed_payload())).hexdigest()


def freeze_candidate_rationale(rationale: CandidateRationale, path: str | Path) -> Path:
    """Write the rationale log once. Refuses to overwrite an existing file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = rationale.sealed_payload()
    payload["rationale_manifest_sha256"] = rationale_manifest_digest(rationale)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
