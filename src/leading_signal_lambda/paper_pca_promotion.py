"""Outcome-blind promotion trial for the paper-compliant subspace-regularized
PCA leading-signal model (λ; 中川慧ほか 2026, see docs/PAPER_PCA_SUB_SPEC.md).

Every hyperparameter of PaperPcaSubModel is pinned to the published paper
(L=60, K=3, q=0.3, Creg=0.1*Ct+0.9*C0, three fixed prior directions), so
there is nothing left for the usual mutation-search RSI to tune here. What
"recursive self-improvement" means for this model is a one-shot, repeatable
*promotion trial*: freeze λ's forecast and a caller-supplied baseline
forecast for the same session before the outcome exists, settle both once
the real outcome is known, and let a Wald SPRT (the same sequential test
recursive_self_improvement.py uses) decide - from genuinely future sessions
only - whether λ is statistically better than the baseline.

This module never attaches to production trading, never edits source, and
never merges a branch. A PROMOTION_PROPOSED verdict only means "λ is
eligible for a human to attach to production"; it does not attach it.

Known, documented limitation (tracked in docs/PAPER_PCA_SUB_SPEC.md): XLC
was created in 2018 and has no price history for the paper's 2010-2014
long-term prior sample. This module does not silently substitute or
fabricate that missing history - callers must supply full_prior_returns
themselves, and using a placeholder/approximate prior sample is a decision
for a human to make and document, not something this code decides.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paper_pca_sub import (
    JAPAN_SECTORS,
    PaperPcaSubModel,
    quantile_long_short_weights,
)
from .recursive_self_improvement import sequential_loss_improvement_test


SCHEMA_VERSION = "paper-pca-promotion-v1"
MIN_PROMOTION_SESSIONS = 20
MIN_EARLY_REJECTION_SESSIONS = 5
DEFAULT_MIN_EFFECT = 0.001

KNOWN_LIMITATIONS = (
    "XLC has no price history before 2018, so the paper's 2010-2014 "
    "long-term prior sample cannot be reproduced from XLC directly. "
    "full_prior_returns must be supplied by the caller; this module does "
    "not silently fill or approximate the missing period.",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_once(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable paper-PCA promotion record: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def initial_state() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": 0,
        "status": "TRIAL_IN_PROGRESS",
        "production_pca_attached": False,
        "automatic_source_push": False,
        "automatic_trade_execution": False,
        "known_limitations": list(KNOWN_LIMITATIONS),
    }


def load_state(path: str | Path | None) -> dict[str, object]:
    if path is None or not Path(path).exists():
        return initial_state()
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported paper-PCA promotion state")
    return state


def _require_japan_order(scores: pd.Series, name: str) -> pd.Series:
    if list(scores.index) != list(JAPAN_SECTORS):
        raise ValueError(f"{name} must use this order: {list(JAPAN_SECTORS)}")
    if scores.isna().any():
        raise ValueError(f"{name} must contain no missing values")
    return scores.astype(float)


def freeze_daily_forecast(
    rolling_returns: pd.DataFrame,
    full_prior_returns: pd.DataFrame,
    current_us_return: pd.Series,
    baseline_scores: pd.Series,
    signal_session: str,
    target_session: str,
    output: str | Path,
) -> Path:
    """Freeze λ's and the baseline's scored forecasts before the outcome exists.

    ``baseline_scores`` is whatever comparison signal the caller wants λ
    held to account against (a zero/no-skill baseline, the existing
    production classifier repurposed for these targets, a naive momentum
    rule, ...); this module deliberately does not choose that for the
    caller, so the promotion trial's fairness is always inspectable from
    the frozen artifact itself.
    """
    model = PaperPcaSubModel().fit(rolling_returns, full_prior_returns)
    candidate_signal = model.predict(current_us_return)
    candidate_scores = _require_japan_order(candidate_signal.japan_standardized_prediction, "candidate_scores")
    baseline_scores = _require_japan_order(baseline_scores, "baseline_scores")

    candidate_weights = quantile_long_short_weights(candidate_scores)
    baseline_weights = quantile_long_short_weights(baseline_scores)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "signal_session": str(signal_session),
        "target_session": str(target_session),
        "candidate_scores": candidate_scores.to_dict(),
        "candidate_weights": candidate_weights.to_dict(),
        "baseline_scores": baseline_scores.to_dict(),
        "baseline_weights": baseline_weights.to_dict(),
        "status": "FROZEN_BEFORE_OUTCOME",
    }
    payload["forecast_sha256"] = _sha(payload)
    return _write_once(Path(output), payload)


def settle_daily_forecast(
    frozen_path: str | Path,
    japan_open_to_close: pd.Series,
    output: str | Path,
) -> Path:
    """Settle a frozen forecast against the realized Japan Open-to-Close return.

    Never mutates the frozen forecast; writes a separate, also-immutable
    settlement artifact, matching every other freeze/settle pair in this
    codebase.
    """
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    actual = _require_japan_order(japan_open_to_close, "japan_open_to_close")

    candidate_weights = pd.Series(frozen["candidate_weights"])[list(JAPAN_SECTORS)]
    baseline_weights = pd.Series(frozen["baseline_weights"])[list(JAPAN_SECTORS)]
    candidate_return = float(candidate_weights @ actual)
    baseline_return = float(baseline_weights @ actual)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "signal_session": frozen["signal_session"],
        "target_session": frozen["target_session"],
        "forecast_sha256": frozen["forecast_sha256"],
        "status": "SETTLED_WITHOUT_FORECAST_MUTATION",
        "candidate_net_return": candidate_return,
        "baseline_net_return": baseline_return,
        # Loss is economic, not a forecast-error metric: this model only
        # ranks sectors for a long/short portfolio (see quantile_long_short_
        # weights), so the quantity the paper actually optimizes for -
        # realized spread return - is the fair comparison, not a magnitude
        # error against an arbitrary standardized score scale.
        "candidate_loss": -candidate_return,
        "baseline_loss": -baseline_return,
    }
    payload["settlement_sha256"] = _sha(payload)
    return _write_once(Path(output), payload)


def evaluate_promotion(
    state: dict[str, object],
    settlements: Iterable[str | Path],
    *,
    min_future_sessions: int = MIN_PROMOTION_SESSIONS,
    min_early_rejection_sessions: int = MIN_EARLY_REJECTION_SESSIONS,
    min_effect: float = DEFAULT_MIN_EFFECT,
    alpha: float = 0.05,
    beta: float = 0.10,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the Wald SPRT over every settled session and decide the trial's fate.

    Mirrors evaluate_and_rotate_candidates in recursive_runtime.py: a
    CONTINUE verdict is a real, valid outcome at any sample size (before or
    after the min_future_sessions floor) and must never be collapsed into a
    forced REJECTED - it just means "keep settling sessions and evaluate
    again later". Promotion itself never happens before
    min_future_sessions, matching the rest of this codebase's floor.
    """
    documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in settlements]
    documents = sorted(documents, key=lambda value: value["target_session"])
    seen: set[str] = set()
    documents = [
        doc for doc in documents if not (doc["target_session"] in seen or seen.add(doc["target_session"]))
    ]

    sessions = len(documents)
    report_base = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_sessions": sessions,
        "production_pca_attached": False,
        "source_code_modified_by_learning": False,
        "known_limitations": list(state.get("known_limitations", KNOWN_LIMITATIONS)),
    }

    if sessions < min_early_rejection_sessions:
        report = {**report_base, "status": "CONTINUE", "report_sha256": ""}
        report["report_sha256"] = _sha({**report, "report_sha256": ""})
        return state, report

    baseline_losses = [float(doc["baseline_loss"]) for doc in documents]
    candidate_losses = [float(doc["candidate_loss"]) for doc in documents]
    evidence = sequential_loss_improvement_test(
        baseline_losses, candidate_losses, min_effect=min_effect, alpha=alpha, beta=beta,
    )

    if evidence.decision == "CONTINUE":
        status = "CONTINUE"
    elif sessions < min_future_sessions:
        # Never promotes on a partial window: only an early REJECT
        # concludes the trial before the floor.
        status = "EARLY_REJECTED" if evidence.decision == "REJECT" else "CONTINUE"
    else:
        status = "PROMOTION_PROPOSED" if evidence.decision == "PROMOTE" else "REJECTED"

    report = {
        **report_base,
        "status": status,
        "mean_loss_improvement": evidence.mean_improvement,
        "log_likelihood_ratio": evidence.log_likelihood_ratio,
        "baseline_mean_net_return": float(pd.Series([-loss for loss in baseline_losses]).mean()),
        "candidate_mean_net_return": float(pd.Series([-loss for loss in candidate_losses]).mean()),
        "human_approval_required": True,
        "autonomous_source_edits": False,
        "autonomous_main_merge": False,
        "trading_authority": False,
        "report_sha256": "",
    }
    report["report_sha256"] = _sha({**report, "report_sha256": ""})

    next_state = {
        **state,
        "status": status,
        "generation": int(state.get("generation", 0)) + (1 if status == "PROMOTION_PROPOSED" else 0),
    }
    return next_state, report
