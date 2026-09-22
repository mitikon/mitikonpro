"""Read-only pairwise lead-lag signal scanner across the full market universe.

2026-09-22: after Stage 1 (calibration_diagnostics) and Stage 2
(edge_diagnostics) showed the existing PCA-subspace classifier has no
verifiable directional edge, the goal shifted to a core/satellite structure:
a buy-and-hold core secures the market's own return, and a satellite sleeve
is only worth building if it can clear a much higher bar (see
benchmark_report.py for the core's own measured floor). This module is the
honest, statistically disciplined way to search for that satellite edge: it
tests every (predictor, target, lag, orientation) combination in the
candidate universe as its own explicit hypothesis, rather than fitting a
model that could quietly overfit thousands of implicit combinations.

Each candidate rule is completely parameter-free and deterministic (no
training window, nothing fit to history): "go long TARGET tomorrow if
PREDICTOR's own trailing LAG-day return was positive" (momentum), or the
mirror-image short-on-positive rule (reversal). Because nothing is fit,
every historical day is a valid, independent test of the rule - there is no
train/test split to get wrong.

Testing thousands of (predictor, target, lag, orientation) combinations on
the same historical data guarantees some will look significant by pure
chance (the same data-snooping risk flagged in calibration_diagnostics and
edge_diagnostics, at much larger scale). This module controls it two ways:

1. Benjamini-Hochberg FDR correction across every hypothesis actually
   tested in one scan - not just the ones that happen to look good.
2. A moving block bootstrap (edge_diagnostics.block_bootstrap_ci /
   one_sample_edge_test) for the per-candidate significance test itself,
   since daily market data is serially correlated and a naive i.i.d. test
   overstates confidence.

A candidate surviving FDR correction on historical data is still only a
*candidate*: like every other candidate parameter in this repository, it
must still clear MarketRecursiveImprovementGate's future-only evaluation
before being trusted. This module changes no prediction or trading logic;
it is invoked explicitly via ``leading-lambda-signal-discovery``.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .collector import DailyMarketCollector, DEFAULT_UNIVERSE
from .edge_diagnostics import DEFAULT_BLOCK_SIZE, one_sample_edge_test


# Every ETF in the universe that can actually be bought as a satellite
# position. The remaining DEFAULT_UNIVERSE symbols (VIX/rates/commodities/FX)
# are still usable as *predictors* but are excluded here as targets since
# they are not directly investable the way an ETF is.
TRADABLE_TARGETS: tuple[str, ...] = (
    "SPY", "QQQ", "DIA", "RSP", "IWM", "SMH", "HYG", "LQD",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
)
SIGNAL_LAGS: tuple[int, ...] = (1, 2, 3, 5)
ORIENTATIONS: tuple[str, ...] = ("momentum", "reversal")
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SCAN_N_BOOTSTRAP = 200
DEFAULT_CONFIRM_N_BOOTSTRAP = 5000


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Standard BH step-up FDR procedure. Returns a reject flag per input p-value.

    Sorts ascending, finds the largest rank k with p_(k) <= (k/m)*alpha, and
    rejects the null for every hypothesis at or below that rank - controlling
    the expected proportion of false discoveries among all rejections,
    unlike a flat per-test alpha which would let false positives pile up
    linearly with the number of hypotheses tested.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    threshold_rank = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= (rank / m) * alpha:
            threshold_rank = rank
    reject = [False] * m
    for rank, index in enumerate(order, start=1):
        if rank <= threshold_rank:
            reject[index] = True
    return reject


def signal_strategy_returns(
    close: pd.DataFrame,
    predictor: str,
    target: str,
    lag: int,
    orientation: str,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.Series:
    """Daily strategy returns for one deterministic, parameter-free rule.

    Position on day t = sign(predictor's trailing `lag`-day return ending at
    t), flipped if orientation == "reversal". That position is held for
    target's return from t to t+1. Nothing is fit to history, so no
    train/test split is needed - this is the raw daily return series a
    trader following the rule mechanically would have earned, net of a flat
    round-trip cost charged whenever the position changes.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}")
    predictor_close = pd.to_numeric(close[predictor], errors="coerce")
    target_close = pd.to_numeric(close[target], errors="coerce")
    predictor_signal = predictor_close.pct_change(lag, fill_method=None)
    target_next_return = target_close.pct_change(fill_method=None).shift(-1)
    position = np.sign(predictor_signal)
    if orientation == "reversal":
        position = -position
    aligned = pd.DataFrame({"position": position, "next_return": target_next_return}).dropna()
    turnover = aligned["position"].diff().abs().fillna(aligned["position"].abs())
    cost = turnover * transaction_cost_bps / 10_000.0
    return (aligned["position"] * aligned["next_return"] - cost).rename(
        f"{predictor}->{target}:lag{lag}:{orientation}"
    )


def _annualized_from_daily_returns(returns: pd.Series) -> dict[str, float]:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    annualized = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    return {
        "annualized_return": annualized,
        "max_drawdown": float(drawdown.min()),
        "ending_equity": float(equity.iloc[-1]),
        "trade_win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
    }


def evaluate_signal(
    close: pd.DataFrame,
    predictor: str,
    target: str,
    lag: int,
    orientation: str,
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_SCAN_N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, object]:
    """One candidate's full evaluation: return profile plus significance test.

    The significance test asks "is the true mean daily strategy return > 0"
    via edge_diagnostics.one_sample_edge_test (block-bootstrap, respects
    serial correlation) - the same machinery Stage 2 used against the PCA
    model, now pointed at this rule's own realized daily returns.
    """
    returns = signal_strategy_returns(close, predictor, target, lag, orientation, transaction_cost_bps)
    n = len(returns)
    if n < block_size:
        return {
            "predictor": predictor, "target": target, "lag": lag, "orientation": orientation,
            "n": n, "insufficient_data": True,
        }
    metrics = _annualized_from_daily_returns(returns)
    significance = one_sample_edge_test(returns.to_numpy(), 0.0, block_size, n_bootstrap, seed)
    return {
        "predictor": predictor,
        "target": target,
        "lag": lag,
        "orientation": orientation,
        "n": n,
        "insufficient_data": False,
        **metrics,
        "p_value_not_above_zero": significance["p_value_not_above_baseline"],
        "daily_return_ci_lower_2.5": significance["ci_lower_2.5"],
        "daily_return_ci_upper_97.5": significance["ci_upper_97.5"],
    }


def scan_pairwise_signals(
    close: pd.DataFrame,
    predictors: tuple[str, ...] = tuple(DEFAULT_UNIVERSE),
    targets: tuple[str, ...] = TRADABLE_TARGETS,
    lags: tuple[int, ...] = SIGNAL_LAGS,
    orientations: tuple[str, ...] = ORIENTATIONS,
    *,
    alpha: float = 0.05,
    min_annualized_return: float = 0.15,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_bootstrap: int = DEFAULT_SCAN_N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, object]:
    """Test every (predictor, target, lag, orientation) combination once.

    predictor == target pairs are skipped (a symbol never "predicts" itself
    here; that is what persistence_baseline in edge_diagnostics already
    covers for the main model). Every other combination is one hypothesis,
    all corrected together by benjamini_hochberg - never re-run or
    cherry-picked after seeing results, which would silently reopen the
    exact data-snooping risk this module exists to control.
    """
    results: list[dict[str, object]] = []
    for target, predictor, lag, orientation in product(targets, predictors, lags, orientations):
        if predictor == target:
            continue
        results.append(
            evaluate_signal(
                close, predictor, target, lag, orientation,
                transaction_cost_bps=transaction_cost_bps, block_size=block_size,
                n_bootstrap=n_bootstrap, seed=seed,
            )
        )
    testable = [row for row in results if not row["insufficient_data"]]
    p_values = [float(row["p_value_not_above_zero"]) for row in testable]
    rejected = benjamini_hochberg(p_values, alpha)
    for row, significant in zip(testable, rejected):
        row["significant_after_fdr"] = bool(significant)
        row["survives_return_bar"] = bool(significant and row["annualized_return"] >= min_annualized_return)
    testable.sort(key=lambda row: row["annualized_return"], reverse=True)
    survivors = [row for row in testable if row["survives_return_bar"]]
    return {
        "hypotheses_tested": len(testable),
        "hypotheses_with_insufficient_data": len(results) - len(testable),
        "alpha": alpha,
        "min_annualized_return": min_annualized_return,
        "significant_after_fdr_count": sum(rejected),
        "survivors_count": len(survivors),
        "survivors": survivors,
        "top_20_by_annualized_return": testable[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only pairwise lead-lag signal scan with FDR correction (changes no prediction logic)"
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end-exclusive", default="auto")
    parser.add_argument("--targets", default=",".join(TRADABLE_TARGETS))
    parser.add_argument("--predictors", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-annualized-return", type=float, default=0.15)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_SCAN_N_BOOTSTRAP)
    parser.add_argument("--output", default="artifacts/signal-discovery")
    parser.add_argument("--exceptional-closures", default="config/exceptional_nyse_closures.json")
    args = parser.parse_args()

    if args.end_exclusive == "auto":
        from .market_calendar import NYSETradingCalendar

        completed = NYSETradingCalendar(exceptional_closures=args.exceptional_closures).last_completed_session()
        end_exclusive = completed.end_exclusive.isoformat()
        print(f"last completed XNYS session: {completed.session_date} (close {completed.close_utc})")
    else:
        end_exclusive = args.end_exclusive

    dataset = DailyMarketCollector().collect(args.start, end_exclusive)
    report = scan_pairwise_signals(
        dataset.close,
        predictors=tuple(args.predictors.split(",")),
        targets=tuple(args.targets.split(",")),
        alpha=args.alpha,
        min_annualized_return=args.min_annualized_return,
        n_bootstrap=args.n_bootstrap,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "signal_scan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"tested {report['hypotheses_tested']} hypotheses, "
        f"{report['significant_after_fdr_count']} significant after FDR, "
        f"{report['survivors_count']} survivors clearing {args.min_annualized_return:.0%} annualized return"
    )
    print(json.dumps(report["survivors"], ensure_ascii=False))


if __name__ == "__main__":
    main()
