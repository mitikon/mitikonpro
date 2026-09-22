import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.signal_discovery import (
    benjamini_hochberg,
    evaluate_signal,
    scan_pairwise_signals,
    signal_strategy_returns,
)


def test_benjamini_hochberg_rejects_only_the_passing_prefix():
    # 5 p-values, alpha=0.05: thresholds are (1/5..5/5)*0.05 =
    # (0.01,0.02,0.03,0.04,0.05). Sorted p-values (0.01,0.02,0.03,0.04,0.20)
    # pass at ranks 1-4 and fail at rank 5 -> reject exactly the first four.
    p_values = [0.01, 0.02, 0.03, 0.04, 0.20]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected == [True, True, True, True, False]


def test_benjamini_hochberg_step_up_reinstates_a_borderline_middle_pvalue():
    # Sorted p-values (0.005, 0.015, 0.035, 0.038, 0.05) against the same
    # thresholds: rank3 (0.035) fails its OWN threshold (0.03), but rank4
    # and rank5 both pass theirs, so the step-up rule still rejects rank3
    # (the largest passing rank is 5, and every hypothesis at or below that
    # rank is rejected, not just the ones that individually passed).
    p_values = [0.035, 0.005, 0.05, 0.038, 0.015]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected == [True, True, True, True, True]


def test_benjamini_hochberg_rejects_nothing_when_the_smallest_p_value_already_fails():
    p_values = [0.5, 0.6, 0.7]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected == [False, False, False]


def test_benjamini_hochberg_handles_empty_input():
    assert benjamini_hochberg([]) == []


def _planted_momentum_close(n=300, seed=1, effect=0.02, noise_std=0.003):
    """PRED's lag-1 return sign deterministically drives TGT's next return."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    predictor_daily_returns = rng.normal(0, 0.01, n)
    predictor_prices = [100.0]
    target_prices = [100.0]
    for i in range(1, n):
        predictor_prices.append(predictor_prices[-1] * (1 + predictor_daily_returns[i]))
        driver = predictor_daily_returns[i - 1] if i - 1 >= 1 else 0.0
        target_return = effect * float(np.sign(driver)) + rng.normal(0, noise_std)
        target_prices.append(target_prices[-1] * (1 + target_return))
    return pd.DataFrame({"PRED": predictor_prices, "TGT": target_prices}, index=index)


def _pure_noise_close(symbols=("A", "B", "C", "D", "E"), n=300, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    data = {
        symbol: 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
        for symbol in symbols
    }
    return pd.DataFrame(data, index=index)


def test_signal_strategy_returns_flips_sign_between_momentum_and_reversal():
    close = _planted_momentum_close()
    momentum = signal_strategy_returns(close, "PRED", "TGT", lag=1, orientation="momentum")
    reversal = signal_strategy_returns(close, "PRED", "TGT", lag=1, orientation="reversal")
    common = momentum.index.intersection(reversal.index)
    # Position is exactly negated; net of the same transaction cost pattern,
    # the two return series should be (almost) mirror images (a small
    # residual is expected since turnover, and thus cost, differs between
    # the two position sequences).
    assert (momentum.loc[common] + reversal.loc[common]).abs().mean() < 0.005


def test_evaluate_signal_detects_a_planted_momentum_relationship():
    close = _planted_momentum_close()
    result = evaluate_signal(close, "PRED", "TGT", lag=1, orientation="momentum", n_bootstrap=200, seed=2)
    assert result["insufficient_data"] is False
    assert result["annualized_return"] > 0.10
    assert result["p_value_not_above_zero"] < 0.05

    mirrored = evaluate_signal(close, "PRED", "TGT", lag=1, orientation="reversal", n_bootstrap=200, seed=2)
    assert mirrored["annualized_return"] < result["annualized_return"]


def test_evaluate_signal_flags_insufficient_data_instead_of_crashing():
    close = _planted_momentum_close(n=10)
    result = evaluate_signal(close, "PRED", "TGT", lag=1, orientation="momentum", block_size=21, n_bootstrap=50)
    assert result["insufficient_data"] is True


def test_scan_pairwise_signals_controls_false_discoveries_in_pure_noise():
    close = _pure_noise_close()
    report = scan_pairwise_signals(
        close,
        predictors=("A", "B", "C", "D"),
        targets=("E",),
        lags=(1, 2, 3, 5),
        orientations=("momentum", "reversal"),
        min_annualized_return=0.15,
        n_bootstrap=100,
        seed=3,
    )
    assert report["hypotheses_tested"] == 4 * 4 * 2
    assert report["survivors_count"] == 0


def test_scan_pairwise_signals_finds_a_planted_signal_among_noise():
    planted = _planted_momentum_close()
    noise = _pure_noise_close(symbols=("A", "B", "C"))
    close = planted.join(noise, how="inner")
    report = scan_pairwise_signals(
        close,
        predictors=("PRED", "A", "B", "C"),
        targets=("TGT",),
        lags=(1, 2, 3, 5),
        orientations=("momentum", "reversal"),
        min_annualized_return=0.10,
        n_bootstrap=100,
        seed=4,
    )
    survivor_keys = {(row["predictor"], row["lag"], row["orientation"]) for row in report["survivors"]}
    assert ("PRED", 1, "momentum") in survivor_keys
