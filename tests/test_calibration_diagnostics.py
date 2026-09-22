import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda import (
    build_leading_features,
    build_training_set,
    calibration_bins,
    calibration_by_temperature,
    class_balance,
    diagnose_target,
    rescale_probabilities,
    walk_forward_validate,
)


def sample_market(rows: int = 760):
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2022-01-03", periods=rows)
    symbols = ["SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP", "VIX9D", "VIX3M"]
    shocks = rng.normal(0.0003, 0.01, size=(rows, len(symbols)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=index, columns=symbols)
    volume = pd.DataFrame(rng.integers(1_000_000, 9_000_000, size=close.shape), index=index, columns=symbols)
    return close, volume


def test_class_balance_shrinks_neutral_class_as_band_narrows():
    close, _ = sample_market()
    next_returns = close["SPY"].pct_change(fill_method=None).shift(-1).dropna()
    narrow = class_balance(next_returns, 0.0001)
    wide = class_balance(next_returns, 0.02)
    # A near-zero band should leave almost nothing in the neutral class; a
    # wide band should pull much more mass into it.
    assert narrow["0"]["share"] < wide["0"]["share"]
    assert sum(row["count"] for row in narrow.values()) == len(next_returns)


def test_calibration_bins_reflect_realized_accuracy():
    close, volume = sample_market()
    features = build_leading_features(close, volume)
    X, y, returns = build_training_set(features, close["SPY"])
    result = walk_forward_validate(X, y, returns, train_size=252, test_size=21, min_samples=60)
    bins = calibration_bins(result.predictions, n_bins=5)
    assert bins, "expected at least one non-empty confidence bin"
    total_n = sum(row["n"] for row in bins)
    assert total_n == len(result.predictions)
    for row in bins:
        assert 0.0 <= row["realized_accuracy"] <= 1.0
        assert 0.0 <= row["mean_stated_confidence"] <= 1.0


def test_diagnose_target_reports_balance_and_calibration_without_changing_predictions():
    close, volume = sample_market()
    report = diagnose_target(close, volume, "SPY", train_size=252)
    assert report["target"] == "SPY"
    assert "0.001" in report["class_balance_by_neutral_band"]
    assert report["calibration"]
    assert 0.0 <= report["overall_trade_coverage"] <= 1.0
    assert "1.0" in report["calibration_by_temperature"]


def test_rescale_probabilities_at_temperature_one_matches_the_stored_confidence():
    row = pd.Series({"prob_neg1": 0.05, "prob_0": 0.15, "prob_1": 0.80})
    predicted_class, confidence = rescale_probabilities(row, 1.0)
    assert predicted_class == 1
    assert confidence == pytest.approx(0.80, abs=1e-9)


def test_rescale_probabilities_never_changes_the_argmax():
    row = pd.Series({"prob_neg1": 0.02, "prob_0": 0.03, "prob_1": 0.95})
    for temperature in (1.0, 2.0, 5.0, 10.0):
        predicted_class, confidence = rescale_probabilities(row, temperature)
        assert predicted_class == 1
    # A higher temperature must pull an overconfident probability down, never up.
    _, low_t_confidence = rescale_probabilities(row, 1.0)
    _, high_t_confidence = rescale_probabilities(row, 5.0)
    assert high_t_confidence < low_t_confidence


def test_calibration_by_temperature_reports_a_gap_per_candidate():
    close, volume = sample_market()
    features = build_leading_features(close, volume)
    X, y, returns = build_training_set(features, close["SPY"])
    result = walk_forward_validate(X, y, returns, train_size=252, test_size=21, min_samples=60)
    sweep = calibration_by_temperature(result.predictions, temperature_candidates=(1.0, 3.0))
    assert set(sweep) == {"1.0", "3.0"}
    for entry in sweep.values():
        assert "overall_calibration_gap" in entry
        assert entry["bins"]
    # A higher temperature must never raise mean stated confidence.
    assert sweep["3.0"]["mean_stated_confidence"] <= sweep["1.0"]["mean_stated_confidence"]
