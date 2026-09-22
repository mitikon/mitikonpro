import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda import (
    block_bootstrap_ci,
    diagnose_edge_significance,
    evaluate_edge_significance,
    majority_class_baseline,
    one_sample_edge_test,
    paired_edge_test,
    per_class_report,
    persistence_baseline,
)


def sample_market(rows: int = 760):
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2022-01-03", periods=rows)
    symbols = ["SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP", "VIX9D", "VIX3M"]
    shocks = rng.normal(0.0003, 0.01, size=(rows, len(symbols)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=index, columns=symbols)
    volume = pd.DataFrame(rng.integers(1_000_000, 9_000_000, size=close.shape), index=index, columns=symbols)
    return close, volume


def test_majority_class_baseline_uses_only_past_labels_no_lookahead():
    # Class 0 for the first 30 rows, class 1 for the remaining 70: the
    # training-window majority only flips to 1 once enough class-1 history
    # has accumulated (at train_size=70, well after the regime shift at
    # row 30) - a lookahead baseline would flip immediately instead.
    index = pd.bdate_range("2024-01-01", periods=100)
    y = pd.Series([0] * 30 + [1] * 70, index=index)
    baseline = majority_class_baseline(y, train_size=40, test_size=10)
    # The first test block (rows 40-49) only saw 30 class-0 vs 10 class-1
    # history -> majority is still 0.
    assert (baseline.iloc[:10] == 0).all()
    # By the last block, the training window (rows 0-89) has accumulated
    # 30 class-0 vs 60 class-1 -> majority has flipped to 1.
    assert baseline.iloc[-1] == 1


def test_persistence_baseline_shifts_by_one_session():
    index = pd.bdate_range("2024-01-01", periods=10)
    y = pd.Series(range(10), index=index)
    baseline = persistence_baseline(y, train_size=3)
    assert list(baseline) == [2, 3, 4, 5, 6, 7, 8]


def test_block_bootstrap_ci_recovers_a_known_rate():
    rng = np.random.default_rng(1)
    values = rng.binomial(1, 0.6, size=500).astype(float)
    result = block_bootstrap_ci(values, block_size=10, n_bootstrap=500, seed=2)
    assert result["ci_lower_2.5"] < 0.6 < result["ci_upper_97.5"]


def test_one_sample_edge_test_detects_a_real_edge():
    rng = np.random.default_rng(3)
    # Clearly, consistently above the 0.33 baseline.
    values = rng.binomial(1, 0.55, size=1000).astype(float)
    result = one_sample_edge_test(values, baseline_rate=1 / 3, block_size=21, n_bootstrap=500, seed=4)
    assert result["p_value_not_above_baseline"] < 0.05
    assert result["ci_lower_2.5"] > 1 / 3


def test_one_sample_edge_test_does_not_falsely_detect_edge_in_pure_noise():
    rng = np.random.default_rng(5)
    # Generated at exactly the baseline rate: there is no real edge here.
    values = rng.binomial(1, 1 / 3, size=1000).astype(float)
    result = one_sample_edge_test(values, baseline_rate=1 / 3, block_size=21, n_bootstrap=500, seed=6)
    assert result["p_value_not_above_baseline"] > 0.05


def test_paired_edge_test_rewards_a_model_that_beats_the_baseline_on_the_same_days():
    rng = np.random.default_rng(7)
    baseline_correct = rng.binomial(1, 0.4, size=800)
    # The model gets every session the baseline gets right, plus more.
    extra_hits = rng.binomial(1, 0.3, size=800)
    model_correct = np.clip(baseline_correct + extra_hits, 0, 1)
    result = paired_edge_test(model_correct, baseline_correct, block_size=21, n_bootstrap=500, seed=8)
    assert result["p_value_not_above_baseline"] < 0.05
    assert result["observed_mean"] > 0


def test_per_class_report_has_support_precision_recall_for_each_class():
    predictions = pd.DataFrame(
        {
            "predicted_class": [1, 1, -1, 0, 1],
            "actual_class": [1, -1, -1, 0, 1],
        }
    )
    report = per_class_report(predictions)
    assert set(report) == {"-1", "0", "1"}
    assert report["1"]["support"] == 2
    assert report["1"]["predicted_count"] == 3
    assert report["1"]["precision"] == pytest.approx(2 / 3)
    assert report["1"]["recall"] == pytest.approx(1.0)


def test_evaluate_edge_significance_reports_all_baselines():
    index = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(9)
    y = pd.Series(rng.integers(-1, 2, size=120), index=index)
    predictions = pd.DataFrame(
        {"predicted_class": rng.integers(-1, 2, size=120), "actual_class": y.to_numpy()},
        index=index,
    )
    result = evaluate_edge_significance(predictions, y, train_size=60, test_size=20, n_bootstrap=200)
    assert result["n"] == 120
    assert "model_vs_uniform_random" in result
    assert "model_vs_majority_class_baseline" in result
    assert "model_vs_persistence_baseline" in result
    assert result["per_class"]


def test_diagnose_edge_significance_runs_end_to_end_without_changing_predictions():
    close, volume = sample_market()
    report = diagnose_edge_significance(close, volume, "SPY", train_size=252, neutral_band=0.005)
    assert report["target"] == "SPY"
    assert report["neutral_band"] == 0.005
    assert "direction_accuracy" in report["walk_forward_metrics"]
    significance = report["edge_significance"]
    assert 0.0 <= significance["model_accuracy"] <= 1.0
    assert 0.0 <= significance["majority_class_baseline_accuracy"] <= 1.0
