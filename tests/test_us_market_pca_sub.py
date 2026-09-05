import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.paper_pca_sub import (
    PAPER_COMPONENTS,
    PAPER_WINDOW,
    correlation_from_returns,
    regularize_correlation,
)
from leading_signal_lambda.us_market_pca_sub import (
    US_LEADING_SECTORS,
    US_MARKET_TARGETS,
    US_MODEL_COLUMNS,
    USMarketPcaSubModel,
    build_us_market_prior_basis,
    prepare_us_market_returns,
)
from leading_signal_lambda.us_market_validation import run_us_market_backtest
from leading_signal_lambda.collector import MarketDataset
from leading_signal_lambda.us_market_experiment import (
    validate_sector_rotation_dataset,
    validate_us_market_dataset,
)


def sample_returns(rows: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, size=(rows, 3))
    loading = rng.normal(size=(3, len(US_MODEL_COLUMNS)))
    noise = rng.normal(0.0, 0.004, size=(rows, len(US_MODEL_COLUMNS)))
    values = common @ loading * 0.25 + noise
    return pd.DataFrame(
        values,
        index=pd.bdate_range(start, periods=rows),
        columns=US_MODEL_COLUMNS,
    )


def test_us_prior_basis_is_three_dimensional_and_orthonormal():
    basis = build_us_market_prior_basis()
    assert basis.shape == (len(US_MODEL_COLUMNS), PAPER_COMPONENTS)
    np.testing.assert_allclose(basis.T @ basis, np.eye(PAPER_COMPONENTS), atol=1e-12)


def test_us_model_keeps_paper_regularization_and_component_count():
    rolling = sample_returns(PAPER_WINDOW, "2022-01-03", 1)
    prior = sample_returns(500, "2019-01-02", 2)
    model = USMarketPcaSubModel().fit(rolling, prior)
    expected = regularize_correlation(
        correlation_from_returns(rolling), model.prior_correlation_, 0.90
    )
    assert model.components_.shape == (len(US_MODEL_COLUMNS), PAPER_COMPONENTS)
    np.testing.assert_allclose(model.regularized_correlation_, expected, atol=1e-12)


def test_us_prediction_is_target_projection_of_leading_sectors():
    rolling = sample_returns(PAPER_WINDOW, "2022-01-03", 3)
    prior = sample_returns(500, "2019-01-02", 4)
    current = sample_returns(1, "2023-01-03", 5).iloc[0]
    model = USMarketPcaSubModel().fit(rolling, prior)
    signal = model.predict(current.loc[list(US_LEADING_SECTORS)])
    expected = model.target_loadings_ @ (
        model.leader_loadings_.T @ signal.leader_standardized.to_numpy()
    )
    assert list(signal.target_standardized_prediction.index) == list(US_MARKET_TARGETS)
    np.testing.assert_allclose(signal.target_standardized_prediction, expected, atol=1e-12)


def test_prepare_returns_rejects_backfill_and_drops_incomplete_rows():
    returns = sample_returns(80, "2022-01-03", 6)
    close = 100.0 * (1.0 + returns).cumprod()
    close.iloc[20, 0] = np.nan
    prepared = prepare_us_market_returns(close)
    assert close.index[20] not in prepared.index
    assert close.index[21] not in prepared.index
    assert not prepared.isna().any().any()


def test_signal_lambda_uses_only_prior_frozen_predictions():
    returns = sample_returns(130, "2022-01-03", 7)
    prior = sample_returns(500, "2019-01-02", 8)
    result = run_us_market_backtest(
        returns, prior, calibration_observations=10, signal_threshold=0.0
    )
    spy = result.predictions.xs("SPY", level="target")
    assert spy["signal_lambda"].iloc[:10].isna().all()
    expected = np.corrcoef(
        spy["raw_score"].iloc[:10], spy["realized_z"].iloc[:10]
    )[0, 1]
    assert spy["signal_lambda"].iloc[10] == pytest.approx(expected)


def test_future_mutation_does_not_change_past_predictions_or_lambda():
    returns = sample_returns(140, "2022-01-03", 9)
    prior = sample_returns(500, "2019-01-02", 10)
    first = run_us_market_backtest(returns, prior, calibration_observations=10)
    mutated = returns.copy()
    mutated.iloc[-5:] *= 20.0
    second = run_us_market_backtest(mutated, prior, calibration_observations=10)
    cutoff = returns.index[-6]
    frozen_columns = ["raw_score", "signal_lambda", "effective_score", "action"]
    pd.testing.assert_frame_equal(
        first.predictions.loc[:cutoff, frozen_columns],
        second.predictions.loc[:cutoff, frozen_columns],
    )


def test_frozen_prediction_file_cannot_be_overwritten(tmp_path):
    returns = sample_returns(100, "2022-01-03", 11)
    prior = sample_returns(500, "2019-01-02", 12)
    result = run_us_market_backtest(returns, prior, calibration_observations=5)
    path = result.save_frozen(tmp_path / "predictions.csv")
    assert path.exists()
    with pytest.raises(FileExistsError):
        result.save_frozen(path)


def test_prior_period_overlap_is_rejected():
    returns = sample_returns(100, "2022-01-03", 13)
    overlapping = sample_returns(100, "2022-01-03", 14)
    with pytest.raises(ValueError, match="prior period"):
        run_us_market_backtest(returns, overlapping)


def test_real_data_experiment_separates_prior_and_evaluation(tmp_path):
    returns = sample_returns(1300, "2019-01-02", 15)
    close = 100.0 * (1.0 + returns).cumprod()
    dataset = MarketDataset(close=close, volume=pd.DataFrame(index=close.index))
    result = validate_us_market_dataset(
        dataset,
        tmp_path,
        prior_end="2021-12-31",
        evaluation_start="2022-01-01",
        calibration_observations=10,
    )
    assert set(result.metrics) == set(US_MARKET_TARGETS)
    assert (tmp_path / "us_pca_sub_frozen_predictions.csv").exists()
    assert (tmp_path / "us_pca_sub_validation_summary.json").exists()


def test_sector_rotation_experiment_saves_state_and_rankings(tmp_path):
    returns = sample_returns(1300, "2019-01-02", 16)
    close = 100.0 * (1.0 + returns).cumprod()
    close["SMH"] = close["XLK"] * 1.02
    close["IWM"] = close["SPY"] * 0.98
    open_price = close.shift(1)
    open_price.iloc[0] = close.iloc[0]
    rng = np.random.default_rng(17)
    volume = pd.DataFrame(
        rng.integers(1_000_000, 9_000_000, size=(len(close), len(US_LEADING_SECTORS))),
        index=close.index,
        columns=US_LEADING_SECTORS,
    )
    dataset = MarketDataset(
        close=close,
        volume=volume,
        open=open_price,
        unadjusted_close=close,
    )
    result = validate_sector_rotation_dataset(
        dataset,
        tmp_path / "rotation",
        prior_end="2021-12-31",
        evaluation_start="2022-01-01",
    )
    assert not result.rankings.empty
    assert (tmp_path / "rotation" / "sector_rotation_decisions.csv").exists()
    assert (tmp_path / "rotation" / "sector_rotation_metrics.json").exists()
    assert (
        tmp_path
        / "rotation"
        / "market_internals_challenger"
        / "sector_rotation_metrics.json"
    ).exists()
    assert (tmp_path / "rotation" / "market_internals_comparison.json").exists()
