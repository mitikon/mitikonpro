import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.paper_backtest import run_paper_backtest
from leading_signal_lambda.paper_pca_sub import (
    JAPAN_SECTORS,
    PAPER_COMPONENTS,
    PAPER_LAMBDA_PRIOR,
    PAPER_WINDOW,
    US_SECTORS,
    PaperPcaSubModel,
    build_prior_basis,
    build_prior_correlation,
    correlation_from_returns,
    quantile_long_short_weights,
    regularize_correlation,
)


COLUMNS = list(US_SECTORS + JAPAN_SECTORS)


def sample_returns(rows: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.008, size=(rows, 3))
    loadings = rng.normal(size=(3, len(COLUMNS)))
    noise = rng.normal(0.0, 0.003, size=(rows, len(COLUMNS)))
    values = common @ loadings * 0.25 + noise
    return pd.DataFrame(
        values,
        index=pd.bdate_range(start, periods=rows),
        columns=COLUMNS,
    )


def test_prior_basis_has_three_orthonormal_vectors():
    basis = build_prior_basis()
    assert basis.shape == (len(COLUMNS), PAPER_COMPONENTS)
    np.testing.assert_allclose(basis.T @ basis, np.eye(PAPER_COMPONENTS), atol=1e-12)


def test_prior_correlation_is_symmetric_with_unit_diagonal():
    prior = sample_returns(250, "2010-01-04", 1)
    correlation = build_prior_correlation(prior, build_prior_basis())
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(correlation), np.ones(len(COLUMNS)), atol=1e-12)


def test_equation_13_uses_ten_percent_sample_and_ninety_percent_prior():
    sample = np.eye(3)
    prior = np.full((3, 3), 0.5)
    expected = 0.10 * sample + 0.90 * prior
    actual = regularize_correlation(sample, prior, PAPER_LAMBDA_PRIOR)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_model_uses_fixed_window_and_three_components():
    rolling = sample_returns(PAPER_WINDOW, "2015-01-05", 2)
    prior = sample_returns(500, "2010-01-04", 3)
    model = PaperPcaSubModel().fit(rolling, prior)
    expected = regularize_correlation(
        correlation_from_returns(rolling), model.prior_correlation_, 0.90
    )
    assert model.components_.shape == (len(COLUMNS), PAPER_COMPONENTS)
    np.testing.assert_allclose(model.regularized_correlation_, expected, atol=1e-12)


def test_prediction_matches_paper_projection_and_reconstruction():
    rolling = sample_returns(PAPER_WINDOW, "2015-01-05", 4)
    prior = sample_returns(500, "2010-01-04", 5)
    model = PaperPcaSubModel().fit(rolling, prior)
    current = sample_returns(1, "2016-01-04", 6).iloc[0].loc[list(US_SECTORS)]
    signal = model.predict(current)
    expected = model.japan_loadings_ @ (
        model.us_loadings_.T @ signal.us_standardized.to_numpy()
    )
    np.testing.assert_allclose(signal.japan_standardized_prediction, expected, atol=1e-12)
    np.testing.assert_allclose(
        signal.transmission.to_numpy(),
        model.japan_loadings_ @ model.us_loadings_.T,
        atol=1e-12,
    )


def test_quantile_portfolio_is_dollar_neutral_and_gross_two():
    scores = pd.Series(np.arange(len(JAPAN_SECTORS)), index=JAPAN_SECTORS)
    weights = quantile_long_short_weights(scores)
    assert (weights > 0).sum() == 5
    assert (weights < 0).sum() == 5
    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().sum() == pytest.approx(2.0)


def test_backtest_rejects_prior_period_overlap():
    returns = sample_returns(100, "2016-01-04", 7)
    target = returns.loc[:, list(JAPAN_SECTORS)]
    overlapping_prior = sample_returns(100, "2016-01-04", 8)
    with pytest.raises(ValueError, match="prior period"):
        run_paper_backtest(returns, target, overlapping_prior)


def test_future_mutation_does_not_change_already_frozen_scores():
    returns = sample_returns(100, "2016-01-04", 9)
    target = sample_returns(100, "2016-01-04", 10).loc[:, list(JAPAN_SECTORS)]
    prior = sample_returns(500, "2010-01-04", 11)
    first = run_paper_backtest(returns, target, prior)

    mutated = returns.copy()
    mutated.iloc[-5:] *= 100.0
    second = run_paper_backtest(mutated, target, prior)
    cutoff = returns.index[-6]
    pd.testing.assert_frame_equal(
        first.frozen_scores.loc[:cutoff], second.frozen_scores.loc[:cutoff]
    )
