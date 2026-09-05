import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.market_internals import (
    MarketInternalRidgeModel,
    build_market_internal_features,
)
from leading_signal_lambda.sector_rotation import SECTOR_ETFS
from leading_signal_lambda.sector_rotation_validation import run_sector_rotation_backtest


def sample_market(rows: int = 420, seed: int = 31):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=rows)
    symbols = list(SECTOR_ETFS) + ["SMH", "IWM", "SPY"]
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(rows, len(symbols))),
        index=index,
        columns=symbols,
    )
    close = 100.0 * (1.0 + returns).cumprod()
    volume = pd.DataFrame(
        rng.integers(1_000_000, 9_000_000, size=(rows, len(SECTOR_ETFS))),
        index=index,
        columns=SECTOR_ETFS,
    )
    return returns, close, volume


def test_volume_shock_uses_only_previous_twenty_days():
    _, close, volume = sample_market(50)
    features = build_market_internal_features(close, volume)
    position = 25
    symbol = "XLK"
    expected = np.log(volume.iloc[position][symbol]) - np.log(
        volume.iloc[position - 20 : position][symbol]
    ).mean()
    assert features.iloc[position][f"volume_shock_{symbol}"] == pytest.approx(expected)


def test_future_volume_mutation_does_not_change_past_features():
    _, close, volume = sample_market(80)
    first = build_market_internal_features(close, volume)
    changed = volume.copy()
    changed.iloc[-5:] *= 100
    second = build_market_internal_features(close, changed)
    pd.testing.assert_frame_equal(first.iloc[:-5], second.iloc[:-5])


def test_market_holiday_does_not_poison_next_twenty_volume_signals():
    _, close, volume = sample_market(80)
    holiday = volume.index[40]
    volume.loc[holiday] = np.nan
    features = build_market_internal_features(close, volume)
    assert features.loc[holiday].filter(like="volume_shock_").isna().all()
    assert features.iloc[41].filter(like="volume_shock_").notna().all()


def test_ridge_model_predicts_all_sector_scores():
    rng = np.random.default_rng(32)
    features = pd.DataFrame(rng.normal(size=(252, 6)))
    targets = pd.DataFrame(
        rng.normal(size=(252, len(SECTOR_ETFS))), columns=SECTOR_ETFS
    )
    model = MarketInternalRidgeModel().fit(features, targets)
    prediction = model.predict(features.iloc[-1])
    assert list(prediction.index) == list(SECTOR_ETFS)
    assert np.isfinite(prediction).all()


def test_challenger_records_base_and_internal_predictions_without_future_leakage():
    returns_all, close, volume = sample_market(430, 33)
    sector_returns = returns_all.loc[:, list(SECTOR_ETFS)]
    prior_rng = np.random.default_rng(34)
    prior = pd.DataFrame(
        prior_rng.normal(0.0, 0.01, size=(500, len(SECTOR_ETFS))),
        index=pd.bdate_range("2019-01-02", periods=500),
        columns=SECTOR_ETFS,
    )
    open_price = close.loc[:, list(SECTOR_ETFS)].shift(1)
    open_price.iloc[0] = close.loc[:, list(SECTOR_ETFS)].iloc[0]
    internal = build_market_internal_features(close, volume)
    first = run_sector_rotation_backtest(
        sector_returns,
        prior,
        open_price,
        close.loc[:, list(SECTOR_ETFS)],
        internal_features=internal,
        calibration_observations=10,
        internal_window=60,
        minimum_internal_samples=30,
    )
    changed_internal = internal.copy()
    changed_internal.iloc[-5:] *= 100.0
    second = run_sector_rotation_backtest(
        sector_returns,
        prior,
        open_price,
        close.loc[:, list(SECTOR_ETFS)],
        internal_features=changed_internal,
        calibration_observations=10,
        internal_window=60,
        minimum_internal_samples=30,
    )
    cutoff = sector_returns.index[-6]
    columns = [
        "raw_score",
        "signal_lambda",
        "base_score",
        "base_lambda",
        "internal_score",
        "internal_lambda",
    ]
    pd.testing.assert_frame_equal(
        first.rankings.loc[:cutoff, columns], second.rankings.loc[:cutoff, columns]
    )
    assert first.rankings["internal_score"].notna().any()


def test_challenger_drops_sparse_past_rows_without_backfilling():
    returns_all, close, volume = sample_market(430, 35)
    sector_returns = returns_all.loc[:, list(SECTOR_ETFS)]
    prior = pd.DataFrame(
        np.random.default_rng(36).normal(0.0, 0.01, size=(500, len(SECTOR_ETFS))),
        index=pd.bdate_range("2019-01-02", periods=500),
        columns=SECTOR_ETFS,
    )
    open_price = close.loc[:, list(SECTOR_ETFS)].shift(1)
    open_price.iloc[0] = close.loc[:, list(SECTOR_ETFS)].iloc[0]
    internal = build_market_internal_features(close, volume)
    internal.iloc[30::25] = np.nan
    result = run_sector_rotation_backtest(
        sector_returns,
        prior,
        open_price,
        close.loc[:, list(SECTOR_ETFS)],
        internal_features=internal,
        calibration_observations=10,
        internal_window=60,
        minimum_internal_samples=30,
    )
    assert result.rankings["internal_score"].notna().any()
