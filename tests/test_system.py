import numpy as np
import pandas as pd

from leading_signal_lambda import (
    DailyMarketCollector,
    LeadingLambdaClassifier,
    build_leading_features,
    build_training_set,
    walk_forward_validate,
)


def sample_market(rows: int = 420):
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2023-01-02", periods=rows)
    symbols = ["SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP", "VIX9D", "VIX3M"]
    shocks = rng.normal(0.0003, 0.01, size=(rows, len(symbols)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=index, columns=symbols)
    volume = pd.DataFrame(rng.integers(1_000_000, 9_000_000, size=close.shape), index=index, columns=symbols)
    return close, volume


def test_lambda_and_variance_are_separate_controls():
    model = LeadingLambdaClassifier(lambda_reg=0.10, variance_target=0.90)
    assert model.lambda_reg == 0.10
    assert model.variance_target == 0.90


def test_pipeline_produces_finite_walk_forward_metrics():
    close, volume = sample_market()
    features = build_leading_features(close, volume)
    X, y, returns = build_training_set(features, close["SPY"])
    result = walk_forward_validate(X, y, returns, train_size=252, test_size=21, min_samples=60)
    assert not result.predictions.empty
    assert set(result.predictions["action"]) <= {"LONG", "SHORT", "NO_TRADE"}
    assert all(np.isfinite(value) for value in result.metrics.values())


def test_unsorted_input_is_rejected():
    close, _ = sample_market(80)
    try:
        build_leading_features(close.sort_index(ascending=False))
    except ValueError as error:
        assert "ascending" in str(error)
    else:
        raise AssertionError("unsorted data must be rejected")


def test_collector_maps_symbols_and_excludes_end_date():
    index = pd.date_range("2024-01-02", periods=4)
    tickers = ["SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP"]
    columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], tickers])
    raw = pd.DataFrame(1.0, index=index, columns=columns)

    def fake_download(**kwargs):
        assert kwargs["interval"] == "1d"
        return raw

    collector = DailyMarketCollector(universe={ticker: ticker for ticker in tickers}, downloader=fake_download)
    result = collector.collect("2024-01-02", "2024-01-05")
    assert list(result.close.columns) == tickers
    assert result.close.index.max() == pd.Timestamp("2024-01-04")


def test_collector_rejects_missing_required_series():
    raw = pd.DataFrame({("Adj Close", "SPY"): [100.0]}, index=[pd.Timestamp("2024-01-02")])
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    collector = DailyMarketCollector(universe={"SPY": "SPY"}, downloader=lambda **kwargs: raw)
    try:
        collector.collect("2024-01-01", "2024-01-03")
    except RuntimeError as error:
        assert "required daily series" in str(error)
    else:
        raise AssertionError("missing required series must be rejected")
