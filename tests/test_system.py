import numpy as np
import pandas as pd
import json

from leading_signal_lambda import (
    DailyMarketCollector,
    LeadingLambdaClassifier,
    build_leading_features,
    build_training_set,
    walk_forward_validate,
)
from leading_signal_lambda.experiment import validate_target
from leading_signal_lambda.market_calendar import NYSETradingCalendar


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


def test_zero_volume_index_series_do_not_remove_all_training_rows():
    close, volume = sample_market(760)
    volume.loc[:, ["VIX9D", "VIX3M"]] = 0
    features = build_leading_features(close, volume)
    X, _, _ = build_training_set(features, close["SPY"])
    assert "volume_ratio_VIX9D" not in features
    assert "volume_ratio_VIX3M" not in features
    assert len(X) > 504


def test_sparse_volume_series_is_ignored_without_future_filling():
    close, volume = sample_market(760)
    volume["RATE_INDEX"] = np.nan
    volume.loc[volume.index[:10], "RATE_INDEX"] = 1.0
    close["RATE_INDEX"] = 100.0
    features = build_leading_features(close, volume)
    assert "volume_ratio_RATE_INDEX" not in features


def test_optional_close_series_with_no_history_is_ignored():
    close, volume = sample_market(760)
    close["OPTIONAL_BROKEN"] = np.nan
    volume["OPTIONAL_BROKEN"] = 0
    features = build_leading_features(close, volume)
    assert not any("OPTIONAL_BROKEN" in column for column in features)


def test_required_close_series_with_insufficient_history_is_rejected():
    close, volume = sample_market(760)
    close.loc[close.index[:720], "SPY"] = np.nan
    try:
        build_leading_features(close, volume)
    except ValueError as error:
        assert "insufficient close coverage" in str(error)
    else:
        raise AssertionError("required sparse price history must be rejected")


def test_stale_optional_close_series_is_ignored():
    close, volume = sample_market(760)
    close.loc[close.index[-20:], "VIX9D"] = np.nan
    features = build_leading_features(close, volume)
    assert not any("VIX9D" in column for column in features)
    assert "vix_term_spread" not in features


def test_market_holiday_does_not_poison_next_twenty_volume_rows():
    close, volume = sample_market(120)
    volume.loc[volume.index[60], "SPY"] = np.nan
    features = build_leading_features(close, volume)
    ratio = features["volume_ratio_SPY"]
    assert pd.isna(ratio.loc[volume.index[60]])
    assert pd.notna(ratio.loc[volume.index[61]])


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
    assert result.open is not None
    assert result.high is not None
    assert result.low is not None


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


def test_validation_report_compares_strategy_and_benchmark(tmp_path):
    close, volume = sample_market(760)
    report = validate_target(close, volume, "SPY", tmp_path, train_size=252)
    assert report.target == "SPY"
    assert np.isfinite(report.annualized_excess_return)
    assert (tmp_path / "spy_frozen_predictions.csv").exists()


def test_validation_writes_diagnostics_before_insufficient_rows_error(tmp_path):
    close, volume = sample_market(420)
    try:
        validate_target(close, volume, "SPY", tmp_path, train_size=504)
    except ValueError as error:
        assert "valid training rows" in str(error)
    else:
        raise AssertionError("insufficient real-data rows must be rejected")
    diagnostics = json.loads((tmp_path / "spy_data_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["raw_rows"] == 420
    assert diagnostics["training_rows"] <= 504


class FakeNYSECalendar:
    def __init__(self):
        self.sessions = pd.bdate_range("2026-08-31", "2026-09-04", tz="UTC")

    def date_to_session(self, value, direction="none"):
        stamp = pd.Timestamp(value, tz="UTC")
        if stamp in self.sessions:
            return stamp
        if direction == "previous":
            return self.sessions[self.sessions <= stamp][-1]
        raise ValueError("not a session")

    def session_close(self, session):
        return session + pd.Timedelta(hours=20)

    def previous_session(self, session):
        return self.sessions[self.sessions.get_loc(session) - 1]


def test_calendar_uses_only_completed_session():
    calendar = NYSETradingCalendar(calendar=FakeNYSECalendar())
    before_close = calendar.last_completed_session(pd.Timestamp("2026-09-04 19:00", tz="UTC"))
    after_close = calendar.last_completed_session(pd.Timestamp("2026-09-04 21:00", tz="UTC"))
    assert before_close.session_date.isoformat() == "2026-09-03"
    assert after_close.session_date.isoformat() == "2026-09-04"


def test_calendar_applies_exceptional_closure(tmp_path):
    path = tmp_path / "closures.json"
    path.write_text('["2026-09-04"]', encoding="utf-8")
    calendar = NYSETradingCalendar(calendar=FakeNYSECalendar(), exceptional_closures=path)
    completed = calendar.last_completed_session(pd.Timestamp("2026-09-04 21:00", tz="UTC"))
    assert completed.session_date.isoformat() == "2026-09-03"
