import pandas as pd
import pytest

from leading_signal_lambda.benchmark_report import buy_and_hold_annualized_return


def test_buy_and_hold_annualized_return_doubles_over_one_year():
    index = pd.bdate_range("2024-01-02", periods=2)
    close = pd.Series([100.0, 200.0], index=[index[0], index[0] + pd.Timedelta(days=365)])
    report = buy_and_hold_annualized_return(close, "2024-01-01", "2025-01-05")
    assert report["total_return"] == pytest.approx(1.0)
    assert report["annualized_return"] == pytest.approx(1.0, abs=0.01)


def test_buy_and_hold_annualized_return_bounds_to_observed_sessions_in_window():
    index = pd.to_datetime(["2020-01-02", "2022-06-15", "2025-12-31", "2026-03-01"])
    close = pd.Series([100.0, 130.0, 180.0, 500.0], index=index)
    # Window excludes the last point (2026-03-01) entirely.
    report = buy_and_hold_annualized_return(close, "2021-01-01", "2025-12-31")
    assert report["start_date"] == "2022-06-15"
    assert report["end_date"] == "2025-12-31"
    assert report["start_price"] == 130.0
    assert report["end_price"] == 180.0


def test_buy_and_hold_annualized_return_raises_with_fewer_than_two_observed_sessions():
    close = pd.Series([100.0], index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="not enough observed sessions"):
        buy_and_hold_annualized_return(close, "2024-01-01", "2024-12-31")


def test_buy_and_hold_annualized_return_drops_nan_rows():
    index = pd.to_datetime(["2024-01-02", "2024-06-01", "2025-01-02"])
    close = pd.Series([100.0, float("nan"), 110.0], index=index)
    report = buy_and_hold_annualized_return(close, "2024-01-01", "2025-01-02")
    assert report["start_price"] == 100.0
    assert report["end_price"] == 110.0
