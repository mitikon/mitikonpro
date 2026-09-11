import json

import numpy as np
import pandas as pd

from leading_signal_lambda.collector import MarketDataset
from leading_signal_lambda.forward import (
    FORWARD_TARGETS,
    carry_forward_same_session,
    freeze_signals,
    generate_forward_signals,
    load_dataset,
    select_primary_trade,
    select_extreme_forecasts,
    settle_frozen_signals,
    write_signal_result_markdown,
    write_signal_result_report,
)
from leading_signal_lambda.signals import REQUIRED_SYMBOLS


class FakeCalendar:
    def next_session(self, value):
        return value + pd.Timedelta(days=1)


def sample_dataset(rows=760):
    index = pd.bdate_range("2022-01-03", periods=rows)
    rng = np.random.default_rng(42)
    symbols = list(REQUIRED_SYMBOLS) + ["DIA", "IWM", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLRE", "XLU", "XLV"]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=(rows, len(symbols))), axis=0)),
        index=index,
        columns=symbols,
    )
    volume = pd.DataFrame(rng.integers(1_000_000, 9_000_000, size=close.shape), index=index, columns=symbols)
    return MarketDataset(close=close, volume=volume)


def test_forward_signal_uses_latest_row_without_known_outcome(tmp_path):
    dataset = sample_dataset()
    records = generate_forward_signals(
        dataset,
        FakeCalendar(),
        generated_at_utc=pd.Timestamp("2026-09-09T02:15:00Z"),
    )
    assert {record.target for record in records} == set(FORWARD_TARGETS)
    assert all(record.signal_session == dataset.close.index[-1].date().isoformat() for record in records)
    assert all(record.training_last_date < record.signal_session for record in records)
    assert all(record.excluded_feature_count >= 0 for record in records)
    assert all(record.status == "PENDING" for record in records)
    assert all(np.isfinite(record.predicted_return) for record in records)
    frozen = freeze_signals(records, tmp_path / "forward.json")
    payload = json.loads(frozen.read_text())
    assert payload["signals"][0]["input_sha256"]
    expected = max(records, key=lambda record: abs(record.predicted_return))
    assert payload["primary_trade"]["target"] == expected.target
    assert payload["extreme_forecasts"]["upside"]["target"] == max(
        records, key=lambda record: record.predicted_return
    ).target
    assert payload["extreme_forecasts"]["downside"]["target"] == min(
        records, key=lambda record: record.predicted_return
    ).target


def test_primary_trade_selection_does_not_use_results():
    signals = [
        {"signal_session": "2026-09-09", "target_session": "2026-09-10", "target": "SPY", "target_category": "市場ETF", "target_name": "S&P 500", "predicted_return": 0.01, "confidence": 0.7, "edge": 0.4, "input_sha256": "a"},
        {"signal_session": "2026-09-09", "target_session": "2026-09-10", "target": "SMH", "target_category": "テーマETF", "target_name": "半導体", "predicted_return": -0.02, "confidence": 0.6, "edge": 0.2, "input_sha256": "a"},
    ]
    selected = select_primary_trade(signals)
    assert selected["target"] == "SMH"
    assert selected["action"] == "SHORT"
    assert selected["position"] == -1
    extremes = select_extreme_forecasts(signals)
    assert extremes["upside"]["target"] == "SPY"
    assert extremes["downside"]["target"] == "SMH"


def test_frozen_signal_cannot_be_overwritten(tmp_path):
    dataset = sample_dataset()
    records = generate_forward_signals(dataset, FakeCalendar())
    path = freeze_signals(records, tmp_path / "forward.json")
    try:
        freeze_signals(records, path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("frozen signal must be immutable")


def test_previous_signal_is_settled_when_target_close_arrives(tmp_path):
    prior = sample_dataset(759)
    records = generate_forward_signals(prior, FakeCalendar())
    frozen = freeze_signals(records, tmp_path / "prior.json")
    complete = sample_dataset(760)
    settled = settle_frozen_signals(frozen, complete, tmp_path / "settled.json")
    assert settled is not None
    payload = json.loads(settled.read_text())
    assert len(payload["settlements"]) == len(FORWARD_TARGETS)
    assert all(item["status"] == "SETTLED" for item in payload["settlements"])
    assert all(item["absolute_divergence_pp"] >= 0 for item in payload["settlements"])
    assert payload["primary_trade"]["status"] == "SETTLED"
    assert set(payload["extreme_forecasts"]) == {"upside", "downside"}
    assert all(
        value["selected_actual_rank"] >= 1 for value in payload["extreme_forecasts"].values()
    )


def test_separate_result_report_ranks_movers_and_deduplicates_history(tmp_path):
    prior = sample_dataset(759)
    frozen = freeze_signals(
        generate_forward_signals(prior, FakeCalendar()), tmp_path / "prior.json"
    )
    settled = settle_frozen_signals(frozen, sample_dataset(760), tmp_path / "settled.json")
    report_path, rows_path, history_path = write_signal_result_report(
        settled,
        tmp_path / "report.json",
        tmp_path / "rows.csv",
        tmp_path / "history.csv",
        trade_history_path=tmp_path / "trade_history.csv",
    )
    report = json.loads(report_path.read_text())
    rows = pd.read_csv(rows_path)
    assert report["daily"]["settled_targets"] == len(FORWARD_TARGETS)
    assert report["daily"]["primary_trade"]["target"]
    assert set(report["daily"]["extreme_forecasts"]) == {"upside", "downside"}
    assert 0.0 <= report["cumulative"]["upside_top1_hit_rate"] <= 1.0
    assert 0.0 <= report["cumulative"]["downside_top1_hit_rate"] <= 1.0
    assert report["cumulative"]["primary_trade_count"] == 1
    assert len(pd.read_csv(tmp_path / "trade_history.csv")) == 1
    assert len(report["daily"]["results"]) == len(FORWARD_TARGETS)
    assert np.isclose(
        report["daily"]["largest_risers"][0]["actual_return"], rows["actual_return"].max()
    )
    assert 0.0 <= report["daily"]["direction_accuracy"] <= 1.0
    assert report["definition"]["no_lookahead"]

    second_history = tmp_path / "history_second.csv"
    write_signal_result_report(
        settled,
        tmp_path / "report_second.json",
        tmp_path / "rows_second.csv",
        second_history,
        history_path,
    )
    assert len(pd.read_csv(second_history)) == len(FORWARD_TARGETS)
    second_report = json.loads((tmp_path / "report_second.json").read_text())
    assert second_report["cumulative"]["direction_accuracy"] == report["daily"]["direction_accuracy"]
    markdown = write_signal_result_markdown(report_path, tmp_path / "report.md")
    assert "実績上昇上位" in markdown.read_text()
    assert "主判定：前日に選定した単独トレード" in markdown.read_text()
    assert "主検証：上昇1位・下落1位の事前選出" in markdown.read_text()


def test_loads_already_collected_csv_without_second_provider_call(tmp_path):
    expected = sample_dataset(100)
    expected.save_csv(tmp_path)
    actual = load_dataset(tmp_path)
    pd.testing.assert_frame_equal(actual.close, expected.close, check_freq=False, check_names=False)
    pd.testing.assert_frame_equal(actual.volume, expected.volume, check_freq=False, check_names=False)


def test_same_session_carries_first_signal_without_recalculation(tmp_path):
    dataset = sample_dataset()
    records = generate_forward_signals(
        dataset,
        FakeCalendar(),
        generated_at_utc=pd.Timestamp("2026-09-09T02:15:00Z"),
    )
    previous = freeze_signals(records, tmp_path / "previous.json")
    carried = carry_forward_same_session(previous, dataset, tmp_path / "current.json")
    assert carried is not None
    assert carried.read_bytes() == previous.read_bytes()


def test_same_session_carries_v4_signal_without_recalculation(tmp_path):
    dataset = sample_dataset()
    records = generate_forward_signals(dataset, FakeCalendar())
    previous = freeze_signals(records, tmp_path / "previous.json")
    payload = json.loads(previous.read_text())
    payload["schema_version"] = "market-forward-v4"
    payload.pop("extreme_forecasts")
    for signal in payload["signals"]:
        signal["schema_version"] = "market-forward-v4"
    previous.write_text(json.dumps(payload), encoding="utf-8")

    carried = carry_forward_same_session(previous, dataset, tmp_path / "current.json")

    assert carried is not None
    assert carried.read_bytes() == previous.read_bytes()
