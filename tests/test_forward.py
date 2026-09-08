import json

import numpy as np
import pandas as pd

from leading_signal_lambda.collector import MarketDataset
from leading_signal_lambda.forward import freeze_signals, generate_forward_signals, load_dataset, settle_frozen_signals
from leading_signal_lambda.signals import REQUIRED_SYMBOLS


class FakeCalendar:
    def next_session(self, value):
        return value + pd.Timedelta(days=1)


def sample_dataset(rows=760):
    index = pd.bdate_range("2022-01-03", periods=rows)
    rng = np.random.default_rng(42)
    symbols = list(REQUIRED_SYMBOLS) + ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLRE", "XLU", "XLV"]
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
    assert {record.target for record in records} == {"SPY", "QQQ"}
    assert all(record.signal_session == dataset.close.index[-1].date().isoformat() for record in records)
    assert all(record.training_last_date < record.signal_session for record in records)
    assert all(record.status == "PENDING" for record in records)
    frozen = freeze_signals(records, tmp_path / "forward.json")
    assert json.loads(frozen.read_text())["signals"][0]["input_sha256"]


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
    assert len(payload["settlements"]) == 2
    assert all(item["status"] == "SETTLED" for item in payload["settlements"])


def test_loads_already_collected_csv_without_second_provider_call(tmp_path):
    expected = sample_dataset(100)
    expected.save_csv(tmp_path)
    actual = load_dataset(tmp_path)
    pd.testing.assert_frame_equal(actual.close, expected.close, check_freq=False, check_names=False)
    pd.testing.assert_frame_equal(actual.volume, expected.volume, check_freq=False, check_names=False)
