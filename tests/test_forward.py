import json
import sys

import numpy as np
import pandas as pd
import pytest

from maintenance_rsi import ExternalDataGuard, MalwareScan, MalwareStatus
from leading_signal_lambda import forward as forward_module
from leading_signal_lambda.collector import MarketDataset
from leading_signal_lambda.forward import (
    FORWARD_TARGETS,
    carry_forward_histories,
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
from leading_signal_lambda.error_classification import ERROR_CATEGORIES
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
    assert all(record.relative_strength_feature_version == "relative-strength-feature-v1" for record in records)
    assert all(record.relative_strength_periods == (5, 7, 14, 21) for record in records)
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


def test_recursive_candidate_parameters_reach_the_prediction_model():
    dataset = sample_dataset()
    baseline = generate_forward_signals(dataset, FakeCalendar())
    candidate = generate_forward_signals(
        dataset,
        FakeCalendar(),
        model_parameters={
            "relative_strength_periods": [7, 14, 28],
            "relative_strength_feature_set": ["level", "velocity3", "cross50"],
            "relative_strength_feature_weight": 1.5,
            "feature_lags": 7,
            "neutral_band": 0.0015,
            "no_trade_threshold": 0.50,
        },
        model_generation=1,
    )
    assert candidate[0].model_generation == 1
    assert candidate[0].model_config_sha256 != baseline[0].model_config_sha256
    assert candidate[0].feature_count != baseline[0].feature_count


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
    assert set(report["daily"]["error_classification"]) == ERROR_CATEGORIES
    assert set(report["cumulative"]["error_classification"]) == ERROR_CATEGORIES
    assert all(isinstance(value, int) for value in report["daily"]["error_classification"].values())
    misclassified_targets = sum(1 for row in report["daily"]["results"] if not row["direction_correct"])
    assert sum(report["daily"]["error_classification"].values()) <= misclassified_targets + 2

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


def test_error_classification_tolerates_pre_migration_history_without_new_columns(tmp_path):
    """A history.csv written before neutral_band/imputed_feature_count existed must not crash."""
    prior = sample_dataset(759)
    frozen = freeze_signals(generate_forward_signals(prior, FakeCalendar()), tmp_path / "prior.json")
    settled = settle_frozen_signals(frozen, sample_dataset(760), tmp_path / "settled.json")

    legacy_history = pd.DataFrame(
        [
            {
                "signal_session": "2026-08-01",
                "target_session": "2026-08-02",
                "target": "SPY",
                "target_category": "市場ETF",
                "target_name": "S&P 500",
                "action": "SHORT",
                "predicted_class": -1,
                "predicted_return": -0.01,
                "confidence": 0.5,
                "edge": 0.1,
                "actual_return": 0.02,
                "actual_class": 1,
                "direction_correct": False,
                "return_error": 0.03,
                "absolute_divergence_pp": 3.0,
                "strategy_return_before_cost": -0.02,
                "input_sha256": "b" * 64,
            }
        ]
    )
    legacy_path = tmp_path / "legacy_history.csv"
    legacy_history.to_csv(legacy_path, index=False)

    report_path, _, _ = write_signal_result_report(
        settled,
        tmp_path / "report_with_legacy.json",
        tmp_path / "rows_with_legacy.csv",
        tmp_path / "history_with_legacy.csv",
        previous_history_path=legacy_path,
    )
    report = json.loads(report_path.read_text())
    assert set(report["cumulative"]["error_classification"]) == ERROR_CATEGORIES
    assert report["cumulative"]["settled_predictions"] == len(FORWARD_TARGETS) + 1


def test_loads_already_collected_csv_without_second_provider_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    expected = sample_dataset(100)
    expected.save_csv(tmp_path)
    actual = load_dataset(tmp_path)
    pd.testing.assert_frame_equal(actual.close, expected.close, check_freq=False, check_names=False)
    pd.testing.assert_frame_equal(actual.volume, expected.volume, check_freq=False, check_names=False)


def test_load_dataset_rejects_and_quarantines_disguised_executable_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    expected = sample_dataset(100)
    expected.save_csv(tmp_path)
    (tmp_path / "daily_close.csv").write_bytes(b"MZmalicious")
    try:
        load_dataset(tmp_path)
    except RuntimeError as error:
        assert "external data guard rejected" in str(error)
    else:
        raise AssertionError("disguised executable content must be rejected")
    assert not (tmp_path / "daily_close.csv").exists()
    assert any((tmp_path / "quarantine").glob("*.csv.quarantine"))


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


def test_cumulative_histories_survive_a_run_without_settlement(tmp_path):
    first = pd.DataFrame(
        [
            {"signal_session": "2026-09-08", "target": "SPY", "actual_return": 0.01},
            {"signal_session": "2026-09-08", "target": "QQQ", "actual_return": 0.02},
        ]
    )
    second = pd.DataFrame(
        [
            {"signal_session": "2026-09-08", "target": "SPY", "actual_return": 0.01},
            {"signal_session": "2026-09-09", "target": "SPY", "actual_return": -0.01},
        ]
    )
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    output = tmp_path / "carried.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    carried = carry_forward_histories(
        [first_path, second_path],
        output,
        deduplicate_by=["signal_session", "target"],
        sort_by=["signal_session", "target"],
    )

    assert carried == output
    rows = pd.read_csv(output)
    assert len(rows) == 3
    assert set(rows["signal_session"]) == {"2026-09-08", "2026-09-09"}

    trade_output = tmp_path / "trades.csv"
    carried_trades = carry_forward_histories(
        [first_path, second_path],
        trade_output,
        deduplicate_by=["signal_session"],
        sort_by=["signal_session", "target"],
    )
    assert carried_trades == trade_output
    trade_rows = pd.read_csv(trade_output)
    assert len(trade_rows) == 2
    assert trade_rows["signal_session"].is_unique


def test_main_cli_bootstraps_and_settles_parallel_candidates(tmp_path, monkeypatch):
    # main() hardcodes a real NYSETradingCalendar with no injection point, so
    # this end-to-end CLI test needs the optional [data] extra; it skips
    # cleanly where that extra isn't installed (e.g. the maintenance-rsi
    # workflow's [test]-only job) rather than failing there.
    pytest.importorskip("exchange_calendars")
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )

    day1 = tmp_path / "raw1"
    sample_dataset(759).save_csv(day1)
    day2 = tmp_path / "raw2"
    sample_dataset(760).save_csv(day2)

    art1 = tmp_path / "art1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leading-lambda-forward",
            "--input-dir", str(day1),
            "--output", str(art1 / "forward_signal.json"),
            "--rsi-state-output", str(art1 / "recursive_rsi_state.json"),
            "--candidate-output", str(art1 / "recursive_rsi_candidate_signal.json"),
        ],
    )
    forward_module.main()

    candidate_paths = [
        art1 / "recursive_rsi_candidate_signal.json",
        art1 / "recursive_rsi_candidate_signal_2.json",
        art1 / "recursive_rsi_candidate_signal_3.json",
    ]
    assert all(path.exists() for path in candidate_paths)
    state_after_bootstrap = json.loads((art1 / "recursive_rsi_state.json").read_text())
    assert len(state_after_bootstrap["candidate_slots"]) == 3

    art2 = tmp_path / "art2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leading-lambda-forward",
            "--input-dir", str(day2),
            "--output", str(art2 / "forward_signal.json"),
            "--previous", str(art1 / "forward_signal.json"),
            "--previous-rsi-state", str(art1 / "recursive_rsi_state.json"),
            "--previous-candidate-signal", str(candidate_paths[0]),
            "--previous-candidate-signal", str(candidate_paths[1]),
            "--previous-candidate-signal", str(candidate_paths[2]),
            "--rsi-state-output", str(art2 / "recursive_rsi_state.json"),
            "--candidate-output", str(art2 / "recursive_rsi_candidate_signal.json"),
            "--settlement-output", str(art2 / "settled_previous_signal.json"),
            "--candidate-settlement-output", str(art2 / "settled_recursive_rsi_candidate.json"),
            "--report-output", str(art2 / "signal_result_report.json"),
            "--report-markdown-output", str(art2 / "signal_result_report.md"),
            "--report-rows-output", str(art2 / "signal_result_rows.csv"),
            "--history-output", str(art2 / "signal_result_history.csv"),
            "--trade-history-output", str(art2 / "selected_trade_history.csv"),
        ],
    )
    forward_module.main()

    assert (art2 / "settled_recursive_rsi_candidate.json").exists()
    assert (art2 / "settled_recursive_rsi_candidate_2.json").exists()
    assert (art2 / "settled_recursive_rsi_candidate_3.json").exists()
    state_after_settle = json.loads((art2 / "recursive_rsi_state.json").read_text())
    assert [len(slot["evaluations"]) for slot in state_after_settle["candidate_slots"]] == [1, 1, 1]
    assert all(slot["pending_trial"] is not None for slot in state_after_settle["candidate_slots"])


def test_main_cli_migrates_a_legacy_single_candidate_state_on_first_run(tmp_path, monkeypatch):
    # Simulates the exact transition the live daily pipeline goes through on
    # its first run after parallel candidates are deployed: an existing v1
    # schema_version state with exactly one prior candidate forecast, now
    # asked to run against a CLI/runtime that wants PARALLEL_CANDIDATES
    # slots. The single carried-over slot must settle normally; the freshly
    # padded slots must be skipped (no prior forecast to compare), not
    # treated as an error; and every slot must end the run with a fresh
    # pending trial so nothing is left unregistered.
    pytest.importorskip("exchange_calendars")
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    from datetime import datetime, timedelta, timezone

    from leading_signal_lambda.forward import DEFAULT_MODEL_PARAMETERS
    from leading_signal_lambda.recursive_runtime import (
        LEGACY_RUNTIME_SCHEMA_VERSION,
        PARALLEL_CANDIDATES,
        _new_candidate,
        candidate_manifest_digest,
        register_frozen_trials,
        write_state,
    )

    day1 = tmp_path / "raw1"
    dataset1 = sample_dataset(759)
    dataset1.save_csv(day1)
    day2 = tmp_path / "raw2"
    sample_dataset(760).save_csv(day2)

    created = datetime(2026, 9, 15, tzinfo=timezone.utc)
    legacy_state = {
        "schema_version": LEGACY_RUNTIME_SCHEMA_VERSION,
        "active_model": {
            "model_id": "market-forward-v7-baseline",
            "generation": 0,
            "parameters": dict(DEFAULT_MODEL_PARAMETERS),
            "promotion_report_sha256": None,
        },
        "attempt": 0,
        "candidate": None,
        "candidate_manifest_sha256": None,
        "trials": [],
        "evaluations": [],
        "pending_trial": None,
        "completed_candidates": [],
        "previous_state_sha256": None,
        "updated_at": created.isoformat(),
    }
    legacy_candidate = _new_candidate(legacy_state, "a" * 40, created - timedelta(microseconds=1))
    legacy_state["attempt"] = 1
    legacy_state["candidate"] = legacy_candidate.sealed_payload()
    legacy_state["candidate_manifest_sha256"] = candidate_manifest_digest(legacy_candidate)

    art1 = tmp_path / "art1"
    art1.mkdir()
    generated_at = pd.Timestamp("2026-09-16T02:15:00Z")
    baseline_records = generate_forward_signals(
        dataset1, FakeCalendar(), generated_at_utc=generated_at,
        model_parameters=dict(legacy_state["active_model"]["parameters"]), model_generation=0,
    )
    baseline_path = freeze_signals(baseline_records, art1 / "forward_signal.json")
    candidate_records = generate_forward_signals(
        dataset1, FakeCalendar(), generated_at_utc=generated_at,
        model_parameters=dict(legacy_candidate.parameters), model_generation=1,
    )
    candidate_path = freeze_signals(candidate_records, art1 / "recursive_rsi_candidate_signal.json")

    # Register the legacy trial the way a real v1 run would have.
    registration_state = {
        **legacy_state,
        "candidate_slots": [
            {
                "attempt": 1,
                "candidate": legacy_state["candidate"],
                "candidate_manifest_sha256": legacy_state["candidate_manifest_sha256"],
                "trials": [],
                "evaluations": [],
                "pending_trial": None,
            }
        ],
    }
    register_frozen_trials(registration_state, baseline_path, [candidate_path])
    legacy_state["pending_trial"] = registration_state["candidate_slots"][0]["pending_trial"]
    legacy_state["trials"] = registration_state["candidate_slots"][0]["trials"]
    legacy_state_path = write_state(legacy_state, art1 / "recursive_rsi_state.json")

    art2 = tmp_path / "art2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leading-lambda-forward",
            "--input-dir", str(day2),
            "--output", str(art2 / "forward_signal.json"),
            "--previous", str(baseline_path),
            "--previous-rsi-state", str(legacy_state_path),
            "--previous-candidate-signal", str(candidate_path),
            "--rsi-state-output", str(art2 / "recursive_rsi_state.json"),
            "--candidate-output", str(art2 / "recursive_rsi_candidate_signal.json"),
            "--settlement-output", str(art2 / "settled_previous_signal.json"),
            "--candidate-settlement-output", str(art2 / "settled_recursive_rsi_candidate.json"),
            "--report-output", str(art2 / "signal_result_report.json"),
            "--report-markdown-output", str(art2 / "signal_result_report.md"),
            "--report-rows-output", str(art2 / "signal_result_rows.csv"),
            "--history-output", str(art2 / "signal_result_history.csv"),
            "--trade-history-output", str(art2 / "selected_trade_history.csv"),
        ],
    )
    forward_module.main()

    state2 = json.loads((art2 / "recursive_rsi_state.json").read_text())
    assert state2["schema_version"] == "market-recursive-runtime-v2"
    assert state2["migrated_from"] == LEGACY_RUNTIME_SCHEMA_VERSION
    assert len(state2["candidate_slots"]) == PARALLEL_CANDIDATES
    evaluations_per_slot = [len(slot["evaluations"]) for slot in state2["candidate_slots"]]
    assert evaluations_per_slot[0] == 1
    assert evaluations_per_slot[1:] == [0] * (PARALLEL_CANDIDATES - 1)
    assert all(slot["pending_trial"] is not None for slot in state2["candidate_slots"])
