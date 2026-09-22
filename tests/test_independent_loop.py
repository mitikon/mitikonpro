import json

import numpy as np
import pandas as pd

from leading_signal_lambda.collector import MarketDataset
from leading_signal_lambda.independent_loop import (
    calendar_run_decision, freeze_shadow_forecasts, initial_state, learn, report_for_date,
    settle_shadow_forecasts,
)
from leading_signal_lambda.signals import REQUIRED_SYMBOLS


class Calendar:
    def next_session(self, value):
        return value + pd.Timedelta(days=1)

    def last_completed_session(self, now_utc=None):
        from leading_signal_lambda.market_calendar import CompletedSession
        return CompletedSession(pd.Timestamp("2026-09-22").date(), pd.Timestamp("2026-09-22T20:00:00Z"), pd.Timestamp("2026-09-23").date())


def dataset(rows=760):
    index = pd.bdate_range("2022-01-03", periods=rows)
    rng = np.random.default_rng(7)
    extras = ["DIA", "IWM", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLRE", "XLU", "XLV"]
    symbols = list(dict.fromkeys([*REQUIRED_SYMBOLS, *extras]))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (rows, len(symbols))), axis=0)), index=index, columns=symbols)
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, close.shape), index=index, columns=symbols)
    return MarketDataset(close, volume)


def test_independent_forecast_is_write_once_and_settled_without_mutation(tmp_path):
    frozen = freeze_shadow_forecasts(dataset(759), Calendar(), initial_state(), tmp_path / "forecast.json")
    before = frozen.read_bytes()
    settled = settle_shadow_forecasts(frozen, dataset(760), tmp_path / "settled.json")
    assert frozen.read_bytes() == before
    assert settled is not None
    value = json.loads(settled.read_text())
    assert value["status"] == "SETTLED_WITHOUT_FORECAST_MUTATION"
    assert len(value["results"]) == 4
    assert report_for_date([settled], value["target_session"])["settlement_sha256"]


def test_learning_requires_20_future_sessions_and_never_attaches_to_pca(tmp_path):
    state = initial_state()
    paths = []
    for index in range(19):
        results = {}
        for band in state["candidate_neutral_bands"]:
            results[f"neutral_band={band:.6f}"] = {
                "neutral_band": band, "direction_accuracy": 0.5 + (0.1 if band == 0.002 else 0),
                "mean_absolute_error": 0.01, "primary_trade_return": 0.001,
                "primary_trade_correct": True, "upside_exact_hit": False, "downside_exact_hit": False, "rows": [],
            }
        payload = {"signal_session": f"2026-01-{index+1:02d}", "target_session": f"2026-02-{index+1:02d}", "results": results}
        path = tmp_path / f"s{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    next_state, report = learn(state, paths)
    assert report["promotion_status"] == "NO_PROMOTION"
    assert next_state["production_pca_attached"] is False


def test_calendar_gate_runs_once_per_completed_market_session(tmp_path):
    first = calendar_run_decision(Calendar())
    assert first["should_run"] is True
    assert first["calendar"] == "XNYS"
    previous = tmp_path / "forecast.json"
    previous.write_text(json.dumps({"signal_session": "2026-09-22"}))
    duplicate = calendar_run_decision(Calendar(), previous)
    assert duplicate["should_run"] is False
    assert duplicate["reason"] == "ALREADY_PROCESSED_OR_MARKET_CLOSED"
