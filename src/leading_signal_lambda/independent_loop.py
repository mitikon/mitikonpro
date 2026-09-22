"""Independent, outcome-locked learning loop for leading-signal lambda.

This module deliberately does not mutate or attach itself to the production
subspace-regularized PCA program.  It runs shadow parameter candidates,
freezes every forecast before the target session, settles them afterwards and
promotes only a candidate that wins on genuinely future sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .forward import (
    FrozenMarketSignal,
    generate_forward_signals,
    load_dataset,
    select_extreme_forecasts,
    select_primary_trade,
)
from .market_calendar import NYSETradingCalendar


SCHEMA_VERSION = "independent-market-rsi-v1"
DEFAULT_NEUTRAL_BANDS = (0.0005, 0.001, 0.0015, 0.002)
MIN_PROMOTION_SESSIONS = 20


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_once(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable RSI record: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def initial_state() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": 1,
        "active_neutral_band": 0.001,
        "candidate_neutral_bands": list(DEFAULT_NEUTRAL_BANDS),
        "minimum_future_sessions": MIN_PROMOTION_SESSIONS,
        "production_pca_attached": False,
        "automatic_source_push": False,
        "automatic_trade_execution": False,
    }


def load_state(path: str | Path | None) -> dict[str, object]:
    if path is None or not Path(path).exists():
        return initial_state()
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported independent RSI state")
    return state


def calendar_run_decision(
    calendar: NYSETradingCalendar,
    previous_forecast: str | Path | None = None,
    now_utc: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Allow one learning cycle per completed XNYS session, never per cron tick."""
    completed = calendar.last_completed_session(now_utc)
    previous_session = None
    if previous_forecast and Path(previous_forecast).exists():
        previous = json.loads(Path(previous_forecast).read_text(encoding="utf-8"))
        previous_session = previous.get("signal_session")
        if previous_session is None:
            raise ValueError("previous forecast has no signal_session")
        pd.Timestamp(previous_session).date()
    should_run = previous_session is None or completed.session_date > pd.Timestamp(previous_session).date()
    return {
        "schema_version": SCHEMA_VERSION,
        "calendar": "XNYS",
        "completed_session": completed.session_date.isoformat(),
        "completed_session_close_utc": completed.close_utc.isoformat(),
        "previous_signal_session": previous_session,
        "next_session": calendar.next_session(completed.session_date).isoformat(),
        "should_run": should_run,
        "reason": "NEW_COMPLETED_SESSION" if should_run else "ALREADY_PROCESSED_OR_MARKET_CLOSED",
    }


def freeze_shadow_forecasts(dataset, calendar, state: dict[str, object], output: str | Path) -> Path:
    candidates: dict[str, object] = {}
    for band in state["candidate_neutral_bands"]:
        records = generate_forward_signals(dataset, calendar, neutral_band=float(band))
        rows = [record.__dict__ for record in records]
        candidates[f"neutral_band={float(band):.6f}"] = {
            "neutral_band": float(band),
            "primary_trade": select_primary_trade(rows),
            "extreme_forecasts": select_extreme_forecasts(rows),
            "signals": rows,
        }
    first = next(iter(candidates.values()))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(state["generation"]),
        "signal_session": first["signals"][0]["signal_session"],
        "target_session": first["signals"][0]["target_session"],
        "active_neutral_band": float(state["active_neutral_band"]),
        "state_sha256": _sha(state),
        "input_sha256": first["signals"][0]["input_sha256"],
        "candidates": candidates,
        "status": "FROZEN_BEFORE_OUTCOME",
    }
    payload["forecast_sha256"] = _sha(payload)
    return _write_once(Path(output), payload)


def settle_shadow_forecasts(frozen_path: str | Path, dataset, output: str | Path) -> Path | None:
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    target_day = pd.Timestamp(frozen["target_session"])
    signal_day = pd.Timestamp(frozen["signal_session"])
    if target_day not in dataset.close.index:
        return None
    results: dict[str, object] = {}
    for candidate_id, candidate in frozen["candidates"].items():
        rows = []
        for signal in candidate["signals"]:
            target = signal["target"]
            before = float(dataset.close.loc[signal_day, target])
            after = float(dataset.close.loc[target_day, target])
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            actual_return = after / before - 1.0
            band = float(signal["neutral_band"])
            actual_class = 1 if actual_return > band else -1 if actual_return < -band else 0
            rows.append({
                "target": target,
                "target_name": signal["target_name"],
                "target_category": signal["target_category"],
                "predicted_class": int(signal["predicted_class"]),
                "predicted_return": float(signal["predicted_return"]),
                "actual_class": actual_class,
                "actual_return": actual_return,
                "direction_correct": int(signal["predicted_class"]) == actual_class,
                "absolute_error": abs(float(signal["predicted_return"]) - actual_return),
            })
        if not rows:
            raise ValueError(
                f"settlement has no finite-price rows for {candidate_id} on "
                f"{signal_day.date()}/{target_day.date()}"
            )
        primary = candidate["primary_trade"]
        selected = next((row for row in rows if row["target"] == primary["target"]), None)
        if selected is None:
            raise ValueError(
                f"settlement is missing the primary trade target {primary['target']!r} "
                f"for {candidate_id} on {signal_day.date()}/{target_day.date()}"
            )
        position = int(primary["position"])
        actual_up = max(rows, key=lambda row: row["actual_return"])["target"]
        actual_down = min(rows, key=lambda row: row["actual_return"])["target"]
        results[candidate_id] = {
            "neutral_band": candidate["neutral_band"],
            "direction_accuracy": float(np.mean([row["direction_correct"] for row in rows])),
            "mean_absolute_error": float(np.mean([row["absolute_error"] for row in rows])),
            "primary_trade_return": position * selected["actual_return"],
            "primary_trade_correct": (position * selected["actual_return"]) > 0.0,
            "upside_exact_hit": candidate["extreme_forecasts"]["upside"]["target"] == actual_up,
            "downside_exact_hit": candidate["extreme_forecasts"]["downside"]["target"] == actual_down,
            "rows": rows,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "signal_session": frozen["signal_session"],
        "target_session": frozen["target_session"],
        "forecast_sha256": frozen["forecast_sha256"],
        "status": "SETTLED_WITHOUT_FORECAST_MUTATION",
        "results": results,
    }
    payload["settlement_sha256"] = _sha(payload)
    return _write_once(Path(output), payload)


def learn(state: dict[str, object], settlements: Iterable[str | Path]) -> tuple[dict[str, object], dict[str, object]]:
    documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in settlements]
    documents = sorted(documents, key=lambda value: value["target_session"])
    seen = set()
    documents = [doc for doc in documents if not (doc["target_session"] in seen or seen.add(doc["target_session"]))]
    summary: dict[str, dict[str, float]] = {}
    candidate_ids = sorted(set.intersection(*(set(doc["results"]) for doc in documents))) if documents else []
    for candidate_id in candidate_ids:
        rows = [doc["results"][candidate_id] for doc in documents]
        summary[candidate_id] = {
            "neutral_band": float(rows[0]["neutral_band"]),
            "sessions": len(rows),
            "direction_accuracy": float(np.mean([row["direction_accuracy"] for row in rows])),
            "mean_absolute_error": float(np.mean([row["mean_absolute_error"] for row in rows])),
            "primary_trade_win_rate": float(np.mean([row["primary_trade_correct"] for row in rows])),
            "mean_primary_trade_return": float(np.mean([row["primary_trade_return"] for row in rows])),
            "upside_top1_hit_rate": float(np.mean([row["upside_exact_hit"] for row in rows])),
            "downside_top1_hit_rate": float(np.mean([row["downside_exact_hit"] for row in rows])),
        }
    active = float(state["active_neutral_band"])
    eligible = [value for value in summary.values() if value["sessions"] >= int(state["minimum_future_sessions"])]
    promoted = False
    chosen = active
    if eligible:
        baseline = next((value for value in eligible if value["neutral_band"] == active), None)
        best = max(eligible, key=lambda value: (value["direction_accuracy"], -value["mean_absolute_error"], value["mean_primary_trade_return"]))
        if baseline and best["neutral_band"] != active and best["direction_accuracy"] > baseline["direction_accuracy"] and best["mean_absolute_error"] <= baseline["mean_absolute_error"] and best["mean_primary_trade_return"] >= baseline["mean_primary_trade_return"]:
            chosen = best["neutral_band"]
            promoted = True
    next_state = {**state, "active_neutral_band": chosen, "generation": int(state["generation"]) + int(promoted)}
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_sessions": len(documents),
        "candidate_scores": summary,
        "previous_active_neutral_band": active,
        "selected_neutral_band": chosen,
        "promotion_status": "PROMOTED_TO_INDEPENDENT_LOOP" if promoted else "NO_PROMOTION",
        "production_pca_attached": False,
        "source_code_modified_by_learning": False,
        "report_sha256": "",
    }
    report["report_sha256"] = _sha({**report, "report_sha256": ""})
    return next_state, report


def report_for_date(settlements: Iterable[str | Path], requested_date: str) -> dict[str, object]:
    date = pd.Timestamp(requested_date).date().isoformat()
    matches = []
    for path in settlements:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if date in {value["signal_session"], value["target_session"]}:
            matches.append(value)
    if not matches:
        raise LookupError(f"no settled report for {date}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent market RSI loop")
    sub = parser.add_subparsers(dest="command", required=True)
    forecast = sub.add_parser("forecast")
    forecast.add_argument("--input-dir", required=True)
    forecast.add_argument("--state")
    forecast.add_argument("--output", required=True)
    forecast.add_argument("--exceptional-closures", default="config/exceptional_nyse_closures.json")
    settle = sub.add_parser("settle")
    settle.add_argument("--input-dir", required=True)
    settle.add_argument("--forecast", required=True)
    settle.add_argument("--output", required=True)
    learn_cmd = sub.add_parser("learn")
    learn_cmd.add_argument("--state")
    learn_cmd.add_argument("--settlement", action="append", default=[])
    learn_cmd.add_argument("--state-output", required=True)
    learn_cmd.add_argument("--report-output", required=True)
    report = sub.add_parser("report")
    report.add_argument("--date", required=True)
    report.add_argument("--settlement", action="append", default=[])
    gate = sub.add_parser("calendar-check")
    gate.add_argument("--previous")
    gate.add_argument("--now-utc")
    gate.add_argument("--exceptional-closures", default="config/exceptional_nyse_closures.json")
    args = parser.parse_args()
    if args.command == "forecast":
        result = freeze_shadow_forecasts(load_dataset(args.input_dir), NYSETradingCalendar(exceptional_closures=args.exceptional_closures), load_state(args.state), args.output)
        print(result.read_text(encoding="utf-8"))
    elif args.command == "settle":
        result = settle_shadow_forecasts(args.forecast, load_dataset(args.input_dir), args.output)
        print(result.read_text(encoding="utf-8") if result else "PENDING")
    elif args.command == "learn":
        state, report_value = learn(load_state(args.state), args.settlement)
        Path(args.state_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.state_output).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_once(Path(args.report_output), report_value)
        print(json.dumps(report_value, ensure_ascii=False, indent=2))
    elif args.command == "report":
        print(json.dumps(report_for_date(args.settlement, args.date), ensure_ascii=False, indent=2))
    else:
        calendar = NYSETradingCalendar(exceptional_closures=args.exceptional_closures)
        now = pd.Timestamp(args.now_utc) if args.now_utc else None
        decision = calendar_run_decision(calendar, args.previous, now)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with Path(output).open("a", encoding="utf-8") as stream:
                stream.write(f"should_run={str(decision['should_run']).lower()}\n")
                stream.write(f"completed_session={decision['completed_session']}\n")
                stream.write(f"next_session={decision['next_session']}\n")
                stream.write(f"reason={decision['reason']}\n")
        print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
