from datetime import datetime, timezone

import pytest

from src.racing_lambda.jra_official_free_ingestion import ingest_snapshot


def test_pre_race_snapshot_accepts_public_observation():
    snap = ingest_snapshot(
        race_id="20260909_TEST_11R",
        phase="PRE_RACE",
        source_url="https://www.jra.go.jp/JRADB/example",
        payload={"going": "良", "weather": "晴", "win_odds": {"1": 4.2}},
        observed_at=datetime(2026, 9, 9, 1, 0, tzinfo=timezone.utc),
        code_commit_sha="abc123",
    )
    assert snap.phase == "PRE_RACE"
    assert snap.payload["going"] == "良"
    assert len(snap.payload_sha256) == 64


def test_result_leakage_is_rejected_from_pre_race():
    with pytest.raises(ValueError, match="result leakage"):
        ingest_snapshot(
            race_id="20260909_TEST_11R",
            phase="PRE_RACE",
            source_url="https://www.jra.go.jp/JRADB/example",
            payload={"finish_order": [1, 2, 3]},
        )


def test_result_is_allowed_only_in_result_phase():
    snap = ingest_snapshot(
        race_id="20260909_TEST_11R",
        phase="RESULT",
        source_url="https://www.jra.go.jp/JRADB/example",
        payload={"finish_order": [1, 2, 3], "payout": {"win": 420}},
    )
    assert snap.phase == "RESULT"


def test_member_services_are_rejected():
    with pytest.raises(ValueError):
        ingest_snapshot(
            race_id="x",
            phase="PRE_RACE",
            source_url="https://www.jra.go.jp/dento/soku.html",
            payload={},
        )
