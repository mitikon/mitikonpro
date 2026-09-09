from datetime import datetime, timezone

import pytest

from racing_lambda import (
    FULL_LEADING_PREDICTION_NAME,
    SIMPLE_LEADING_PREDICTION_NAME,
    FullLeadingPredictionLambda,
    SimpleLeadingPredictionLambda,
    SimpleLeadingSignalLambdaV02,
    build_jra_training_frame,
    ingest_snapshot,
    odds_snapshots_from_official,
)


def pre_race(race_id: str, minute: int, supports: list[tuple[str, float, float]]):
    return ingest_snapshot(
        race_id=race_id,
        phase="PRE_RACE",
        source_url="https://www.jra.go.jp/",
        observed_at=datetime(2026, 9, 9, 3, minute, tzinfo=timezone.utc),
        payload={
            "market_support": [
                {
                    "horse_id": horse_id,
                    "support": {"win": win, "place": place},
                }
                for horse_id, win, place in supports
            ]
        },
    )


def test_names_are_explicit_and_backward_compatible():
    assert FULL_LEADING_PREDICTION_NAME == "本格先行予測λ"
    assert SIMPLE_LEADING_PREDICTION_NAME == "簡易式先行予測λ"
    assert SimpleLeadingPredictionLambda is SimpleLeadingSignalLambdaV02
    assert FullLeadingPredictionLambda is not SimpleLeadingPredictionLambda


def test_public_jra_pre_race_snapshots_feed_full_model_features():
    snapshots = [
        pre_race("R1", 0, [("1", 0.20, 0.30), ("2", 0.10, 0.15), ("3", 0.05, 0.08)]),
        pre_race("R1", 5, [("1", 0.24, 0.34), ("2", 0.09, 0.16), ("3", 0.07, 0.10)]),
    ]
    rows = odds_snapshots_from_official(snapshots)
    assert len(rows) == 6
    frame = build_jra_training_frame([snapshots])
    assert frame.shape[0] == 3
    assert "win_change" in frame.columns
    assert "place_vs_win" in frame.columns


def test_result_snapshot_can_never_enter_full_leading_prediction():
    result = ingest_snapshot(
        race_id="R1",
        phase="RESULT",
        source_url="https://www.jra.go.jp/",
        observed_at=datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc),
        payload={
            "official_result": ["1", "2", "3"],
            "market_support": [
                {"horse_id": "1", "support": {"win": 0.30, "place": 0.40}}
            ],
        },
    )
    with pytest.raises(ValueError, match="RESULT snapshots cannot enter"):
        odds_snapshots_from_official([result])


def test_pre_race_ingestion_rejects_result_leakage_before_learning():
    with pytest.raises(ValueError, match="result leakage"):
        ingest_snapshot(
            race_id="R1",
            phase="PRE_RACE",
            source_url="https://www.jra.go.jp/",
            observed_at=datetime(2026, 9, 9, 3, 0, tzinfo=timezone.utc),
            payload={"finish_order": ["1", "2", "3"]},
        )
