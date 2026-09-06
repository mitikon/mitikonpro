from racing_lambda.historical_validation_inventory import (
    HISTORICAL_VALIDATION_TARGETS,
    next_external_validation_target,
)


def test_inventory_preserves_missing_data_instead_of_backfilling():
    targets = {row.race_id: row for row in HISTORICAL_VALIDATION_TARGETS}
    assert targets["2026-08-02-niigata-07"].frozen_marks == ("8", "6", "10", "13")
    assert targets["2026-07-19-hakodate-2yo"].frozen_marks == ()


def test_next_external_target_is_ibis_summer_dash():
    target = next_external_validation_target()
    assert target.race_id == "2026-08-02-niigata-07"
    assert target.official_top5 == ("6", "11", "4", "17", "16")
    assert target.feature_recovery == "recoverable_from_uploaded_racecards"
