from racing_lambda.chukyo_weakness_diagnostics import (
    chukyo_2yo_component_table,
    chukyo_2yo_weakness_findings,
)


def test_component_table_preserves_frozen_partial_ranking():
    rows = chukyo_2yo_component_table()
    assert tuple(row["horse_id"] for row in rows) == ("4", "6", "2", "5", "9", "7", "3", "8", "1")


def test_critical_structural_weaknesses_are_explicit():
    findings = chukyo_2yo_weakness_findings()
    critical = {finding.code for finding in findings if finding.severity == "critical"}
    assert critical == {
        "RECENT_FORM_SATURATION",
        "SURFACE_MISMATCH_OVERVALUATION",
        "FRONT_POSITION_BIAS",
        "MARKET_GAP_DOUBLE_COUNT",
    }


def test_diagnostics_do_not_reference_target_finish_order():
    text = " ".join(
        finding.explanation + " " + finding.proposed_direction
        for finding in chukyo_2yo_weakness_findings()
    )
    assert "正式結果" not in text
    assert "finishing_order" not in text
