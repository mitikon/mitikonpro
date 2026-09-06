from racing_lambda.ibis_external_validation import ibis_external_validation


def test_ibis_external_validation_records_portability_failure_without_reweighting():
    result = ibis_external_validation()
    assert result.ranking[:5] == ("16", "1", "9", "6", "12")
    assert result.official_top5 == ("6", "11", "4", "17", "16")
    assert result.top5_hits == 2
    assert "course" in result.finding.lower()
