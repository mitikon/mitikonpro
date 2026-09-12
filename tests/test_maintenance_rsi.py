import json
from pathlib import Path

import pandas as pd
import pytest

from maintenance_rsi import ExternalDataGuard, MalwareScan, MalwareStatus, audit_repository, validate_market_frames


def test_current_repository_passes_maintenance_integrity_controls():
    root = Path(__file__).resolve().parents[1]
    report = audit_repository(root)
    assert report.status == "PASS", [finding.to_dict() for finding in report.findings]
    payload = report.to_dict()
    assert payload["autonomous_main_merge"] is False
    assert payload["trading_authority"] is False


def test_external_data_guard_accepts_valid_json_without_executing_it(tmp_path, monkeypatch):
    source = tmp_path / "market.json"
    source.write_text(json.dumps({"symbol": "SPY", "close": 100.0}), encoding="utf-8")
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    result = ExternalDataGuard().inspect(source, require_antivirus=True)
    assert result.accepted
    assert len(result.sha256) == 64


def test_external_data_guard_rejects_disguised_executable(tmp_path, monkeypatch):
    source = tmp_path / "market.csv"
    source.write_bytes(b"MZmalicious")
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    result = ExternalDataGuard().inspect(source)
    assert not result.accepted
    assert "executable content signature detected" in result.reasons


def test_external_data_guard_rejects_symlink_without_reading_target(tmp_path):
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "market.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        ExternalDataGuard().inspect(link)


def test_external_data_guard_fails_closed_when_antivirus_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "market.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        ExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.UNAVAILABLE, None, "not installed")),
    )
    result = ExternalDataGuard().inspect(source)
    assert not result.accepted
    assert "mandatory antivirus scanner is unavailable" in result.reasons


def test_external_data_guard_rejects_oversized_input_before_scanner(tmp_path, monkeypatch):
    source = tmp_path / "large.csv"
    source.write_bytes(b"123456")

    def scanner_must_not_run(path):
        raise AssertionError("oversized input must be rejected before scanning")

    monkeypatch.setattr(ExternalDataGuard, "scan_malware", staticmethod(scanner_must_not_run))
    result = ExternalDataGuard(max_bytes=5).inspect(source)
    assert not result.accepted
    assert result.sha256 == ""


def test_quarantine_uses_content_hash_and_removes_untrusted_source(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("{}", encoding="utf-8")
    destination = ExternalDataGuard.quarantine(source, tmp_path / "quarantine")
    assert destination.exists()
    assert destination.name.endswith(".json.quarantine")
    assert not source.exists()


def test_market_frame_poisoning_is_blocked():
    index = pd.to_datetime(["2026-09-10", "2026-09-09", "2026-09-09"])
    close = pd.DataFrame({"SPY": [100.0, -1.0, 101.0]}, index=index)
    volume = pd.DataFrame({"SPY": [10.0, -2.0, 12.0]}, index=index)
    reasons = validate_market_frames(close, volume, required_symbols=("SPY", "QQQ"))
    assert "market timestamps are not ascending" in reasons
    assert "duplicate market timestamps detected" in reasons
    assert any("QQQ" in reason for reason in reasons)
    assert "non-positive close detected" in reasons
    assert "negative volume detected" in reasons


def test_documented_negative_futures_price_can_be_allow_listed():
    index = pd.to_datetime(["2020-04-20"])
    close = pd.DataFrame({"SPY": [281.59], "OIL": [-37.63]}, index=index)
    volume = pd.DataFrame({"SPY": [100.0], "OIL": [100.0]}, index=index)
    reasons = validate_market_frames(
        close,
        volume,
        required_symbols=("SPY",),
        allow_non_positive_symbols=("OIL",),
    )
    assert reasons == ()
