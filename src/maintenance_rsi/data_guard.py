"""Quarantine-first validation for external market data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

import numpy as np
import pandas as pd


class MalwareStatus(str, Enum):
    CLEAN = "CLEAN"
    INFECTED = "INFECTED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class MalwareScan:
    status: MalwareStatus
    scanner: str | None
    detail: str


@dataclass(frozen=True)
class DataInspection:
    path: str
    sha256: str
    size_bytes: int
    accepted: bool
    reasons: tuple[str, ...]
    malware: MalwareScan


class ExternalDataGuard:
    """Inspect external files before any parser or prediction code sees them."""

    ALLOWED_EXTENSIONS = frozenset({".csv", ".json"})
    EXECUTABLE_MAGIC = (b"MZ", b"\x7fELF", b"#!")

    def __init__(self, max_bytes: int = 50_000_000) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)

    @staticmethod
    def scan_malware(path: Path) -> MalwareScan:
        scanner = shutil.which("clamscan")
        if scanner is None:
            return MalwareScan(MalwareStatus.UNAVAILABLE, None, "no supported scanner is installed")
        completed = subprocess.run(
            [scanner, "--no-summary", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            shell=False,
        )
        detail = (completed.stdout + completed.stderr).strip()[-2000:]
        if completed.returncode == 0:
            return MalwareScan(MalwareStatus.CLEAN, "clamscan", detail)
        if completed.returncode == 1:
            return MalwareScan(MalwareStatus.INFECTED, "clamscan", detail)
        return MalwareScan(MalwareStatus.ERROR, "clamscan", detail)

    def inspect(self, path: str | Path, *, require_antivirus: bool = True) -> DataInspection:
        source = Path(path)
        reasons: list[str] = []
        if source.is_symlink():
            raise ValueError("symbolic links are prohibited")
        if not source.is_file():
            raise ValueError("external data path must be a regular file")
        size = source.stat().st_size
        if size > self.max_bytes:
            return DataInspection(
                path=str(source),
                sha256="",
                size_bytes=size,
                accepted=False,
                reasons=("file exceeds configured size limit",),
                malware=MalwareScan(
                    MalwareStatus.UNAVAILABLE,
                    None,
                    "oversized input was rejected before reading or scanning",
                ),
            )
        if source.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            reasons.append("file extension is not allow-listed")
        raw = source.read_bytes()
        if b"\x00" in raw:
            reasons.append("NUL byte detected")
        if raw.startswith(self.EXECUTABLE_MAGIC):
            reasons.append("executable content signature detected")
        digest = sha256(raw).hexdigest()
        if not reasons:
            try:
                if source.suffix.lower() == ".json":
                    json.loads(raw.decode("utf-8"))
                else:
                    pd.read_csv(source, nrows=5)
            except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
                reasons.append(f"content parsing failed: {type(exc).__name__}")
        malware = self.scan_malware(source)
        if malware.status in {MalwareStatus.INFECTED, MalwareStatus.ERROR}:
            reasons.append(f"malware scan status is {malware.status.value}")
        if require_antivirus and malware.status is MalwareStatus.UNAVAILABLE:
            reasons.append("mandatory antivirus scanner is unavailable")
        return DataInspection(
            path=str(source),
            sha256=digest,
            size_bytes=size,
            accepted=not reasons,
            reasons=tuple(reasons),
            malware=malware,
        )

    @staticmethod
    def quarantine(path: str | Path, quarantine_root: str | Path) -> Path:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("quarantine source must be a regular non-symlink file")
        destination_root = Path(quarantine_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        destination_root.chmod(0o700)
        digest = sha256(source.read_bytes()).hexdigest()
        destination = destination_root / f"{digest}{source.suffix.lower()}.quarantine"
        if destination.exists():
            raise FileExistsError(f"quarantine target already exists: {destination}")
        moved = Path(shutil.move(str(source), str(destination)))
        moved.chmod(0o600)
        return moved


def validate_market_frames(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    required_symbols: Sequence[str],
    allow_non_positive_symbols: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return blocking reasons for poisoned, stale, or malformed market frames."""
    reasons: list[str] = []
    if close.empty or volume.empty:
        reasons.append("market frames must be non-empty")
        return tuple(reasons)
    if not close.index.equals(volume.index):
        reasons.append("close and volume indexes differ")
    if not close.index.is_monotonic_increasing:
        reasons.append("market timestamps are not ascending")
    if close.index.has_duplicates:
        reasons.append("duplicate market timestamps detected")
    missing = sorted(set(required_symbols) - set(close.columns))
    if missing:
        reasons.append(f"required symbols are missing: {missing}")
    close_values = close.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    volume_values = volume.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isinf(close_values).any() or np.isinf(volume_values).any():
        reasons.append("infinite market value detected")
    protected_close = close.drop(columns=list(allow_non_positive_symbols), errors="ignore")
    protected_values = protected_close.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_close = protected_values[np.isfinite(protected_values)]
    if finite_close.size and (finite_close <= 0.0).any():
        reasons.append("non-positive close detected")
    finite_volume = volume_values[np.isfinite(volume_values)]
    if finite_volume.size and (finite_volume < 0.0).any():
        reasons.append("negative volume detected")
    return tuple(reasons)
