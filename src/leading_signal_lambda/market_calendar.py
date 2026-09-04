from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import json

import pandas as pd


@dataclass(frozen=True)
class CompletedSession:
    session_date: date
    close_utc: pd.Timestamp
    end_exclusive: date


class NYSETradingCalendar:
    """NYSE（XNYS）の休日・夏時間・短縮取引を考慮する営業日判定器。

    exchange_calendarsを通常規則の基準とし、予測不能な臨時休場は
    JSONの例外日リストで除外できる。固定した「永久日付表」にはしない。
    """

    def __init__(self, calendar=None, exceptional_closures: str | Path | None = None) -> None:
        self._calendar = calendar or self._load_calendar()
        self._exceptional_closures = self._load_exceptions(exceptional_closures)

    @staticmethod
    def _load_calendar():
        try:
            import exchange_calendars as xcals
        except ImportError as error:
            raise RuntimeError("install the data extra: pip install '.[data]'") from error
        return xcals.get_calendar("XNYS")

    @staticmethod
    def _load_exceptions(path: str | Path | None) -> set[date]:
        if path is None:
            return set()
        source = Path(path)
        if not source.exists():
            return set()
        values = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("exceptional closure file must contain a JSON date list")
        return {date.fromisoformat(value) for value in values}

    def last_completed_session(self, now_utc: pd.Timestamp | None = None) -> CompletedSession:
        now = now_utc or pd.Timestamp.now(tz="UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        candidate = self._calendar.date_to_session(now.date(), direction="previous")
        while True:
            session_day = candidate.date()
            close = self._calendar.session_close(candidate)
            if close.tzinfo is None:
                close = close.tz_localize("UTC")
            else:
                close = close.tz_convert("UTC")
            if close <= now and session_day not in self._exceptional_closures:
                return CompletedSession(session_day, close, session_day + timedelta(days=1))
            candidate = self._calendar.previous_session(candidate)

    def is_session(self, value: str | date) -> bool:
        day = date.fromisoformat(value) if isinstance(value, str) else value
        if day in self._exceptional_closures:
            return False
        try:
            self._calendar.date_to_session(day, direction="none")
            return True
        except ValueError:
            return False
