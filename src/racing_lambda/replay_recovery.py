"""Recovery planning for historical layer-2 replay.

The goal is to determine exactly what still has to be recovered before the
current race-day adapter can run faithfully.  This module never substitutes
neutral values for missing historical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .historical_evidence_2026 import CHUKYO_2YO_PAST_RUNS


@dataclass(frozen=True)
class HorseRecoveryGap:
    horse_id: str
    missing: tuple[str, ...]


@dataclass(frozen=True)
class RaceRecoveryPlan:
    race_id: str
    ready_for_full_layer2: bool
    gaps: tuple[HorseRecoveryGap, ...]
    race_level_missing: tuple[str, ...]


def chukyo_2yo_recovery_plan() -> RaceRecoveryPlan:
    gaps: list[HorseRecoveryGap] = []
    for horse_id, runs in sorted(CHUKYO_2YO_PAST_RUNS.items(), key=lambda item: int(item[0])):
        missing: set[str] = set()
        for run in runs:
            # layer2_live_input.PastRun requires field size in order to normalize
            # finish and positions.  Project history preserved finishes but not
            # the field sizes for these races.
            missing.add("past_run_field_size")
            # Passing positions are required for a faithful pace/position score.
            missing.add("passing_positions")
        gaps.append(HorseRecoveryGap(horse_id=horse_id, missing=tuple(sorted(missing))))

    race_level_missing = (
        "monthly_course_distance_surface_stats",
        "monthly_going_weather_season_stats",
        "monthly_meeting_frequency_stats",
    )
    ready = not race_level_missing and all(not gap.missing for gap in gaps)
    return RaceRecoveryPlan(
        race_id="2026-08-30-chukyo-07",
        ready_for_full_layer2=ready,
        gaps=tuple(gaps),
        race_level_missing=race_level_missing,
    )
