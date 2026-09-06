"""Inventory of historical races available for leakage-safe validation.

Only races with evidence that a prediction existed before the official result are
listed.  Completeness is explicit; missing pre-race marks/features are never
invented from the result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationTarget:
    race_id: str
    race_name: str
    date: str
    venue: str
    status: str
    frozen_marks: tuple[str, ...]
    official_top5: tuple[str, ...] | None
    feature_recovery: str
    note: str = ""


HISTORICAL_VALIDATION_TARGETS: tuple[ValidationTarget, ...] = (
    ValidationTarget(
        race_id="2026-08-30-chukyo-07",
        race_name="中京2歳ステークス G3",
        date="2026-08-30",
        venue="中京",
        status="performance_detail_validated",
        frozen_marks=("7", "3", "4", "9", "6"),
        official_top5=("9", "4", "7", "6", "3"),
        feature_recovery="high",
        note="過去走・通過順位・騎手・調教師・事前オッズを復元済み。月次統計は未復元。",
    ),
    ValidationTarget(
        race_id="2026-08-30-niigata-08",
        race_name="新潟記念 G3",
        date="2026-08-30",
        venue="新潟",
        status="registered_needs_numeric_runs",
        frozen_marks=("8", "10", "5", "11", "6"),
        official_top5=("5", "3", "8", "2", "9"),
        feature_recovery="medium",
        note="全馬の完全な過去走時計・上がり・通過順位が未復元。性能詳細λの外部検証には不足。",
    ),
    ValidationTarget(
        race_id="2026-08-02-niigata-07",
        race_name="アイビスサマーダッシュ G3",
        date="2026-08-02",
        venue="新潟",
        status="course_specialization_complete_base_score_recovery_needed",
        frozen_marks=("8", "6", "10", "13"),
        official_top5=("6", "11", "4", "17", "16"),
        feature_recovery="course_complete_base_continuous_score_missing",
        note=(
            "全17頭の発走前JRA芝1000m成績を復元し、コース固有適性λは完全カバレッジ。"
            "元の外部bridgeは連続スコアを保存せず順位だけ保存したため、総合再融合では順位を偽の距離へ変換せず、"
            "連続base scoreの再構築を先に行う。"
        ),
    ),
    ValidationTarget(
        race_id="2026-07-19-kokura-kinen",
        race_name="小倉記念 G3",
        date="2026-07-19",
        venue="小倉",
        status="next_recovery_target",
        frozen_marks=("14", "18", "1", "16", "6"),
        official_top5=("1", "17", "18", "6", "3"),
        feature_recovery="low_to_medium",
        note="事前上位5頭と正式上位5頭は履歴確認済み。性能詳細λ用の全馬数値復元が必要。",
    ),
    ValidationTarget(
        race_id="2026-07-19-hakodate-2yo",
        race_name="函館2歳ステークス G3",
        date="2026-07-19",
        venue="函館",
        status="result_confirmed_prediction_recovery_needed",
        frozen_marks=(),
        official_top5=None,
        feature_recovery="low",
        note="結果CSV履歴はあるが、現在の構造化保存では完全な事前順位が未復元。",
    ),
)


def next_external_validation_target() -> ValidationTarget:
    """Return the best next race for external reproduction testing."""
    for target in HISTORICAL_VALIDATION_TARGETS:
        if target.status == "next_recovery_target":
            return target
    raise RuntimeError("no next validation target is registered")
