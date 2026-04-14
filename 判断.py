# -*- coding: utf-8 -*-
"""
同一人物判定アルゴリズム
- 顔特徴
- 服装色
- 荷物情報
- 身長
- 時間情報
を用いて、入場人物と退場人物が同一人物かどうかを判定する。

※ このコードにはテストデータ作成部分は含めていない
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import List, Optional, Tuple, Dict, Any


# =========================================================
# 1. データ構造
# =========================================================

@dataclass
class PersonFeatures:
    """
    1人分の特徴量を保持するクラス
    """
    person_id: int
    face_vector: List[float]              # 顔特徴ベクトル
    shirt_color: Tuple[int, int, int]     # 上衣RGB
    pants_color: Tuple[int, int, int]     # 下衣RGB
    bag: bool                             # 荷物の有無
    bag_size: float                       # 荷物サイズ（0.0なら無しでも可）
    height: float                         # 推定身長(cm)
    timestamp: datetime                   # 入場/退場時刻


# =========================================================
# 2. 基本ユーティリティ
# =========================================================

def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    値を[min_value, max_value]範囲に制限する
    """
    return max(min_value, min(value, max_value))


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    ユークリッド距離を計算
    """
    if len(vec1) != len(vec2):
        raise ValueError("face_vectorの長さが一致していません。")

    return sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    コサイン類似度を計算
    戻り値: -1.0 ～ 1.0
    """
    if len(vec1) != len(vec2):
        raise ValueError("face_vectorの長さが一致していません。")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# =========================================================
# 3. 特徴別の類似度計算
# =========================================================

def compare_face(face_vec1: List[float], face_vec2: List[float]) -> float:
    """
    顔特徴ベクトルの類似度を0.0～1.0で返す
    コサイン類似度を使用
    """
    sim = cosine_similarity(face_vec1, face_vec2)

    # cosine_similarityは -1 ~ 1 なので 0 ~ 1 に変換
    normalized = (sim + 1.0) / 2.0
    return clamp(normalized)


def color_distance_rgb(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    RGB色の距離を計算
    """
    return sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2 +
        (c1[2] - c2[2]) ** 2
    )


def compare_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    RGB色の類似度を0.0～1.0で返す
    """
    dist = color_distance_rgb(c1, c2)

    # RGB空間の最大距離
    max_dist = sqrt((255 ** 2) * 3)

    similarity = 1.0 - (dist / max_dist)
    return clamp(similarity)


def compare_bag(bag1: bool, bag2: bool, size1: float, size2: float) -> float:
    """
    荷物情報の類似度を0.0～1.0で返す
    """
    # どちらも荷物なし
    if not bag1 and not bag2:
        return 1.0

    # 一方だけ荷物あり
    if bag1 != bag2:
        return 0.0

    # 両方荷物あり → サイズ比較
    # サイズ差が小さいほど高得点
    diff = abs(size1 - size2)

    # サイズ差50を超えるとかなり違うとみなす例
    similarity = 1.0 - (diff / 50.0)
    return clamp(similarity)


def compare_height(h1: float, h2: float) -> float:
    """
    身長の類似度を0.0～1.0で返す
    """
    diff = abs(h1 - h2)

    # 20cm差でかなり低い評価になるよう設定
    similarity = 1.0 - (diff / 20.0)
    return clamp(similarity)


def compare_time(entry_time: datetime, exit_time: datetime, max_minutes: float = 180.0) -> float:
    """
    入場時刻と退場時刻の時間差から類似度を計算
    ※ 近すぎても遠すぎても状況次第だが、ここでは単純に
       max_minutes以内なら徐々に減点する形
    """
    diff_minutes = abs((exit_time - entry_time).total_seconds()) / 60.0

    similarity = 1.0 - (diff_minutes / max_minutes)
    return clamp(similarity)


# =========================================================
# 4. 総合スコア計算
# =========================================================

def calculate_match_score(entry_person: PersonFeatures, exit_person: PersonFeatures) -> Dict[str, float]:
    """
    各特徴の類似度と総合スコアを返す
    """

    face_score = compare_face(entry_person.face_vector, exit_person.face_vector)
    shirt_score = compare_color(entry_person.shirt_color, exit_person.shirt_color)
    pants_score = compare_color(entry_person.pants_color, exit_person.pants_color)
    bag_score = compare_bag(
        entry_person.bag,
        exit_person.bag,
        entry_person.bag_size,
        exit_person.bag_size
    )
    height_score = compare_height(entry_person.height, exit_person.height)
    time_score = compare_time(entry_person.timestamp, exit_person.timestamp)

    # 가중치는 나중에 실험으로 조정 가능
    total_score = (
        face_score * 0.45 +
        shirt_score * 0.15 +
        pants_score * 0.10 +
        bag_score * 0.10 +
        height_score * 0.10 +
        time_score * 0.10
    )

    total_score = clamp(total_score)

    return {
        "face_score": face_score,
        "shirt_score": shirt_score,
        "pants_score": pants_score,
        "bag_score": bag_score,
        "height_score": height_score,
        "time_score": time_score,
        "total_score": total_score
    }


# =========================================================
# 5. 同一人物判定
# =========================================================

def is_same_person(score: float, threshold: float = 0.75) -> bool:
    """
    総合スコアが閾値以上なら同一人物と判定
    """
    return score >= threshold


# =========================================================
# 6. 最適候補探索
# =========================================================

def find_best_match(
    entry_people: List[PersonFeatures],
    exit_person: PersonFeatures,
    threshold: float = 0.75
) -> Optional[Dict[str, Any]]:
    """
    退場人物に対して、最もスコアの高い入場人物を探す
    """

    if not entry_people:
        return None

    best_result: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for entry_person in entry_people:
        scores = calculate_match_score(entry_person, exit_person)
        total_score = scores["total_score"]

        if total_score > best_score:
            best_score = total_score
            best_result = {
                "matched_person_id": entry_person.person_id,
                "is_same_person": is_same_person(total_score, threshold),
                "scores": scores,
                "entry_time": entry_person.timestamp,
                "exit_time": exit_person.timestamp
            }

    return best_result


# =========================================================
# 7. 異常判定補助
# =========================================================

def detect_bag_change(entry_person: PersonFeatures, exit_person: PersonFeatures) -> bool:
    """
    荷物の有無やサイズが大きく変化したか判定
    """
    if entry_person.bag != exit_person.bag:
        return True

    if entry_person.bag and exit_person.bag:
        if abs(entry_person.bag_size - exit_person.bag_size) > 20.0:
            return True

    return False


def detect_long_stay(entry_person: PersonFeatures, current_time: datetime, limit_minutes: float = 120.0) -> bool:
    """
    長時間未退室かどうかを判定
    """
    stayed_minutes = (current_time - entry_person.timestamp).total_seconds() / 60.0
    return stayed_minutes > limit_minutes


# =========================================================
# 8. 結果整理
# =========================================================

def build_match_report(
    entry_person: PersonFeatures,
    exit_person: PersonFeatures,
    threshold: float = 0.75
) -> Dict[str, Any]:
    """
    2人を直接比較し、判定結果をまとめる
    """
    scores = calculate_match_score(entry_person, exit_person)
    same_person = is_same_person(scores["total_score"], threshold)
    bag_changed = detect_bag_change(entry_person, exit_person)

    report = {
        "entry_person_id": entry_person.person_id,
        "exit_person_id": exit_person.person_id,
        "same_person": same_person,
        "bag_changed": bag_changed,
        "scores": scores,
        "message": ""
    }

    if same_person and not bag_changed:
        report["message"] = "同一人物であり、荷物変化もありません。"
    elif same_person and bag_changed:
        report["message"] = "同一人物の可能性は高いですが、荷物変化が検出されました。"
    else:
        report["message"] = "同一人物ではない可能性が高いです。"

    return report


# =========================================================
# 9. チーム連携用インターフェース
# =========================================================

def process_exit_person(
    entry_people: List[PersonFeatures],
    exit_person: PersonFeatures,
    threshold: float = 0.75
) -> Dict[str, Any]:
    """
    システム全体で使う想定の処理関数
    退場人物が来たときに最も近い入場人物を探し、判定結果を返す
    """

    best_match = find_best_match(entry_people, exit_person, threshold)

    if best_match is None:
        return {
            "success": False,
            "message": "比較対象となる入場データがありません。"
        }

    return {
        "success": True,
        "matched_person_id": best_match["matched_person_id"],
        "is_same_person": best_match["is_same_person"],
        "scores": best_match["scores"],
        "entry_time": best_match["entry_time"],
        "exit_time": best_match["exit_time"],
    }


# =========================================================
# 10. 今後追加しやすい拡張ポイント
# =========================================================
# - 荷物タイプ比較（バックパック、スーツケース、手提げ袋など）
# - 服装色をRGBではなくHSVやヒストグラムで比較
# - 顔特徴の正規化処理
# - 閾値自動最適化
# - 誤判定ログ保存
# - DB連携