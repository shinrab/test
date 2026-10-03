import json
import sys
import numpy as np
from datetime import datetime


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def vector_distance(v1, v2):
    if v1 is None or v2 is None:
        return None

    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)

    if a.shape != b.shape:
        return None

    return float(np.linalg.norm(a - b))


def similarity_from_distance(distance):
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - distance))


def bool_score(v1, v2):
    return 1.0 if v1 == v2 else 0.0


def color_score(color1, color2):
    if color1 is None or color2 is None:
        return 0.0

    h1, s1, v1 = color1["h"], color1["s"], color1["v"]
    h2, s2, v2 = color2["h"], color2["s"], color2["v"]

    dh = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 90.0
    ds = abs(s1 - s2) / 255.0
    dv = abs(v1 - v2) / 255.0

    diff = dh * 0.5 + ds * 0.25 + dv * 0.25
    return max(0.0, min(1.0, 1.0 - diff))


def attribute_score(attr1, attr2):
    keys = [
        "hat",
        "glasses",
        "short_sleeve",
        "long_sleeve",
        "trousers",
        "handbag",
        "shoulderbag",
        "backpack"
    ]

    scores = []

    for key in keys:
        if key in attr1 and key in attr2:
            scores.append(bool_score(attr1[key], attr2[key]))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def time_score(entry_time, exit_time):
    try:
        t1 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
        diff_seconds = abs((t2 - t1).total_seconds())

        return 1.0 - min(diff_seconds / 86400.0, 1.0)
    except Exception:
        return 0.0

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return None

    a = np.asarray(v1, dtype=np.float32).flatten()
    b = np.asarray(v2, dtype=np.float32).flatten()

    if a.shape != b.shape or a.size == 0:
        return None

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denominator < 1e-12:
        return None

    similarity = float(np.dot(a, b) / denominator)
    return float(np.clip(similarity, 0.0, 1.0))


def judge_pair(entry_person, exit_person):
    reid_dist = vector_distance(
        entry_person.get("reid"),
        exit_person.get("reid")
    )

    face_dist = vector_distance(
        entry_person.get("face"),
        exit_person.get("face")
    )

    reid_sim = cosine_similarity(
    entry_person.get("reid"),
    exit_person.get("reid")
    )
    face_sim = cosine_similarity(
        entry_person.get("face"),
        exit_person.get("face")
    )

    attr_score = attribute_score(
        entry_person.get("attributes", {}),
        exit_person.get("attributes", {})
    )

    col_score = color_score(
        entry_person.get("color"),
        exit_person.get("color")
    )

    t_score = time_score(
        entry_person.get("timestamp", ""),
        exit_person.get("timestamp", "")
    )

    # Re-ID縺悟叙繧後※縺�ｋ蝣ｴ蜷医�Re-ID繧呈怙蜆ｪ蜈�
    if reid_sim is not None:

        weighted_sum = (
            reid_sim * 0.75 +
            col_score * 0.07 +
            attr_score * 0.03
        )

        weight_sum = 0.75 + 0.07 + 0.03

        if face_sim is not None:
            weighted_sum += face_sim * 0.15
            weight_sum += 0.15

        total_score = weighted_sum / weight_sum

        is_same = total_score >= 0.75

    # Re-ID縺後↑縺��ｴ蜷医�鬘斐ｒ荳ｭ蠢�↓蛻､譁ｭ
    elif face_sim is not None:
        total_score = (
            face_sim * 0.75 +
            col_score * 0.15 +
            attr_score * 0.05 +
            t_score * 0.05
        )

        is_same = total_score >= 0.65

    # Re-ID繧る｡斐ｂ縺ｪ縺��ｴ蜷医�陬懷勧諠��ｱ縺�縺代〒蛻､譁ｭ
    else:
        total_score = (
            col_score * 0.60 +
            attr_score * 0.25 +
            t_score * 0.15
        )

        is_same = total_score >= 0.75

    return {
        "entry_id": entry_person.get("id"),
        "exit_id": exit_person.get("id"),
        "is_same_person": bool(is_same),
        "scores": {
            "reid_distance": reid_dist,
            "reid_similarity": reid_sim,
            "face_distance": face_dist,
            "face_similarity": face_sim,
            "attribute_score": attr_score,
            "color_score": col_score,
            "time_score": t_score,
            "total_score": total_score
        }
    }


def judge_all(entry_data, exit_data):
    comparisons = []

    for entry_person in entry_data:
        for exit_person in exit_data:
            comparisons.append(judge_pair(entry_person, exit_person))

    comparisons.sort(
        key=lambda x: x["scores"]["total_score"],
        reverse=True
    )

    best = comparisons[0] if comparisons else None

    return {
        "success": best is not None,
        "best_match": best,
        "all_comparisons": comparisons
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("菴ｿ縺�婿: python3 judge.py entry.json exit.json")
        sys.exit(1)

    entry_path = sys.argv[1]
    exit_path = sys.argv[2]

    entry_data = load_json(entry_path)
    exit_data = load_json(exit_path)

    result = judge_all(entry_data, exit_data)

    print("=== 蜷御ｸ莠ｺ迚ｩ蛻､螳夂ｵ先棡 ===")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    with open("judge_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)