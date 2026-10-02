from database import (
    get_inside_people,
    get_person,
    leave_store,
    save_log
)

from feature_extractor import extract_person_feature
from judge import judge_pair


MATCH_THRESHOLD = 0.75


# =========================
# 謇ｾ譛逶ｸ莨ｼ蝨ｨ蠎嶺ｺｺ蜻�
# =========================
def find_best_match(feature):

    inside_people = get_inside_people()

    best_pid = None
    best_score = 0

    for p in inside_people:

        pid = p["person_id"]

        person = get_person(pid)

        if person is None:
            continue

        try:

            result = judge_pair(
                person,
                feature
            )

            score = result["scores"]["total_score"]

            if score > best_score:

                best_score = score
                best_pid = pid

        except:
            continue

    return best_pid, best_score


# =========================
# 蜃ｺ蜿｣螟�炊
# =========================
def process_exit(frame_rgb, bbox):

    feature = extract_person_feature(
        frame_rgb,
        bbox
    )

    if feature is None:
        return None

    pid, score = find_best_match(feature)

    # =========================
    # 謇ｾ蛻ｰ蛹ｹ驟�
    # =========================
    if pid is not None and score >= MATCH_THRESHOLD:

        leave_store(pid)

        save_log(
            "EXIT",
            f"person {pid} leave store"
        )

        return {
            "status": "EXIT",
            "person_id": pid,
            "score": score
        }

    # =========================
    # 譛ｪ謇ｾ蛻ｰ
    # =========================
    save_log(
        "UNKNOWN_EXIT",
        "unknown person detected",
        feature
    )

    return {
        "status": "UNKNOWN",
        "person_id": None,
        "score": score
    }