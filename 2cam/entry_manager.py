from database import (
    load_all_persons,
    save_person,
    enter_store,
    is_inside
)

from feature_extractor import extract_person_feature
from judge import judge_pair


MATCH_THRESHOLD = 0.75

person_cache = {}
loaded = False


# =========================
# 蜉�霓ｽ豌ｸ荵�ｺｫ莉ｽ蠎�
# =========================
def load_persons():

    global loaded

    if loaded:
        return

    persons = load_all_persons()

    for p in persons:

        person_cache[p["person_id"]] = p["person"]

    loaded = True


# =========================
# 謇ｾ譛逶ｸ莨ｼ逧�ｺｺ
# =========================
def find_best(feature):

    best_id = None
    best_score = 0

    for pid, person in person_cache.items():

        try:

            result = judge_pair(
                person,
                feature
            )

            score = result["scores"]["total_score"]
            print(
                f"[DETAIL] ID={pid} "
                f"ReID={result['scores']['reid_similarity']} "
                f"色={result['scores']['color_score']:.3f} "
                f"属性={result['scores']['attribute_score']:.3f} "
                f"総合点={score:.3f}"
            )

            if score > best_score:

                best_score = score
                best_id = pid

        except:
            continue

    return best_id, best_score


# =========================
# 蜈･蜿｣螟�炊
# =========================
def process_entry(frame_rgb, bbox):

    load_persons()

    feature = extract_person_feature(
        frame_rgb,
        bbox
    )

    if feature is None:
        return None

    pid, score = find_best(feature)

    print(f"[MATCH] 既存の人物={pid}, 類似度={score:.3f}")

    # =========================
    # 閠�｡ｾ螳｢
    # =========================
    if pid is not None and score >= MATCH_THRESHOLD:

        if not is_inside(pid):
            enter_store(pid)

        return pid

    # =========================
    # 譁ｰ鬘ｾ螳｢
    # =========================
    pid = save_person(feature)

    person_cache[pid] = feature

    enter_store(pid)

    return pid