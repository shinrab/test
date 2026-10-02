import numpy as np
from database import get_inside_people, save_entry
from feature_extractor import extract_person_feature
from judge import judge_pair

# =========================
# 郛灘ｭ�
# =========================
track_cache = {}        # track_id -> person_id
track_last_seen = {}    # track_id -> frame_id

person_cache = {}       # pid -> feature

db_cache = []
db_loaded = False
max_db_id = 0

THRESHOLD = 0.65
MAX_AGE = 50

frame_counter = 0

# 櫨 譁ｰ蠅橸ｼ夐亟豁｢驥榊､榊�謨ｰ謐ｮ蠎�
written_person = set()


# =========================
# DB蜉�霓ｽ�亥宵荳谺｡��
# =========================
def load_db():
    global db_cache, db_loaded, max_db_id

    if not db_loaded:
        db_cache = get_inside_people()
        db_loaded = True

        if len(db_cache) > 0:
            max_db_id = max([p["db_id"] for p in db_cache])
            for p in db_cache:
                person_cache[p["db_id"]] = p["person"]
                written_person.add(p["db_id"])  # 櫨 蟾ｲ蟄伜惠逧�ｹ滓��ｮｰ
        else:
            max_db_id = 0


# =========================
# 蠢ｫ騾溷源驟�
# =========================
def fast_match(feature, cache_dict):

    best_id = None
    best_score = 0

    for pid, feat in list(cache_dict.items())[:20]:

        try:
            score = judge_pair(feat, feature)["scores"]["total_score"]
        except:
            continue

        if score > best_score:
            best_score = score
            best_id = pid

    return best_id, best_score


# =========================
# track螟�炊
# =========================
def process_track(frame_rgb, track):

    global max_db_id

    tid = track["track_id"]
    bbox = track["bbox"]

    # 蟾ｲ蟄伜惠逶ｴ謗･霑泌屓
    if tid in track_cache:
        track_last_seen[tid] = frame_counter
        return track_cache[tid]

    # 謠仙叙迚ｹ蠕�
    feature = extract_person_feature(frame_rgb, bbox)

    if feature is None:
        return None

    load_db()

    # 蜈亥源驟榊�蟄�
    pid, score = fast_match(feature, person_cache)

    if pid is not None and score >= THRESHOLD:
        track_cache[tid] = pid
        track_last_seen[tid] = frame_counter
        return pid

    # 蜀榊源驟好B
    pid, score = fast_match(feature, {p["db_id"]: p["person"] for p in db_cache})

    # 譁ｰ莠ｺ
    if pid is None or score < THRESHOLD:
        max_db_id += 1
        pid = max_db_id

        # 櫨 蜿ｪ蜀吩ｸ谺｡謨ｰ謐ｮ蠎�
        if pid not in written_person:
            save_entry({"id": pid, "person": feature})
            written_person.add(pid)

        db_cache.append({"db_id": pid, "person": feature})

    track_cache[tid] = pid
    track_last_seen[tid] = frame_counter
    person_cache[pid] = feature

    return pid


# =========================
# 荳ｻ蜈･蜿｣
# =========================
def update_persons(frame_rgb, tracks):

    global frame_counter
    frame_counter += 1

    results = []

    # 貂�炊霑�悄track
    expired = []

    for tid, last in track_last_seen.items():
        if frame_counter - last > MAX_AGE:
            expired.append(tid)

    for tid in expired:
        track_cache.pop(tid, None)
        track_last_seen.pop(tid, None)

    # 螟�炊tracks
    for t in tracks:

        pid = process_track(frame_rgb, t)

        if pid is None:
            continue

        results.append({
            "track_id": t["track_id"],
            "person_id": pid,
            "bbox": t["bbox"]
        })

    return results