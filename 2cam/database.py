import sqlite3
import json
from datetime import datetime

DB_PATH = "people_log_reid_test_v3.db"


# =========================
# 蛻晏ｧ句喧謨ｰ謐ｮ蠎�
# =========================
def init_db():

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 蠖灘燕蝨ｨ蠎嶺ｺｺ蜻�
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inside_people (
        person_id INTEGER PRIMARY KEY,
        entry_time TEXT NOT NULL
    )
    """)

    # 譌･蠢�
    cur.execute("""
    CREATE TABLE IF NOT EXISTS event_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        data TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # 豌ｸ荵�ｺｫ莉ｽ蠎�
    cur.execute("""
    CREATE TABLE IF NOT EXISTS person_master (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_data TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# =========================
# 菫晏ｭ俶ｰｸ荵�ｺｫ莉ｽ
# =========================
def save_person(person):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO person_master
        (person_data, created_at)
        VALUES (?, ?)
        """,
        (
            json.dumps(person, ensure_ascii=False),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    person_id = cur.lastrowid

    conn.commit()
    conn.close()

    return person_id


# =========================
# 闔ｷ蜿門�驛ｨ霄ｫ莉ｽ
# =========================
def load_all_persons():

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    SELECT person_id, person_data
    FROM person_master
    """)

    rows = cur.fetchall()

    conn.close()

    persons = []

    for row in rows:

        persons.append({
            "person_id": row[0],
            "person": json.loads(row[1])
        })

    return persons


# =========================
# 闔ｷ蜿門黒荳ｪ莠ｺ
# =========================
def get_person(person_id):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT person_data
        FROM person_master
        WHERE person_id=?
        """,
        (person_id,)
    )

    row = cur.fetchone()

    conn.close()

    if row is None:
        return None

    return json.loads(row[0])


# =========================
# 譖ｴ譁ｰ霄ｫ莉ｽ迚ｹ蠕�
# =========================
def update_person(person_id, person):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE person_master
        SET person_data=?
        WHERE person_id=?
        """,
        (
            json.dumps(person, ensure_ascii=False),
            person_id
        )
    )

    conn.commit()
    conn.close()


# =========================
# 譏ｯ蜷ｦ蝨ｨ蠎�
# =========================
def is_inside(person_id):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*)
        FROM inside_people
        WHERE person_id=?
        """,
        (person_id,)
    )

    count = cur.fetchone()[0]

    conn.close()

    return count > 0


# =========================
# 蜈･蠎�
# =========================
def enter_store(person_id):

    if is_inside(person_id):
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO inside_people
        (person_id, entry_time)
        VALUES (?, ?)
        """,
        (
            person_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()


# =========================
# 遖ｻ蠎�
# =========================
def leave_store(person_id):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        DELETE FROM inside_people
        WHERE person_id=?
        """,
        (person_id,)
    )

    conn.commit()
    conn.close()


# =========================
# 闔ｷ蜿門ｽ灘燕蝨ｨ蠎嶺ｺｺ蜻�
# =========================
def get_inside_people():

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT person_id, entry_time
        FROM inside_people
        """
    )

    rows = cur.fetchall()

    conn.close()

    people = []

    for row in rows:

        people.append({
            "person_id": row[0],
            "entry_time": row[1]
        })

    return people


# =========================
# 譌･蠢�
# =========================
def save_log(
    event_type,
    message,
    data=None
):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO event_logs
        (
            event_type,
            message,
            data,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            event_type,
            message,
            json.dumps(
                data,
                ensure_ascii=False
            ) if data else None,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()
    conn.close()