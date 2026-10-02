import cv2
from database import init_db
from tracker import update_tracker
from entry_manager import process_entry


# =========================
# 蜈･蜿｣ROI
# =========================
ENTRY_ROI = (
    100, 300,
    550, 470
)

# 蟾ｲ隗ｦ蜿奏rack
processed_tracks = set()


# =========================
# bbox荳ｭ蠢�せ
# =========================
def center_of_bbox(bbox):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return cx, cy


# =========================
# ROI蛻､螳�
# =========================
def in_entry_roi(bbox):

    cx, cy = center_of_bbox(bbox)

    x1, y1, x2, y2 = ENTRY_ROI

    return (
        x1 <= cx <= x2
        and
        y1 <= cy <= y2
    )


# =========================
# main
# =========================
def run():

    init_db()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        global ENTRY_ROI
        ENTRY_ROI = (0, 0, frame.shape[1] - 1, frame.shape[0] - 1)

        tracks = update_tracker(frame_rgb)

        # ROI譏ｾ遉ｺ
        cv2.rectangle(
            frame,
            (ENTRY_ROI[0], ENTRY_ROI[1]),
            (ENTRY_ROI[2], ENTRY_ROI[3]),
            (0, 255, 0),
            2
        )

        for t in tracks:

            tid = t["track_id"]
            bbox = t["bbox"]

            x1, y1, x2, y2 = bbox

            # 譏ｾ遉ｺ譯�
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"T:{tid}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # 蟾ｲ螟�炊霑�
            if tid in processed_tracks:
                continue

            # 霑帛�ROI
            if in_entry_roi(bbox):

                pid = process_entry(
                    frame_rgb,
                    bbox
                )

                processed_tracks.add(tid)

                print(
                    f"[ENTRY] "
                    f"track={tid} "
                    f"person={pid}"
                )

        cv2.imshow(
            "Entry Camera",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()