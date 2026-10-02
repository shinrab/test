import cv2
from database import init_db
from tracker import update_tracker
from exit_manager import process_exit




# =========================
# 蜃ｺ蜿｣ROI
# =========================
EXIT_ROI = (
    100, 300,
    550, 470
)
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
def in_exit_roi(bbox):

    cx, cy = center_of_bbox(bbox)

    x1, y1, x2, y2 = EXIT_ROI

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

        tracks = update_tracker(frame_rgb)

        cv2.rectangle(
            frame,
            (EXIT_ROI[0], EXIT_ROI[1]),
            (EXIT_ROI[2], EXIT_ROI[3]),
            (0, 0, 255),
            2
        )

        for t in tracks:

            tid = t["track_id"]
            bbox = t["bbox"]

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"T:{tid}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            if tid in processed_tracks:
                continue

            if not in_exit_roi(bbox):
                continue

            result = process_exit(
                frame_rgb,
                bbox
            )

            processed_tracks.add(tid)

            if result["status"] == "EXIT":

                print(
                    f"[EXIT] "
                    f"person={result['person_id']} "
                    f"score={result['score']:.3f}"
                )

            else:

                print(
                    "[UNKNOWN EXIT]"
                )

        cv2.imshow(
            "Exit Camera",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()