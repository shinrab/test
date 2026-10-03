import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


# =========================
# YOLOv5
# =========================
yolo_model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",
    pretrained=True
)

yolo_model.classes = [0]
yolo_model.conf = 0.6

YOLO_SIZE = 640


# =========================
# DeepSORT
# =========================
tracker = DeepSort(
    max_age=15,
    n_init=1,
    max_iou_distance=0.5,
    nn_budget=50,
    embedder="mobilenet"
)


# =========================
# YOLO髯埼｢�
# =========================
frame_id = 0

DETECT_INTERVAL = 5

last_dets = []


# =========================
# detect
# =========================
def detect(frame):

    frame = np.ascontiguousarray(frame)

    results = yolo_model(
        frame,
        size=YOLO_SIZE
    )

    pred = results.xyxy[0].cpu().numpy()

    dets = []

    if pred is None or len(pred) == 0:
        return dets

    for x1, y1, x2, y2, conf, cls in pred:

        if conf < 0.7:
            continue

        dets.append([
            [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],
            float(conf),
            0
        ])

    return dets


# =========================
# update tracker
# =========================
def update_tracker(frame):

    global frame_id
    global last_dets

    frame = np.ascontiguousarray(frame)

    if frame_id % DETECT_INTERVAL == 0:

        last_dets = detect(frame)

    frame_id += 1

    tracks = tracker.update_tracks(
        last_dets,
        frame=frame
    )

    results = []

    for t in tracks:

        if not t.is_confirmed():
            continue

        l, top, r, bottom = t.to_ltrb()

        results.append({
            "track_id": int(t.track_id),
            "bbox": [
                int(l),
                int(top),
                int(r),
                int(bottom)
            ]
        })

    return results