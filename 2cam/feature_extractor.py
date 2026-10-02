import numpy as np
import cv2
import torch
import torchreid
from PIL import Image
from datetime import datetime
from paddle.inference import Config, create_predictor


# =========================
# ReID
# =========================
from pathlib import Path

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=False
)

weight_path = (
    Path(__file__).resolve().parent
    / "osnet_x1_0_market1501.pth"
)

torchreid.utils.load_pretrained_weights(
    reid_model,
    str(weight_path)
)

reid_model.eval()

print("[ReID] Model loaded successfully.")


# =========================
# Paddle attribute
# =========================
config = Config("inference.pdmodel", "inference.pdiparams")
config.disable_gpu()
config.disable_mkldnn()
predictor = create_predictor(config)

attribute_map = {
    0: "hat",
    1: "glasses",
    2: "short_sleeve",
    3: "long_sleeve",
    11: "trousers",
    15: "handbag",
    16: "shoulderbag",
    17: "backpack"
}


# =========================
# attribute
# =========================
def extract_attributes(img):

    img = img.resize((192, 256))
    img = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    inp = predictor.get_input_handle(predictor.get_input_names()[0])
    inp.copy_from_cpu(img.astype(np.float32))

    predictor.run()

    out = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()[0]

    return {v: bool(out[k] > 0.5) for k, v in attribute_map.items()}


# =========================
# color
# =========================
def extract_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return {
        "h": float(np.mean(hsv[:, :, 0])),
        "s": float(np.mean(hsv[:, :, 1])),
        "v": float(np.mean(hsv[:, :, 2]))
    }


# =========================
# reid
# =========================
def extract_reid(crop):

    img = cv2.resize(crop, (128, 256))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    tensor = torch.from_numpy(img).float()

    with torch.no_grad():
        feat = reid_model(tensor).numpy().flatten()

    feat = feat / (np.linalg.norm(feat) + 1e-6)

    return feat


# =========================
# main
# =========================
def extract_person_feature(frame_rgb, bbox):

    h, w = frame_rgb.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop_img = Image.fromarray(crop)

    return {
        "attributes": extract_attributes(crop_img),
        "color": extract_color(crop),
        "reid": extract_reid(crop).tolist(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }