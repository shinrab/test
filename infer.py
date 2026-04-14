import torch
import paddle
from paddle.inference import Config, create_predictor
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# 1. YOLOv5 加载
# -------------------------
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.classes = [0]  # 只检测人类 (class 0)

# -------------------------
# 2. Paddle 属性识别模型路径
# -------------------------
model_dir = r"C:\PaddleDetection\output_inference\PPLCNet_x1_0_person_attribute_945_infer"
pdmodel = f"{model_dir}/inference.pdmodel"
pdparams = f"{model_dir}/inference.pdiparams"

config = Config(pdmodel, pdparams)
config.disable_mkldnn()  # CPU 下禁用 MKL-DNN
predictor = create_predictor(config)

# -------------------------
# 3. 属性识别预处理函数
# -------------------------
def preprocess_image(img, input_size=(256,192)):
    img = img.resize(input_size)
    img = np.array(img).astype("float32") / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# 4. 标签
# -------------------------
label_list = [
    'Hat','Glasses','ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid','UpperSplice',
    'LowerStripe','LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress','boots','HandBag',
    'ShoulderBag','Backpack','HoldObjectsInFront','AgeOver60','Age18-60','AgeLess18','Female',
    'Front','Side','Back'
]

# -------------------------
# 5. 测试图片路径
# -------------------------
img_path = r"C:\PaddleDetection\output_inference\PPLCNet_x1_0_person_attribute_945_infer\test_image.jpg"
orig_img = Image.open(img_path).convert("RGB")

# -------------------------
# 6. YOLO 检测人物
# -------------------------
results = yolo_model(np.array(orig_img))
detections = results.xyxy[0]  # tensor [x1, y1, x2, y2, conf, class]

# -------------------------
# 7. 在原图上画框 + 预测属性
# -------------------------
draw = ImageDraw.Draw(orig_img)
font = ImageFont.load_default()  # 默认字体，小一点可读

for det in detections:
    x1, y1, x2, y2 = map(int, det[:4])
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # 裁剪人物
    crop_img = orig_img.crop((x1, y1, x2, y2))
    input_array = preprocess_image(crop_img)

    # 推理属性
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(input_array)
    predictor.run()
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()[0]  # 1x26 array

    # 选择概率 > 0.5 的属性
    attrs = [label_list[i] for i, v in enumerate(result) if v > 0.5]
    text = ",".join(attrs)

    bbox = draw.textbbox((0, 0), text, font=font)  # 返回 (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

# 文字居中在框内
    text_x = x1 + (x2 - x1 - text_w) // 2
    text_y = y1 + (y2 - y1 - text_h) // 2

    draw.text((text_x, text_y), text, fill="red", font=font)

    print(f"人物框 [{x1},{y1},{x2},{y2}] 属性: {text}")

# -------------------------
# 8. 保存结果
# -------------------------
output_path = "result.jpg"
orig_img.save(output_path)
print(f"可视化结果已保存: {output_path}")