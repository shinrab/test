import cv2
import numpy as np

from feature_extractor import extract_reid


def get_feature(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"사진을 읽을 수 없습니다: {path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_reid(image_rgb)


def compare(name, a, b):
    distance = float(np.linalg.norm(a - b))

    # 현재 judge.py에서 사용하는 방식
    current_similarity = max(0.0, min(1.0, 1.0 - distance))

    # 비교용으로 함께 확인하는 코사인 유사도
    cosine = float(
        np.dot(a, b)
        / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    )

    print(f"\n{name}")
    print(f"特徴距離:       {distance:.4f}")
    print(f"現在類似度:     {current_similarity:.4f}")
    print(f"コサイン類似度:   {cosine:.4f}")


me1 = get_feature("me1.jpg")
me1_again = get_feature("me1.jpg")
me2 = get_feature("me2.jpg")
other = get_feature("other.jpg")

compare("① 同じ写真: me1 ↔ me1", me1, me1_again)
compare("② 同じ人: me1 ↔ me2", me1, me2)
compare("③ 異なる人: me1 ↔ other", me1, other)