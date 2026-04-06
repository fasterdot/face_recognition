from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
PROFILE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"

FACE_CLASSIFIER = cv2.CascadeClassifier(str(CASCADE_PATH))
PROFILE_CLASSIFIER = cv2.CascadeClassifier(str(PROFILE_CASCADE_PATH))

HOG_DESCRIPTOR = cv2.HOGDescriptor(
    _winSize=(96, 96),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def preprocess_gray_face(gray_face: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_face)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def _clean_boxes(boxes: list[tuple[int, int, int, int]], min_area: int) -> list[tuple[int, int, int, int]]:
    cleaned: list[tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        if w * h < min_area:
            continue
        cleaned.append((int(x), int(y), int(w), int(h)))
    cleaned.sort(key=lambda b: b[2] * b[3], reverse=True)
    return cleaned


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / max(1, union))


def _nms(boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.35) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for candidate in boxes:
        if all(_iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def detect_faces(gray: np.ndarray, min_neighbors: int = 5) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape[:2]
    min_side = max(32, int(min(h, w) * 0.08))
    min_size = (min_side, min_side)
    min_area = min_side * min_side

    eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    variants = [gray, eq, clahe, blurred]

    all_boxes: list[tuple[int, int, int, int]] = []
    for img in variants:
        for neighbors in (min_neighbors, 4, 3):
            frontal = FACE_CLASSIFIER.detectMultiScale(
                img,
                scaleFactor=1.05,
                minNeighbors=neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            all_boxes.extend((int(x), int(y), int(wb), int(hb)) for (x, y, wb, hb) in frontal)

        profile = PROFILE_CLASSIFIER.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        all_boxes.extend((int(x), int(y), int(wb), int(hb)) for (x, y, wb, hb) in profile)

        flipped = cv2.flip(img, 1)
        profile_flipped = PROFILE_CLASSIFIER.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, wb, hb in profile_flipped:
            all_boxes.append((int(w - (x + wb)), int(y), int(wb), int(hb)))

    unique = list(dict.fromkeys(all_boxes))
    cleaned = _clean_boxes(unique, min_area=min_area)
    return _nms(cleaned, iou_threshold=0.35)


def extract_hog_features_from_face(gray_face: np.ndarray) -> np.ndarray:
    processed = preprocess_gray_face(gray_face)
    resized = cv2.resize(processed, (96, 96), interpolation=cv2.INTER_AREA)
    hog = HOG_DESCRIPTOR.compute(resized)
    if hog is None:
        return np.array([], dtype=np.float32)
    return hog.flatten().astype(np.float32)


def extract_hog_from_image(image_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0:
        return None, None

    features = extract_hog_features_from_face(crop)
    if features.size == 0:
        return None, None
    return features, (x, y, w, h)
