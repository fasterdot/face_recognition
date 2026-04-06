from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from Model.features import detect_faces, extract_hog_features_from_face


@dataclass
class Prediction:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class FaceClassifierService:
    def __init__(self, model_path: Path, unknown_threshold: float | None = None) -> None:
        with model_path.open("rb") as f:
            payload = pickle.load(f)
        learned_threshold = float(payload.get("unknown_threshold", 0.45)) if isinstance(payload, dict) else 0.45
        self.model = self._unwrap_model(payload)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Artefact invalide, relancez `python Model/train_face_classifier.py`.")
        self.unknown_threshold = float(unknown_threshold if unknown_threshold is not None else learned_threshold)

    @staticmethod
    def _unwrap_model(payload: object) -> object:
        current = payload
        for _ in range(5):
            if not isinstance(current, dict):
                break
            if "model" in current:
                current = current["model"]
                continue
            if "pipeline" in current:
                current = current["pipeline"]
                continue
            if "classifier" in current:
                current = current["classifier"]
                continue
            break
        return current

    def predict_faces(self, image_bgr: np.ndarray) -> list[Prediction]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if not faces:
            return []

        features: list[np.ndarray] = []
        valid_faces: list[tuple[int, int, int, int]] = []
        for x, y, w, h in faces:
            crop = gray[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            feat = extract_hog_features_from_face(crop)
            if feat.size == 0:
                continue
            features.append(feat)
            valid_faces.append((x, y, w, h))

        if not features:
            return []

        X = np.array(features)
        probabilities = self.model.predict_proba(X)
        classes = self.model.classes_
        predictions: list[Prediction] = []
        for i, probs in enumerate(probabilities):
            idx = int(np.argmax(probs))
            label = str(classes[idx])
            confidence = float(probs[idx])
            if confidence < self.unknown_threshold:
                label = "Inconnu"
            predictions.append(Prediction(label=label, confidence=confidence, bbox=valid_faces[i]))
        return predictions


def draw_predictions(image_bgr: np.ndarray, predictions: list[Prediction]) -> np.ndarray:
    output = image_bgr.copy()
    for pred in predictions:
        x, y, w, h = pred.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{pred.label} ({pred.confidence:.2f})",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output
