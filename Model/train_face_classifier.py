from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Model.features import extract_hog_from_image


@dataclass
class TrainingResult:
    model: Pipeline
    labels: list[str]
    train_accuracy: float
    val_accuracy: float
    num_samples: int
    unknown_threshold: float


def _augment_image(image_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    outputs = [gray]
    outputs.append(cv2.convertScaleAbs(gray, alpha=1.15, beta=12))
    outputs.append(cv2.convertScaleAbs(gray, alpha=0.85, beta=-12))
    h, w = gray.shape
    center = (w // 2, h // 2)
    outputs.append(
        cv2.warpAffine(
            gray,
            cv2.getRotationMatrix2D(center, 8, 1.0),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    )
    outputs.append(
        cv2.warpAffine(
            gray,
            cv2.getRotationMatrix2D(center, -8, 1.0),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    )
    return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in outputs]


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for augmented in _augment_image(image):
                features, _ = extract_hog_from_image(augmented)
                if features is None:
                    continue
                X.append(features)
                y.append(label)
    if not X:
        raise ValueError("Aucun visage exploitable trouve dans le dataset.")
    return np.array(X), np.array(y)


def balance_dataset(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for label, count in counts.items():
        idx = np.where(y == label)[0]
        if count < max_count:
            extra = rng.choice(idx, size=max_count - count, replace=True)
            idx = np.concatenate([idx, extra])
        x_parts.append(X[idx])
        y_parts.append(y[idx])
    X_bal = np.vstack(x_parts)
    y_bal = np.concatenate(y_parts)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def train_model(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> TrainingResult:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))]
    )
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)
    best_prob = np.max(val_prob, axis=1)
    y_pred = model.predict(X_val)
    wrong_probs = best_prob[y_pred != y_val]
    unknown_threshold = float(np.percentile(wrong_probs, 70)) if wrong_probs.size > 0 else 0.45
    unknown_threshold = float(np.clip(unknown_threshold, 0.35, 0.7))
    return TrainingResult(
        model=model,
        labels=sorted(set(y.tolist())),
        train_accuracy=float(model.score(X_train, y_train)),
        val_accuracy=float(model.score(X_val, y_val)),
        num_samples=len(y),
        unknown_threshold=unknown_threshold,
    )


def save_artifacts(result: TrainingResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "face_classifier.pkl").open("wb") as f:
        pickle.dump({"model": result.model, "unknown_threshold": result.unknown_threshold}, f)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": result.labels,
                "num_samples": result.num_samples,
                "train_accuracy": result.train_accuracy,
                "val_accuracy": result.val_accuracy,
                "unknown_threshold": result.unknown_threshold,
            },
            f,
            indent=2,
        )


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Entraine un classifieur de visages multi-classes.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Model/Dependancies/Images"))
    parser.add_argument("--output-dir", type=Path, default=Path("Model/artifacts"))
    parser.add_argument("--neighbors", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.dataset_dir)
    X, y = balance_dataset(X, y)
    if len(set(y.tolist())) < 2:
        raise ValueError("Au moins 2 classes sont necessaires pour classifier.")
    result = train_model(X, y, n_neighbors=args.neighbors)
    save_artifacts(result, args.output_dir)
    print("Entrainement termine")
    print(f"Classes: {', '.join(result.labels)}")
    print(f"Nombre d'echantillons: {result.num_samples}")
    print(f"Accuracy train: {result.train_accuracy:.4f}")
    print(f"Accuracy validation: {result.val_accuracy:.4f}")
    print(f"Seuil inconnu: {result.unknown_threshold:.3f}")


if __name__ == "__main__":
    main()
