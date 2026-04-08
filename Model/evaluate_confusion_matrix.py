"""
Genere la matrice de confusion sur l'ensemble de VALIDATION (meme split que l'entrainement).

Utilisation:
  source .venv/bin/activate
  pip install matplotlib
  python Model/evaluate_confusion_matrix.py

Sortie par defaut:
  Doc/figures/confusion_matrix.png
  Doc/figures/classification_report.txt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Model.train_face_classifier import balance_dataset, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrice de confusion sur le jeu de validation.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Model/Dependancies/Images"))
    parser.add_argument("--model-path", type=Path, default=Path("Model/artifacts/face_classifier.pkl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Doc/figures"),
        help="Dossier pour PNG et rapport texte.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Modele introuvable: {args.model_path}. Lance d'abord: python Model/train_face_classifier.py")

    X, y = load_dataset(args.dataset_dir)
    X, y = balance_dataset(X, y)
    # Meme decoupage que dans train_model() (random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with args.model_path.open("rb") as f:
        payload = pickle.load(f)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    y_pred = model.predict(X_val)
    labels = sorted(np.unique(np.concatenate([y_val, y_pred])))

    cm = confusion_matrix(y_val, y_pred, labels=labels)
    report = classification_report(y_val, y_pred, labels=labels, digits=3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Vrai label",
        xlabel="Label predit",
        title="Matrice de confusion (ensemble de validation, 20%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    png_path = args.output_dir / "confusion_matrix.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure enregistree: {png_path}")
    print(f"Rapport enregistre: {report_path}")


if __name__ == "__main__":
    main()
