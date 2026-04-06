from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Model.inference import FaceClassifierService, draw_predictions

MODEL_PATH = ROOT_DIR / "Model" / "artifacts" / "face_classifier.pkl"

st.set_page_config(page_title="Classification des visages", layout="wide")
st.title("Classification des membres du groupe")

if not MODEL_PATH.exists():
    st.error("Modele introuvable. Lance: python Model/train_face_classifier.py")
    st.stop()

service = FaceClassifierService(MODEL_PATH)
source = st.radio("Source de l'image", ["Upload", "Camera"], horizontal=True)
uploaded_image = (
    st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png", "webp"])
    if source == "Upload"
    else st.camera_input("Capturer une image")
)

if uploaded_image is not None:
    image_rgb = np.array(Image.open(uploaded_image).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    predictions = service.predict_faces(image_bgr)
    rendered_rgb = cv2.cvtColor(draw_predictions(image_bgr, predictions), cv2.COLOR_BGR2RGB)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Image originale")
        st.image(image_rgb, use_container_width=True)
    with c2:
        st.subheader("Resultat du modele")
        st.image(rendered_rgb, use_container_width=True)

    st.markdown("### Personnes detectees")
    if not predictions:
        st.warning("Aucun visage detecte.")
    else:
        for i, pred in enumerate(predictions, start=1):
            st.write(f"{i}. {pred.label} - confiance: {pred.confidence:.2f}")
