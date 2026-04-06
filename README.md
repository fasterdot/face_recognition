# TP Machine Learning - Classification et deploiement

Ce projet implemente:
- entrainement d'un classifieur de visages
- detection + classification de plusieurs visages
- deploiement web (upload + camera)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset attendu

Structure:

`Model/Dependancies/Images/<nom_personne>/*.jpg`

Exemple:
- `Model/Dependancies/Images/david/...`
- `Model/Dependancies/Images/khan/...`

## Entrainement

```bash
python Model/train_face_classifier.py
```

Fichiers generes:
- `Model/artifacts/face_classifier.pkl`
- `Model/artifacts/metrics.json`

## Lancer la webapp

```bash
python -m streamlit run Webapp/app.py
```

## Notes

- Le dossier `.venv/` n'est pas versionne.
- Les artefacts de modele sont regenerables (`Model/artifacts/` ignore par git).
- Pour de meilleures performances, ajouter plus d'images variees par personne.
# TP Machine Learning - Classification et deploiement

Ce projet implemente la partie **entrainement + deploiement web** de l'enonce:
- classification des membres du groupe
- detection + classification de plusieurs visages sur une meme photo
- interface web avec upload et capture camera

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Entrainement du modele

Le dataset est attendu avec 1 dossier par classe:
`Model/Dependancies/Images/<nom_personne>/*.jpg`

Commande:

```bash
python Model/train_face_classifier.py
```

Artefacts generes:
- `Model/artifacts/face_classifier.pkl`
- `Model/artifacts/metrics.json`

## 3) Lancer la webapp

```bash
streamlit run Webapp/app.py
```

Fonctionnalites:
- Upload d'une image locale
- Capture directe depuis la camera
- Detection de chaque visage
- Classification par personne + score de confiance

## 4) Notes importantes

- L'enonce demande idealement 3 classes (trinome). Le code fonctionne avec 2+ classes.
- Si vous ajoutez une nouvelle personne, relancez l'entrainement.
- La detection de visage est faite avec OpenCV (cascade Haar), ce qui evite les problemes d'installation de `dlib/face-recognition`.
