# TP Machine Learning - Classification et deploiement

Ce projet implemente:
- entrainement d'un classifieur de visages
- detection + classification de plusieurs visages
- deploiement web (upload + camera)

## 1) Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 2) Entrainement du modele

Le dataset est attendu avec 1 dossier par classe:
`Model/Dependancies/Images/<nom_personne>/*.jpg`

Commande (Linux/macOS/Windows):

```bash
python Model/train_face_classifier.py
```

Artefacts generes:
- `Model/artifacts/face_classifier.pkl`
- `Model/artifacts/metrics.json`

## Matrice de confusion (evaluation)

Apres l'entrainement, pour generer une image PNG et un rapport precision/rappel/F1:

```bash
python Model/evaluate_confusion_matrix.py
```

Fichiers crees:
- `Doc/figures/confusion_matrix.png`
- `Doc/figures/classification_report.txt`

Le script utilise le **meme decoupage** train/validation que `train_face_classifier.py` (20 % validation, `random_state=42`) et le modele sauvegarde dans `Model/artifacts/face_classifier.pkl`.

## 3) Lancer la webapp

Linux / macOS:

```bash
streamlit run Webapp/app.py
```

Windows:

```powershell
python -m streamlit run Webapp/app.py
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
- Le dossier `.venv/` n'est pas versionne.
- Les artefacts de modele sont regenerables (`Model/artifacts/` ignore par git).
- Pour de meilleures performances, ajouter plus d'images variees par personne.
