# Diabetic Retinopathy Detection — EfficientNet-B3 + Explainability (Grad-CAM)

**Author:** Rajesh Kumar Jha — `@404rajesh`

**Project type:** Final-year minor project

A complete end-to-end pipeline to detect Diabetic Retinopathy (DR) from retinal fundus images using a deep learning model (EfficientNet-B3). The project includes preprocessing, training (5-fold CV), evaluation with Quadratic Weighted Kappa (QWK), Grad-CAM explainability, and a Streamlit demo app for inference.

---

## Quick demo (deployed)

**Streamlit app (live):** `https://404rajesh-retinopathy.streamlit.app/`


## Highlights / Results

* **Model:** EfficientNet-B3 (timm)
* **Metric (QWK):** **0.8649** (full evaluation)
* **Accuracy:** ~82.9% (on held-out set used for evaluation)
* **Deliverables:** preprocessing, training scripts, evaluation scripts, Grad-CAM, Streamlit app, trained weights hosted on HuggingFace
* **Weights :** `https://huggingface.co/404rajesh/dr-detection-efficientnet-b3/resolve/main/best_fold1.pth`

## What this repo contains (full project)

```
Diabetic-Retinopathy-Detection/
│
├── src/                            # core python modules
│   ├── model.py                    # timm EfficientNet loader (create_model)
│   ├── dataset.py                  # PyTorch Dataset + transforms
│   ├── preprocess.py               # preprocessing for training + helper funcs
│   ├── gradcam.py                  # Grad-CAM implementation
│   ├── utils.py                    # helpers: save_checkpoint, metrics etc.
│   ├── train.py                    # training + 5-fold CV
│   ├── evaluate.py                 # full evaluation + confusion matrix
│   └── main.py                     # Streamlit app (UI + inference)
│
├── requirements.txt
├── .streamlit/config.toml
├── README.md
└──  best_fold1.pth                   (Final weight used in project)
```

## How to run (local)

1. Create and activate a Python venv:

```bash
python3 -m venv env
source env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit demo locally:

```bash
streamlit run src/main.py
```

## Training (if you want to retrain)

**Prepare data** (APTOS dataset), create preprocessed images using `src/preprocess.py`:

```bash
python3 src/preprocess.py --data_dir path/to/raw_train_images --out_dir preprocessed_data/train_images --csv data/train.csv --size 224
```

Train (example):

```bash
python3 src/train.py \
  --csv data/train.csv \
  --preproc_dir preprocessed_data/train_images \
  --ckpt_dir outputs \
  --img_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --model_name efficientnet_b3 \
  --folds 5 \
  --stage1 \
  --seed 42
```

## Evaluation

Generate confusion matrix + classification report:

```bash
python3 src/evaluate.py --csv data/train.csv --preproc_dir preprocessed_data/train_images --weights outputs/best_fold1.pth --output_dir eval_outputs
```

Outputs: `eval_outputs/confusion_matrix.png`, `eval_outputs/predictions.csv`, terminal classification report + QWK.


## Deployment (Streamlit Cloud — free)

1. Push your repo to GitHub (see below).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and login with GitHub.
3. Create a new app → select your repo → set the main file to `src/main.py` → Deploy.


## Files to upload to this GitHub repo (recommended checklist)

**Upload these files/folders to the repo root:**

* `requirements.txt`
* `.streamlit/config.toml`
* `README.md` (this file)
* `best_fold1.pth`
* `src/` folder with:

  * `model.py`
  * `main.py` (Streamlit app)
  * `preprocess.py`
  * `gradcam.py`
  * (optional) `train.py`, `dataset.py` — if you want to share training pipeline
* `.gitignore` (ensure `preprocessed_data/`, `outputs/`, `weights/`, large files excluded)

**Do NOT upload:**

* `preprocessed_data/` (large image folders)
* raw dataset files (APTOS)
* large checkpoint files >50MB (use HF or Drive)


## Git – Simple push commands

From your project root:

```bash
git init
git add .
git commit -m "Initial commit "
git branch -M main
git remote add origin (your repo link)
git push -u origin main
```
