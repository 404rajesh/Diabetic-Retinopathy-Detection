# Diabetic Retinopathy Detection — EfficientNet-B3 + Grad-CAM Explainability

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-1.14-red?logo=pytorch) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-orange?logo=streamlit) ![License](https://img.shields.io/badge/License-MIT-green)

**Author:** Rajesh Kumar Jha — `@404rajesh`
**Project Type:** Final-year Minor Project

A complete end-to-end pipeline to detect **Diabetic Retinopathy (DR)** from retinal fundus images using **EfficientNet-B3**. Includes preprocessing, 5-fold cross-validation training, evaluation with **Quadratic Weighted Kappa (QWK)**, **Grad-CAM explainability**, and a **Streamlit demo app** for inference.

---

## 🚀 Live Demo

Try the app online:
**[Streamlit App (Live)](https://404rajesh-retinopathy.streamlit.app/)**

---

## 📊 Highlights / Results

* **Model:** EfficientNet-B3 (timm)
* **QWK Score:** 0.8649 (full evaluation)
* **Accuracy:** ~82.9% (on held-out evaluation set)
* **Deliverables:** preprocessing, training, evaluation, Grad-CAM, Streamlit app
* **Weights:** [HuggingFace link](https://huggingface.co/404rajesh/dr-detection-efficientnet-b3/resolve/main/best_fold1.pth)

---

## 🗂 Project Structure

```
Diabetic-Retinopathy-Detection/
│
├── src/                            
│   ├── model.py          # EfficientNet loader
│   ├── dataset.py        # PyTorch Dataset + transforms
│   ├── preprocess.py     # Preprocessing + helper functions
│   ├── gradcam.py        # Grad-CAM implementation
│   ├── utils.py          # Helper functions: save_checkpoint, metrics
│   ├── train.py          # Training + 5-fold CV
│   ├── evaluate.py       # Evaluation + confusion matrix
│   └── main.py           # Streamlit app (UI + inference)
│
├── requirements.txt
├── .streamlit/config.toml
├── README.md
└── best_fold1.pth        # Trained model weights
```

---

## ⚙️ Local Setup & Usage

1. **Create and activate a Python virtual environment:**

```bash
python3 -m venv env
source env/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run Streamlit app locally:**

```bash
streamlit run src/main.py
```

Open the displayed URL in your browser and upload retinal images to see predictions with Grad-CAM.

---

## 🏋️‍♂️ Training (Optional)

**Prepare preprocessed data** using `preprocess.py`:

```bash
python3 src/preprocess.py --data_dir path/to/raw_train_images \
                          --out_dir preprocessed_data/train_images \
                          --csv data/train.csv \
                          --size 224
```

**Train model (example):**

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

---

## 📈 Evaluation

Generate **confusion matrix** and **classification report**:

```bash
python3 src/evaluate.py \
  --csv data/train.csv \
  --preproc_dir preprocessed_data/train_images \
  --weights outputs/best_fold1.pth \
  --output_dir eval_outputs
```

Outputs include:

* `eval_outputs/confusion_matrix.png`
* `eval_outputs/predictions.csv`
* Terminal classification report + QWK score

---

## ☁️ Deployment (Streamlit Cloud)

1. Push repository to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io) and login with GitHub.
3. Create a new app → select your repo → set main file to `src/main.py` → Deploy.

---

## 📄 References

* [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection)
* PyTorch & timm library
* Streamlit documentation

---
