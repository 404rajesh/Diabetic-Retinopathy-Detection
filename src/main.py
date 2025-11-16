import streamlit as st
import cv2
import numpy as np
import torch
from model import create_model
from gradcam import generate_gradcam
from preprocess import enhance_image
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import requests
import os

# =========================================================
#  PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="wide",
    page_icon="🔬"
)

st.markdown("""
    <style>
        body { background-color: #0E1117; }
        .main { background-color: #0E1117; }
        h1, h2, h3, h4, h5 { color: #fafafa; }
        .css-1aumxhk, .css-ffhzg2 { background-color: #161A23; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Diabetic Retinopathy Detection (AI Powered)")
st.write("Upload a retinal fundus image to detect diabetic retinopathy and view heatmap visualization.")

# =========================================================
#  MODEL DOWNLOAD & LOAD
# =========================================================
@st.cache_resource
def download_model():
    st.info("Downloading model from HuggingFace Hub...")
    model_path = hf_hub_download(
        repo_id="404rajesh/dr-detection-efficientnet-b3",
        filename="best_fold1.pth",
        repo_type="model"
    )
    st.success("Model downloaded successfully!")
    return model_path


@st.cache_resource
def load_model(model_path, model_name="efficientnet_b3"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=model_name, num_classes=5, pretrained=False)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


# Download and load model
MODEL_PATH = download_model()
model, device = load_model(MODEL_PATH)

CLASS_NAMES = {
    0: "0 - No DR",
    1: "1 - Mild",
    2: "2 - Moderate",
    3: "3 - Severe",
    4: "4 - Proliferative DR"
}

# =========================================================
#  IMAGE PREPROCESSING
# =========================================================
def preprocess_for_model(img, size=224):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = enhance_image(img, size=size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

# =========================================================
#  FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Original Image")
        st.image(img, width=350)

    input_tensor = preprocess_for_model(img)

    with st.spinner("Analyzing image with AI model..."):
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probabilities))

        # GradCAM
        temp_path = "/tmp/temp_gradcam.png"
        img.save(temp_path)
        pred_cam_class, heatmap = generate_gradcam(temp_path, MODEL_PATH, "efficientnet_b3")

    with col2:
        st.subheader(" Grad-CAM Heatmap")
        st.image(heatmap, width=350)

    st.markdown("---")
    st.subheader(" Prediction Result")
    st.success(f"**Detected: {CLASS_NAMES[pred_class]}**")

    st.subheader(" Prediction Confidence")
    plt.figure(figsize=(6,4))
    plt.bar(range(5), probabilities, color=["#4CAF50", "#FFC107", "#03A9F4", "#E91E63", "#9C27B0"])
    plt.xticks(range(5), CLASS_NAMES.values(), rotation=45, ha="right")
    plt.title("Model Confidence Score")
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown("---")
    st.info("Note: This result includes preprocessing + Grad-CAM + confidence visualization")
