import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

from model import create_model
from gradcam import generate_gradcam
from preprocess import enhance_image

# =========================================================
#  PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="wide",
    page_icon="🔬"
)

st.title("🔬 Diabetic Retinopathy Detection (AI Powered)")
st.write("Upload a retinal fundus image to detect diabetic retinopathy and view Grad-CAM heatmap.")

# =========================================================
#  MODEL LOAD (LOCAL FILE)
# =========================================================
MODEL_PATH = "best_fold1.pth"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model("efficientnet_b3", num_classes=5, pretrained=False)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

CLASS_NAMES = {
    0: "0 - No DR",
    1: "1 - Mild",
    2: "2 - Moderate",
    3: "3 - Severe",
    4: "4 - Proliferative DR"
}

# =========================================================
#  PREPROCESSING
# =========================================================
def preprocess_for_model(img, size=224):
    img = np.array(img)
    img = enhance_image(img, size=size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

# =========================================================
#  UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Original Image")
        st.image(img, width=350)

    input_tensor = preprocess_for_model(img).to(device)

    with st.spinner("Analyzing..."):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probabilities))

        # GradCAM
        temp_path = "temp.png"
        img.save(temp_path)
        _, heatmap = generate_gradcam(temp_path, MODEL_PATH, "efficientnet_b3")

    with col2:
        st.subheader(" Grad-CAM Heatmap")
        st.image(heatmap, width=350)

    st.markdown("---")
    st.subheader(" Prediction Result")
    st.success(f"**Detected: {CLASS_NAMES[pred_class]}**")

    st.subheader(" Confidence")
    plt.figure(figsize=(6, 4))
    plt.bar(range(5), probabilities)
    plt.xticks(range(5), CLASS_NAMES.values(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)
