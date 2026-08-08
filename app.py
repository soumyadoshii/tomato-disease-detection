import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import streamlit as st
from torchvision import transforms

# Import pipeline functions from src/
from src.models import get_model
from src.explainability import generate_gradcam, generate_lime

# Page Configuration
st.set_page_config(
    page_title="Tomato Disease Detection & XAI",
    page_icon="🍅",
    layout="wide"
)

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew"
]

MODEL_MAPPING = {
    "ResNet50": "resnet50",
    "EfficientNet-B0": "efficientnet_b0",
    "DenseNet-121": "densenet121",
    "ConvNeXt-Tiny": "convnext_tiny",
    "Vision Transformer (ViT-B/16)": "vit_b_16"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Preprocessing Transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


@st.cache_resource
def load_trained_model(selected_arch_key: str):
    """Loads and caches selected model instance and weight checkpoint."""
    model_name = MODEL_MAPPING[selected_arch_key]
    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    
    weights_path = os.path.join("saved_models", f"best_{model_name}.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        st.sidebar.success(f"Loaded saved weights: {weights_path}")
    else:
        st.sidebar.warning(f"No checkpoint found at '{weights_path}'. Using initialized weights for demo.")
        
    model = model.to(device)
    model.eval()
    return model, model_name


# --- UI Header ---
st.title("🍅 Tomato Leaf Disease Detection System")
st.markdown(
    "Upload a tomato leaf image to identify potential diseases and generate "
    "**Explainable AI (Grad-CAM & LIME)** feature visualisations."
)

# Sidebar
st.sidebar.header("Model Configuration")
selected_model_name = st.sidebar.selectbox("Select Model Architecture", list(MODEL_MAPPING.keys()))
run_gradcam = st.sidebar.checkbox("Generate Grad-CAM Heatmap", value=True)
run_lime = st.sidebar.checkbox("Generate LIME Explanation", value=False)

# Main Workspace
col_input, col_results = st.columns([1, 1])

with col_input:
    st.subheader("1. Input Leaf Image")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Uploaded Image", use_container_width=True)

if uploaded_file is not None:
    # Model Loading
    model, arch_id = load_trained_model(selected_model_name)

    # Preprocess Image
    img_tensor = val_transform(raw_image).unsqueeze(0).to(device)
    
    # Unnormalized numpy array [224, 224, 3] for LIME and visualization
    resized_raw = raw_image.resize((224, 224))
    img_np_224 = np.array(resized_raw, dtype=np.float32) / 255.0

    # Prediction Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        
    top_prob, top_class_idx = torch.topk(probabilities, 3)
    predicted_label = CLASS_NAMES[top_class_idx[0].item()]
    confidence = top_prob[0].item() * 100

    with col_results:
        st.subheader("2. Prediction Analysis")
        if predicted_label.lower() == "healthy":
            st.success(f"**Diagnosis:** {predicted_label}")
        else:
            st.error(f"**Diagnosis:** {predicted_label}")
            
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        # Top-3 Class Probabilities Chart
        st.markdown("**Top 3 Probabilities:**")
        top3_data = {
            CLASS_NAMES[top_class_idx[i].item()]: float(top_prob[i].item() * 100)
            for i in range(3)
        }
        st.bar_chart(top3_data)

    st.markdown("---")
    st.subheader("3. Explainable AI (XAI) Visualizations")
    
    xai_col1, xai_col2 = st.columns(2)

    # Grad-CAM Visualization
    if run_gradcam:
        with xai_col1:
            st.markdown("#### Grad-CAM Heatmap")
            with st.spinner("Calculating Grad-CAM activations..."):
                cam_vis, _ = generate_gradcam(model, img_tensor, arch_id)
                st.image(cam_vis, caption="Grad-CAM Attention Region", use_container_width=True)

    # LIME Visualization
    if run_lime:
        with xai_col2:
            st.markdown("#### LIME Superpixel Segmentation")
            with st.spinner("Generating LIME superpixels (this may take a few seconds on CPU)..."):
                lime_vis, _ = generate_lime(model, img_np_224, device, num_samples=150)
                st.image(lime_vis, caption="LIME Positive Feature Superpixels", use_container_width=True)
else:
    st.info("Upload a tomato leaf image above to begin detection and visual analysis.")