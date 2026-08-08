import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_target_layer(model, model_name: str):
    """
    Returns the target convolutional/feature layer required for Grad-CAM activation mapping.
    """
    model_name = model_name.lower()
    if "resnet" in model_name:
        return [model.layer4[-1]]
    elif "efficientnet" in model_name:
        return [model.features[-1]]
    elif "densenet" in model_name:
        return [model.features.denseblock4]
    elif "convnext" in model_name:
        return [model.features[-1]]
    elif "vit" in model_name:
        return [model.encoder.layers[-1].ln_1]
    else:
        raise ValueError(f"Unsupported model name for Grad-CAM targeting: {model_name}")


def generate_gradcam(model, input_tensor: torch.Tensor, model_name: str):
    """
    Generates a Grad-CAM heatmap visualization overlay on top of the input image tensor.
    
    Args:
        model: Loaded PyTorch neural network model
        input_tensor: Normalized image tensor of shape (1, 3, 224, 224)
        model_name: String identifier for the target model architecture
        
    Returns:
        visualization: RGB image array (224, 224, 3) with heatmap overlay in range [0, 255]
    """
    model.eval()
    target_layers = get_target_layer(model, model_name)

    # Reshape transform function required specifically for Vision Transformer architectures
    reshape_transform = None
    if "vit" in model_name.lower():
        def vit_reshape_transform(tensor):
            result = tensor[:, 1:, :].reshape(tensor.size(0), 14, 14, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        reshape_transform = vit_reshape_transform

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # Calculate activation heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # Un-normalize PyTorch tensor back to standard RGB numpy array [0.0, 1.0]
    img_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    # Overlay Grad-CAM heatmap onto base image
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def generate_lime(model, image_np: np.ndarray, device, num_samples: int = 250):
    """
    Generates LIME local feature explanations showing superpixels driving predictions.
    
    Args:
        model: PyTorch model
        image_np: Unnormalized RGB float numpy array (224, 224, 3) with values in range [0, 1]
        device: 'cuda' or 'cpu'
        num_samples: Number of random perturbations to evaluate
        
    Returns:
        lime_boundaried: Visual RGB image highlighting critical decision superpixels
        top_label: Predicted integer label ID
    """
    model.eval()
    explainer = lime_image.LimeImageExplainer()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    def predict_fn(images):
        """Batch prediction pipeline wrapper for LIME superpixel perturbations."""
        tensors = torch.stack([preprocess(img.astype(np.float32)) for img in images]).to(device)
        with torch.no_grad():
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    explanation = explainer.explain_instance(
        image_np.astype(np.float64),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_boundaried = mark_boundaries(temp, mask)
    return lime_boundaried, top_label


# --- Independent Verification Script ---
if __name__ == "__main__":
    print("--- Verifying Explainability Functions (Grad-CAM & LIME) ---")
    from models import get_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_img_np = np.random.rand(224, 224, 3).astype(np.float32)

    model_names = ["resnet50", "efficientnet_b0", "densenet121", "convnext_tiny", "vit_b_16"]

    for name in model_names:
        print(f"Testing Grad-CAM target layer setup for: {name}...")
        model = get_model(name, num_classes=11, pretrained=False).to(device)
        vis, cam_map = generate_gradcam(model, dummy_input, name)
        print(f"  ✓ Grad-CAM output shape: {vis.shape}")

    print("\nGrad-CAM target layers successfully resolved for all 5 models.")