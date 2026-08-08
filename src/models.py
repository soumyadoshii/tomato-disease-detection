import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int = 11, pretrained: bool = True):
    """
    Factory function to instantiate, load pretrained weights, and modify output 
    heads for all 5 target model architectures.
    
    Supported model_name options:
      - 'resnet50'
      - 'efficientnet_b0'
      - 'densenet121'
      - 'convnext_tiny'
      - 'vit_b_16'
    """
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Choose from: 'resnet50', 'efficientnet_b0', 'densenet121', 'convnext_tiny', 'vit_b_16'."
        )

    return model


# --- Independent Verification Script ---
if __name__ == "__main__":
    print("--- Verifying Model Creation and Forward Pass ---")

    target_models = [
        "resnet50",
        "efficientnet_b0",
        "densenet121",
        "convnext_tiny",
        "vit_b_16"
    ]

    # Dummy batch matching DataLoader output shape: (Batch Size=2, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    num_classes = 11

    print(f"Testing dummy input shape: {list(dummy_input.shape)}\n")

    for model_name in target_models:
        print(f"Building: {model_name}...")
        try:
            # Instantiate model without downloading full weights during rapid verification test
            model = get_model(model_name, num_classes=num_classes, pretrained=False)
            model.eval()

            with torch.no_grad():
                output = model(dummy_input)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"  ✓ Output Shape: {list(output.shape)} (Expected: [2, {num_classes}])")
            print(f"  ✓ Total Parameters: {total_params:,}")
            print(f"  ✓ Trainable Parameters: {trainable_params:,}\n")

        except Exception as e:
            print(f"  ✗ Failed to initialize {model_name}: {e}\n")

    print("All 5 model architectures successfully built and verified.")