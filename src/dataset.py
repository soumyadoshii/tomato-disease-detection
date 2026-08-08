import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard ImageNet Mean and Standard Deviation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size=224):
    """
    Returns data augmentation and preprocessing pipelines for training, 
    validation, and testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def create_dataloaders(data_dir="data", batch_size=32, num_workers=2, img_size=224):
    """
    Loads train, valid, and test sets using ImageFolder and returns DataLoaders 
    along with class labels.
    """
    train_tf, val_tf = get_transforms(img_size=img_size)

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_tf)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=val_tf)
    test_dataset = datasets.ImageFolder(root=test_path, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader, train_dataset.classes


# --- Independent Verification Script ---
if __name__ == "__main__":
    print("--- Verifying Dataset & DataLoader Setup ---")

    # Locate project root 'data' folder whether run from project root or inside src/
    if os.path.exists("data"):
        data_path = "data"
    elif os.path.exists("../data"):
        data_path = "../data"
    else:
        raise FileNotFoundError("Could not find the 'data/' directory.")

    print(f"Loading datasets from: {os.path.abspath(data_path)}")

    BATCH_SIZE = 16
    train_loader, valid_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_path,
        batch_size=BATCH_SIZE,
        num_workers=0  # Set to 0 for debugging execution
    )

    print(f"\nDiscovered {len(class_names)} Classes:")
    for idx, cls_name in enumerate(class_names):
        print(f"  [{idx}] {cls_name}")

    print(f"\nDataset Sample Counts:")
    print(f"  Train count : {len(train_loader.dataset)}")
    print(f"  Valid count : {len(valid_loader.dataset)}")
    print(f"  Test count  : {len(test_loader.dataset)}")

    # Extract a single batch to verify shapes and distribution
    images, labels = next(iter(train_loader))

    print("\nSingle Batch Inspection:")
    print(f"  Image Batch Shape : {images.shape}  (Batch, Channels, Height, Width)")
    print(f"  Label Batch Shape : {labels.shape}")
    print(f"  Pixel Value Range : Min={images.min().item():.3f}, Max={images.max().item():.3f}")

    label_counts = Counter(labels.tolist())
    print("\nBatch Label Distribution:")
    for label_idx, count in sorted(label_counts.items()):
        print(f"  Class {label_idx} ({class_names[label_idx]}): {count} sample(s)")

    print("\nDataset pipeline verified successfully.")