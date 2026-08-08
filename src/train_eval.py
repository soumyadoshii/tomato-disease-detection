import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from dataset import create_dataloaders
from models import get_model


def calculate_metrics(y_true, y_pred, y_probs, num_classes=11):
    """
    Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC (One-vs-Rest).
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics across all classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ROC-AUC computation (multiclass, One-vs-Rest)
    try:
        # Convert integer targets to one-hot encoding for multiclass ROC-AUC calculation
        y_true_onehot = np.eye(num_classes)[y_true]
        roc_auc = roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"Warning: ROC-AUC computation failed ({e}). Defaulting to 0.0.")
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one epoch of model training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, num_classes=11):
    """
    Evaluates the model on validation or test set and returns loss + comprehensive metrics.
    """
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    eval_loss = running_loss / total
    metrics = calculate_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_probs=np.array(all_probs),
        num_classes=num_classes
    )
    metrics["loss"] = eval_loss

    return metrics


def train_model(model_name, data_dir="data", epochs=10, batch_size=32, lr=1e-4, save_dir="saved_models"):
    """
    Full training pipeline for a given model architecture. Saves best weights based on validation accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=========================================")
    print(f" Starting Training: {model_name.upper()} on {device}")
    print(f"=========================================")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_{model_name}.pt")

    train_loader, valid_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )

    model = get_model(model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, valid_loader, criterion, device, num_classes=len(class_names))

        elapsed_time = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}%"
        )

        # Checkpoint best model checkpoint based on validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), save_path)
            print(f"   Saved new best model checkpoint to '{save_path}'")

    print(f"\nTraining Complete for {model_name}. Best Val Accuracy: {best_val_acc*100:.2f}%")

    # Evaluate Best Checkpoint on Test Set
    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes=len(class_names))

    print(f"--- Final Test Set Results [{model_name}] ---")
    print(f"  Accuracy  : {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1-Score  : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")

    return test_metrics


if __name__ == "__main__":
    # Test run with 1 epoch using ResNet50 to verify training and evaluation steps
    if os.path.exists("data"):
        data_path = "data"
    elif os.path.exists("../data"):
        data_path = "../data"
    else:
        raise FileNotFoundError("Could not find the 'data/' directory.")

    train_model(
        model_name="resnet50",
        data_dir=data_path,
        epochs=1,
        batch_size=32,
        lr=1e-4
    )