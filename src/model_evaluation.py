"""
model_evaluation.py
-------------------
Evaluate PyTorch models for hotel booking cancellation prediction.
Includes:
- Classification reports
- Confusion matrices
- ROC curve and AUC
- Calibration curve
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


# -----------------------------------------------------------
# Core Evaluation Functions
# -----------------------------------------------------------

def evaluate_classification(model, X_tensor, y_tensor, device, label_names=["Not Canceled", "Canceled"]):
    """
    Evaluate a trained model on a given dataset (validation or test).
    Prints a classification report and returns predicted labels.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        preds = torch.argmax(outputs, dim=1)

    y_true = y_tensor.cpu().numpy()
    y_pred = preds.cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    return y_true, y_pred


# -----------------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", cmap='Blues'):
    """
    Plot a confusion matrix with Seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=["Not Canceled", "Canceled"],
                yticklabels=["Not Canceled", "Canceled"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


# -----------------------------------------------------------
# ROC Curve & AUC
# -----------------------------------------------------------

def plot_roc_curve(model, X_tensor, y_tensor, device, model_name="Model"):
    """
    Plot ROC curve and display AUC value for a given model.
    """
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_tensor.to(device)), dim=1)[:, 1].cpu().numpy()
        actuals = y_tensor.cpu().numpy()

    fpr, tpr, _ = roc_curve(actuals, probs)
    auc_score = roc_auc_score(actuals, probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})", linewidth=2.2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc_score


# -----------------------------------------------------------
# Calibration Curve
# -----------------------------------------------------------

def plot_calibration_curve(models, X_tensor, y_tensor, device, model_names):
    """
    Plot calibration curves for one or more trained models.
    """
    plt.figure(figsize=(8, 6))

    for model_name, model in zip(model_names, models):
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor.to(device)).detach().cpu()
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        actuals = y_tensor.cpu().numpy()

        prob_true, prob_pred = calibration_curve(actuals, probs, n_bins=10)
        sns.lineplot(x=prob_pred, y=prob_true, marker='o', label=f"{model_name}")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Combined ROC for Multiple Models
# -----------------------------------------------------------

def plot_roc_comparison(models_dict, X_tensor, y_tensor, device, dataset_name="Validation"):
    """
    Plot ROC curves comparing multiple models on the same dataset.
    """
    plt.figure(figsize=(8, 6))
    for model_name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_tensor.to(device)), dim=1)[:, 1].cpu().numpy()
            actuals = y_tensor.cpu().numpy()

        fpr, tpr, _ = roc_curve(actuals, probs)
        auc_val = roc_auc_score(actuals, probs)
        sns.lineplot(x=fpr, y=tpr, label=f"{model_name} (AUC={auc_val:.3f})", linewidth=2.2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Comparison - {dataset_name} Set")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
