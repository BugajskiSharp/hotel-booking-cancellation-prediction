"""
model_comparison.py
-------------------
Compare multiple PyTorch models (e.g., Overfitting vs Regularized) using:
- ROC curves across train/validation/test sets
- AUC bar charts
- Calibration plots
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve


# -----------------------------------------------------------
# Helper Function: Generate Prediction DataFrame
# -----------------------------------------------------------

def get_predictions_df(model, X_tensor, y_tensor, device):
    """
    Get predicted probabilities and actual labels in a DataFrame.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device)).detach().cpu()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # probability of class 1
    df = pd.DataFrame({'p_1': probs, 'actual': y_tensor.cpu().numpy()})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# -----------------------------------------------------------
# ROC Curve Comparison (for multiple datasets)
# -----------------------------------------------------------

def plot_roc_per_dataset(models_dict, X_tensors, y_tensors, device,
                         dataset_names=['Train', 'Validation', 'Test']):
    """
    Plot ROC curves comparing multiple models across datasets.
    Each dataset gets its own ROC plot.

    Args:
        models_dict: dict of model name -> model object
        X_tensors: list of tensors for [train, val, test]
        y_tensors: list of tensors for [train, val, test]
        device: torch device
        dataset_names: list of dataset names
    """
    for X_tensor, y_tensor, ds_name in zip(X_tensors, y_tensors, dataset_names):
        plt.figure(figsize=(8, 6))
        for model_name, model in models_dict.items():
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor.to(device)).detach().cpu()
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            actuals = y_tensor.cpu().numpy()

            fpr, tpr, _ = roc_curve(actuals, probs)
            auc_val = roc_auc_score(actuals, probs)
            sns.lineplot(x=fpr, y=tpr, label=f'{model_name} (AUC={auc_val:.3f})', linewidth=2.2)

        # Add reference diagonal
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier')
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title(f'ROC Curve Comparison - {ds_name} Set', fontsize=15)
        plt.legend(fontsize=11)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------
# AUC Bar Chart Comparison
# -----------------------------------------------------------

def plot_auc_bar_chart(auc_results):
    """
    Plot a grouped bar chart comparing AUC scores across datasets for multiple models.

    Args:
        auc_results: pandas DataFrame with columns:
                     ['Dataset', 'Overfitted Model', 'Regularized Model']
    """
    auc_melted = auc_results.melt(id_vars='Dataset', var_name='Model', value_name='AUC')
    plt.figure(figsize=(8, 6))
    sns.barplot(data=auc_melted, x='Dataset', y='AUC', hue='Model',
                palette=['skyblue', 'red'])
    plt.title('AUC Comparison Across Datasets', fontsize=14)
    plt.ylim(0.90, 1.00)
    plt.xlabel("Dataset")
    plt.ylabel("AUC")
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Calibration Curve Comparison
# -----------------------------------------------------------

def plot_calibration_comparison(p1_actual_overfit, p1_actual_reg):
    """
    Plot calibration curves comparing overfit and regularized models.

    Args:
        p1_actual_overfit: DataFrame with columns ['p_1', 'actual']
        p1_actual_reg: DataFrame with columns ['p_1', 'actual']
    """
    y_true_reg = p1_actual_reg['actual'].astype(int)
    y_prob_reg = p1_actual_reg['p_1']

    y_true_overfit = p1_actual_overfit['actual'].astype(int)
    y_prob_overfit = p1_actual_overfit['p_1']

    prob_true_reg, prob_pred_reg = calibration_curve(y_true_reg, y_prob_reg, n_bins=10)
    prob_true_overfit, prob_pred_overfit = calibration_curve(y_true_overfit, y_prob_overfit, n_bins=10)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=prob_pred_overfit, y=prob_true_overfit, marker='o', label='Overfitting Model', color='blue')
    sns.lineplot(x=prob_pred_reg, y=prob_true_reg, marker='o', label='Regularized Model', color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title('Calibration Curves Comparison', fontsize=15)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
