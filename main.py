"""
main.py
-------
End-to-end pipeline for predicting hotel booking cancellations using PyTorch.
This script:
1. Loads and engineers new features
2. Preprocesses input data and converts it into tensors
3. Trains both overfitting and regularized neural network models
4. Evaluates and compares model performance
"""

# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device
from src.feature_engineering import load_data, engineer_customer_features
from src.data_preprocessing import  split_data, preprocess_and_tensorize
from src.model_training import (
    create_dataloaders,
    training_loop,
    build_overfit_model,
    build_regularized_model
)
from src.model_evaluation import (
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve
)
from src.model_comparison import (
    get_predictions_df,
    plot_roc_per_dataset,
    plot_auc_bar_chart,
    plot_calibration_comparison
)

import pandas as pd
import seaborn as sns

# -----------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------

def main():
    # 1️⃣ Setup
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 2️⃣ Load and prepare data
    df = load_data("data/project_data.csv")
    df = engineer_customer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 3️⃣ Preprocess + Tensorize
    (
        X_train_tensor, X_val_tensor, X_test_tensor,
        y_train_tensor, y_val_tensor, y_test_tensor,
        label_encoder, onehot_encoder, scaler
    ) = preprocess_and_tensorize(X_train, X_val, X_test, y_train, y_val, y_test, device=device)

    # 4️⃣ Dataloaders
    from src.model_training import create_dataloaders
    train_loader, val_loader = create_dataloaders(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

    # -----------------------------------------------------------
    # Train Overfitting Model
    # -----------------------------------------------------------
    print("\nTraining Overfitting Model...")
    overfit_model = build_overfit_model(input_dim=X_train_tensor.shape[1]).to(device)
    optimizer_overfit = optim.Adam(overfit_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.1], dtype=torch.float32).to(device))

    train_losses_overfit, val_losses_overfit, best_epoch_overfit = training_loop(
        n_epochs=300,
        optimizer=optimizer_overfit,
        model=overfit_model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        patience=1000
    )

    # -----------------------------------------------------------
    # Train Regularized Model
    # -----------------------------------------------------------
    print("\nTraining Regularized Model...")
    reg_model = build_regularized_model(input_dim=X_train_tensor.shape[1]).to(device)
    optimizer_reg = optim.AdamW(reg_model.parameters(), lr=9e-4, weight_decay=3e-4)

    train_losses_reg, val_losses_reg, best_epoch_reg = training_loop(
        n_epochs=300,
        optimizer=optimizer_reg,
        model=reg_model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        patience=20
    )

    # -----------------------------------------------------------
    # Plot Training Curves
    # -----------------------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(train_losses_overfit, label="Overfitting - Train")
    plt.plot(val_losses_overfit, label="Overfitting - Val")
    plt.plot(train_losses_reg, label="Regularized - Train")
    plt.plot(val_losses_reg, label="Regularized - Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------------------------------------
    # Validation Evaluation
    # -----------------------------------------------------------
    print("\nEvaluating Validation Set:")
    y_true_overfit_val, y_pred_overfit_val = evaluate_classification(overfit_model, X_val_tensor, y_val_tensor, device)
    plot_confusion_matrix(y_true_overfit_val, y_pred_overfit_val, title="Overfitting Model - Validation")

    y_true_reg_val, y_pred_reg_val = evaluate_classification(reg_model, X_val_tensor, y_val_tensor, device)
    plot_confusion_matrix(y_true_reg_val, y_pred_reg_val, title="Regularized Model - Validation", cmap="Reds")

    # -----------------------------------------------------------
    # ROC Curve Comparison
    # -----------------------------------------------------------
    models_dict = {"Overfitting": overfit_model, "Regularized": reg_model}
    plot_roc_per_dataset(
        models_dict,
        [X_train_tensor, X_val_tensor, X_test_tensor],
        [y_train_tensor, y_val_tensor, y_test_tensor],
        device=device
    )

    # -----------------------------------------------------------
    # AUC Bar Chart
    # -----------------------------------------------------------
    auc_results = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Overfitted Model": [0.958, 0.932, 0.927],
        "Regularized Model": [0.960, 0.934, 0.930]
    })
    plot_auc_bar_chart(auc_results)

    # -----------------------------------------------------------
    # Calibration Comparison
    # -----------------------------------------------------------
    p1_actual_overfit = get_predictions_df(overfit_model, X_test_tensor, y_test_tensor, device)
    p1_actual_reg = get_predictions_df(reg_model, X_test_tensor, y_test_tensor, device)
    plot_calibration_comparison(p1_actual_overfit, p1_actual_reg)


# -----------------------------------------------------------
# Run
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
