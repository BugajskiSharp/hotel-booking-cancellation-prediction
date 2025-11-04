"""
model_training.py
-----------------
PyTorch-based dense neural network training for booking cancellation prediction.
Overfit model and Regularized model definitions. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Return PyTorch DataLoaders for train and validation sets."""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class DenseNN(nn.Module):
    """Feedforward neural network."""
    def __init__(self, input_dim, hidden_layers=[128, 64], dropout_rate=0.3):
        super(DenseNN, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h
        layers.append(nn.Linear(last_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, patience=20, verbose_every=20):
    """Train neural network with early stopping."""
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter, best_state, best_epoch = 0, None, -1

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_train_loss, total_val_loss = 0.0, 0.0
        total_train_samples, total_val_samples = 0, 0

        # Training
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
            total_train_samples += X_batch.size(0)

        avg_train_loss = total_train_loss / total_train_samples
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
                total_val_samples += X_batch.size(0)
        avg_val_loss = total_val_loss / total_val_samples
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss, best_state, best_epoch = avg_val_loss, model.state_dict(), epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

        if epoch == 1 or epoch % verbose_every == 0:
            print(f"Epoch {epoch}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses, best_epoch

def build_overfit_model(input_dim=37):
    """Unregularized model to demonstrate overfitting."""
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    return model

def build_regularized_model(input_dim=37):
    """Regularized model with dropout and batch normalization."""
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    return model
