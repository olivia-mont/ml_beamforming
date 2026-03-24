"""
ml/train.py
===========
Trains the BeamMLP model using cross-entropy loss (Equation 8 in spec).

What this file does in plain English:
  - Loads training and validation datasets.
  - Runs the training loop: feed data in, compute loss, update weights.
  - Saves the best model checkpoint (based on validation accuracy).
  - Uses early stopping: if validation loss doesn't improve for `patience`
    epochs, we stop training early to avoid wasting time.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml

from ml.model import BeamMLP
from channel.dataset import load_dataset


# ============================================================
# PyTorch Dataset wrapper
# ============================================================
class BeamDataset(Dataset):
    """
    Wraps our numpy arrays into a PyTorch Dataset.

    PyTorch's DataLoader needs this format to handle batching,
    shuffling, etc. automatically.
    """

    def __init__(self, z: np.ndarray, labels: np.ndarray):
        """
        Args:
            z:      (n_samples, n_probes) float array of measurements.
            labels: (n_samples,) int array of beam pair class indices.
        """
        # Normalize z to zero mean, unit std — helps neural net training
        self.z_mean = z.mean(axis=0, keepdims=True)
        self.z_std = z.std(axis=0, keepdims=True) + 1e-8  # avoid div by zero

        self.z = torch.tensor((z - self.z_mean) / self.z_std, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.z[idx], self.labels[idx]


def train_model(
    train_ds: BeamDataset,
    val_ds: BeamDataset,
    input_dim: int,
    n_classes: int,
    cfg: dict,
    save_path: str,
    device: str = None,
) -> dict:
    """
    Full training loop.

    Args:
        train_ds:  Training dataset.
        val_ds:    Validation dataset.
        input_dim: Length of z vector.
        n_classes: Number of beam pair classes (1024).
        cfg:       Config dict loaded from default.yaml.
        save_path: Where to save the best model checkpoint.
        device:    'cuda', 'mps', or 'cpu'. Auto-detected if None.

    Returns:
        history dict with train_loss, val_loss, val_acc per epoch.
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"  Using device: {device}")

    # Hyperparameters from config
    epochs = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['learning_rate']
    wd = cfg['training']['weight_decay']
    patience = cfg['training']['patience']
    hidden_dims = cfg['model']['hidden_dims']
    dropout = cfg['model']['dropout']

    # Model, loss, optimizer
    model = BeamMLP(input_dim, n_classes, hidden_dims, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Learning rate scheduler: reduce LR if val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):

        # --- Training phase ---
        model.train()
        train_losses = []
        for z_batch, label_batch in train_loader:
            z_batch = z_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(z_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation phase ---
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for z_batch, label_batch in val_loader:
                z_batch = z_batch.to(device)
                label_batch = label_batch.to(device)

                logits = model(z_batch)
                loss = criterion(logits, label_batch)
                val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label_batch).sum().item()
                val_total += len(label_batch)

        avg_val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(avg_val_loss)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.1f}%")

        # --- Save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'n_classes': n_classes,
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'z_mean': train_ds.z_mean,
                'z_std': train_ds.z_std,
                'epoch': epoch,
                'val_acc': val_acc,
            }, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {save_path}")
    return history


def load_trained_model(checkpoint_path: str, device: str = 'cpu') -> tuple:
    """
    Load a saved model checkpoint.

    Returns:
        model:  BeamMLP in eval mode.
        z_mean, z_std: normalization stats (needed to preprocess new inputs).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # z_mean has shape (1, input_dim) — always reliable way to get input_dim
    input_dim = int(ckpt['z_mean'].shape[1])

    model = BeamMLP(
        input_dim=input_dim,
        n_classes=ckpt['n_classes'],
        hidden_dims=ckpt['hidden_dims'],
        dropout=ckpt['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['z_mean'], ckpt['z_std']


if __name__ == "__main__":
    print("Run eval/run_all.py to train and evaluate.")