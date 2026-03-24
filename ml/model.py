"""
ml/model.py
===========
Defines the MLP (Multi-Layer Perceptron) neural network that predicts
the best beam pair from pilot measurements.

What this file does in plain English:
  - Input:  z — a vector of T pilot power measurements.
  - Output: a probability score for each of the 1024 possible beam pairs.
  - We pick the beam pair with the highest score as our prediction.

Architecture:
  z (input) -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> 1024 logits
"""

import torch
import torch.nn as nn


class BeamMLP(nn.Module):
    """
    Multi-Layer Perceptron for beam pair prediction.

    Takes a vector of pilot power measurements z and outputs
    logits (unnormalized scores) over all possible beam pairs.

    Args:
        input_dim:    Length of z (= T * Rx_beams_probed).
        n_classes:    Number of beam pairs (= F_size * W_size = 1024).
        hidden_dims:  List of hidden layer sizes, e.g. [256, 256, 128].
        dropout:      Dropout probability (regularization — prevents overfitting).
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 1024,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Build the layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))  # Fully connected layer
            layers.append(nn.ReLU())                    # Non-linearity
            layers.append(nn.Dropout(dropout))          # Randomly zero some neurons (prevents overfitting)
            prev_dim = h_dim

        # Final output layer: one score per beam pair class
        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

        # Count and print parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  BeamMLP: input={input_dim}, hidden={hidden_dims}, "
              f"output={n_classes}, params={n_params:,}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: (batch_size, input_dim) float tensor of pilot measurements.

        Returns:
            logits: (batch_size, n_classes) — raw scores, one per beam pair.
                    Apply softmax to get probabilities.
        """
        return self.net(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely beam pair index for each sample.

        Returns:
            (batch_size,) int tensor of predicted class indices.
        """
        with torch.no_grad():
            logits = self.forward(z)
            return torch.argmax(logits, dim=1)

    def predict_with_confidence(self, z: torch.Tensor) -> tuple:
        """
        Predict beam pair and return confidence (max softmax probability).

        Useful for the calibration add-on.

        Returns:
            predictions: (batch_size,) int tensor.
            confidence:  (batch_size,) float tensor in [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(z)
            probs = torch.softmax(logits, dim=1)
            confidence, predictions = probs.max(dim=1)
        return predictions, confidence


if __name__ == "__main__":
    print("Testing BeamMLP...")
    model = BeamMLP(input_dim=64, n_classes=1024, hidden_dims=[256, 256, 128], dropout=0.3)

    # Fake batch of 8 samples
    z_fake = torch.randn(8, 64)
    logits = model(z_fake)
    print(f"  Input shape:  {z_fake.shape}")
    print(f"  Output shape: {logits.shape}  (expected: [8, 1024])")

    preds = model.predict(z_fake)
    print(f"  Predictions:  {preds}  (each should be in [0, 1023])")
    print("Model OK.")