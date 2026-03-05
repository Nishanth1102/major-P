"""
model.py
--------
PyTorch MLP model for binary classification (Benign vs Attack).
Designed for tabular network traffic features from CIC-IDS 2018.

Architecture:
  Input -> FC(256) -> BN -> ReLU -> Dropout
        -> FC(128) -> BN -> ReLU -> Dropout
        -> FC(64)  -> BN -> ReLU -> Dropout
        -> FC(1)   [logits, no sigmoid — use BCEWithLogitsLoss]
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary intrusion detection.

    Parameters
    ----------
    input_dim : int
        Number of input features (determined after preprocessing).
    hidden_dims : list of int
        Sizes of hidden layers. Default: [256, 128, 64].
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
    ) -> None:
        super(MLP, self).__init__()

        layers = OrderedDict()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers[f"fc{i}"]      = nn.Linear(prev_dim, hidden_dim)
            layers[f"bn{i}"]      = nn.BatchNorm1d(hidden_dim)
            layers[f"relu{i}"]    = nn.ReLU()
            layers[f"dropout{i}"] = nn.Dropout(p=dropout)
            prev_dim = hidden_dim

        # Output layer — raw logits for BCEWithLogitsLoss
        layers["output"] = nn.Linear(prev_dim, 1)

        self.network = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # shape: (batch, 1)


# ---------------------------------------------------------------------------
# Flower compatibility helpers
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model weights as a list of NumPy arrays.
    Flower uses this format to transfer parameters between server and clients.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load aggregated global weights (from server) back into the local model.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Training & Evaluation helpers
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> float:
    """
    Train the model for a given number of local epochs.

    Returns
    -------
    float
        Average training loss over all epochs.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # (batch, 1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        total_loss += epoch_loss / len(train_loader)
        print(f"  [Local] Epoch {epoch + 1}/{epochs}  loss: {epoch_loss / len(train_loader):.4f}")

    return total_loss / epochs


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on locally held-out test data.

    Returns
    -------
    loss : float
    accuracy : float  (between 0 and 1)
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()

            # Convert logits → binary predictions
            preds   = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total_samples
    return avg_loss, accuracy
