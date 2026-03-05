"""
client.py
---------
Flower FL client for Federated Intrusion Detection.

Usage (run in a separate terminal per client):
    python client.py --client_id 0 --num_clients 2
    python client.py --client_id 1 --num_clients 2

Each client:
  1. Loads its own Non-IID partition of the CIC-IDS 2018 data
  2. Connects to the Flower server (default: localhost:8080)
  3. Runs local training each round and sends updated weights to the server
  4. Evaluates using the aggregated global model after each round
"""

import argparse
import warnings
import torch
import flwr as fl
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

from model import MLP, get_parameters, set_parameters, train, evaluate
from data_utils import get_client_data

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR      = "./data/raw"       # Folder containing CIC-IDS 2018 CSVs
SERVER_ADDR   = "127.0.0.1:8080"  # Must match server.py
BATCH_SIZE    = 256
LOCAL_EPOCHS  = 3                  # Epochs per FL round (keep low for simulation)
LEARNING_RATE = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Flower Client Class
# ---------------------------------------------------------------------------

class IDSClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for Intrusion Detection.

    Holds local data, trains locally each round, and exchanges model
    parameters with the Flower server via gRPC.
    """

    def __init__(
        self,
        client_id: int,
        num_clients: int,
    ) -> None:
        super().__init__()

        self.client_id   = client_id
        self.num_clients = num_clients

        # -------------------------------------------------------------------
        # Load this client's Non-IID data partition
        # -------------------------------------------------------------------
        X_train, y_train, X_test, y_test = get_client_data(
            data_dir    = DATA_DIR,
            client_id   = client_id,
            num_clients = num_clients,
        )

        # Store input dimensionality to build the model correctly
        self.input_dim = X_train.shape[1]

        # Convert NumPy → PyTorch TensorDatasets
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        # -------------------------------------------------------------------
        # Instantiate local model and move to device
        # -------------------------------------------------------------------
        self.model = MLP(input_dim=self.input_dim).to(DEVICE)
        print(
            f"\n[Client {client_id}] Ready | "
            f"Device: {DEVICE} | "
            f"Train batches: {len(self.train_loader)} | "
            f"Test batches: {len(self.test_loader)}"
        )

    # ------------------------------------------------------------------
    # Flower interface: get_parameters
    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current model weights as a list of NumPy arrays.
        Called by the server to collect parameters for aggregation.
        """
        return get_parameters(self.model)

    # ------------------------------------------------------------------
    # Flower interface: set_parameters (internal helper)
    # ------------------------------------------------------------------
    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self.model, parameters)

    # ------------------------------------------------------------------
    # Flower interface: fit
    # ------------------------------------------------------------------
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Load global weights, run local training, return updated weights.

        Parameters: global model weights from server.
        Returns:    (updated_weights, num_train_samples, metrics_dict)
        """
        round_num = config.get("server_round", "?")
        print(f"\n[Client {self.client_id}] === Round {round_num} — FIT ===")

        # Load aggregated global weights
        self._set_parameters(parameters)

        # Local training
        avg_loss = train(
            model        = self.model,
            train_loader = self.train_loader,
            epochs       = LOCAL_EPOCHS,
            device       = DEVICE,
            lr           = LEARNING_RATE,
        )

        num_samples = sum(len(batch[0]) for batch in self.train_loader)
        print(f"[Client {self.client_id}] Train loss: {avg_loss:.4f}  Samples: {num_samples}")

        return get_parameters(self.model), num_samples, {"train_loss": float(avg_loss)}

    # ------------------------------------------------------------------
    # Flower interface: evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Load global weights, evaluate on local test set, report metrics.

        Returns: (loss, num_test_samples, metrics_dict)
        """
        print(f"\n[Client {self.client_id}] === EVALUATE ===")

        # Load latest global weights for evaluation
        self._set_parameters(parameters)

        loss, accuracy = evaluate(
            model       = self.model,
            test_loader = self.test_loader,
            device      = DEVICE,
        )

        num_samples = sum(len(batch[0]) for batch in self.test_loader)
        print(
            f"[Client {self.client_id}] "
            f"Loss: {loss:.4f}  Accuracy: {accuracy * 100:.2f}%  "
            f"Samples: {num_samples}"
        )

        return float(loss), num_samples, {"accuracy": float(accuracy)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Flower FL Client — CIC-IDS 2018")
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="Zero-based index of this client (e.g., 0, 1, 2, ...)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=2,
        help="Total number of clients in the federation (default: 2)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=SERVER_ADDR,
        help=f"Server address (default: {SERVER_ADDR})",
    )
    args = parser.parse_args()

    if args.client_id >= args.num_clients:
        raise ValueError(
            f"client_id ({args.client_id}) must be < num_clients ({args.num_clients})"
        )

    print(f"\n{'=' * 50}")
    print(f" FL Client {args.client_id} starting")
    print(f" Server: {args.server}  |  Total clients: {args.num_clients}")
    print(f"{'=' * 50}")

    # Start Flower client — connects to the server and waits for rounds
    fl.client.start_client(
        server_address = args.server,
        client         = IDSClient(
            client_id   = args.client_id,
            num_clients = args.num_clients,
        ).to_client(),
    )


if __name__ == "__main__":
    main()
