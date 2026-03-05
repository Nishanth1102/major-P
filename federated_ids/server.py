"""
server.py
---------
Flower FL server using FedAvg for Federated Intrusion Detection.

Usage:
    python server.py

The server:
  1. Starts and listens for client connections on localhost:8080
  2. Runs NUM_ROUNDS of federated averaging
  3. Aggregates client weights using weighted averaging
  4. Logs per-round accuracy to console and saves to results/round_metrics.csv
"""

import os
import csv
import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_ADDR       = "0.0.0.0:8080"    # Listen on all interfaces
NUM_ROUNDS        = 10                 # Total FL rounds
NUM_CLIENTS       = 2                  # Expected number of clients

# Minimum clients needed to start a round
MIN_FIT_CLIENTS   = NUM_CLIENTS        # All clients must participate in training
MIN_EVAL_CLIENTS  = NUM_CLIENTS        # All clients must evaluate
MIN_AVAIL_CLIENTS = NUM_CLIENTS        # All clients must be available

# Results directory
RESULTS_DIR       = "./results"
METRICS_FILE      = os.path.join(RESULTS_DIR, "round_metrics.csv")


# ---------------------------------------------------------------------------
# Metrics Aggregation Functions
# ---------------------------------------------------------------------------

def weighted_average_accuracy(
    metrics: List[Tuple[int, Metrics]]
) -> Dict[str, float]:
    """
    Aggregate accuracy from all clients using weighted average.
    Each client's accuracy is weighted by its number of evaluation samples.

    This function is called by Flower after every evaluation round.
    """
    # metrics is a list of (num_examples, {"accuracy": value}) tuples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_acc   = sum(
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ) / total_examples

    print(f"\n  [Server] Aggregated Accuracy: {weighted_acc * 100:.2f}%")
    return {"accuracy": weighted_acc}


def weighted_average_loss(
    metrics: List[Tuple[int, Metrics]]
) -> Dict[str, float]:
    """
    Aggregate training loss from all clients using weighted average.
    Called by Flower after every training round's fit phase.
    """
    total_examples  = sum(num_examples for num_examples, _ in metrics)
    weighted_loss   = sum(
        num_examples * m.get("train_loss", 0.0)
        for num_examples, m in metrics
    ) / total_examples

    print(f"  [Server] Aggregated Train Loss: {weighted_loss:.4f}")
    return {"train_loss": weighted_loss}


# ---------------------------------------------------------------------------
# Round Logger: writes metrics to CSV after each round
# ---------------------------------------------------------------------------

class RoundLogger:
    """
    Logs per-round training and evaluation metrics to a CSV file.
    Used as a strategy wrapper to intercept Flower round results.
    """

    def __init__(self, base_strategy: fl.server.strategy.Strategy) -> None:
        self.strategy = base_strategy
        self._round   = 0

        # Create results directory and CSV with header
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "train_loss", "accuracy"])
        print(f"  [Logger] Results will be saved to: {METRICS_FILE}")

    def __getattr__(self, name):
        """Delegate all unknown attributes to the wrapped strategy."""
        return getattr(self.strategy, name)

    def aggregate_fit(self, server_round, results, failures):
        """Called after clients finish local training."""
        self._round = server_round
        print(f"\n{'=' * 50}")
        print(f" Round {server_round} / {NUM_ROUNDS} — Aggregating FIT")
        print(f"{'=' * 50}")
        return self.strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        """Called after clients finish evaluation; logs accuracy to CSV."""
        print(f"\n Round {server_round} — Aggregating EVALUATE")
        output = self.strategy.aggregate_evaluate(server_round, results, failures)

        # Extract aggregated metrics
        if output is not None:
            loss_agg, metrics_agg = output
            accuracy = metrics_agg.get("accuracy", 0.0) if metrics_agg else 0.0
            train_loss = 0.0  # placeholder (fit metrics not easily accessible here)

            # Append row to CSV
            with open(METRICS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([server_round, f"{train_loss:.4f}", f"{accuracy:.4f}"])

            print(
                f"  [Logger] Round {server_round} saved: "
                f"accuracy={accuracy * 100:.2f}%"
            )

        return output

    def configure_fit(self, server_round, parameters, client_manager):
        """Pass server_round to clients via config so they can log it."""
        config    = {"server_round": server_round}
        fit_ins   = self.strategy.configure_fit(server_round, parameters, client_manager)

        # Inject round number into each client's config
        updated = []
        for client, fit_instruction in fit_ins:
            from flwr.common import FitIns
            new_fit = FitIns(fit_instruction.parameters, {**fit_instruction.config, **config})
            updated.append((client, new_fit))
        return updated

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)

    def initialize_parameters(self, client_manager):
        return self.strategy.initialize_parameters(client_manager)

    def evaluate(self, server_round, parameters):
        return self.strategy.evaluate(server_round, parameters)


# ---------------------------------------------------------------------------
# Build Strategy
# ---------------------------------------------------------------------------

def build_strategy() -> fl.server.strategy.Strategy:
    """
    Configure FedAvg strategy with metric aggregation functions.
    Wrap it with RoundLogger to save metrics to CSV.
    """
    fedavg = fl.server.strategy.FedAvg(
        # Minimum clients to start a round
        min_fit_clients           = MIN_FIT_CLIENTS,
        min_evaluate_clients      = MIN_EVAL_CLIENTS,
        min_available_clients     = MIN_AVAIL_CLIENTS,

        # Fraction of clients to sample (1.0 = all clients participate)
        fraction_fit              = 1.0,
        fraction_evaluate         = 1.0,

        # Custom aggregation functions for logging
        fit_metrics_aggregation_fn      = weighted_average_loss,
        evaluate_metrics_aggregation_fn = weighted_average_accuracy,
    )

    return RoundLogger(fedavg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'=' * 50}")
    print(f" Flower Federated Learning Server")
    print(f" Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}")
    print(f" Address: {SERVER_ADDR}")
    print(f"{'=' * 50}\n")

    strategy = build_strategy()

    # Start the server — blocks until all rounds complete
    fl.server.start_server(
        server_address = SERVER_ADDR,
        config         = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy       = strategy,
    )

    print(f"\n[Server] Training complete. Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
