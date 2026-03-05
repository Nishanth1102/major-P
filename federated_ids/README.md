# Federated Intrusion Detection with Flower + PyTorch

A research-ready Federated Learning (FL) simulation for binary network intrusion detection (Benign vs. Attack) using:
- **FL Framework**: [Flower (flwr) 1.8.0](https://flower.dev)
- **ML Framework**: PyTorch 2.2+
- **OS**: Windows (Python 3.12 virtual environment)

## Dataset
You can download it from Kaggle:
https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv/data
---

## Project Structure

```
federated_ids/
├── data/
│   └── raw/            ← Place CIC-IDS 2018 CSV files here
├── results/
│   └── round_metrics.csv   ← Auto-generated after training
├── model.py            ← MLP model definition + train/evaluate helpers
├── data_utils.py       ← Data loading, cleaning, Non-IID splitting
├── client.py           ← Flower FL client
├── server.py           ← Flower FL server (FedAvg)
└── requirements.txt    ← Python dependencies
```

---

## Step 1 — Download and Place the Dataset

1. Go to the [CIC-IDS 2018 dataset page](https://www.unb.ca/cic/datasets/ids-2018.html) and download one or more CSV files (e.g., `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`).
2. Place the downloaded `.csv` files inside `federated_ids/data/raw/`.

---

## Step 2 — Set Up Virtual Environment

Open **PowerShell** in `federated_ids/`:

```powershell
# Navigate to project
cd C:\Users\nisha\OneDrive\Desktop\major-P\federated_ids

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **Note**: If you get a script execution policy error, run:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## Step 3 — Configure the Experiment

Open `server.py` and adjust the top-level constants if needed:

| Constant | Default | Description |
|---|---|---|
| `NUM_ROUNDS` | `10` | Number of FL rounds |
| `NUM_CLIENTS` | `2` | Number of simulated clients |
| `SERVER_ADDR` | `0.0.0.0:8080` | Server listen address |

Open `client.py` and adjust similarly:

| Constant | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data/raw` | Path to CSV directory |
| `LOCAL_EPOCHS` | `3` | Local epochs per FL round |
| `BATCH_SIZE` | `256` | Mini-batch size |

---

## Step 4 — Run the Simulation (3 terminals)

Open **3 separate PowerShell terminals** inside `federated_ids/`.  
Activate the venv in each terminal first:

```powershell
.\venv\Scripts\Activate.ps1
```

### Terminal 1 — Start the Server
```powershell
python server.py
```
Wait until you see `Flower server running...` before starting clients.

### Terminal 2 — Start Client 0
```powershell
python client.py --client_id 0 --num_clients 2
```

### Terminal 3 — Start Client 1
```powershell
python client.py --client_id 1 --num_clients 2
```

Training will begin automatically once both clients connect.

---

## Step 5 — Scaling to More Clients

To simulate **N clients**, update `NUM_CLIENTS` in `server.py`, then launch N client terminals:

```powershell
# server.py: set NUM_CLIENTS = 4

# Then run 4 client terminals:
python client.py --client_id 0 --num_clients 4
python client.py --client_id 1 --num_clients 4
python client.py --client_id 2 --num_clients 4
python client.py --client_id 3 --num_clients 4
```

---

## Step 6 — Reading Results

After all rounds complete, `results/round_metrics.csv` contains:

```
round,train_loss,accuracy
1,0.1823,0.9412
2,0.1234,0.9561
...
```

You can load and plot this with:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/round_metrics.csv")
plt.plot(df["round"], df["accuracy"] * 100)
plt.xlabel("FL Round")
plt.ylabel("Accuracy (%)")
plt.title("Federated Learning — Global Accuracy per Round")
plt.grid(True)
plt.show()
```

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────┐
│                  Flower Server (FedAvg)              │
│  - Aggregates weights from all clients each round    │
│  - Logs accuracy to results/round_metrics.csv        │
└────────────────┬────────────────┬────────────────────┘
                 │                │
         ┌───────▼──────┐  ┌──────▼──────┐
         │   Client 0   │  │   Client 1  │
         │  Non-IID     │  │  Non-IID    │
         │  Partition 0 │  │  Partition 1│
         │  MLP (local) │  │  MLP (local)│
         └──────────────┘  └─────────────┘
```

### Non-IID Data Splitting

Each client receives a different distribution of traffic classes via **Dirichlet sampling** (α=0.5). This simulates realistic scenarios where different network sensors observe different traffic patterns — standard in FL research.

### Model

- 4-layer MLP: `Input → 256 → 128 → 64 → 1`
- BatchNorm1d + ReLU + Dropout(0.3) after each hidden layer
- Loss: `BCEWithLogitsLoss` | Optimizer: `Adam`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No CSV files found` | Check that CSVs are in `data/raw/` |
| `Connection refused` | Start `server.py` before clients |
| `CUDA not available` | Normal — code falls back to CPU automatically |
| `Script cannot be loaded` | Run `Set-ExecutionPolicy RemoteSigned` |
| Clients time out | Increase `NUM_ROUNDS` wait or check server is running |
