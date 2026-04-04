"""
data_utils.py
-------------
Data preprocessing and Non-IID partitioning for CIC-IDS 2018.

Pipeline:
  1. Load all CSV files from a directory
  2. Strip whitespace from column names
  3. Encode labels: "Benign" -> 0, everything else -> 1
  4. Drop irrelevant / non-numeric columns
  5. Replace infinite values and drop NaN rows
  6. Scale features with StandardScaler
  7. Split into train/test
  8. Partition across clients using Dirichlet Non-IID sampling (alpha=0.5)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are not useful as features (identifiers, timestamps, label)
_COLS_TO_DROP = [
    "Label",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
]

# Dirichlet concentration parameter — lower = more heterogeneous
_DIRICHLET_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Step 1: Loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Read all CSV files from `data_dir` and concatenate them into one DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to folder containing CIC-IDS 2018 CSV files.

    Returns
    -------
    pd.DataFrame
        Combined raw dataset.
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Please place CIC-IDS 2018 CSV files there."
        )

    dfs = []
    for fname in csv_files:
        fpath = os.path.join(data_dir, fname)
        print(f"  Loading: {fname}")
        df = pd.read_csv(fpath, low_memory=False)
        # Strip leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows loaded: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# Step 2: Cleaning & Encoding
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean and encode the raw CIC-IDS 2018 DataFrame.

    Returns
    -------
    X : np.ndarray  shape (N, num_features)
    y : np.ndarray  shape (N,)   binary labels: 0=Benign, 1=Attack
    """
    # --- Label encoding ---
    # "Benign" → 0; any attack category → 1
    if "Label" not in df.columns:
        raise ValueError("Dataset is missing a 'Label' column.")

    y = (df["Label"].str.strip() != "Benign").astype(int).values

    # --- Drop non-feature columns ---
    cols_to_drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Cast to numeric, coercing any string artifacts (like repeated headers) to NaN ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Keep only numeric columns ---
    df = df.select_dtypes(include=[np.number])

    # --- Replace ±inf with NaN, then drop ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_mask = ~df.isnull().any(axis=1)

    X = df[valid_mask].values.astype(np.float32)
    y = y[valid_mask]

    print(f"  Features: {X.shape[1]}   Samples after cleaning: {X.shape[0]:,}")
    print(f"  Benign: {(y == 0).sum():,}   Attack: {(y == 1).sum():,}")
    return X, y


# ---------------------------------------------------------------------------
# Step 3: Normalization
# ---------------------------------------------------------------------------

def normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data and transform both splits.
    Scaler is fit ONLY on X_train to prevent data leakage.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


# ---------------------------------------------------------------------------
# Step 4: Non-IID Dirichlet Partitioning
# ---------------------------------------------------------------------------

def dirichlet_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = _DIRICHLET_ALPHA,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition (X, y) into `num_clients` Non-IID splits using Dirichlet sampling.

    Each client gets a different proportion of classes, simulating the real-world
    scenario where network sensors see different traffic profiles.

    Parameters
    ----------
    X          : feature array
    y          : binary label array
    num_clients: number of FL clients
    alpha      : Dirichlet concentration (lower = more heterogeneous)
    seed       : reproducibility seed

    Returns
    -------
    List of (X_client, y_client) tuples, one per client.
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)  # [0, 1]

    # For each class, sample a Dirichlet distribution over clients
    # This gives each client a different mixture of classes
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx   = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        # Dirichlet proportions for this class across all clients
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # Convert proportions to actual sample counts
        splits = (proportions * len(cls_idx)).astype(int)
        # Fix rounding error so we use exactly all samples
        splits[-1] = len(cls_idx) - splits[:-1].sum()

        start = 0
        for cid, count in enumerate(splits):
            client_indices[cid].extend(cls_idx[start : start + count].tolist())
            start += count

    # Build per-client datasets
    client_datasets = []
    for cid, idx in enumerate(client_indices):
        idx_arr    = np.array(idx)
        rng.shuffle(idx_arr)
        X_c = X[idx_arr]
        y_c = y[idx_arr]
        benign_pct = (y_c == 0).mean() * 100
        print(
            f"  Client {cid}: {len(y_c):,} samples  "
            f"Benign: {benign_pct:.1f}%  Attack: {100 - benign_pct:.1f}%"
        )
        client_datasets.append((X_c, y_c))

    return client_datasets


# ---------------------------------------------------------------------------
# Public API: get data for a specific client
# ---------------------------------------------------------------------------

def get_client_data(
    data_dir: str,
    client_id: int,
    num_clients: int,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end data pipeline for a single FL client.

    Parameters
    ----------
    data_dir    : path to folder with CIC-IDS 2018 CSV files
    client_id   : index of this client (0-based)
    num_clients : total number of clients
    test_size   : fraction of each client's data to hold out for evaluation
    seed        : random seed for reproducibility

    Returns
    -------
    X_train, y_train, X_test, y_test — all as NumPy float32 arrays
    """
    print(f"\n[DataUtils] Loading data for client {client_id}/{num_clients - 1}")

    # Step 1: Load
    df = load_dataset(data_dir)

    # Step 2: Clean & encode
    X, y = preprocess(df)

    # Step 3: Dirichlet Non-IID split (returns full arrays per client)
    client_datasets = dirichlet_split(X, y, num_clients=num_clients, seed=seed)
    X_c, y_c = client_datasets[client_id]

    # Step 4: Local train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_c, y_c, test_size=test_size, random_state=seed, stratify=y_c
    )

    # Step 5: Normalize (fit only on train)
    X_train, X_test, _ = normalize(X_train, X_test)

    print(
        f"  Train: {len(y_train):,}   Test: {len(y_test):,}   "
        f"Features: {X_train.shape[1]}"
    )
    return X_train, y_train, X_test, y_test
