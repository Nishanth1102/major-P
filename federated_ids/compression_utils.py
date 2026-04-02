"""
compression_utils.py
--------------------
Reusable gradient compression helpers for Federated Learning.
"""

import numpy as np
from typing import List, Tuple

def encode_sparse(
    params_delta: List[np.ndarray],
    k_ratio: float = 0.10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply Top-K sparsification and FP16 quantization to a list of delta arrays.
    Returns the sparse payload for transmission and the effective dense delta
    used for local error tracking.
    """
    if not (0.0 < k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be in (0, 1]. Got {k_ratio}")

    sparse_payload = []
    dense_deltas = []

    for layer in params_delta:
        shape_arr = np.array(layer.shape, dtype=np.int32)
        flat = layer.flatten()
        k = max(1, int(len(flat) * k_ratio))

        # Find the k-th largest absolute value as the threshold
        if k < len(flat):
            threshold = np.partition(np.abs(flat), -k)[-k]
        else:
            threshold = 0.0

        # Extract top-K indices and values
        indices = np.where(np.abs(flat) >= threshold)[0].astype(np.int32)
        
        # simulated fp16 precision
        values = flat[indices].astype(np.float16).astype(np.float32)

        sparse_payload.extend([shape_arr, indices, values])

        # Reconstruct the effective dense layer for error tracking
        dense_layer_flat = np.zeros_like(flat)
        dense_layer_flat[indices] = values
        dense_deltas.append(dense_layer_flat.reshape(layer.shape))

    return sparse_payload, dense_deltas


def decode_sparse(
    sparse_payload: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Reconstructs the dense delta list from the sparse network payload.
    """
    reconstructed_deltas = []
    for i in range(0, len(sparse_payload), 3):
        shape = tuple(sparse_payload[i])
        indices = sparse_payload[i+1].astype(int)
        values = sparse_payload[i+2]
        
        dense_flat = np.zeros(np.prod(shape), dtype=np.float32)
        dense_flat[indices] = values
        reconstructed_deltas.append(dense_flat.reshape(shape))
        
    return reconstructed_deltas


def compression_stats(
    original_deltas: List[np.ndarray],
    sparse_payload: List[np.ndarray],
) -> float:
    """
    Compute and print the byte-size reduction due to compression.
    """
    # Total bytes if transmitted as dense float32
    orig_bytes = sum(arr.nbytes for arr in original_deltas)

    # Effective bytes after compression:
    # shape array + indices array (int32 = 4 bytes) + values array (fp16 effective = 2 bytes)
    comp_bytes = 0
    for i in range(0, len(sparse_payload), 3):
        comp_bytes += sparse_payload[i].nbytes       # shape
        comp_bytes += sparse_payload[i+1].nbytes     # indices (int32 default)
        # We count values as 2 bytes conceptually since it represents fp16 data
        comp_bytes += len(sparse_payload[i+2]) * 2
        
    comp_bytes = max(comp_bytes, 1)

    reduction_pct = (1.0 - comp_bytes / orig_bytes) * 100.0

    print(
        f"\n  [Compression] Original (Dense)  : {orig_bytes / 1024:.2f} KB"
    )
    print(
        f"  [Compression] Compressed Payload: {comp_bytes / 1024:.2f} KB"
    )
    print(
        f"  [Compression] Reduction         : {reduction_pct:.1f}%"
    )

    return reduction_pct
