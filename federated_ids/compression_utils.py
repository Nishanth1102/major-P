"""
compression_utils.py
--------------------
Reusable gradient compression helpers for Federated Learning.

Functions
---------
top_k_sparsify(params, k_ratio)
    Keep only the top k% highest-magnitude values in each parameter array;
    zero out the rest (Top-K sparsification).

quantize_fp16(params)
    Simulate FP16 quantization by casting float32 → float16 → float32.
    This quantizes the precision while keeping the dtype as float32 so that
    Flower's gRPC serializer remains fully compatible.

compression_stats(original, compressed)
    Print and return the byte-size reduction achieved after compression.

Usage example
-------------
    from compression_utils import top_k_sparsify, quantize_fp16, compression_stats

    updated = get_parameters(model)           # List[np.ndarray] float32
    sparse  = top_k_sparsify(updated, 0.10)  # Keep top 10%
    quant   = quantize_fp16(sparse)           # Quantize precision
    compression_stats(updated, quant)         # Log savings
"""

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# 1. Top-K Sparsification
# ---------------------------------------------------------------------------

def top_k_sparsify(
    params: List[np.ndarray],
    k_ratio: float = 0.10,
) -> List[np.ndarray]:
    """
    Apply Top-K sparsification to a list of parameter arrays.

    For each array, only the top `k_ratio` fraction of values (by absolute
    magnitude) are kept; all other values are set to zero. This reduces the
    number of non-zero values that need to be transmitted.

    Parameters
    ----------
    params  : List of NumPy arrays (model weights / gradients), float32.
    k_ratio : Fraction of values to KEEP (0 < k_ratio <= 1.0).
              Default 0.10 keeps the top 10% — discards 90%.

    Returns
    -------
    List[np.ndarray]  Same shapes as input, dtype float32.
    """
    if not (0.0 < k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be in (0, 1]. Got {k_ratio}")

    sparsified = []
    for layer in params:
        flat = layer.flatten()                      # Work on a 1-D view
        k    = max(1, int(len(flat) * k_ratio))     # At least 1 value kept

        # Find the k-th largest absolute value as the threshold
        threshold = np.partition(np.abs(flat), -k)[-k]

        # Zero out everything below the threshold
        mask         = np.abs(layer) >= threshold
        sparse_layer = (layer * mask).astype(np.float32)
        sparsified.append(sparse_layer)

    return sparsified


# ---------------------------------------------------------------------------
# 2. FP16 Quantization
# ---------------------------------------------------------------------------

def quantize_fp16(
    params: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Simulate FP16 quantization on a list of parameter arrays.

    Casts each float32 array → float16 → float32.
    The round-trip clips precision to FP16 resolution (~3 decimal digits)
    while keeping the final dtype as float32, which is required by Flower.

    Parameters
    ----------
    params : List of NumPy arrays, dtype float32.

    Returns
    -------
    List[np.ndarray]  Same shapes, dtype float32 (FP16-precision values).
    """
    return [layer.astype(np.float16).astype(np.float32) for layer in params]


# ---------------------------------------------------------------------------
# 3. Communication Cost Logger
# ---------------------------------------------------------------------------

def compression_stats(
    original: List[np.ndarray],
    compressed: List[np.ndarray],
) -> float:
    """
    Compute and print the byte-size reduction due to compression.

    Compares the raw float32 byte footprint of `original` against `compressed`.
    Note: both inputs are float32 arrays; the savings measured here represent
    the *effective* reduction from sparsification (90% zeros compress well in
    practice) and the precision reduction from FP16 quantization.

    Parameters
    ----------
    original   : Original parameter list (float32).
    compressed : Compressed parameter list (float32, sparsified + quantized).

    Returns
    -------
    float  Percentage reduction in bytes (0–100).
    """
    # Total bytes if transmitted as float32
    orig_bytes = sum(arr.nbytes for arr in original)

    # Effective bytes after compression:
    # - Sparsified arrays: non-zero values only need ~4 bytes each (float32)
    # - Quantized arrays: precision is FP16, so effective cost is ~2 bytes each
    # We approximate: count non-zero elements × 2 bytes (FP16 effective size)
    comp_bytes = sum(
        int(np.count_nonzero(arr)) * 2   # 2 bytes per non-zero (FP16)
        for arr in compressed
    )
    comp_bytes = max(comp_bytes, 1)  # Avoid division by zero edge case

    reduction_pct = (1.0 - comp_bytes / orig_bytes) * 100.0

    print(
        f"\n  [Compression] Original : {orig_bytes / 1024:.2f} KB (float32)"
    )
    print(
        f"  [Compression] Compressed: {comp_bytes / 1024:.2f} KB "
        f"(Top-K sparse + FP16 effective)"
    )
    print(
        f"  [Compression] Reduction : {reduction_pct:.1f}%"
    )

    return reduction_pct
