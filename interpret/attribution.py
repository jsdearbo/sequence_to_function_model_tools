"""
Attribution methods for interpreting genomic sequence models.

Wraps gReLU's attribution infrastructure to provide element-level
attribution scoring with configurable prediction aggregation
over tasks and genomic positions.
"""

import logging
import time
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import grelu.transforms.prediction_transforms
    from grelu.interpret.score import get_attributions
    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def get_attributions_for_element(
    model: Any,
    input_seq: str,
    selected_bins: Sequence[int],
    tasks: List[str],
    device: str = "cuda",
    method: str = "saliency",
    genome: str = "mm10",
    n_shuffles: int = 10,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Compute attribution scores for a specific genomic element.

    Aggregates model predictions over the specified bins and tasks
    before computing gradients, so the attribution reflects the
    model's sensitivity w.r.t. the element of interest.

    Args:
        model: gReLU LightningModel.
        input_seq: DNA sequence string (length = model input length).
        selected_bins: Output bin indices corresponding to the element.
        tasks: Task names to aggregate over.
        device: Torch device.
        method: Attribution method ('saliency', 'deeplift', 'integrated_gradients').
        genome: Genome assembly for sequence handling.
        n_shuffles: Number of dinucleotide shuffles for DeepLIFT/IG baselines.
        batch_size: Batch size for attribution computation.

    Returns:
        Attribution array, typically shape (1, L, 4).
    """
    if not _HAS_GRELU:
        raise ImportError("gReLU is required for attribution methods.")

    aggregate = grelu.transforms.prediction_transforms.Aggregate(
        tasks=tasks,
        positions=selected_bins,
        length_aggfunc="mean",
        task_aggfunc="mean",
        model=model,
    )

    logger.info(
        "Computing %s attributions over %d bins, %d tasks",
        method, len(selected_bins), len(tasks),
    )
    start = time.time()

    attrs = get_attributions(
        model,
        seqs=[input_seq],
        genome=genome,
        prediction_transform=aggregate,
        device=device,
        method=method,
        seed=0,
        hypothetical=False,
        n_shuffles=n_shuffles,
        batch_size=batch_size,
    )

    logger.info("Attribution time: %.1fs", time.time() - start)
    return attrs


def attribution_native_only(
    attrs: np.ndarray,
    seq: str,
) -> np.ndarray:
    """
    Zero out attribution values at non-native bases.

    For each position, keeps only the attribution for the base actually
    present in the input sequence and sets the other three channels to zero.
    Useful for visualizing importance of the actual sequence rather than
    hypothetical contributions.

    Args:
        attrs: Attribution array, shape (..., L, 4).
        seq: DNA sequence of length L.

    Returns:
        Masked attribution array with same shape.
    """
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    masked = np.zeros_like(attrs)

    for i, base in enumerate(seq.upper()):
        idx = base_to_idx.get(base)
        if idx is not None:
            masked[..., i, idx] = attrs[..., i, idx]

    return masked
