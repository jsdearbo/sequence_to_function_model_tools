"""
TF-MoDISco integration for motif discovery from attribution data.

Runs TF-MoDISco on pre-computed attribution maps to discover recurring
sequence motifs that drive model predictions, and maps them against
known motif databases.
"""

import logging
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import grelu.interpret.modisco
    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def run_modisco(
    model: Any,
    input_seqs: Sequence[str],
    attributions: np.ndarray,
    out_dir: str,
    genome: str = "mm10",
    meme_file: str = "CISBP_RNA_DNA_ENCODED",
    method: str = "completed",
    window: int = 5000,
    sliding_window_size: int = 21,
    flank_size: int = 10,
    device: str = "cuda",
    batch_size: int = 1024,
    num_workers: int = 16,
    seed: int = 0,
) -> None:
    """
    Run TF-MoDISco analysis on attribution data.

    Discovers recurring motif patterns in attribution maps and compares
    them to a reference motif database via TOMTOM alignment.

    Args:
        model: gReLU LightningModel.
        input_seqs: The sequences that attributions were computed on.
        attributions: Attribution array from ``get_attributions_for_element``.
        out_dir: Directory for MoDISco output files.
        genome: Genome assembly.
        meme_file: Motif database name or path for TOMTOM comparison.
        method: Attribution method used ('completed', 'saliency', etc.).
        window: Window size for seqlet scanning.
        sliding_window_size: MoDISco sliding window parameter (controls
            maximum seqlet length = window + 2 * flank_size).
        flank_size: Flank size for seqlet extension.
        device: Compute device.
        batch_size: Batch size for internal ISM (if applicable).
        num_workers: DataLoader workers.
        seed: Random seed for reproducibility.
    """
    if not _HAS_GRELU:
        raise ImportError("gReLU is required for TF-MoDISco.")

    logger.info("Running TF-MoDISco: window=%d, seqlet_max=%d", window, sliding_window_size + 2 * flank_size)

    grelu.interpret.modisco.run_modisco(
        model,
        seqs=input_seqs,
        genome=genome,
        meme_file=meme_file,
        method=method,
        out_dir=out_dir,
        batch_size=batch_size,
        devices=device,
        num_workers=num_workers,
        window=window,
        seed=seed,
        attributions=attributions,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
    )

    logger.info("TF-MoDISco analysis completed. Results in %s", out_dir)
