"""
In silico mutagenesis (ISM) pipeline for genomic sequence models.

Systematically introduces every possible single-nucleotide variant across
a region and measures the predicted effect, revealing which positions are
functionally important to the model.
"""

import logging
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import grelu.interpret.score
    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def run_ism(
    model: Any,
    input_seq: Sequence[str],
    prediction_transform: Any,
    start_pos: int,
    end_pos: int,
    device: int = 0,
    batch_size: int = 8,
    compare_func: str = "log2FC",
) -> pd.DataFrame:
    """
    Run in silico mutagenesis via gReLU.

    Mutates every position in [start_pos, end_pos) to all three alternate
    bases and compares predictions to the reference using the specified
    comparison function.

    Args:
        model: gReLU LightningModel.
        input_seq: Reference sequence(s).
        prediction_transform: Callable that reduces model output to a scalar
            (e.g. Aggregate transform over specific bins/tasks).
        start_pos: Start position for mutagenesis (0-based, relative to input).
        end_pos: End position for mutagenesis (exclusive).
        device: GPU device index.
        batch_size: Number of mutant sequences per batch.
        compare_func: How to compare mutant vs reference ('log2FC', 'diff', 'abs_diff').

    Returns:
        DataFrame with ISM scores per position and alternate allele.
    """
    if not _HAS_GRELU:
        raise ImportError("gReLU is required for ISM.")

    logger.info(
        "Running ISM on positions %d–%d (%d mutations)",
        start_pos, end_pos, (end_pos - start_pos) * 3,
    )
    start_time = time.time()

    ism_df = grelu.interpret.score.ISM_predict(
        seqs=input_seq,
        model=model,
        prediction_transform=prediction_transform,
        devices=[device],
        batch_size=batch_size,
        num_workers=1,
        start_pos=start_pos,
        end_pos=end_pos,
        return_df=True,
        compare_func=compare_func,
    )

    logger.info("ISM completed in %.1fs", time.time() - start_time)
    return ism_df


def predict_sequence(
    model: Any,
    input_seqs: Sequence[str],
    device: str = "cuda",
) -> np.ndarray:
    """
    Run model predictions on input sequences.

    Args:
        model: Model with ``predict_on_seqs`` method.
        input_seqs: List of DNA sequences.
        device: Torch device string.

    Returns:
        Prediction array.
    """
    logger.info("Running prediction on %d sequences", len(input_seqs))
    start_time = time.time()
    preds = model.predict_on_seqs(input_seqs, device=device)
    logger.info("Prediction shape: %s, time: %.1fs", preds.shape, time.time() - start_time)
    return preds
