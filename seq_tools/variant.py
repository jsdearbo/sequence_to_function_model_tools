"""
Variant generation utilities for in silico mutagenesis and variant effect prediction.

Provides functions to systematically generate single-nucleotide variants (SNVs)
and score them against a reference sequence, supporting both targeted and
saturating mutagenesis strategies.
"""

from typing import Optional

import numpy as np
import pandas as pd


BASES = ["A", "C", "G", "T"]


def generate_snvs(
    seq: str,
    positions: Optional[list[int]] = None,
) -> list[dict]:
    """
    Generate all single-nucleotide variants at specified positions.

    For each position, produces 3 alternate alleles (excluding the reference base).

    Args:
        seq: Reference DNA sequence.
        positions: 0-based positions to mutate. If None, mutates all positions.

    Returns:
        List of dicts with keys: position, ref, alt, sequence.
    """
    if positions is None:
        positions = list(range(len(seq)))

    seq_upper = seq.upper()
    variants = []
    for pos in positions:
        ref = seq_upper[pos]
        for alt in BASES:
            if alt == ref:
                continue
            mutant = seq[:pos] + alt + seq[pos + 1:]
            variants.append({
                "position": pos,
                "ref": ref,
                "alt": alt,
                "sequence": mutant,
            })
    return variants


def generate_window_variants(
    seq: str,
    start: int,
    end: int,
) -> list[dict]:
    """
    Saturating mutagenesis: generate all SNVs within a window [start, end).

    Args:
        seq: Full reference sequence.
        start: 0-based start of the mutation window (inclusive).
        end: 0-based end of the mutation window (exclusive).

    Returns:
        List of variant dicts (see ``generate_snvs``).
    """
    positions = list(range(max(0, start), min(end, len(seq))))
    return generate_snvs(seq, positions)


def variants_to_dataframe(variants: list[dict]) -> pd.DataFrame:
    """
    Convert a list of variant dicts to a DataFrame (without the full sequence).

    Returns:
        DataFrame with columns [position, ref, alt].
    """
    return pd.DataFrame([
        {"position": v["position"], "ref": v["ref"], "alt": v["alt"]}
        for v in variants
    ])


def score_variants(
    variants: list[dict],
    model,
    prediction_transform=None,
    ref_pred: Optional[np.ndarray] = None,
    device: str = "cuda",
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Score a list of variants by predicting on mutant sequences and comparing
    to reference.

    Requires a model with a ``predict_on_seqs`` method (e.g. gReLU LightningModel).

    Args:
        variants: Output of ``generate_snvs`` or ``generate_window_variants``.
        model: Model with ``predict_on_seqs(seqs, device=...)`` method.
        prediction_transform: Optional callable to reduce predictions to a scalar
            per sequence (e.g. aggregation over tasks/positions).
        ref_pred: Pre-computed reference prediction. If None, the first variant's
            original sequence is used as reference (assumes all share the same ref).
        device: Torch device string.
        batch_size: Number of sequences per prediction batch.

    Returns:
        DataFrame with columns [position, ref, alt, ref_score, alt_score, log2fc].
    """
    seqs = [v["sequence"] for v in variants]

    # Batch prediction
    all_preds = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        preds = model.predict_on_seqs(batch, device=device)
        if prediction_transform is not None:
            preds = prediction_transform(preds)
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    # Reduce to scalar if multi-dimensional
    if all_preds.ndim > 1:
        alt_scores = all_preds.mean(axis=tuple(range(1, all_preds.ndim)))
    else:
        alt_scores = all_preds

    # Reference prediction
    if ref_pred is None:
        # Reconstruct reference from the first variant
        v0 = variants[0]
        ref_seq = v0["sequence"][:v0["position"]] + v0["ref"] + v0["sequence"][v0["position"] + 1:]
        ref_out = model.predict_on_seqs([ref_seq], device=device)
        if prediction_transform is not None:
            ref_out = prediction_transform(ref_out)
        if ref_out.ndim > 1:
            ref_score = float(ref_out.mean())
        else:
            ref_score = float(ref_out[0])
    else:
        ref_score = float(ref_pred) if np.ndim(ref_pred) == 0 else float(ref_pred.mean())

    results = []
    for v, alt_s in zip(variants, alt_scores):
        alt_val = float(alt_s)
        log2fc = np.log2(alt_val / ref_score) if ref_score > 0 and alt_val > 0 else 0.0
        results.append({
            "position": v["position"],
            "ref": v["ref"],
            "alt": v["alt"],
            "ref_score": ref_score,
            "alt_score": alt_val,
            "log2fc": log2fc,
        })

    return pd.DataFrame(results)
