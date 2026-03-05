"""
Sequence encoding and format conversion for genomic sequences.

Provides one-hot encoding, reverse complement, and genome-aware sequence
fetching via gReLU.
"""

import numpy as np
import pandas as pd
from typing import Optional

# gReLU is an optional dependency — used for genome-aware fetching
try:
    import grelu.sequence.format
    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


# Mapping tables
_ONEHOT_MAP = {
    "A": [1, 0, 0, 0], "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0], "T": [0, 0, 0, 1],
    "a": [1, 0, 0, 0], "c": [0, 1, 0, 0],
    "g": [0, 0, 1, 0], "t": [0, 0, 0, 1],
    "N": [0, 0, 0, 0], "n": [0, 0, 0, 0],
}
_DECODE_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence into shape (L, 4) with channel order ACGT.

    Ambiguous bases (N) are encoded as all zeros.
    """
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        enc = _ONEHOT_MAP.get(base)
        if enc is not None:
            out[i] = enc
    return out


def decode_one_hot(arr: np.ndarray) -> str:
    """
    Decode a (L, 4) one-hot array back to a DNA string.

    All-zero rows are decoded as 'N'.
    """
    indices = np.argmax(arr, axis=1)
    row_sums = arr.sum(axis=1)
    return "".join(
        _DECODE_MAP[idx] if row_sums[i] > 0 else "N"
        for i, idx in enumerate(indices)
    )


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def normalize_chrom(chrom: str, genome: str) -> str:
    """
    Normalize chromosome names for a given genome assembly.

    - hg38: ensures 'chr' prefix
    - GRCm39: strips 'chr' prefix
    - mm10: keeps as-is (UCSC convention with 'chr')
    """
    chrom = str(chrom)
    if genome == "hg38":
        if not chrom.startswith("chr"):
            return "chr" + chrom
    elif genome == "GRCm39":
        if chrom.startswith("chr"):
            return chrom[3:]
    return chrom


def fetch_sequence(
    intervals: pd.DataFrame,
    genome: str,
) -> str:
    """
    Fetch a genomic sequence for an interval using gReLU.

    Args:
        intervals: DataFrame with columns [chrom, start, end] and optionally [strand].
        genome: Assembly name (e.g. 'hg38', 'mm10', 'GRCm39').

    Returns:
        DNA sequence string. If strand is '-', returns reverse complement.

    Raises:
        ImportError: If gReLU is not installed.
    """
    if not _HAS_GRELU:
        raise ImportError(
            "gReLU is required for genome-aware fetching. "
            "Install with: pip install grelu"
        )

    ivals = intervals.copy()
    ivals["chrom"] = ivals["chrom"].apply(lambda x: normalize_chrom(x, genome))

    seq = grelu.sequence.format.convert_input_type(
        ivals, output_type="strings", genome=genome
    )[0]
    return seq
