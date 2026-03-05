"""
FASTA reading and windowed sequence iteration.

Provides a thin pysam wrapper and a generator for tiling transcript regions
into fixed-size windows suitable for model input.
"""

import logging
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pysam
except ImportError:
    pysam = None


class FastaReader:
    """
    Thin wrapper around pysam.FastaFile with 0-based half-open coordinates.

    Example::

        reader = FastaReader("genome.fa")
        seq = reader.fetch("chr1", 1000, 2000)
    """

    def __init__(self, fasta_path: str):
        if pysam is None:
            raise ImportError(
                "pysam is required for FASTA access. Install with: pip install pysam"
            )
        self.fasta = pysam.FastaFile(fasta_path)

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch sequence for a 0-based half-open interval [start, end)."""
        return self.fasta.fetch(chrom, start, end)

    @property
    def references(self) -> list[str]:
        """Return available chromosome/contig names."""
        return list(self.fasta.references)

    def close(self):
        self.fasta.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_windows(
    regions: pd.DataFrame,
    window_bp: int,
    stride_bp: int,
) -> Iterator[Tuple[str, int, int]]:
    """
    Yield (chrom, start, end) windows tiling across the union of transcript regions.

    Windows are generated per chromosome from the minimum start to the maximum
    end of all regions on that chromosome.

    Args:
        regions: DataFrame with columns [Chromosome, Start, End].
        window_bp: Window size in base pairs.
        stride_bp: Step size between windows.

    Yields:
        Tuples of (chromosome, window_start, window_end).
    """
    required = {"Chromosome", "Start", "End"}
    missing = required - set(regions.columns)
    if missing:
        raise ValueError(f"regions missing columns: {sorted(missing)}")

    for chrom, sub in regions.groupby("Chromosome"):
        start = int(sub["Start"].min())
        end = int(sub["End"].max())

        if end - start < window_bp:
            continue

        for w_start in range(start, end - window_bp + 1, stride_bp):
            yield chrom, w_start, w_start + window_bp
