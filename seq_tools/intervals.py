"""
Genomic interval generation, centering, and bin-coordinate conversion.

Provides utilities for tiling genomes into windows, centering input intervals
around elements of interest, and converting genome coordinates to model
output bin indices (e.g. for Borzoi-class models).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# gReLU is optional — used only for blacklist filtering
try:
    import grelu.data.preprocess
    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def load_chromosome_lengths(assembly_report: str) -> pd.DataFrame:
    """
    Load chromosome lengths from a UCSC/NCBI assembly report TSV.

    Expects columns 'UCSC-style-name' and 'Sequence-Length'.

    Returns:
        DataFrame with columns [chrom, length].
    """
    df = pd.read_csv(assembly_report, sep="\t")
    out = df[["UCSC-style-name", "Sequence-Length"]].copy()
    out.columns = ["chrom", "length"]
    return out


def generate_intervals(
    chromosome_lengths: pd.DataFrame,
    seq_len: int,
    stride: Optional[int] = None,
) -> pd.DataFrame:
    """
    Tile chromosomes into fixed-length intervals.

    Args:
        chromosome_lengths: DataFrame with columns [chrom, length].
        seq_len: Length of each interval in base pairs.
        stride: Step size between window starts. Defaults to ``seq_len``
            (non-overlapping). Use ``seq_len // 3`` for dense tiling.

    Returns:
        DataFrame with columns [chrom, start, end].
    """
    if stride is None:
        stride = seq_len

    rows: list[dict] = []
    for _, row in chromosome_lengths.iterrows():
        chrom = row["chrom"]
        length = int(row["length"])
        for start in range(0, length, stride):
            end = start + seq_len
            if end <= length:
                rows.append({"chrom": chrom, "start": start, "end": end})
    return pd.DataFrame(rows)


def make_input_interval(
    chrom: str,
    start: int,
    end: int,
    seq_len: int,
    strand: str = "+",
) -> pd.DataFrame:
    """
    Create a model input interval centered on the given coordinates.

    The returned interval has length ``seq_len`` and is centered on the
    midpoint of [start, end).
    """
    center = (start + end) // 2
    input_start = center - seq_len // 2
    input_end = center + seq_len // 2
    return pd.DataFrame({
        "chrom": [chrom],
        "start": [input_start],
        "end": [input_end],
        "strand": [strand],
    })


def make_eval_interval(
    chrom: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """
    Create an evaluation interval (the region whose prediction is of interest).
    """
    return pd.DataFrame({"chrom": [chrom], "start": [start], "end": [end]})


def get_element_coords(
    row: pd.Series,
    measurement: str,
) -> tuple[str, int, int, str]:
    """
    Extract (chrom, start, end, strand) from a row for a given measurement type.

    Args:
        row: A pandas Series with genomic coordinate columns.
        measurement: One of:
            - 'element_only': uses row['start'] and row['end']
            - 'whole_transcript': uses row['tscript_start'] and row['tscript_end']

    Raises:
        ValueError: If measurement is unknown or required columns are missing.
    """
    chrom = row["chrom"]
    strand = row.get("strand", "+")

    if measurement == "element_only":
        return chrom, int(row["start"]), int(row["end"]), strand

    if measurement == "whole_transcript":
        if "tscript_start" not in row or "tscript_end" not in row:
            raise ValueError(
                "'whole_transcript' requires 'tscript_start' and 'tscript_end' columns."
            )
        return chrom, int(row["tscript_start"]), int(row["tscript_end"]), strand

    raise ValueError(f"Unknown measurement: {measurement!r}")


def genome_to_output_bins(
    input_start: int,
    input_end: int,
    eval_start: int,
    eval_end: int,
    bin_size: int = 32,
    output_window: int = 196_608,
) -> list[int]:
    """
    Convert genome-space evaluation coordinates to model output bin indices.

    Handles the central output window of Borzoi-class models where only the
    inner ``output_window`` bp of the input produce predictions.

    Args:
        input_start: Genomic start of the model input window.
        input_end: Genomic end of the model input window.
        eval_start: Genomic start of the region of interest.
        eval_end: Genomic end of the region of interest.
        bin_size: Resolution of the model output (bp per bin).
        output_window: Length of the model's output window in bp.

    Returns:
        List of 0-based bin indices corresponding to the evaluation region.

    Raises:
        ValueError: If the evaluation interval does not overlap the output window.
    """
    input_center = (input_start + input_end) // 2
    output_half = output_window // 2
    out_start = max(input_center - output_half, input_start)
    out_end = min(input_center + output_half, input_end)

    # Clamp eval to output window
    ev_start = max(eval_start, out_start)
    ev_end = min(eval_end, out_end)

    if ev_start >= ev_end:
        raise ValueError(
            f"Eval [{eval_start}, {eval_end}) does not overlap output window "
            f"[{out_start}, {out_end})"
        )

    rel_start = ev_start - out_start
    rel_end = ev_end - out_start
    bin_start = rel_start // bin_size
    bin_end = (rel_end + bin_size - 1) // bin_size  # ceil

    n_bins = output_window // bin_size
    bin_start = max(0, min(bin_start, n_bins))
    bin_end = max(0, min(bin_end, n_bins))

    if bin_start >= bin_end:
        raise ValueError(
            f"Empty bin range: rel [{rel_start}, {rel_end}), "
            f"bins [{bin_start}, {bin_end})"
        )

    return list(range(bin_start, bin_end))


def filter_blacklist(
    intervals: pd.DataFrame,
    genome: Optional[str] = None,
    blacklist_path: Optional[str] = None,
    window: int = 1,
) -> pd.DataFrame:
    """
    Remove intervals overlapping blacklisted regions.

    Uses gReLU's blacklist filter when available. Accepts either a genome name
    (to use the built-in blacklist) or a path to a custom BED file.

    Args:
        intervals: DataFrame with columns [chrom, start, end].
        genome: Assembly name for built-in blacklist (e.g. 'mm10').
        blacklist_path: Path to a custom blacklist BED file.
        window: Extension around blacklisted regions in bp.
    """
    if not _HAS_GRELU:
        raise ImportError("gReLU is required for blacklist filtering.")

    kwargs = {"window": window}
    if blacklist_path:
        kwargs["blacklist"] = blacklist_path
    elif genome:
        kwargs["genome"] = genome
    else:
        raise ValueError("Provide either 'genome' or 'blacklist_path'.")

    return grelu.data.preprocess.filter_blacklist(intervals, **kwargs)
