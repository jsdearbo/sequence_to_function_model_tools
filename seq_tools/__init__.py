"""
seq_tools: Utilities for genomic sequence encoding, interval manipulation,
FASTA access, splicing label generation, and variant construction.
"""

from seq_tools.encoding import one_hot_encode, decode_one_hot, reverse_complement, normalize_chrom
from seq_tools.intervals import generate_intervals, make_input_interval, genome_to_output_bins
from seq_tools.variant import generate_snvs, generate_window_variants, score_variants

__all__ = [
    "one_hot_encode",
    "decode_one_hot",
    "reverse_complement",
    "normalize_chrom",
    "generate_intervals",
    "make_input_interval",
    "genome_to_output_bins",
    "generate_snvs",
    "generate_window_variants",
    "score_variants",
]
