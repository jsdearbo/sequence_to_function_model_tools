"""
seq_tools: Utilities for genomic sequence encoding, interval manipulation,
FASTA access, splicing label generation, and variant construction.
"""

from seq_tools.encoding import one_hot_encode, decode_one_hot, reverse_complement, normalize_chrom
from seq_tools.intervals import generate_intervals, make_input_interval, genome_to_output_bins
from seq_tools.variant import generate_snvs, generate_window_variants, score_variants
from seq_tools.visualization import (
    plot_ism_heatmap,
    plot_prediction_track,
    plot_gene_model,
    plot_attribution,
    multi_track_figure,
)

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
    "plot_ism_heatmap",
    "plot_prediction_track",
    "plot_gene_model",
    "plot_attribution",
    "multi_track_figure",
]
