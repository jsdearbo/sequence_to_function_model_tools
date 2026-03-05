"""Tests for seq_tools.intervals."""

import pandas as pd
from seq_tools.intervals import (
    generate_intervals,
    make_input_interval,
    genome_to_output_bins,
)


def test_generate_intervals_nonoverlapping():
    chroms = pd.DataFrame({"chrom": ["chr1"], "length": [1000]})
    intervals = generate_intervals(chroms, seq_len=200)
    assert len(intervals) == 5  # 0-200, 200-400, ..., 800-1000
    assert all(intervals["end"] - intervals["start"] == 200)


def test_generate_intervals_with_stride():
    chroms = pd.DataFrame({"chrom": ["chr1"], "length": [500]})
    intervals = generate_intervals(chroms, seq_len=200, stride=100)
    # Windows: 0-200, 100-300, 200-400, 300-500 = 4
    assert len(intervals) == 4


def test_generate_intervals_skips_short():
    chroms = pd.DataFrame({"chrom": ["chr1"], "length": [50]})
    intervals = generate_intervals(chroms, seq_len=100)
    assert len(intervals) == 0


def test_make_input_interval_centering():
    iv = make_input_interval("chr1", 1000, 2000, seq_len=524288)
    assert iv["end"].iloc[0] - iv["start"].iloc[0] == 524288
    center = (iv["start"].iloc[0] + iv["end"].iloc[0]) / 2
    assert center == 1500  # midpoint of 1000-2000


def test_genome_to_output_bins():
    bins = genome_to_output_bins(
        input_start=0,
        input_end=524288,
        eval_start=262000,
        eval_end=262320,
        bin_size=32,
        output_window=196608,
    )
    assert len(bins) > 0
    assert all(isinstance(b, int) for b in bins)


def test_genome_to_output_bins_raises_on_no_overlap():
    import pytest
    with pytest.raises(ValueError):
        genome_to_output_bins(
            input_start=0,
            input_end=524288,
            eval_start=0,
            eval_end=100,
            bin_size=32,
            output_window=196608,
        )
