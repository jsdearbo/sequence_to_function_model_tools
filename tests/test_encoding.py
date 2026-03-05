"""Tests for seq_tools.encoding."""

import numpy as np
from seq_tools.encoding import (
    one_hot_encode,
    decode_one_hot,
    reverse_complement,
    normalize_chrom,
)


def test_one_hot_roundtrip():
    seq = "ACGTACGT"
    encoded = one_hot_encode(seq)
    assert encoded.shape == (8, 4)
    assert encoded.dtype == np.float32
    decoded = decode_one_hot(encoded)
    assert decoded == seq


def test_one_hot_ambiguous():
    encoded = one_hot_encode("N")
    assert encoded.sum() == 0.0
    assert decode_one_hot(encoded) == "N"


def test_one_hot_case_insensitive():
    upper = one_hot_encode("ACGT")
    lower = one_hot_encode("acgt")
    np.testing.assert_array_equal(upper, lower)


def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"  # palindrome
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("ATCG") == "CGAT"


def test_normalize_chrom_hg38():
    assert normalize_chrom("1", "hg38") == "chr1"
    assert normalize_chrom("chr1", "hg38") == "chr1"


def test_normalize_chrom_grcm39():
    assert normalize_chrom("chr1", "GRCm39") == "1"
    assert normalize_chrom("1", "GRCm39") == "1"


def test_normalize_chrom_mm10():
    assert normalize_chrom("chr1", "mm10") == "chr1"
