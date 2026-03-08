"""Tests for seq_tools.variant."""

import numpy as np
import pandas as pd
import pytest

from seq_tools.variant import (
    generate_snvs,
    generate_window_variants,
    variants_to_dataframe,
    score_variants,
)


class TestGenerateSnvs:
    def test_all_positions(self):
        variants = generate_snvs("ACGT")
        assert len(variants) == 4 * 3  # 4 positions × 3 alts

    def test_specific_positions(self):
        variants = generate_snvs("ACGT", positions=[1, 3])
        assert len(variants) == 2 * 3

    def test_excludes_reference_allele(self):
        variants = generate_snvs("A")
        alts = {v["alt"] for v in variants}
        assert "A" not in alts
        assert alts == {"C", "G", "T"}

    def test_mutant_sequence_correct(self):
        variants = generate_snvs("ACGT", positions=[0])
        for v in variants:
            assert v["sequence"][0] == v["alt"]
            assert v["sequence"][1:] == "CGT"

    def test_returns_dict_format(self):
        variants = generate_snvs("AC")
        for v in variants:
            assert set(v.keys()) == {"position", "ref", "alt", "sequence"}

    def test_case_insensitive(self):
        variants = generate_snvs("acgt")
        # References should be uppercase
        for v in variants:
            assert v["ref"] in "ACGT"


class TestGenerateWindowVariants:
    def test_window_bounds(self):
        seq = "A" * 100
        variants = generate_window_variants(seq, start=10, end=20)
        positions = {v["position"] for v in variants}
        assert positions == set(range(10, 20))

    def test_clamped_to_sequence(self):
        seq = "ACGT"
        variants = generate_window_variants(seq, start=-5, end=100)
        positions = {v["position"] for v in variants}
        assert positions == {0, 1, 2, 3}

    def test_empty_window(self):
        variants = generate_window_variants("ACGT", start=2, end=2)
        assert len(variants) == 0


class TestVariantsToDataframe:
    def test_columns(self):
        variants = generate_snvs("AC")
        df = variants_to_dataframe(variants)
        assert list(df.columns) == ["position", "ref", "alt"]

    def test_excludes_sequence(self):
        variants = generate_snvs("AC")
        df = variants_to_dataframe(variants)
        assert "sequence" not in df.columns

    def test_row_count(self):
        variants = generate_snvs("ACGT")
        df = variants_to_dataframe(variants)
        assert len(df) == 12


class TestScoreVariants:
    def _make_mock_model(self):
        """Mock model that returns GC content as score."""
        class MockModel:
            def predict_on_seqs(self, seqs, device="cpu"):
                scores = []
                for s in seqs:
                    gc = (s.upper().count("G") + s.upper().count("C")) / max(len(s), 1)
                    scores.append(gc + 0.1)  # avoid zero
                return np.array(scores)
        return MockModel()

    def test_output_columns(self):
        model = self._make_mock_model()
        variants = generate_snvs("ACGT", positions=[0, 1])
        result = score_variants(variants, model, device="cpu")
        assert list(result.columns) == ["position", "ref", "alt", "ref_score", "alt_score", "log2fc"]

    def test_output_length(self):
        model = self._make_mock_model()
        variants = generate_snvs("ACGT", positions=[0])
        result = score_variants(variants, model, device="cpu")
        assert len(result) == 3

    def test_log2fc_direction(self):
        model = self._make_mock_model()
        # Mutating A→G at position 0 increases GC, so log2fc should be positive
        variants = generate_snvs("AAAA", positions=[0])
        result = score_variants(variants, model, device="cpu")
        g_row = result[result["alt"] == "G"]
        assert g_row["log2fc"].iloc[0] > 0

    def test_with_prediction_transform(self):
        model = self._make_mock_model()
        variants = generate_snvs("ACGT", positions=[0])
        result = score_variants(
            variants, model,
            prediction_transform=lambda x: x * 2,
            device="cpu",
        )
        assert len(result) == 3
