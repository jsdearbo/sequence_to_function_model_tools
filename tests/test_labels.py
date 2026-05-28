"""Tests for seq_tools.labels."""

import numpy as np
import pandas as pd
import pytest

try:
    import pyranges as pr
    _HAS_PYRANGES = True
except ImportError:
    _HAS_PYRANGES = False

from seq_tools.labels import (
    _disjoint_labels_with_priority,
    collapse_exon_coords_weighted,
    generate_soft_labels,
)


# ---------------------------------------------------------------------------
# _disjoint_labels_with_priority
# ---------------------------------------------------------------------------

class TestDisjointLabelsWithPriority:
    """Tests for the core sweep-line disjoint labeling algorithm."""

    def _t(self, chrom, start, end, strand="+"):
        return pd.DataFrame({"Chromosome": [chrom], "Start": [start], "End": [end], "Strand": [strand]})

    def _e(self, records, strand="+"):
        rows = [{"Chromosome": c, "Start": s, "End": e, "Strand": strand} for c, s, e in records]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Chromosome", "Start", "End", "Strand"])

    def _p(self, records, strand="+"):
        rows = [
            {"Chromosome": c, "Start": s, "End": e, "psi": psi, "confidence": conf, "Strand": strand}
            for c, s, e, psi, conf in records
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Chromosome", "Start", "End", "Strand", "psi", "confidence"]
        )

    def _no_p(self):
        return self._p([])

    def test_output_columns(self):
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([("chr1", 0, 1000)]),
            self._no_p(),
        )
        assert set(out.columns) >= {"Chromosome", "Start", "End", "p_exon", "confidence", "Strand"}

    def test_empty_transcripts_returns_empty(self):
        t = pd.DataFrame(columns=["Chromosome", "Start", "End", "Strand"])
        out = _disjoint_labels_with_priority(t, self._e([]), self._no_p())
        assert out.empty

    def test_pure_exon_is_one(self):
        """Transcript fully covered by an exon → p_exon=1.0 throughout."""
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([("chr1", 0, 1000)]),
            self._no_p(),
        )
        assert not out.empty
        assert (out["p_exon"] == 1.0).all()

    def test_no_exon_annotation_is_zero(self):
        """Transcript with no exon annotations → p_exon=0.0 throughout."""
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([]),
            self._no_p(),
        )
        assert not out.empty
        assert (out["p_exon"] == 0.0).all()

    def test_exon_intron_boundary(self):
        """Exon in first half, intron in second half — clear split at 500."""
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([("chr1", 0, 500)]),
            self._no_p(),
        )
        assert not out[out["p_exon"] == 1.0].empty
        assert not out[out["p_exon"] == 0.0].empty
        assert out[out["p_exon"] == 1.0]["End"].max() <= 500
        assert out[out["p_exon"] == 0.0]["Start"].min() >= 500

    def test_psi_overrides_exon_annotation(self):
        """PSI annotation in a subregion takes priority over the exon background."""
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([("chr1", 0, 1000)]),
            self._p([("chr1", 300, 700, 0.4, 5.0)]),
        )
        psi_seg = out[(out["Start"] >= 300) & (out["End"] <= 700)]
        assert not psi_seg.empty
        assert np.allclose(psi_seg["p_exon"].values, 0.4)
        # Flanking regions should remain fully exonic
        left = out[out["End"] <= 300]
        right = out[out["Start"] >= 700]
        if not left.empty:
            assert np.allclose(left["p_exon"].values, 1.0)
        if not right.empty:
            assert np.allclose(right["p_exon"].values, 1.0)

    def test_segments_cover_full_transcript(self):
        """All segments together exactly span the input transcript (no gaps)."""
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([("chr1", 0, 400), ("chr1", 600, 1000)]),
            self._no_p(),
        )
        total_bp = (out["End"] - out["Start"]).sum()
        assert total_bp == 1000

    def test_multiple_chromosomes_labeled_independently(self):
        """Segments from different chromosomes appear in output."""
        t = pd.DataFrame({
            "Chromosome": ["chr1", "chr2"],
            "Start": [0, 0],
            "End": [1000, 500],
            "Strand": ["+", "+"],
        })
        e = pd.DataFrame({
            "Chromosome": ["chr1", "chr2"],
            "Start": [0, 0],
            "End": [1000, 500],
            "Strand": ["+", "+"],
        })
        out = _disjoint_labels_with_priority(t, e, self._no_p())
        assert set(out["Chromosome"].unique()) == {"chr1", "chr2"}

    def test_multiple_psi_events_averaged(self):
        """Overlapping PSI events in the same region are averaged."""
        out = _disjoint_labels_with_priority(
            self._t("chr1", 0, 1000),
            self._e([("chr1", 0, 1000)]),
            self._p([
                ("chr1", 200, 800, 0.2, 1.0),
                ("chr1", 200, 800, 0.8, 1.0),
            ]),
        )
        psi_seg = out[(out["Start"] >= 200) & (out["End"] <= 800)]
        assert not psi_seg.empty
        assert psi_seg["p_exon"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# collapse_exon_coords_weighted
# ---------------------------------------------------------------------------

class TestCollapseExonCoordsWeighted:

    def _row(self, chrom, start, end, strand, ijc_list, sjc_list):
        d = {"Chromosome": chrom, "Start": start, "End": end, "Strand": strand}
        for i, (ic, sc) in enumerate(zip(ijc_list, sjc_list)):
            d[f"ijc_{i}"] = ic
            d[f"sjc_{i}"] = sc
        d["coverage"] = sum(ijc_list) + sum(sjc_list)
        return d

    def test_empty_returns_empty(self):
        df = pd.DataFrame(columns=["Chromosome", "Start", "End", "Strand", "ijc_0", "sjc_0", "coverage"])
        assert collapse_exon_coords_weighted(df).empty

    def test_single_row_psi(self):
        """pooled PSI = ijc / (ijc + sjc) for a single row."""
        df = pd.DataFrame([self._row("chr1", 100, 200, "+", [80], [20])])
        out = collapse_exon_coords_weighted(df)
        assert len(out) == 1
        assert out["psi"].iloc[0] == pytest.approx(0.8)

    def test_duplicate_coords_collapsed_to_one(self):
        """Identical coordinates collapse into one row with summed counts."""
        row = self._row("chr1", 100, 200, "+", [40], [10])
        df = pd.DataFrame([row, row.copy()])
        out = collapse_exon_coords_weighted(df)
        assert len(out) == 1
        assert out["psi"].iloc[0] == pytest.approx(0.8)  # 80/(80+20)

    def test_different_coords_not_merged(self):
        """Rows with distinct coordinates remain separate."""
        df = pd.DataFrame([
            self._row("chr1", 100, 200, "+", [10], [10]),
            self._row("chr1", 300, 400, "+", [10], [10]),
        ])
        assert len(collapse_exon_coords_weighted(df)) == 2

    def test_read_weighted_not_naive_average(self):
        """Counts are summed before PSI recomputation — not a naive average of PSI values."""
        # High-coverage row (PSI=0.9) and low-coverage row (PSI=0.1) at same locus.
        # Naive avg = 0.5; weighted = 910/1100 ≈ 0.827.
        df = pd.DataFrame([
            self._row("chr1", 100, 200, "+", [900], [100]),
            self._row("chr1", 100, 200, "+", [10], [90]),
        ])
        out = collapse_exon_coords_weighted(df)
        assert len(out) == 1
        expected = 910 / (910 + 190)
        assert out["psi"].iloc[0] == pytest.approx(expected, abs=1e-6)

    def test_confidence_increases_with_coverage(self):
        """Higher junction read depth → higher confidence (log1p scale)."""
        low = collapse_exon_coords_weighted(
            pd.DataFrame([self._row("chr1", 100, 200, "+", [5], [5])])
        )
        high = collapse_exon_coords_weighted(
            pd.DataFrame([self._row("chr1", 100, 200, "+", [500], [500])])
        )
        assert high["confidence"].iloc[0] > low["confidence"].iloc[0]

    def test_output_has_confidence_column(self):
        df = pd.DataFrame([self._row("chr1", 100, 200, "+", [50], [50])])
        assert "confidence" in collapse_exon_coords_weighted(df).columns


# ---------------------------------------------------------------------------
# generate_soft_labels  (requires pyranges)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_PYRANGES, reason="pyranges not installed")
class TestGenerateSoftLabels:

    def _gtf(self, records):
        """Build a minimal StringTie-like PyRanges."""
        rows = []
        for chrom, t_start, t_end, strand, exons in records:
            rows.append({
                "Chromosome": chrom, "Start": t_start, "End": t_end,
                "Strand": strand, "Feature": "transcript",
                "transcript_id": f"t_{chrom}_{t_start}",
            })
            for e_start, e_end in exons:
                rows.append({
                    "Chromosome": chrom, "Start": e_start, "End": e_end,
                    "Strand": strand, "Feature": "exon",
                    "transcript_id": f"t_{chrom}_{t_start}",
                })
        return pr.PyRanges(pd.DataFrame(rows))

    def _se(self, records, strand="+"):
        """Build a minimal rMATS SE PyRanges with ijc/sjc columns."""
        rows = []
        for chrom, start, end, psi in records:
            inc = int(round(psi * 100))
            skp = 100 - inc
            rows.append({
                "Chromosome": chrom, "Start": start, "End": end, "Strand": strand,
                "psi_0": psi, "ijc_0": inc, "sjc_0": skp, "coverage": 100,
            })
        return pr.PyRanges(pd.DataFrame(rows))

    def test_output_schema(self):
        gtf = self._gtf([("chr1", 0, 1000, "+", [(0, 500), (700, 1000)])])
        out = generate_soft_labels(gtf, pr.PyRanges())
        assert set(out.columns) >= {"Chromosome", "Start", "End", "p_exon", "confidence", "Strand"}

    def test_no_rmats_exons_are_one(self):
        """Without rMATS, annotated exon regions get p_exon=1.0."""
        gtf = self._gtf([("chr1", 0, 1000, "+", [(0, 500)])])
        out = generate_soft_labels(gtf, pr.PyRanges())
        assert any(out["p_exon"] == 1.0)

    def test_no_rmats_introns_are_zero(self):
        """Without rMATS, inter-exon regions get p_exon=0.0."""
        gtf = self._gtf([("chr1", 0, 1000, "+", [(0, 400), (600, 1000)])])
        out = generate_soft_labels(gtf, pr.PyRanges())
        assert any(out["p_exon"] == 0.0)

    def test_rmats_psi_in_output(self):
        """rMATS SE events produce intermediate p_exon values."""
        gtf = self._gtf([("chr1", 0, 2000, "+", [(0, 2000)])])
        se = self._se([("chr1", 500, 1500, 0.3)])
        out = generate_soft_labels(gtf, se)
        mid = out.loc[(out["p_exon"] > 0.0) & (out["p_exon"] < 1.0), "p_exon"]
        assert not mid.empty
        assert abs(mid.iloc[0] - 0.3) < 1e-4

    def test_ri_events_incorporated(self):
        """Retained intron events contribute PSI values alongside SE events."""
        gtf = self._gtf([("chr1", 0, 2000, "+", [(0, 2000)])])
        se = self._se([("chr1", 200, 600, 0.6)])
        ri = self._se([("chr1", 1000, 1800, 0.8)])
        out = generate_soft_labels(gtf, se, rmats_ri=ri)
        mid = out.loc[(out["p_exon"] > 0.0) & (out["p_exon"] < 1.0)]
        assert len(mid) >= 2

    def test_none_rmats_inputs_accepted(self):
        """None for optional rMATS inputs is accepted (treated as empty)."""
        gtf = self._gtf([("chr1", 0, 1000, "+", [(0, 1000)])])
        out = generate_soft_labels(gtf, pr.PyRanges(), rmats_ri=None, rmats_a3ss=None)
        assert not out.empty
