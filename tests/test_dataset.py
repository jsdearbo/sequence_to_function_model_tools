"""Tests for training.dataset (GenomicWindowDataset, rasterize_window)."""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from training.dataset import GenomicWindowDataset, rasterize_window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_shard(
    directory: Path,
    shard_idx: int,
    n_examples: int,
    seq_len: int = 100,
    n_bins: int = 10,
    n_tasks: int = 2,
) -> Path:
    rng = np.random.default_rng(shard_idx)
    seq = rng.random((n_examples, seq_len, 4)).astype(np.float32)
    targets = rng.random((n_examples, n_bins, n_tasks)).astype(np.float32)
    mask = (rng.random((n_examples, n_bins, n_tasks)) > 0.3).astype(np.float32)
    path = directory / f"shard_{shard_idx:04d}.npz"
    np.savez(path, sequence=seq, targets=targets, mask=mask)
    return path


# ---------------------------------------------------------------------------
# GenomicWindowDataset
# ---------------------------------------------------------------------------

class TestGenomicWindowDataset:

    def test_len_single_shard(self, tmp_path):
        _write_shard(tmp_path, 0, n_examples=4)
        ds = GenomicWindowDataset(str(tmp_path))
        assert len(ds) == 4

    def test_len_multiple_shards(self, tmp_path):
        _write_shard(tmp_path, 0, n_examples=3)
        _write_shard(tmp_path, 1, n_examples=5)
        ds = GenomicWindowDataset(str(tmp_path))
        assert len(ds) == 8

    def test_getitem_returns_dict_with_required_keys(self, tmp_path):
        _write_shard(tmp_path, 0, n_examples=2)
        ds = GenomicWindowDataset(str(tmp_path))
        sample = ds[0]
        assert set(sample.keys()) == {"sequence", "targets", "mask"}

    def test_getitem_tensor_dtype(self, tmp_path):
        _write_shard(tmp_path, 0, n_examples=2)
        ds = GenomicWindowDataset(str(tmp_path))
        sample = ds[0]
        for key in ("sequence", "targets", "mask"):
            assert sample[key].dtype == torch.float32

    def test_getitem_tensor_shapes(self, tmp_path):
        _write_shard(tmp_path, 0, n_examples=2, seq_len=100, n_bins=10, n_tasks=3)
        ds = GenomicWindowDataset(str(tmp_path))
        sample = ds[0]
        assert sample["sequence"].shape == (100, 4)
        assert sample["targets"].shape == (10, 3)
        assert sample["mask"].shape == (10, 3)

    def test_no_shards_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GenomicWindowDataset(str(tmp_path))

    def test_shard_caching(self, tmp_path):
        """Accessing items triggers shard loads; cache tracks current shard index."""
        _write_shard(tmp_path, 0, n_examples=3)
        _write_shard(tmp_path, 1, n_examples=2)
        ds = GenomicWindowDataset(str(tmp_path))

        ds[0]   # shard 0
        assert ds._cached_shard_idx == 0

        ds[3]   # shard 1 (indices 3–4 map to shard 1)
        assert ds._cached_shard_idx == 1

        ds[1]   # back to shard 0
        assert ds._cached_shard_idx == 0

    def test_same_shard_no_reload(self, tmp_path):
        """Accessing items within the same shard does not reload from disk."""
        _write_shard(tmp_path, 0, n_examples=4)
        ds = GenomicWindowDataset(str(tmp_path))
        ds[0]  # Loads shard 0
        cached_id = id(ds._cached_data)
        ds[2]  # Still shard 0 — cached_data object should not change
        assert id(ds._cached_data) == cached_id

    def test_transform_applied(self, tmp_path):
        """Optional transform callable is applied to each sample."""
        _write_shard(tmp_path, 0, n_examples=2)

        def negate_sequence(sample):
            sample["sequence"] = sample["sequence"] * -1
            return sample

        ds = GenomicWindowDataset(str(tmp_path), transform=negate_sequence)
        assert (ds[0]["sequence"] <= 0).all()

    def test_dataloader_compatible(self, tmp_path):
        """Dataset works as a drop-in PyTorch DataLoader source."""
        _write_shard(tmp_path, 0, n_examples=4, seq_len=50, n_bins=5, n_tasks=2)
        ds = GenomicWindowDataset(str(tmp_path))
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["sequence"].shape == (2, 50, 4)
        assert batch["targets"].shape == (2, 5, 2)
        assert batch["mask"].shape == (2, 5, 2)

    def test_all_items_accessible(self, tmp_path):
        """Every index in [0, len(ds)) can be accessed without error."""
        _write_shard(tmp_path, 0, n_examples=3)
        _write_shard(tmp_path, 1, n_examples=2)
        ds = GenomicWindowDataset(str(tmp_path))
        for i in range(len(ds)):
            sample = ds[i]
            assert "sequence" in sample


# ---------------------------------------------------------------------------
# rasterize_window
# ---------------------------------------------------------------------------

class TestRasterizeWindow:

    def _df(self, intervals):
        """Build label DataFrame from (start, end, p_exon) triples."""
        rows = [{"Start": s, "End": e, "p_exon": p} for s, e, p in intervals]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Start", "End", "p_exon"])

    def test_empty_label_df_returns_zeros(self):
        targets, events = rasterize_window(0, 100, 10, self._df([]), n_bins=10)
        assert np.all(targets == 0.0)
        assert np.all(events == 0.0)

    def test_output_shapes(self):
        targets, events = rasterize_window(0, 100, 10, self._df([]), n_bins=10)
        assert targets.shape == (10,)
        assert events.shape == (10,)

    def test_single_full_bin(self):
        """Interval spanning exactly one bin → that bin's target = p_exon."""
        targets, _ = rasterize_window(0, 100, 10, self._df([(10, 20, 1.0)]), n_bins=10)
        assert targets[1] == pytest.approx(1.0)
        assert targets[0] == 0.0
        assert targets[2] == 0.0

    def test_interval_outside_window_ignored(self):
        targets, events = rasterize_window(0, 100, 10, self._df([(200, 300, 1.0)]), n_bins=10)
        assert np.all(targets == 0.0)
        assert np.all(events == 0.0)

    def test_partial_bin_overlap(self):
        """Interval overlapping only part of a bin — events records bp of overlap."""
        targets, events = rasterize_window(0, 100, 10, self._df([(5, 10, 1.0)]), n_bins=10)
        assert events[0] == pytest.approx(5.0)
        assert targets[0] == pytest.approx(1.0)

    def test_weighted_mean_across_two_intervals(self):
        """Two intervals with different p_exon in the same bin → overlap-weighted mean."""
        # Bin 0 = [0, 10). Left half p=0.8, right half p=0.2 → weighted mean = 0.5
        targets, events = rasterize_window(
            0, 100, 10,
            self._df([(0, 5, 0.8), (5, 10, 0.2)]),
            n_bins=10,
        )
        assert targets[0] == pytest.approx(0.5)
        assert events[0] == pytest.approx(10.0)

    def test_nan_p_exon_skipped(self):
        """Intervals with NaN p_exon do not contribute to any bin."""
        targets, events = rasterize_window(
            0, 100, 10,
            self._df([(0, 10, float("nan"))]),
            n_bins=10,
        )
        assert targets[0] == 0.0
        assert events[0] == 0.0

    def test_full_window_coverage(self):
        """Interval spanning the entire window fills all bins with p_exon."""
        targets, events = rasterize_window(0, 100, 10, self._df([(0, 100, 0.5)]), n_bins=10)
        assert np.allclose(targets, 0.5)
        assert np.allclose(events, 10.0)

    def test_non_zero_window_start(self):
        """Window offset from genome origin is handled correctly."""
        # window [1000, 1100), bin_size=10, single interval [1010, 1020)
        targets, _ = rasterize_window(1000, 1100, 10, self._df([(1010, 1020, 0.7)]), n_bins=10)
        assert targets[1] == pytest.approx(0.7)
        assert targets[0] == 0.0
