"""
Dataset classes for windowed genomic sequence data.

Provides a PyTorch Dataset that pairs one-hot encoded DNA sequences with
multitask bin-level targets (e.g. PSI labels, BigWig coverage), reading
from sharded .npz files produced by the data conversion pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from seq_tools.encoding import one_hot_encode
from seq_tools.fasta import FastaReader

logger = logging.getLogger(__name__)


class GenomicWindowDataset(Dataset):
    """
    PyTorch Dataset over sharded .npz files of windowed genomic data.

    Each shard contains:
        - ``sequence``: one-hot encoded DNA, shape (N, L, 4)
        - ``targets``: multitask labels, shape (N, n_bins, n_tasks)
        - ``mask``: binary mask, shape (N, n_bins, n_tasks)

    Loads shards lazily and caches the current shard in memory.

    Args:
        shard_dir: Directory containing ``shard_*.npz`` files.
        transform: Optional callable applied to (sequence, targets, mask).
    """

    def __init__(
        self,
        shard_dir: str,
        transform=None,
    ):
        self.shard_dir = Path(shard_dir)
        self.transform = transform

        self.shard_paths = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files in {shard_dir}")

        # Index: (shard_idx, index_within_shard) for each example
        self._index: list[tuple[int, int]] = []
        self._shard_sizes: list[int] = []

        for shard_idx, path in enumerate(self.shard_paths):
            # Peek at the first array to get shard size without loading all data
            with np.load(path) as data:
                n = data["sequence"].shape[0]
            self._shard_sizes.append(n)
            for j in range(n):
                self._index.append((shard_idx, j))

        # Cache
        self._cached_shard_idx: int = -1
        self._cached_data: dict = {}

        logger.info(
            "Loaded %d examples from %d shards in %s",
            len(self._index), len(self.shard_paths), shard_dir,
        )

    def _load_shard(self, shard_idx: int):
        if shard_idx != self._cached_shard_idx:
            self._cached_data = dict(np.load(self.shard_paths[shard_idx]))
            self._cached_shard_idx = shard_idx

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        shard_idx, local_idx = self._index[idx]
        self._load_shard(shard_idx)

        seq = self._cached_data["sequence"][local_idx]      # (L, 4)
        targets = self._cached_data["targets"][local_idx]    # (n_bins, n_tasks)
        mask = self._cached_data["mask"][local_idx]          # (n_bins, n_tasks)

        sample = {
            "sequence": torch.from_numpy(seq).float(),
            "targets": torch.from_numpy(targets).float(),
            "mask": torch.from_numpy(mask).float(),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def rasterize_window(
    window_start: int,
    window_end: int,
    bin_size: int,
    label_df,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize label intervals into fixed bins for a genomic window.

    For each bin, computes the overlap-weighted mean p_exon from
    overlapping label intervals.

    Args:
        window_start: Genomic start of the window.
        window_end: Genomic end of the window.
        bin_size: Size of each output bin in bp.
        label_df: DataFrame with columns [Start, End, p_exon].
        n_bins: Number of output bins.

    Returns:
        Tuple of (targets, events) arrays, each shape (n_bins,).
        ``targets[b]`` = overlap-weighted mean p_exon for bin b.
        ``events[b]``  = total bp of event overlap in bin b.
    """
    targets = np.zeros(n_bins, dtype=np.float32)
    events = np.zeros(n_bins, dtype=np.float32)

    if label_df.empty:
        return targets, events

    for _, row in label_df.iterrows():
        s = int(row.Start)
        e = int(row.End)

        if e <= window_start or s >= window_end:
            continue

        s_in = max(s, window_start)
        e_in = min(e, window_end)

        b0 = (s_in - window_start) // bin_size
        b1 = (e_in - window_start + bin_size - 1) // bin_size

        p = row.p_exon
        if np.isnan(p):
            continue

        for b in range(int(b0), int(b1)):
            bin_s = window_start + b * bin_size
            bin_e = bin_s + bin_size

            ov = max(0, min(e_in, bin_e) - max(s_in, bin_s))
            if ov <= 0:
                continue

            targets[b] += ov * float(p)
            events[b] += ov

    nz = events > 0
    targets[nz] /= events[nz]

    return targets, events
