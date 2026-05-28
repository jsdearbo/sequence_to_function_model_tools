# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `seq_tools/visualization.py` — matplotlib-based plotting utilities requiring no grelu or genome files:
  - `plot_ism_heatmap` — ISM log2FC result as a diverging-colormap heatmap (ACGT rows)
  - `plot_prediction_track` — 1D bin-level model prediction as a filled area chart
  - `plot_gene_model` — exon/intron track from a soft-label DataFrame, colored by p_exon
  - `plot_attribution` — per-position saliency bar chart or tangermeme logo (falls back gracefully)
  - `multi_track_figure` — stacked, shared-x-axis multi-panel figure layout
- All five visualization functions exposed in `seq_tools.__init__.__all__`
- `examples/04_interpretation_workflow.ipynb` — end-to-end interpretation notebook:
  - ISM with `seq_tools.variant` (no grelu required)
  - API tour of `interpret.ism`, `interpret.attribution`, `interpret.modisco`
  - Synthetic attribution and gene model visualization
  - Composite multi-track figure combining all tracks
- `tests/test_labels.py` — comprehensive tests for the splicing label generation pipeline:
  - `_disjoint_labels_with_priority`: priority ordering, exon/intron segmentation, PSI override, multi-chrom
  - `collapse_exon_coords_weighted`: read-weighted PSI aggregation vs. naive average
  - `generate_soft_labels`: schema, exon/intron assignment, rMATS PSI integration (pyranges-gated)
- `tests/test_dataset.py` — tests for `GenomicWindowDataset` and `rasterize_window`:
  - Shard loading, caching behavior, transform support, DataLoader compatibility
  - Bin boundary alignment, partial overlap weighting, NaN handling

### Changed
- `training/multitask_head.py` — module docstring now includes a design note explaining the output-collapse problem and how per-task `BatchNorm1d` in `SplitHead` prevents it
- `pyproject.toml` — added `matplotlib >= 3.5` as a core dependency

## [0.1.0] — 2025-01-01

### Added
- Sequence encoding: one-hot encoding, genome-aware fetching, reverse complement
- Genomic intervals: chromosome tiling, window centering, output bin mapping
- Splicing label generation: hard/soft PSI labels from rMATS + StringTie with priority-based disjoint segmentation
- Variant construction: SNV generation and scoring for in silico mutagenesis
- Custom losses: PSI-aware, Bhattacharyya, masked MSE/Poisson for multitask training
- Multitask heads: nonlinear, split-head, and cell-type conditional architectures
- LoRA fine-tuning: low-rank adaptation for Conv1d and Linear layers with weight merging
- Interpretability: attribution wrappers, ISM pipelines, TF-MoDISco integration
- CI via GitHub Actions with Python 3.10–3.12 matrix
