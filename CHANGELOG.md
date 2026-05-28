# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
