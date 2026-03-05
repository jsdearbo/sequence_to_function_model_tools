# sequence-foundation-model-tools

Reusable tools for training, fine-tuning, and interpreting genomic sequence-to-function models.

Built around [gReLU](https://github.com/Genentech/gReLU) and designed for Borzoi-class foundation models. Provides:

- **Sequence encoding** — one-hot encoding, genome-aware fetching, reverse complement
- **Genomic intervals** — chromosome tiling, window centering, output bin mapping
- **Splicing label generation** — hard/soft PSI labels from rMATS + StringTie, with priority-based disjoint segmentation
- **Variant construction** — SNV generation and scoring for in silico mutagenesis
- **Custom losses** — PSI-aware, Bhattacharyya, masked MSE/Poisson for multitask training
- **Multitask heads** — nonlinear, split-head, and cell-type conditional architectures
- **LoRA fine-tuning** — lightweight low-rank adaptation for Conv1d and Linear layers with weight merging
- **Interpretability** — attribution wrappers, ISM pipelines, TF-MoDISco integration

## Installation

```bash
pip install -e .
```

Core dependencies (numpy, pandas, torch) are required. Optional dependencies for full functionality:

```bash
pip install grelu pysam pyranges
```

## Structure

```
seq_tools/              Core sequence utilities
├── encoding.py         One-hot encoding, reverse complement, genome-aware fetch
├── intervals.py        Genomic interval generation, centering, bin conversion
├── fasta.py            FASTA reading, windowed iteration
├── labels.py           Splicing label generation (rMATS + StringTie)
└── variant.py          SNV generation and variant scoring

training/               Model training infrastructure
├── losses.py           PSI, Bhattacharyya, masked MSE/Poisson losses
├── multitask_head.py   Nonlinear, split, conditional, and calibrated heads
├── dataset.py          Windowed genomic dataset from sharded .npz files
└── finetune.py         LoRA injection, merging, and parameter counting

interpret/              Model interpretability
├── attribution.py      Element-level attribution with prediction aggregation
├── ism.py              In silico mutagenesis pipeline
└── modisco.py          TF-MoDISco motif discovery integration
```

## Quick Examples

### Encode a sequence

```python
from seq_tools.encoding import one_hot_encode, reverse_complement

seq = "ACGTACGTACGT"
encoded = one_hot_encode(seq)        # (12, 4) float32
rc = reverse_complement(seq)          # "ACGTACGTACGT"
```

### Generate tiled intervals

```python
from seq_tools.intervals import generate_intervals
import pandas as pd

chroms = pd.DataFrame({"chrom": ["chr1", "chr2"], "length": [248956422, 242193529]})
intervals = generate_intervals(chroms, seq_len=524288, stride=174762)
```

### Apply LoRA to a model

```python
from training.finetune import LoRAConfig, inject_lora, merge_lora, count_trainable_params

cfg = LoRAConfig(rank=8, alpha=16)
inject_lora(model.model, cfg)
print(count_trainable_params(model.model))
# {'trainable': 2_400_000, 'total': 300_000_000, 'pct_trainable': 0.8}

# After training, merge for zero-overhead inference:
merge_lora(model.model)
```

### Custom multitask heads

```python
from training.multitask_head import SplitHead, ConditionalHead

# Per-task normalization prevents output collapse
head = SplitHead(in_channels=1920, hidden=512, task_hidden=256, out_channels=3)

# Cell-type conditioned predictions
cond_head = ConditionalHead(in_channels=1920, n_celltypes=5, out_channels=1)
pred = cond_head(trunk_features, cell_type_id=torch.tensor([2]))
```

### Generate splicing labels

```python
from seq_tools.labels import load_stringtie_gtf, load_rmats_se, generate_soft_labels

gtf = load_stringtie_gtf("stringtie.gtf")
se = load_rmats_se("SE.MATS.JCEC.txt", min_coverage=20)
labels = generate_soft_labels(gtf, se)
# DataFrame: [Chromosome, Start, End, p_exon, confidence, Strand]
```

## Design Philosophy

This toolkit extends [gReLU](https://github.com/Genentech/gReLU) rather than replacing it. gReLU provides the model loading, dataset primitives, and core prediction infrastructure. This repo adds the task-specific components needed for splicing regulation research:

- **Label generation** that integrates rMATS alternative splicing calls with transcript structure
- **Loss functions** designed for the unique properties of PSI prediction
- **Head architectures** that handle the multitask nature of cell-type-specific splicing
- **LoRA** adapted for the mixed Conv1d/Linear + transformer architecture of Borzoi

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy, Pandas

Optional:
- [gReLU](https://github.com/Genentech/gReLU) — model loading, attribution, ISM
- [pysam](https://github.com/pysam-developers/pysam) — FASTA reading
- [PyRanges](https://github.com/pyranges/pyranges) — splicing label generation

## License

MIT
