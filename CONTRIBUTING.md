# Contributing

Thanks for your interest in contributing to **sequence-foundation-model-tools**!

## Development Setup

```bash
git clone https://github.com/jsdearbo/sequence_to_function_model_tools.git
cd sequence_to_function_model_tools
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

Tests are designed to run without GPU, genome files, or optional dependencies (gReLU, pysam, pyranges). Core module tests use only numpy, pandas, and torch.

## Project Structure

```
seq_tools/       Sequence encoding, intervals, FASTA, labels, variants
training/        Losses, heads, dataset, LoRA fine-tuning
interpret/       Attribution, ISM, TF-MoDISco (requires gReLU)
tests/           Unit tests (pytest)
examples/        Jupyter notebooks demonstrating usage
```

## Guidelines

- **Tests**: Add tests for new functionality in `tests/`. Follow existing naming (`test_<module>.py`).
- **Dependencies**: Keep core modules free of optional dependencies. Use `try/except` import guards for gReLU, pysam, and pyranges.
- **Style**: Follow existing code style. Type hints for public function signatures.
- **Notebooks**: Example notebooks in `examples/` should work with synthetic data (no external files).

## Adding a New Module

1. Add the source file to the appropriate package (`seq_tools/`, `training/`, or `interpret/`)
2. Add exports to the package `__init__.py`
3. Write tests in `tests/test_<module>.py`
4. Update `README.md` if the module adds a new capability

## Reporting Issues

Open a GitHub issue with:
- What you expected vs what happened
- Minimal code to reproduce
- Python/PyTorch version info
