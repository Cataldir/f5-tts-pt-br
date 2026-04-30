# Contributing to F5-TTS PT-BR

Thank you for considering contributing to this project! This guide explains how to set up your environment, run tests, and submit changes.

---

## Principles

1. **Reproducibility** — `clone → install → run` must work without manual adjustments.
2. **Clarity** — Any contributor should understand the pipeline without asking the author.
3. **Quality** — All code must pass linting, formatting, and tests before merging.
4. **Lightweight** — No large files (models, datasets, audio) in version control.

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA-capable GPU for training (6GB+ VRAM)
- FFmpeg installed and on PATH

### Install

```bash
git clone https://github.com/Cataldir/f5-tts-pt-br.git
cd f5-tts-pt-br

# Using uv (recommended)
uv pip install -e ".[train,dev]"

# Or using pip
pip install -e ".[train,dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Development Workflow

### Running checks locally

```bash
# Lint and format
make lint
make format

# Run tests
make test

# Or run pre-commit on all files
pre-commit run --all-files
```

### Adding new scripts

1. Place scripts in `scripts/` directory.
2. Add a docstring with usage instructions at the top of the file.
3. Use `argparse` for CLI arguments with sensible defaults.
4. Add a corresponding test in `tests/`.

### Modifying the training config

1. Edit `configs/finetune_ptbr.yaml`.
2. Document the rationale in the PR description.
3. Keep VRAM constraints in mind (target: RTX 3060 6GB).

---

## What NOT to commit

- Model checkpoints (`.pt`, `.safetensors`, `.bin`)
- Audio files (`.wav`, `.mp3`)
- Downloaded datasets (`data/raw/`, `data/processed/`)
- HuggingFace cache (`.cache/`)
- Generated outputs (`outputs/`)

These are all in `.gitignore`. Use HuggingFace Hub for model distribution.

---

## Pull Request Checklist

- [ ] Code passes `make lint` without errors.
- [ ] Tests pass with `make test`.
- [ ] New functionality has corresponding tests.
- [ ] PR description explains the "why" not just the "what".
- [ ] No large binary files added to the repository.
- [ ] Config changes are documented with rationale.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the pipeline design and decision log.

---

## Code Style

- **Formatter**: ruff (line length 100)
- **Type hints**: Required on all public functions
- **Docstrings**: Google style, required on modules and public functions
- **Imports**: Sorted by ruff (isort-compatible)

---

## Releasing a Model

1. Train locally or on a cloud GPU.
2. Test with `make inference`.
3. Upload with `make upload` (requires HuggingFace token).
4. Update `MODEL_CARD.md` with training details and metrics.
5. Tag the commit: `git tag v0.x.0 && git push --tags`.
