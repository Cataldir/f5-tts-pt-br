# Architecture

This document describes the pipeline architecture, design decisions, and key trade-offs.

---

## Pipeline Overview

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Download   │────▶│   Prepare   │────▶│    Train    │────▶│  Inference   │
│  (CV 17.0)  │     │  (24kHz)    │     │ (fine-tune) │     │   (test)     │
└─────────────┘     └─────────────┘     └─────────────┘     └──────────────┘
       │                    │                   │                    │
       ▼                    ▼                   ▼                    ▼
   data/raw/          data/processed/     checkpoints/          outputs/
   metadata.csv       train.csv           model_last.safet...   sample_XX.wav
   wavs/              val.csv
                      wavs/ (24kHz)
```

---

## Design Decisions

### ADR-001: F5-TTS as Base Architecture

**Context**: Need a TTS model that supports Portuguese, fits in 6GB VRAM, and delivers fast inference with voice cloning.

**Decision**: Fine-tune F5-TTS v1 Base (300M params, flow matching, DiT backbone).

**Alternatives considered**:
| Model | VRAM | Speed | PT-BR | Verdict |
|-------|------|-------|-------|---------|
| Fish Speech S2 (4B) | 24GB | Fast | Yes | Too large for target GPU |
| XTTS v2 | 4GB | Slow (2-3x RTF) | Yes | Unacceptable latency |
| CosyVoice 3 | 3GB | Fast | **No** | No Portuguese support |
| F5-TTS Base (300M) | 4GB | Fast (0.04 RTF) | Finetune | **Selected** |

**Consequences**: Requires fine-tuning effort but delivers optimal speed/quality/VRAM balance.

---

### ADR-002: Common Voice as Training Data

**Context**: Need freely available, high-quality PT-BR speech data with transcriptions.

**Decision**: Mozilla Common Voice 17.0 Portuguese subset, quality-filtered (≥2 upvotes, 2-30s duration).

**Rationale**:
- CC0 license (no restrictions on derivative models)
- 100+ hours available after filtering
- Diverse speakers (improves generalization)
- Streaming download (no full dataset needed locally)

---

### ADR-003: 6GB VRAM Training Strategy

**Context**: Training target is an RTX 3060 Laptop GPU with 6GB VRAM.

**Decision**: Combination of:
1. Gradient checkpointing (`checkpoint_activations: True`)
2. 8-bit AdamW optimizer (`bnb_optimizer: True`)
3. Reduced batch size (9600 frames vs 38400 default)
4. Gradient accumulation (4 steps, simulating 4x larger batch)
5. FP16 mixed precision

**Trade-off**: ~2-3x slower training per epoch, but fits in memory without quality loss.

---

### ADR-004: Pinyin Tokenizer Reuse

**Context**: F5-TTS uses a pinyin-based tokenizer for Chinese+English. Portuguese needs different characters.

**Decision**: Use F5-TTS's built-in `prepare_csv_wavs` which generates a custom `vocab.txt` from the training data, then set `tokenizer: custom` in the config.

**Mechanism**: The preparation step scans all text, builds a character vocabulary, and the training script uses this vocab file via `tokenizer_path`.

---

## File Structure

```text
f5-tts-pt-br/
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint + test on push/PR
├── configs/
│   └── finetune_ptbr.yaml      # Training hyperparameters
├── docs/
│   └── architecture.md         # This file
├── scripts/
│   ├── download_dataset.py     # Step 1: Stream Common Voice PT
│   ├── prepare_dataset.py      # Step 2: Resample + normalize
│   ├── train.py                # Step 3: Fine-tune with accelerate
│   ├── inference.py            # Step 4: Test voice cloning
│   └── upload_to_hf.py        # Step 5: Push to HuggingFace
├── tests/
│   └── test_pipeline.py        # Smoke tests
├── .pre-commit-config.yaml     # Code quality hooks
├── Makefile                    # Task runner
├── pyproject.toml              # Package definition
├── CONTRIBUTING.md             # How to contribute
├── MODEL_CARD.md               # HuggingFace model card
├── README.md                   # Project overview
└── run_pipeline.ps1            # Windows one-click orchestration
```

---

## Integration with Video Pipeline

Once trained, the model integrates with the narration pipeline in the `i` repository:

```python
# src/automation/video-pipeline/narration/providers/f5tts.py
# Automatically downloads from HuggingFace: Cataldir/F5-TTS-pt-br
```

Environment variables:
- `F5TTS_PTBR_CKPT`: Override checkpoint path (local file)
- `NARRATION_PROVIDER=f5tts`: Select F5-TTS provider
- `NARRATION_LANGUAGE=pt`: Set language to Portuguese

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| RTF (Real-Time Factor) | < 0.2 | On RTX 3060 |
| Voice similarity (SIM) | > 0.70 | vs reference audio |
| Word Error Rate | < 5% | On PT-BR test set |
| VRAM (inference) | < 5GB | Single sample |
| Training time | < 24h | 100h data, RTX 3060 |
