# F5-TTS PT-BR: Brazilian Portuguese Voice Cloning & TTS

Fine-tuned [F5-TTS v1 Base](https://github.com/SWivid/F5-TTS) for Brazilian Portuguese (PT-BR) with zero-shot voice cloning support.

## Highlights

- **Fast inference**: ~10-30x faster than XTTS v2 (RTF ~0.04-0.15 on RTX 3060)
- **Voice cloning**: Clone any voice from 5-30s reference audio
- **Flow matching**: Non-autoregressive architecture = no character limits, consistent quality
- **Trained on**: Mozilla Common Voice 17.0 PT-BR (100+ hours, quality-filtered)
- **Base model**: F5-TTS v1 Base (300M params, fits in 4-6GB VRAM)

## Quick Start

### Installation

```bash
git clone https://github.com/Cataldir/f5-tts-pt-br.git
cd f5-tts-pt-br
pip install -e .
```

### Inference (using pre-trained model from HuggingFace)

```python
from f5_tts.api import F5TTS

model = F5TTS(device="cuda", ckpt_file="hf://Cataldir/F5-TTS-pt-br/model_last.safetensors")
model.infer(
    ref_file="reference.wav",  # 5-30s of target voice
    ref_text="Transcrição do áudio de referência.",
    gen_text="Qualquer texto em português que você quiser sintetizar.",
    file_wave="output.wav",
    speed=1.0,
)
```

### CLI Inference

```bash
python scripts/inference.py \
    --text "Olá, tudo bem? Este é um teste do modelo." \
    --ref-audio path/to/reference.wav \
    --ref-text "Transcrição do áudio de referência." \
    --output output.wav
```

## Training from Scratch

If you want to reproduce the fine-tuning or train on your own data:

### 1. Download Dataset

```bash
# Downloads Common Voice PT-BR (streaming, ~100 hours)
# Requires HuggingFace account with Common Voice agreement
python scripts/download_dataset.py \
    --output-dir data/raw \
    --max-hours 100 \
    --min-upvotes 2 \
    --hf-token YOUR_HF_TOKEN
```

### 2. Prepare Dataset

```bash
# Resamples to 24kHz, normalizes text, splits train/val
python scripts/prepare_dataset.py \
    --input-csv data/raw/metadata.csv \
    --output-dir data/processed
```

### 3. Fine-tune

```bash
# Single GPU training (RTX 3060 6GB compatible)
python scripts/train.py \
    --dataset-csv data/processed/train.csv \
    --output-dir checkpoints \
    --mixed-precision fp16
```

Training takes approximately 12-24 hours on an RTX 3060 6GB with 100 hours of data.

### 4. Test

```bash
python scripts/inference.py --checkpoint checkpoints/model_last.safetensors
```

### 5. Upload to HuggingFace

```bash
python scripts/upload_to_hf.py --checkpoint checkpoints/model_last.safetensors
```

## Configuration

The training config is at [`configs/finetune_ptbr.yaml`](configs/finetune_ptbr.yaml).

Key settings optimized for 6GB VRAM:
- `batch_size_per_gpu: 9600` (frames, reduced from 38400)
- `grad_accumulation_steps: 4` (simulates larger batch)
- `checkpoint_activations: True` (gradient checkpointing)
- `bnb_optimizer: True` (8-bit AdamW)
- `learning_rate: 1.0e-5` (lower for fine-tuning stability)

## Model Architecture

| Component | Value |
|-----------|-------|
| Base model | F5-TTS v1 Base |
| Architecture | DiT (Diffusion Transformer) |
| Parameters | ~300M |
| Sample rate | 24kHz |
| Mel channels | 100 |
| Vocoder | Vocos |
| Tokenizer | Custom (Portuguese characters) |

## Dataset

Trained on Mozilla Common Voice 17.0 Portuguese subset:
- Quality filtered (≥2 upvotes)
- Duration filtered (2-30 seconds)
- Resampled to 24kHz mono
- Text normalized for Portuguese

## Integration with Video Pipeline

This model integrates with the narration pipeline by setting the checkpoint path:

```bash
export F5TTS_PTBR_CKPT=/path/to/checkpoints/model_last.safetensors
export NARRATION_PROVIDER=f5tts
export NARRATION_LANGUAGE=pt
```

## License

- **Code**: MIT License
- **Model weights**: CC-BY-NC-4.0 (inherited from F5-TTS base model training data)
- **Training data**: Mozilla Common Voice (CC0)

## Citation

```bibtex
@misc{cataldi2026f5ttsptbr,
    title={F5-TTS PT-BR: Brazilian Portuguese Voice Cloning with Flow Matching},
    author={Ricardo Cataldi},
    year={2026},
    url={https://github.com/Cataldir/f5-tts-pt-br}
}
```

## Acknowledgements

- [F5-TTS](https://github.com/SWivid/F5-TTS) by Yushen Chen et al.
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for the training data
- Community fine-tuners (French, Spanish, Italian, etc.) whose work inspired this project
