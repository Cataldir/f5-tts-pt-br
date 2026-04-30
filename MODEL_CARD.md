---
license: cc-by-nc-4.0
language:
  - pt
tags:
  - tts
  - text-to-speech
  - voice-cloning
  - f5-tts
  - portuguese
  - pt-br
  - flow-matching
library_name: f5-tts
pipeline_tag: text-to-speech
datasets:
  - mozilla-foundation/common_voice_17_0
base_model: SWivid/F5-TTS
---

# F5-TTS PT-BR: Brazilian Portuguese Voice Cloning

Fine-tuned [F5-TTS v1 Base](https://github.com/SWivid/F5-TTS) for Brazilian Portuguese with zero-shot voice cloning.

## Usage

```python
from f5_tts.api import F5TTS

model = F5TTS(
    device="cuda",
    ckpt_file="hf://Cataldir/F5-TTS-pt-br/model_last.safetensors",
    vocab_file="hf://Cataldir/F5-TTS-pt-br/vocab.txt",
)
model.infer(
    ref_file="reference.wav",
    ref_text="Transcrição do áudio de referência.",
    gen_text="Qualquer texto em português que você quiser sintetizar.",
    file_wave="output.wav",
)
```

## Model Details

| Key | Value |
|-----|-------|
| Base model | F5-TTS v1 Base (300M params) |
| Architecture | DiT + Flow Matching |
| Training data | Common Voice 17.0 PT (~100 hours) |
| Sample rate | 24kHz |
| Vocoder | Vocos |
| VRAM required | ~4GB inference |

## Training

Fine-tuned from F5-TTS v1 Base checkpoint using:
- Common Voice 17.0 Portuguese (quality-filtered, 100+ hours)
- 50 epochs, lr=1e-5, fp16
- Single RTX 3060 6GB GPU

## Config

```yaml
dim: 1024
depth: 22
heads: 16
ff_mult: 2
text_dim: 512
conv_layers: 4
```

## Limitations

- Quality depends on reference audio quality
- Best with 10-30 second reference clips
- Inherits F5-TTS base model limitations on very long text

## License

CC-BY-NC-4.0 (inherited from F5-TTS training data license)

## Citation

```bibtex
@misc{cataldi2026f5ttsptbr,
    title={F5-TTS PT-BR: Brazilian Portuguese Voice Cloning with Flow Matching},
    author={Ricardo Cataldi},
    year={2026},
    url={https://github.com/Cataldir/f5-tts-pt-br}
}
```
