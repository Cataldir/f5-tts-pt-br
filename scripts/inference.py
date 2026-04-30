"""Quick inference test for the fine-tuned F5-TTS PT-BR model.

Usage:
    python scripts/inference.py --text "Olá, como você está?" --output output.wav
    python scripts/inference.py --text "Olá" --ref-audio path/to/reference.wav
    python scripts/inference.py --checkpoint checkpoints/model_best.safetensors
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "model_last.safetensors"

# Sample texts for testing
SAMPLE_TEXTS = [
    "Olá, tudo bem? Meu nome é Ricardo e estou testando o modelo de síntese de voz.",
    "O aprendizado de máquina está transformando a forma como interagimos com computadores.",
    "Hoje o tempo está muito bom para dar uma caminhada no parque.",
    "A inteligência artificial já é parte do nosso dia a dia.",
    "Vamos construir o futuro juntos, com tecnologia e criatividade.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F5-TTS PT-BR Inference")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (default: runs all sample texts)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Output WAV file or directory (default: outputs/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to fine-tuned checkpoint (default: auto-detect latest)",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help="Reference audio for voice cloning (uses built-in default if not set)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="",
        help="Transcript of reference audio (auto-ASR if empty)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)",
    )
    return parser.parse_args()


def find_checkpoint(explicit: Path | None) -> str:
    """Find the best checkpoint to use."""
    if explicit and explicit.is_file():
        return str(explicit)

    # Search checkpoints directory
    ckpt_dir = REPO_ROOT / "checkpoints"
    if ckpt_dir.is_dir():
        # Prefer model_last, then highest numbered
        for name in ["model_last.safetensors", "model_last.pt"]:
            candidate = ckpt_dir / name
            if candidate.is_file():
                return str(candidate)

        # Find highest numbered checkpoint
        safetensors = sorted(ckpt_dir.glob("model_*.safetensors"))
        if safetensors:
            return str(safetensors[-1])

        pt_files = sorted(ckpt_dir.glob("model_*.pt"))
        if pt_files:
            return str(pt_files[-1])

    # No local checkpoint — return empty to use base model
    return ""


def find_vocab(checkpoint_path: str) -> str | None:
    """Find vocab.txt near the checkpoint or in prepared data."""
    if checkpoint_path:
        ckpt_dir = Path(checkpoint_path).parent
        vocab = ckpt_dir / "vocab.txt"
        if vocab.is_file():
            return str(vocab)

    # Check prepared data
    prepared_vocab = REPO_ROOT / "checkpoints" / "prepared" / "vocab.txt"
    if prepared_vocab.is_file():
        return str(prepared_vocab)

    return None


def run_inference(args: argparse.Namespace) -> None:
    from f5_tts.api import F5TTS

    checkpoint = find_checkpoint(args.checkpoint)
    vocab_path = find_vocab(checkpoint)

    print(f"Loading model...")
    print(f"  Checkpoint: {checkpoint or '(base model)'}")
    print(f"  Vocab: {vocab_path or '(default)'}")
    print(f"  Device: {args.device}")

    # Initialize model
    kwargs = {"device": args.device}
    if checkpoint:
        kwargs["ckpt_file"] = checkpoint
    if vocab_path:
        kwargs["vocab_file"] = vocab_path

    model = F5TTS(**kwargs)

    # Determine texts to synthesize
    texts = [args.text] if args.text else SAMPLE_TEXTS

    # Prepare output
    output = args.output
    if output.suffix == ".wav":
        output.parent.mkdir(parents=True, exist_ok=True)
        output_files = [output]
    else:
        output.mkdir(parents=True, exist_ok=True)
        output_files = [output / f"sample_{i:02d}.wav" for i in range(len(texts))]

    # Run inference
    ref_audio = str(args.ref_audio) if args.ref_audio else None
    ref_text = args.ref_text

    for i, (text, out_path) in enumerate(zip(texts, output_files)):
        print(f"\n[{i + 1}/{len(texts)}] Generating: {text[:60]}...")
        start = time.perf_counter()

        kwargs_infer = {
            "gen_text": text,
            "file_wave": str(out_path),
            "speed": args.speed,
            "remove_silence": True,
        }
        if ref_audio:
            kwargs_infer["ref_file"] = ref_audio
            kwargs_infer["ref_text"] = ref_text

        model.infer(**kwargs_infer)

        elapsed = time.perf_counter() - start
        print(f"  -> {out_path} ({elapsed:.2f}s)")

    print(f"\nDone! Generated {len(texts)} samples.")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
