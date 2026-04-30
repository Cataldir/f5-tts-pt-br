"""Download and extract Portuguese (PT-BR) audio dataset for F5-TTS fine-tuning.

Supports multiple dataset sources:
  - CML-TTS (default, public, no auth, already 24kHz)
  - Common Voice 17.0 (requires HF token + license acceptance)
  - Multilingual LibriSpeech / MLS (open, no auth required)

Uses Audio(decode=False) + soundfile to avoid torchcodec/FFmpeg DLL issues on Windows.

Usage:
    python scripts/download_dataset.py --output-dir data/raw --max-hours 100
    python scripts/download_dataset.py --dataset mls --max-hours 50
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PT-BR audio for F5-TTS")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save WAV files and metadata CSV",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cml_tts",
        choices=["cml_tts", "common_voice", "mls"],
        help="Dataset source (default: cml_tts, public and already 24kHz)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=100.0,
        help="Maximum hours of audio to download (default: 100)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum audio duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--min-upvotes",
        type=int,
        default=2,
        help="Minimum up_votes to include (Common Voice only)",
    )
    parser.add_argument(
        "--max-levenshtein",
        type=float,
        default=0.3,
        help="Maximum levenshtein ratio to filter noisy transcripts (CML-TTS only)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes using soundfile (avoids torchcodec dependency)."""
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _load_dataset_streaming(args: argparse.Namespace):
    """Load the chosen dataset in streaming mode with decode=False."""
    if args.dataset == "cml_tts":
        print(f"Loading CML-TTS - Portuguese ({args.split} split)...")
        print("  (Public dataset, no authentication, already 24kHz)")
        ds = load_dataset(
            "ylacombe/cml-tts",
            "portuguese",
            split=args.split,
            streaming=True,
        )
    elif args.dataset == "mls":
        print(f"Loading Multilingual LibriSpeech - Portuguese ({args.split} split)...")
        ds = load_dataset(
            "facebook/multilingual_librispeech",
            "portuguese",
            split=args.split,
            streaming=True,
        )
    else:
        print(f"Loading Common Voice 17.0 PT ({args.split} split)...")
        token = args.hf_token or __import__("os").environ.get("HF_TOKEN")
        if not token:
            print("ERROR: Common Voice requires a HuggingFace token.")
            print("  Set HF_TOKEN env var or use --hf-token, or use --dataset cml_tts.")
            sys.exit(1)
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "pt",
            split=args.split,
            streaming=True,
            token=token,
        )

    # Disable audio decoding to avoid torchcodec (broken on Windows)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def _get_text(sample: dict, dataset: str) -> str:
    """Extract transcript text from sample based on dataset format."""
    if dataset == "common_voice":
        return sample.get("sentence", "").strip()
    if dataset == "mls":
        return sample.get("transcript", sample.get("text", "")).strip()
    return sample.get("text", "").strip()


def _quality_filter(sample: dict, args: argparse.Namespace) -> bool:
    """Return True if sample passes quality filters."""
    if args.dataset == "common_voice":
        if sample.get("up_votes", 0) < args.min_upvotes:
            return False
    if args.dataset == "cml_tts":
        levenshtein = sample.get("levenshtein", 0)
        if levenshtein and levenshtein > args.max_levenshtein:
            return False
    return True


def download_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metadata.csv"
    max_seconds = args.max_hours * 3600
    total_seconds = 0.0
    written = 0
    skipped = 0

    ds = _load_dataset_streaming(args)

    print(f"Target: {args.max_hours} hours, duration: {args.min_duration}-{args.max_duration}s")
    print(f"Output: {output_dir}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "duration"])

        for sample in tqdm(ds, desc="Downloading", unit="samples"):
            if total_seconds >= max_seconds:
                print(f"\nReached target of {args.max_hours} hours.")
                break

            # Quality filter
            if not _quality_filter(sample, args):
                skipped += 1
                continue

            # Get transcript
            sentence = _get_text(sample, args.dataset)
            if not sentence:
                skipped += 1
                continue

            # Decode audio from raw bytes
            audio_bytes = sample["audio"].get("bytes")
            if not audio_bytes:
                skipped += 1
                continue

            try:
                array, sr = _decode_audio(audio_bytes)
            except Exception:
                skipped += 1
                continue

            duration = len(array) / sr

            # Duration filter
            if duration < args.min_duration or duration > args.max_duration:
                skipped += 1
                continue

            # Save WAV at original sample rate (cml-tts is already 24kHz)
            filename = f"ptbr_{written:06d}.wav"
            wav_path = wavs_dir / filename
            sf.write(str(wav_path), array, sr)

            # Write metadata
            writer.writerow([str(wav_path), sentence, f"{duration:.3f}"])

            total_seconds += duration
            written += 1

            if written % 500 == 0:
                hours = total_seconds / 3600
                print(f"  [{written} samples, {hours:.1f}h, {skipped} skipped]")

    hours = total_seconds / 3600
    print(f"\nDone! {written} samples, {hours:.1f} hours total.")
    print(f"Skipped {skipped} samples (quality/duration filters).")
    print(f"Metadata: {csv_path}")
    print(f"WAVs: {wavs_dir}")


def main() -> None:
    args = parse_args()
    try:
        download_dataset(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial data is still usable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
