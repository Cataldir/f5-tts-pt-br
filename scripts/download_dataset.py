"""Download and extract Common Voice PT-BR dataset for F5-TTS fine-tuning.

Uses the HuggingFace datasets library to stream Common Voice 17.0 Portuguese
subset. Filters for high-quality samples and exports as WAV + metadata CSV.

Usage:
    python scripts/download_dataset.py --output-dir data/raw --max-hours 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Common Voice PT-BR for F5-TTS")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save WAV files and metadata CSV",
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
        help="Minimum up_votes to include a sample (default: 2)",
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


def download_common_voice(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metadata.csv"
    max_seconds = args.max_hours * 3600
    total_seconds = 0.0
    written = 0
    skipped = 0

    print(f"Loading Common Voice 17.0 PT ({args.split} split)...")
    print(f"Target: {args.max_hours} hours, duration range: {args.min_duration}-{args.max_duration}s")
    print(f"Output: {output_dir}")

    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "pt",
        split=args.split,
        streaming=True,
        token=args.hf_token,
        trust_remote_code=True,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])

        for sample in tqdm(ds, desc="Downloading", unit="samples"):
            if total_seconds >= max_seconds:
                print(f"\nReached target of {args.max_hours} hours.")
                break

            # Quality filter: minimum upvotes
            up_votes = sample.get("up_votes", 0)
            if up_votes < args.min_upvotes:
                skipped += 1
                continue

            # Extract audio array and sample rate
            audio = sample["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]
            duration = len(array) / sr

            # Duration filter
            if duration < args.min_duration or duration > args.max_duration:
                skipped += 1
                continue

            # Get transcript
            sentence = sample.get("sentence", "").strip()
            if not sentence:
                skipped += 1
                continue

            # Save WAV
            filename = f"cv_pt_{written:06d}.wav"
            wav_path = wavs_dir / filename
            sf.write(str(wav_path), array, sr)

            # Write metadata (absolute path for F5-TTS compatibility)
            writer.writerow([str(wav_path), sentence])

            total_seconds += duration
            written += 1

            if written % 500 == 0:
                hours = total_seconds / 3600
                print(f"  [{written} samples, {hours:.1f}h collected, {skipped} skipped]")

    hours = total_seconds / 3600
    print(f"\nDone! {written} samples, {hours:.1f} hours total.")
    print(f"Skipped {skipped} samples (quality/duration filters).")
    print(f"Metadata: {csv_path}")
    print(f"WAVs: {wavs_dir}")


def main() -> None:
    args = parse_args()
    try:
        download_common_voice(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial data is still usable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
