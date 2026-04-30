"""Prepare downloaded audio data into F5-TTS training format.

Takes the raw metadata.csv (audio_file|text) and produces the directory
structure expected by F5-TTS's prepare_csv_wavs:
  - Resamples audio to 24kHz mono WAV
  - Normalizes text (basic Portuguese normalization)
  - Produces final metadata CSV with absolute paths
  - Splits into train/val sets

Usage:
    python scripts/prepare_dataset.py --input-csv data/raw/metadata.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import unicodedata
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

TARGET_SR = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare F5-TTS PT-BR dataset")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/metadata.csv"),
        help="Input metadata CSV from download step",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Fraction of data for validation (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def normalize_portuguese_text(text: str) -> str:
    """Basic Portuguese text normalization for TTS training."""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Expand common abbreviations
    abbrevs = {
        "sr.": "senhor",
        "sra.": "senhora",
        "dr.": "doutor",
        "dra.": "doutora",
        "prof.": "professor",
        "profa.": "professora",
        "etc.": "etcétera",
        "nº": "número",
        "n.º": "número",
    }
    for abbr, expansion in abbrevs.items():
        text = text.replace(abbr, expansion)

    # Remove unusual characters but keep Portuguese diacritics
    text = re.sub(r"[^\w\s\u00C0-\u00FF.,!?;:\-'\"()]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def resample_audio(input_path: Path, output_path: Path) -> float | None:
    """Resample audio to target sample rate and mono. Returns duration or None on failure."""
    try:
        import numpy as np

        data, sr = sf.read(str(input_path))

        # Convert to mono if stereo
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if needed
        if sr != TARGET_SR:
            try:
                import torchaudio
                import torch

                tensor = torch.from_numpy(data).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                tensor = resampler(tensor)
                data = tensor.squeeze(0).numpy()
            except ImportError:
                # Fallback: simple linear interpolation (less quality)
                duration = len(data) / sr
                new_len = int(duration * TARGET_SR)
                x_old = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, new_len)
                data = np.interp(x_new, x_old, data)

        # Normalize volume
        peak = np.abs(data).max()
        if peak > 0:
            data = data * (0.95 / peak)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), data, TARGET_SR)

        return len(data) / TARGET_SR
    except Exception as e:
        print(f"  Warning: failed to process {input_path}: {e}")
        return None


def prepare_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Read input CSV
    input_csv = args.input_csv.resolve()
    if not input_csv.is_file():
        print(f"Error: {input_csv} not found. Run download_dataset.py first.")
        return

    print(f"Reading {input_csv}...")
    rows: list[tuple[str, str]] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], row[1]))

    print(f"Found {len(rows)} samples. Processing...")

    # Process each sample
    processed: list[tuple[str, str, float]] = []
    for i, (audio_path, text) in enumerate(tqdm(rows, desc="Processing")):
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            continue

        # Normalize text
        norm_text = normalize_portuguese_text(text)
        if len(norm_text) < 5:
            continue

        # Resample and save
        out_wav = wavs_dir / f"ptbr_{i:06d}.wav"
        duration = resample_audio(audio_path, out_wav)
        if duration is None:
            continue

        processed.append((str(out_wav), norm_text, duration))

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(processed)

    val_count = max(1, int(len(processed) * args.val_ratio))
    val_set = processed[:val_count]
    train_set = processed[val_count:]

    # Write train CSV
    train_csv = output_dir / "train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for audio_file, text, _ in train_set:
            writer.writerow([audio_file, text])

    # Write val CSV
    val_csv = output_dir / "val.csv"
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for audio_file, text, _ in val_set:
            writer.writerow([audio_file, text])

    total_hours = sum(d for _, _, d in processed) / 3600
    train_hours = sum(d for _, _, d in train_set) / 3600
    val_hours = sum(d for _, _, d in val_set) / 3600

    print(f"\nDone!")
    print(f"  Total: {len(processed)} samples, {total_hours:.1f} hours")
    print(f"  Train: {len(train_set)} samples, {train_hours:.1f} hours -> {train_csv}")
    print(f"  Val:   {len(val_set)} samples, {val_hours:.1f} hours -> {val_csv}")
    print(f"  WAVs:  {wavs_dir} (24kHz mono)")


def main() -> None:
    args = parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
