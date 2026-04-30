"""Fine-tune F5-TTS v1 Base model on Portuguese (PT-BR) dataset.

This script orchestrates the complete fine-tuning pipeline:
1. Prepares the dataset using F5-TTS's built-in prepare_csv_wavs
2. Downloads the pretrained checkpoint if needed
3. Launches training with accelerate

Usage:
    python scripts/train.py --dataset-csv data/processed/train.csv --output-dir checkpoints
    python scripts/train.py --resume  # Resume from last checkpoint
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = None  # Set by detect_python()


def detect_python() -> str:
    """Find the Python with F5-TTS installed."""
    # Check if f5_tts is importable in current Python
    try:
        import f5_tts  # noqa: F401

        return sys.executable
    except ImportError:
        pass

    # Check common locations
    candidates = [
        Path(sys.prefix) / "Scripts" / "python.exe",
        Path(sys.prefix) / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            result = subprocess.run(
                [str(candidate), "-c", "import f5_tts"],
                capture_output=True,
            )
            if result.returncode == 0:
                return str(candidate)

    print("Error: f5_tts not found. Install with: pip install f5-tts")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS for PT-BR")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train.csv",
        help="Path to prepared train.csv (audio_file|text format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "finetune_ptbr.yaml",
        help="Training config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size_per_gpu (frames)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation (if already done)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "no"],
        help="Mixed precision mode (default: fp16)",
    )
    return parser.parse_args()


def step_prepare_dataset(dataset_csv: Path, output_dir: Path) -> Path:
    """Run F5-TTS's prepare_csv_wavs to create Arrow dataset + vocab."""
    prepared_dir = output_dir / "prepared"
    if (prepared_dir / "raw.arrow").is_file():
        print(f"[prepare] Dataset already prepared at {prepared_dir}")
        return prepared_dir

    print(f"[prepare] Preparing dataset from {dataset_csv}...")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "f5_tts.train.datasets.prepare_csv_wavs",
        str(dataset_csv),
        str(prepared_dir),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("[prepare] Error: Dataset preparation failed.")
        sys.exit(1)

    print(f"[prepare] Done. Output: {prepared_dir}")
    return prepared_dir


def step_download_pretrained() -> Path:
    """Ensure the pretrained F5-TTS v1 Base checkpoint is available."""
    try:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1250000.safetensors",
        )
        print(f"[pretrained] Checkpoint: {ckpt_path}")
        return Path(ckpt_path)
    except Exception as e:
        print(f"[pretrained] Warning: Could not download pretrained model: {e}")
        print("[pretrained] Training will start from scratch.")
        return Path("")


def step_train(args: argparse.Namespace, prepared_dir: Path) -> None:
    """Launch F5-TTS training with accelerate."""
    vocab_path = prepared_dir / "vocab.txt"
    if not vocab_path.is_file():
        print(f"Error: vocab.txt not found at {vocab_path}")
        sys.exit(1)

    # Build accelerate launch command
    cmd = [
        sys.executable,
        "-m",
        "accelerate",
        "launch",
        f"--mixed_precision={args.mixed_precision}",
    ]

    # Find the train.py module path
    train_module = Path(sys.prefix) / "Lib" / "site-packages" / "f5_tts" / "train" / "train.py"
    if not train_module.is_file():
        # Try site-packages search
        import f5_tts

        train_module = Path(f5_tts.__file__).parent / "train" / "train.py"

    cmd.append(str(train_module))

    # Config overrides via Hydra
    cmd.append(f"--config-name=finetune_ptbr")
    cmd.append(f"--config-path={args.config.parent.resolve()}")

    # Override tokenizer path to point to our vocab
    cmd.append(f"++model.tokenizer_path={vocab_path}")
    cmd.append(f"++ckpts.save_dir={args.output_dir}")

    # Apply CLI overrides
    if args.epochs:
        cmd.append(f"++optim.epochs={args.epochs}")
    if args.batch_size:
        cmd.append(f"++datasets.batch_size_per_gpu={args.batch_size}")
    if args.learning_rate:
        cmd.append(f"++optim.learning_rate={args.learning_rate}")

    print(f"\n[train] Launching training...")
    print(f"[train] Command: {' '.join(cmd)}")
    print(f"[train] Config: {args.config}")
    print(f"[train] Vocab: {vocab_path}")
    print(f"[train] Output: {args.output_dir}")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        print(f"\n[train] Training exited with code {result.returncode}")
        sys.exit(result.returncode)

    print("\n[train] Training complete!")


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("F5-TTS PT-BR Fine-tuning Pipeline")
    print("=" * 60)
    print(f"  Dataset CSV:  {args.dataset_csv}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Config:       {args.config}")
    print(f"  Precision:    {args.mixed_precision}")
    print("=" * 60)

    # Step 1: Prepare dataset
    if not args.skip_prepare:
        prepared_dir = step_prepare_dataset(args.dataset_csv, args.output_dir)
    else:
        prepared_dir = args.output_dir / "prepared"
        if not (prepared_dir / "raw.arrow").is_file():
            print("Error: --skip-prepare but no prepared dataset found.")
            sys.exit(1)

    # Step 2: Ensure pretrained model is cached
    step_download_pretrained()

    # Step 3: Train
    step_train(args, prepared_dir)


if __name__ == "__main__":
    main()
