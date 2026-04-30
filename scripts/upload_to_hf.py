"""Upload fine-tuned model to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py --checkpoint checkpoints/model_last.safetensors
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload F5-TTS PT-BR model to HuggingFace")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.safetensors or .pt)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Path to vocab.txt (auto-detected if not set)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Cataldir/F5-TTS-pt-br",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create or get repo
    print(f"Creating/accessing repo: {args.repo_id}")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # Prepare upload directory
    upload_dir = REPO_ROOT / ".cache" / "hf_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint
    ckpt_dest = upload_dir / "model_last.safetensors"
    shutil.copy2(args.checkpoint, ckpt_dest)
    print(f"  Checkpoint: {args.checkpoint} -> {ckpt_dest.name}")

    # Copy vocab
    vocab_src = args.vocab
    if not vocab_src:
        # Auto-detect
        for candidate in [
            args.checkpoint.parent / "vocab.txt",
            REPO_ROOT / "checkpoints" / "prepared" / "vocab.txt",
        ]:
            if candidate.is_file():
                vocab_src = candidate
                break

    if vocab_src and vocab_src.is_file():
        shutil.copy2(vocab_src, upload_dir / "vocab.txt")
        print(f"  Vocab: {vocab_src}")

    # Copy model card
    model_card = REPO_ROOT / "MODEL_CARD.md"
    if model_card.is_file():
        shutil.copy2(model_card, upload_dir / "README.md")

    # Copy config
    config = REPO_ROOT / "configs" / "finetune_ptbr.yaml"
    if config.is_file():
        shutil.copy2(config, upload_dir / "config.yaml")

    # Upload
    print(f"\nUploading to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload F5-TTS PT-BR fine-tuned model",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")

    # Cleanup
    shutil.rmtree(upload_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
