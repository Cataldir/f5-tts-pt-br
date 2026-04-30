"""Basic smoke tests for the F5-TTS PT-BR pipeline."""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def test_scripts_exist():
    """All pipeline scripts are present."""
    expected = [
        "download_dataset.py",
        "prepare_dataset.py",
        "train.py",
        "inference.py",
        "upload_to_hf.py",
    ]
    for script in expected:
        assert (SCRIPTS_DIR / script).is_file(), f"Missing: scripts/{script}"


def test_config_exists():
    """Training config is present and valid YAML."""
    import yaml

    config_path = REPO_ROOT / "configs" / "finetune_ptbr.yaml"
    assert config_path.is_file()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["model"]["name"] == "F5TTS_v1_Base"
    assert config["model"]["mel_spec"]["target_sample_rate"] == 24000
    assert config["datasets"]["batch_size_per_gpu"] <= 12800  # Must fit in 6GB


def test_text_normalization():
    """Portuguese text normalization works correctly."""
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    from prepare_dataset import normalize_portuguese_text

    # Basic normalization
    assert normalize_portuguese_text("Olá Mundo!") == "olá mundo!"

    # Abbreviation expansion
    assert "senhor" in normalize_portuguese_text("O Sr. Silva chegou.")
    assert "professor" in normalize_portuguese_text("O Prof. falou.")

    # Whitespace normalization
    assert normalize_portuguese_text("  espaços   extras  ") == "espaços extras"

    # Preserves Portuguese diacritics
    result = normalize_portuguese_text("Ação, coração, não, pão")
    assert "ação" in result
    assert "coração" in result


def test_f5tts_importable():
    """F5-TTS package is installed and importable."""
    import f5_tts  # noqa: F401
    from f5_tts.api import F5TTS  # noqa: F401
