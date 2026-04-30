"""Generate a talking avatar video from F5-TTS audio + reference photo(s).

Uses EchoMimic to animate a reference face image with generated PT-BR speech.
Pipeline: reference photo + audio wav → talking head video (mp4).

Usage:
    python scripts/generate_avatar.py --audio outputs/sample_00.wav --image photos/me.jpg
    python scripts/generate_avatar.py --audio outputs/ --image photos/me.jpg --output videos/
    python scripts/generate_avatar.py --text "Olá, mundo!" --image photos/me.jpg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate talking avatar from audio + photo")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Input WAV file or directory of WAVs (from F5-TTS inference)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize first (runs F5-TTS inference, then animates)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Reference face image (front-facing portrait photo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "videos",
        help="Output video file or directory (default: videos/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="F5-TTS checkpoint for --text mode (auto-detect if not set)",
    )
    parser.add_argument(
        "--echomimic-dir",
        type=Path,
        default=None,
        help="Path to EchoMimic repo (default: ../EchoMimic or auto-clone)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output video width (default: 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output video height (default: 512)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Denoising steps (default: 30, lower=faster)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale (default: 2.5)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output video FPS (default: 24)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    return parser.parse_args()


def find_echomimic(explicit: Path | None) -> Path:
    """Locate EchoMimic installation."""
    if explicit and explicit.is_dir():
        return explicit

    # Check common locations
    candidates = [
        REPO_ROOT.parent / "EchoMimic",
        REPO_ROOT.parent / "echomimic",
        REPO_ROOT.parent / "echomimic_v2",
        Path.home() / "Github" / "EchoMimic",
    ]
    for candidate in candidates:
        if (candidate / "infer_audio2vid.py").is_file():
            return candidate

    return Path("")


def setup_echomimic(target_dir: Path) -> Path:
    """Clone and set up EchoMimic if not present."""
    if (target_dir / "infer_audio2vid.py").is_file():
        print(f"[avatar] EchoMimic found at {target_dir}")
        return target_dir

    target_dir = REPO_ROOT.parent / "EchoMimic"
    if (target_dir / "infer_audio2vid.py").is_file():
        return target_dir

    print("[avatar] Cloning EchoMimic...")
    subprocess.run(
        ["git", "clone", "https://github.com/antgroup/echomimic.git", str(target_dir)],
        check=True,
    )

    print("[avatar] Downloading EchoMimic pretrained weights...")
    subprocess.run(
        ["git", "lfs", "install"],
        cwd=str(target_dir),
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "huggingface_hub", "download",
            "BadToBest/EchoMimic",
            "--local-dir", str(target_dir / "pretrained_weights"),
        ],
        check=True,
    )

    print("[avatar] EchoMimic setup complete.")
    return target_dir


def synthesize_audio(text: str, checkpoint: Path | None, output_wav: Path) -> Path:
    """Run F5-TTS inference to produce audio from text."""
    print(f"[avatar] Synthesizing audio: {text[:60]}...")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "inference.py"),
        "--text", text,
        "--output", str(output_wav),
    ]
    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[avatar] Error: F5-TTS inference failed.")
        sys.exit(1)

    return output_wav


def generate_avatar_video(
    audio_path: Path,
    image_path: Path,
    output_path: Path,
    echomimic_dir: Path,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg_scale: float = 2.5,
    fps: int = 24,
    seed: int = 42,
    device: str = "cuda",
) -> Path:
    """Generate talking avatar video using EchoMimic."""
    print(f"[avatar] Generating video...")
    print(f"  Audio: {audio_path}")
    print(f"  Image: {image_path}")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Steps: {steps}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write a temporary animation config for EchoMimic
    import yaml

    config = {
        "test_cases": {
            str(image_path.resolve()): [str(audio_path.resolve())]
        }
    }

    config_path = echomimic_dir / "configs" / "prompts" / "animation_ptbr.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run EchoMimic inference
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(echomimic_dir),
    }

    cmd = [
        sys.executable,
        str(echomimic_dir / "infer_audio2vid.py"),
        "--config", str(config_path),
        "--W", str(width),
        "--H", str(height),
        "--steps", str(steps),
        "--cfg", str(cfg_scale),
        "--fps", str(fps),
        "--seed", str(seed),
        "--save_dir", str(output_path.parent),
    ]

    result = subprocess.run(cmd, env=env, cwd=str(echomimic_dir))

    if result.returncode != 0:
        # Fallback: try the accelerated version
        print("[avatar] Standard inference failed, trying accelerated...")
        cmd[1] = str(echomimic_dir / "infer_audio2vid_accelerate.py")
        result = subprocess.run(cmd, env=env, cwd=str(echomimic_dir))

    if result.returncode != 0:
        print("[avatar] Error: EchoMimic inference failed.")
        print("[avatar] Make sure EchoMimic is properly installed with pretrained weights.")
        sys.exit(1)

    # Find the generated video (EchoMimic saves with its own naming)
    generated = sorted(output_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if generated:
        latest = generated[-1]
        if latest != output_path:
            shutil.move(str(latest), str(output_path))

    print(f"[avatar] Video saved: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()

    # Resolve EchoMimic location
    echomimic_dir = find_echomimic(args.echomimic_dir)
    if not echomimic_dir.is_dir():
        print("[avatar] EchoMimic not found. Setting up...")
        echomimic_dir = setup_echomimic(REPO_ROOT.parent / "EchoMimic")

    # Validate reference image
    if not args.image.is_file():
        print(f"Error: Reference image not found: {args.image}")
        sys.exit(1)

    # Get audio file(s)
    audio_files: list[Path] = []

    if args.text:
        # Synthesize audio first
        output_wav = REPO_ROOT / "outputs" / "avatar_tts.wav"
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        synthesize_audio(args.text, args.checkpoint, output_wav)
        audio_files = [output_wav]
    elif args.audio:
        if args.audio.is_dir():
            audio_files = sorted(args.audio.glob("*.wav"))
            if not audio_files:
                print(f"Error: No WAV files found in {args.audio}")
                sys.exit(1)
        elif args.audio.is_file():
            audio_files = [args.audio]
        else:
            print(f"Error: Audio not found: {args.audio}")
            sys.exit(1)
    else:
        print("Error: Must provide --audio or --text")
        sys.exit(1)

    # Generate video for each audio
    output = args.output
    if output.suffix in (".mp4", ".avi", ".mkv"):
        # Single output file
        generate_avatar_video(
            audio_path=audio_files[0],
            image_path=args.image,
            output_path=output,
            echomimic_dir=echomimic_dir,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            fps=args.fps,
            seed=args.seed,
            device=args.device,
        )
    else:
        # Directory output
        output.mkdir(parents=True, exist_ok=True)
        for i, audio in enumerate(audio_files):
            video_name = f"avatar_{audio.stem}.mp4"
            generate_avatar_video(
                audio_path=audio,
                image_path=args.image,
                output_path=output / video_name,
                echomimic_dir=echomimic_dir,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                fps=args.fps,
                seed=args.seed,
                device=args.device,
            )

    print(f"\n[avatar] All done! Videos in: {output}")


if __name__ == "__main__":
    main()
