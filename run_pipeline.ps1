<#
.SYNOPSIS
    End-to-end F5-TTS PT-BR fine-tuning pipeline.
    Downloads data, prepares it, trains the model, and tests inference.

.DESCRIPTION
    Run this script to execute the full pipeline:
      1. Download Common Voice PT-BR (streaming)
      2. Prepare dataset (resample, normalize, split)
      3. Fine-tune F5-TTS v1 Base
      4. Run inference test

.PARAMETER MaxHours
    Maximum hours of audio to download (default: 100)

.PARAMETER SkipDownload
    Skip the download step (if data already exists)

.PARAMETER SkipPrepare
    Skip the prepare step (if already processed)

.PARAMETER HfToken
    HuggingFace token for Common Voice access

.EXAMPLE
    .\run_pipeline.ps1 -MaxHours 50 -HfToken "hf_..."
    .\run_pipeline.ps1 -SkipDownload -SkipPrepare  # Resume training only
#>

param(
    [int]$MaxHours = 100,
    [switch]$SkipDownload,
    [switch]$SkipPrepare,
    [string]$HfToken = $env:HF_TOKEN,
    [string]$MixedPrecision = "fp16",
    [int]$BatchSize = 0,
    [int]$Epochs = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# Fall back to video-pipeline venv if local doesn't exist
if (-not (Test-Path $Python)) {
    $Python = "C:\Users\ricar\Github\i\src\automation\video-pipeline\.venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " F5-TTS PT-BR Fine-tuning Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Python: $Python"
Write-Host "  Max Hours: $MaxHours"
Write-Host "  Mixed Precision: $MixedPrecision"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Download dataset
if (-not $SkipDownload) {
    Write-Host "[1/4] Downloading Common Voice PT-BR..." -ForegroundColor Yellow
    $downloadArgs = @(
        "$RepoRoot\scripts\download_dataset.py",
        "--output-dir", "$RepoRoot\data\raw",
        "--max-hours", $MaxHours
    )
    if ($HfToken) {
        $downloadArgs += "--hf-token"
        $downloadArgs += $HfToken
    }
    & $Python @downloadArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Download failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[1/4] Download complete." -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping download (--SkipDownload)" -ForegroundColor DarkGray
}
Write-Host ""

# Step 2: Prepare dataset
if (-not $SkipPrepare) {
    Write-Host "[2/4] Preparing dataset..." -ForegroundColor Yellow
    & $Python "$RepoRoot\scripts\prepare_dataset.py" `
        --input-csv "$RepoRoot\data\raw\metadata.csv" `
        --output-dir "$RepoRoot\data\processed"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Preparation failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[2/4] Preparation complete." -ForegroundColor Green
} else {
    Write-Host "[2/4] Skipping preparation (--SkipPrepare)" -ForegroundColor DarkGray
}
Write-Host ""

# Step 3: Train
Write-Host "[3/4] Starting fine-tuning..." -ForegroundColor Yellow
$trainArgs = @(
    "$RepoRoot\scripts\train.py",
    "--dataset-csv", "$RepoRoot\data\processed\train.csv",
    "--output-dir", "$RepoRoot\checkpoints",
    "--mixed-precision", $MixedPrecision
)
if ($SkipPrepare) {
    $trainArgs += "--skip-prepare"
}
if ($BatchSize -gt 0) {
    $trainArgs += "--batch-size"
    $trainArgs += $BatchSize
}
if ($Epochs -gt 0) {
    $trainArgs += "--epochs"
    $trainArgs += $Epochs
}
& $Python @trainArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[3/4] Training complete." -ForegroundColor Green
Write-Host ""

# Step 4: Inference test
Write-Host "[4/4] Running inference test..." -ForegroundColor Yellow
& $Python "$RepoRoot\scripts\inference.py" `
    --checkpoint "$RepoRoot\checkpoints\model_last.safetensors" `
    --output "$RepoRoot\outputs"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Inference test failed (model may need more training)" -ForegroundColor DarkYellow
} else {
    Write-Host "[4/4] Inference test complete." -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Pipeline Complete!" -ForegroundColor Cyan
Write-Host " Outputs: $RepoRoot\outputs\" -ForegroundColor Cyan
Write-Host " Model:   $RepoRoot\checkpoints\" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Listen to outputs/ samples"
Write-Host "  2. Upload: python scripts/upload_to_hf.py --checkpoint checkpoints/model_last.safetensors"
Write-Host "  3. Use in video pipeline: set F5TTS_PTBR_CKPT=checkpoints/model_last.safetensors"
