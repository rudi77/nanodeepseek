# NanoDeepSeek Training Script
# This script starts the training process with configurable parameters

param(
    [int]$Dim = 512,
    [int]$Layers = 8,
    [int]$Heads = 8,
    [int]$MaxSeqLen = 128, # später auf 512
    [int]$BatchSize = 8,
    [float]$LearningRate = 0.0003,
    [int]$Epochs = 1,
    [string]$SaveDir = "checkpoints",
    [string]$DataFile = ""
)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  NanoDeepSeek Training" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model Dimension:    $Dim" -ForegroundColor White
Write-Host "  Layers:             $Layers" -ForegroundColor White
Write-Host "  Attention Heads:    $Heads" -ForegroundColor White
Write-Host "  Max Sequence Length: $MaxSeqLen" -ForegroundColor White
Write-Host "  Batch Size:         $BatchSize" -ForegroundColor White
Write-Host "  Learning Rate:      $LearningRate" -ForegroundColor White
Write-Host "  Epochs:             $Epochs" -ForegroundColor White
Write-Host "  Save Directory:     $SaveDir" -ForegroundColor White
if ($DataFile) {
    Write-Host "  Data File:          $DataFile" -ForegroundColor White
} else {
    Write-Host "  Data File:          (using sample data)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
$venvPath = "python\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $venvPath
}

# Build command
$cmd = "python python/train.py --dim $Dim --n_layers $Layers --n_heads $Heads --max_seq_len $MaxSeqLen --batch_size $BatchSize --learning_rate $LearningRate --epochs $Epochs --save_dir $SaveDir"

if ($DataFile) {
    $cmd += " --data_file `"$DataFile`""
}

# Execute training
Write-Host "Starting training..." -ForegroundColor Green
Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""

Invoke-Expression $cmd

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host "  Training completed successfully!" -ForegroundColor Green
    Write-Host "==================================================" -ForegroundColor Green
} else {
    Write-Host "==================================================" -ForegroundColor Red
    Write-Host "  Training failed with exit code: $exitCode" -ForegroundColor Red
    Write-Host "==================================================" -ForegroundColor Red
}

exit $exitCode
