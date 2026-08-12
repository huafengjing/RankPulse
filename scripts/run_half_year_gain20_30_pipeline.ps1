param(
    [int]$LookbackDays = 180,
    [string]$OutputDir = "outputs/half_year_gain20_30_signals"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = $Root

$Python = "C:\Users\hua\AppData\Local\Programs\Python\Python312\python.exe"
$LogDir = Join-Path $Root "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$StartedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Output "[$StartedAt] Half-year gain20-30 pipeline started"
Write-Output "Workspace: $Root"
Write-Output "LookbackDays: $LookbackDays"
Write-Output "OutputDir: $OutputDir"

Write-Output "Step 1/2: downloading Binance USDT-M klines..."
& $Python -m src.main download --lookback-days $LookbackDays --all-symbols

Write-Output "Step 2/2: exporting first Top10 gain 20-30 signals..."
& $Python scripts\export_gain20_30_top10_signals.py --lookback-days $LookbackDays --output-dir $OutputDir

$FinishedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Output "[$FinishedAt] Half-year gain20-30 pipeline finished"
