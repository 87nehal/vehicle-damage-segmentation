$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendRoot
$env:PYTHONPATH = Join-Path $backendRoot "src"

Write-Host "Starting Vehicle Damage FastAPI on http://127.0.0.1:8001 ..."
python -m vehicle_damage.api
