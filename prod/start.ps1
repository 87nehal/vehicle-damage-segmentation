$ErrorActionPreference = "Stop"

$prodRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $prodRoot "backend"
$frontendRoot = Join-Path $prodRoot "frontend"
$logsRoot = Join-Path $prodRoot "logs"
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

function Test-ModelApi {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/health" -TimeoutSec 2
        return [bool]($health.damage.ok -and $health.damage.model_loaded)
    } catch {
        return $false
    }
}

$backendProcess = $null
$previousPythonPath = $env:PYTHONPATH
try {
    if (Test-ModelApi) {
        Write-Host "Reusing the model API already running on port 8001."
    } else {
        $env:PYTHONPATH = Join-Path $backendRoot "src"
        $stdoutLog = Join-Path $logsRoot "backend.out.log"
        $stderrLog = Join-Path $logsRoot "backend.err.log"
        $backendProcess = Start-Process -FilePath "python" `
            -ArgumentList @("-m", "vehicle_damage.api") `
            -WorkingDirectory $backendRoot `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -WindowStyle Hidden `
            -PassThru

        Write-Host "Loading the model API (this can take up to 90 seconds on first start)..."
        $ready = $false
        for ($attempt = 0; $attempt -lt 180; $attempt++) {
            if (Test-ModelApi) {
                $ready = $true
                break
            }
            if ($backendProcess.HasExited) {
                throw "The backend exited with code $($backendProcess.ExitCode). See $stderrLog"
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) {
            throw "The model API did not become ready within 90 seconds. See $stderrLog"
        }
    }

    Set-Location $frontendRoot
    Write-Host "Starting the frontend at http://localhost:3000 ..."
    & npm.cmd run dev:web
    if ($LASTEXITCODE -ne 0) {
        throw "The frontend exited with code $LASTEXITCODE"
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }
}
