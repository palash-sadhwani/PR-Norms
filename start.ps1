$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Python = Join-Path $Backend ".venv\Scripts\python.exe"

$BackendUrl = "http://127.0.0.1:8000"
$FrontendUrl = "http://127.0.0.1:5173"
$HealthUrl = "$BackendUrl/health"

# Check required files and dependencies
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Backend Python environment not found:" -ForegroundColor Red
    Write-Host $Python
    exit 1
}

if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Host "ERROR: Frontend dependencies are missing." -ForegroundColor Red
    Write-Host "Run 'npm ci' inside the frontend directory first."
    exit 1
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "       PR-Norms Startup" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check whether the backend is already healthy
$BackendReady = $false

try {
    $Response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
    if ($Response.status -eq "ok") {
        $BackendReady = $true
    }
} catch {
    $BackendReady = $false
}

# Launch backend only if it is not already running
if ($BackendReady) {
    Write-Host "Backend is already healthy. Reusing it." -ForegroundColor Green
} else {
    Write-Host "Starting backend..." -ForegroundColor Yellow

    $BackendCommand = @"
`$ErrorActionPreference = 'Stop'
`$env:DATABASE_URL = 'sqlite+aiosqlite:///./presentation.db'
`$env:LOCAL_DEMO_MODE = 'true'

Set-Location '$Backend'

Write-Host 'Running database migrations...' -ForegroundColor Cyan
& '$Python' -m alembic upgrade head

if (`$LASTEXITCODE -ne 0) {
    Write-Host 'ERROR: Database migration failed.' -ForegroundColor Red
    exit 1
}

Write-Host 'Starting FastAPI...' -ForegroundColor Green
& '$Python' -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"@

    Start-Process powershell.exe `
        -WorkingDirectory $Backend `
        -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command", $BackendCommand
        )

    # Wait up to 90 seconds for the backend to become healthy
    Write-Host "Waiting for backend health check..." -ForegroundColor Yellow

    $Timeout = 90
    $Elapsed = 0

    while ($Elapsed -lt $Timeout) {
        Start-Sleep -Seconds 2
        $Elapsed += 2

        try {
            $Response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3

            if ($Response.status -eq "ok") {
                $BackendReady = $true
                break
            }
        } catch {
            # Backend is still starting
        }

        Write-Progress `
            -Activity "Starting PR-Norms backend" `
            -Status "Waiting for health check..." `
            -PercentComplete ([Math]::Min(100, ($Elapsed / $Timeout) * 100))
    }

    Write-Progress -Activity "Starting PR-Norms backend" -Completed

    if (-not $BackendReady) {
        Write-Host ""
        Write-Host "ERROR: Backend did not become healthy within 90 seconds." -ForegroundColor Red
        Write-Host "Check the backend PowerShell window for errors."
        Write-Host "Health endpoint: $HealthUrl"
        exit 1
    }
}

Write-Host "Backend health check passed." -ForegroundColor Green

# Check whether the frontend is already responding
$FrontendReady = $false

try {
    $Response = Invoke-WebRequest `
        -Uri $FrontendUrl `
        -TimeoutSec 3 `
        -UseBasicParsing

    if ($Response.StatusCode -eq 200) {
        $FrontendReady = $true
    }
} catch {
    $FrontendReady = $false
}

# Launch frontend only if it is not already responding
if ($FrontendReady) {
    Write-Host "Frontend is already responding. Reusing it." -ForegroundColor Green
} else {
    Write-Host "Starting frontend..." -ForegroundColor Yellow

    $FrontendCommand = @"
Set-Location '$Frontend'
Write-Host 'Starting PR-Norms frontend...' -ForegroundColor Cyan
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
"@

    Start-Process powershell.exe `
        -WorkingDirectory $Frontend `
        -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command", $FrontendCommand
        )

    # Wait up to 30 seconds for the frontend
    $FrontendReady = $false

    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2

        try {
            $Response = Invoke-WebRequest `
                -Uri $FrontendUrl `
                -TimeoutSec 3 `
                -UseBasicParsing

            if ($Response.StatusCode -eq 200) {
                $FrontendReady = $true
                break
            }
        } catch {
            # Frontend is still starting
        }
    }
}

Write-Host ""

if ($FrontendReady) {
    Write-Host "PR-Norms is ready!" -ForegroundColor Green
    Write-Host "Frontend: $FrontendUrl"
    Write-Host "Backend:  $BackendUrl"
    Write-Host "Health:   $HealthUrl"

    Start-Process $FrontendUrl
} else {
    Write-Host "WARNING: Frontend did not respond within 30 seconds." -ForegroundColor Yellow
    Write-Host "Check the frontend PowerShell window."
    Write-Host "Frontend URL: $FrontendUrl"
}

Write-Host ""
Write-Host "Keep the server windows open while using PR-Norms."