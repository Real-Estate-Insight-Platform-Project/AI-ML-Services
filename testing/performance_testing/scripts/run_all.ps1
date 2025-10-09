Param(
    [string]$EnvFile = "../.env.example"
)

Write-Host "=== Performance & Load Testing Suite ===" -ForegroundColor Green
Write-Host "Loading environment from $EnvFile" -ForegroundColor Yellow

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -and $_ -notmatch '^#') {
            $name, $value = $_.Split('=', 2)
            if ($name -and $value) {
                [System.Environment]::SetEnvironmentVariable($name, $value)
                Write-Host "Set $name" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "No .env file found at $EnvFile - using defaults" -ForegroundColor Yellow
}

# Ensure logs directory exists
$logsDir = "../logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force
}

Write-Host "`n=== Running k6 Performance Tests ===" -ForegroundColor Green

# k6 recommender test
Write-Host "`nRunning k6 recommender test..." -ForegroundColor Cyan
try {
    k6 run ../k6/recommender.k6.js | Tee-Object -FilePath ../logs/k6-recommender.log
    if ($LASTEXITCODE -eq 0) {
            Write-Host "[PASS] Recommender test passed" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Recommender test failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to run recommender test: $_" -ForegroundColor Red
}

# k6 tiles test
Write-Host "`nRunning k6 tiles test..." -ForegroundColor Cyan
try {
    k6 run ../k6/tiles.k6.js | Tee-Object -FilePath ../logs/k6-tiles.log
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[PASS] Tiles test passed" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Tiles test failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to run tiles test: $_" -ForegroundColor Red
}

# k6 SQL Agent test
Write-Host "`nRunning k6 SQL Agent test..." -ForegroundColor Cyan
try {
    k6 run ../k6/sql_agent.k6.js | Tee-Object -FilePath ../logs/k6-sql-agent.log
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[PASS] SQL Agent test passed" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] SQL Agent test failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to run SQL Agent test: $_" -ForegroundColor Red
}

Write-Host "`n=== Setting up Locust Environment ===" -ForegroundColor Green

# Use existing Python virtual environment from AI-ML-Services root
$venv = "$PSScriptRoot\..\..\..\venv"
Write-Host "Using existing Python virtual environment at: $venv" -ForegroundColor Yellow

if (!(Test-Path $venv)) {
    Write-Host "[ERROR] Virtual environment not found at $venv" -ForegroundColor Red
    Write-Host "Please ensure you have a Python virtual environment set up in the AI-ML-Services root directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "Installing/updating Locust dependencies from root requirements.txt..." -ForegroundColor Yellow
& "$venv\Scripts\pip.exe" install -r ..\..\..\requirements.txt

Write-Host "`n=== Running Locust Load Test ===" -ForegroundColor Green
Write-Host "Running Locust (All Services) - 5 users, 2 spawn rate, 2 minutes..." -ForegroundColor Cyan

try {
    & "$venv\Scripts\locust.exe" -f ../locust/sql_agent_locust.py --headless -u 5 -r 2 -t 2m --csv=../logs/locust-sql-agent --only-summary
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[PASS] Locust test completed successfully" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Locust test failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to run Locust test: $_" -ForegroundColor Red
}

Write-Host "`n=== Testing Complete ===" -ForegroundColor Green
Write-Host "Check the logs/ directory for detailed results:" -ForegroundColor Yellow
Write-Host "  - k6-recommender.log" -ForegroundColor Gray
Write-Host "  - k6-tiles.log" -ForegroundColor Gray
Write-Host "  - k6-sql-agent.log" -ForegroundColor Gray
Write-Host "  - k6-tiles-summary.json" -ForegroundColor Gray
Write-Host "  - k6-sql-agent-summary.json" -ForegroundColor Gray
Write-Host "  - locust-sql-agent_stats.csv" -ForegroundColor Gray
Write-Host "  - locust-sql-agent_stats_history.csv" -ForegroundColor Gray
Write-Host "  - locust-sql-agent_failures.csv" -ForegroundColor Gray

Write-Host "`nPerformance Summary:" -ForegroundColor Yellow
Write-Host "  - k6 tests validate response times and error rates against SLA thresholds" -ForegroundColor Gray
Write-Host "  - Locust provides user-driven load testing with realistic interaction patterns" -ForegroundColor Gray
Write-Host "  - Check threshold failures in k6 logs for SLA violations" -ForegroundColor Gray
Write-Host "  - Review Locust CSV files for detailed performance metrics" -ForegroundColor Gray