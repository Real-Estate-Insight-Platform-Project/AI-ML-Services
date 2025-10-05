#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-../.env.example}"

echo "=== Performance & Load Testing Suite ==="
echo "Loading environment from $ENV_FILE"

if [ -f "$ENV_FILE" ]; then
    # Export environment variables from .env file
    set -a
    source "$ENV_FILE"
    set +a
    echo "Environment loaded successfully"
else
    echo "No .env file found at $ENV_FILE - using defaults"
fi

# Ensure logs directory exists
mkdir -p ../logs

echo
echo "=== Running k6 Performance Tests ==="

# k6 recommender test
echo
echo "Running k6 recommender test..."
if k6 run ../k6/recommender.k6.js | tee ../logs/k6-recommender.log; then
    echo "[PASS] Recommender test passed"
else
    echo "[FAIL] Recommender test failed (exit code: $?)"
fi

# k6 tiles test
echo
echo "Running k6 tiles test..."
if k6 run ../k6/tiles.k6.js | tee ../logs/k6-tiles.log; then
    echo "[PASS] Tiles test passed"
else
    echo "[FAIL] Tiles test failed (exit code: $?)"
fi

# k6 SQL Agent test
echo
echo "Running k6 SQL Agent test..."
if k6 run ../k6/sql_agent.k6.js | tee ../logs/k6-sql-agent.log; then
    echo "[PASS] SQL Agent test passed"
else
    echo "[FAIL] SQL Agent test failed (exit code: $?)"
fi

echo
echo "=== Setting up Locust Environment ==="

# Use existing Python virtual environment from AI-ML-Services root
VENV="$(dirname "$0")/../../../venv"
echo "Using existing Python virtual environment at: $VENV"

if [ ! -d "$VENV" ]; then
    echo "[ERROR] Virtual environment not found at $VENV"
    echo "Please ensure you have a Python virtual environment set up in the AI-ML-Services root directory"
    exit 1
fi

echo "Installing/updating Locust dependencies from root requirements.txt..."
"$VENV/bin/pip" install -r ../../../requirements.txt

echo
echo "=== Running Locust Load Test ==="
echo "Running Locust (SQL Agent) - 50 users, 5 spawn rate, 15 minutes..."

if "$VENV/bin/locust" -f ../locust/sql_agent_locust.py --headless -u 50 -r 5 -t 15m --csv=../logs/locust-sql-agent --only-summary; then
    echo "[PASS] Locust test completed successfully"
else
    echo "[FAIL] Locust test failed (exit code: $?)"
fi

echo
echo "=== Testing Complete ==="
echo "Check the logs/ directory for detailed results:"
echo "  - k6-recommender.log"
echo "  - k6-tiles.log"
echo "  - k6-sql-agent.log"
echo "  - k6-tiles-summary.json"
echo "  - k6-sql-agent-summary.json"
echo "  - locust-sql-agent_stats.csv"
echo "  - locust-sql-agent_stats_history.csv"
echo "  - locust-sql-agent_failures.csv"

echo
echo "Performance Summary:"
echo "  - k6 tests validate response times and error rates against SLA thresholds"
echo "  - Locust provides user-driven load testing with realistic interaction patterns"
echo "  - Check threshold failures in k6 logs for SLA violations"
echo "  - Review Locust CSV files for detailed performance metrics"