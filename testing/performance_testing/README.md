# Performance & Load Testing Suite

This directory contains a comprehensive performance and load testing suite for your three backend services:
- **Agent Recommender System**
- **Risk Map Tiles Service**
- **SQL Agent Service**

## 🏗️ Architecture Overview

The testing suite uses industry-standard tools:
- **k6** for HTTP performance testing with SLA thresholds
- **Locust** for user-flow load testing with realistic interaction patterns

## 📁 Directory Structure

```
testing/performance_testing/
├── README.md                    # This file
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git ignore rules
├── datasets/                    # Test data
│   ├── tiles.csv               # Sample tile coordinates (z,x,y)
│   └── sql_agent_prompts.json  # Sample SQL queries
├── k6/                         # k6 performance test scripts
│   ├── recommender.k6.js       # Agent recommender load testing
│   └── tiles.k6.js             # Risk map tiles performance testing
├── locust/                     # Locust load test scripts
│   └── sql_agent_locust.py     # SQL agent user-flow testing
├── scripts/                    # Automation scripts
│   ├── run_all.ps1            # Windows PowerShell runner
│   └── run_all.sh             # Linux/macOS bash runner
└── logs/                      # Test outputs (auto-created, gitignored)
    ├── k6-recommender.log
    ├── k6-tiles.log
    ├── k6-tiles-summary.json
    ├── locust-sql-agent_stats.csv
    ├── locust-sql-agent_stats_history.csv
    └── locust-sql-agent_failures.csv
```

## 🚀 Quick Start

### Prerequisites

1. **Install k6**:
   - **Windows**: `choco install k6` or `winget install k6`
   - **macOS**: `brew install k6`
   - **Linux**: See [k6 installation guide](https://grafana.com/docs/k6/latest/set-up/install-k6/)

2. **Python Virtual Environment**: Ensure you have a Python virtual environment set up in the AI-ML-Services root directory with all dependencies installed

3. **Running Services**: Ensure your three backend services are running on their respective ports

### Configuration

1. The scripts use `.env.example` by default, but you can copy it to `.env` and customize if needed:
   ```bash
   cp .env.example .env  # Optional - scripts use .env.example by default
   ```

2. Edit `.env.example` (or your custom `.env`) to match your service URLs:
   ```env
   RECOMMENDER_BASE=http://127.0.0.1:8000
   RISKMAP_BASE=http://127.0.0.1:8000
   SQL_AGENT_BASE=http://127.0.0.1:8000
   SQL_AGENT_API_KEY=your_api_key_if_needed
   ```

### Running Tests

#### Windows PowerShell
```powershell
cd testing/performance_testing/scripts
./run_all.ps1
```

#### Linux/macOS
```bash
cd testing/performance_testing/scripts
chmod +x run_all.sh
./run_all.sh
```

## 🧪 Test Scenarios

### k6 Performance Tests

#### Agent Recommender (`recommender.k6.js`)
- **Baseline**: Single-user profiling (200 iterations, 3min max)
- **Ramp**: Gradual load increase (10→100 RPS over 5min)
- **Spike**: Instant load spike (200 RPS for 1min)
- **Soak**: Sustained load (60 RPS for 20min)

**SLA Thresholds**:
- ❌ `<1%` error rate
- ❌ `p95 <300ms` response time
- ❌ `p95 <500ms` during spike (relaxed)

#### Risk Map Tiles (`tiles.k6.js`)
- **z10_city**: City-level tiles (150 RPS, 5min)
- **z6_region**: Regional tiles (80 RPS, 5min)
- **z0_world**: World-level tiles (20 RPS, 2min)

**SLA Thresholds**:
- ❌ `<1%` error rate
- ❌ City tiles: `p95 <200ms`
- ❌ Regional tiles: `p95 <350ms`
- ❌ World tiles: `p95 <800ms`

### Locust Load Tests

#### SQL Agent (`sql_agent_locust.py`)
- **User Simulation**: 50 concurrent users
- **Spawn Rate**: 5 users/second
- **Duration**: 15 minutes
- **Think Time**: 0.2-1.0 seconds between requests

**Test Patterns**:
- Random SQL queries from prompt dataset
- Health check endpoints (2x frequency)
- Realistic user interaction timing

## 📊 Understanding Results

### k6 Results
- **PASS/FAIL**: Determined by threshold violations
- **Key Metrics**: 
  - `http_req_duration`: Response times (p50, p95, p99)
  - `http_req_failed`: Error rate percentage
  - `http_reqs`: Requests per second
- **Output**: Console summary + detailed logs in `logs/`

### Locust Results
- **CSV Files**: Detailed statistics in `logs/locust-sql-agent_*.csv`
- **Key Metrics**:
  - Average/min/max response times
  - Requests per second
  - Failure rates by endpoint
  - User concurrency patterns

### Success Criteria

✅ **Passing Tests**:
- All k6 thresholds are met
- Locust shows <5% failure rate
- Response times within acceptable ranges
- No critical errors in logs

❌ **Failing Tests**:
- Threshold violations in k6 output
- High failure rates in Locust CSV
- Timeouts or connection errors
- Memory/resource exhaustion

## 🔧 Customization

### Modifying Test Parameters

#### k6 Configuration
Edit the `options` object in k6 scripts:
```javascript
export const options = {
  scenarios: {
    your_scenario: {
      executor: "constant-arrival-rate",
      rate: 100,        // requests per second
      duration: "5m",   // test duration
      // ... other options
    }
  },
  thresholds: {
    http_req_duration: ["p(95)<500"],  // 95th percentile < 500ms
    http_req_failed: ["rate<0.02"],    // <2% error rate
  }
};
```

#### Locust Configuration
Modify the class parameters in `sql_agent_locust.py`:
```python
class SQLAgentUser(HttpUser):
    wait_time = between(1, 3)  # Think time between requests
    
    @task(3)  # Task weight (higher = more frequent)
    def your_task(self):
        # Your test logic
```

### Adding New Test Data

#### Tile Coordinates (`datasets/tiles.csv`)
Add new z/x/y coordinates in CSV format:
```csv
zoom,x,y
12,1024,1536
13,2048,3072
```

#### SQL Queries (`datasets/sql_agent_prompts.json`)
Add new test queries to the JSON array:
```json
[
  "Your new SQL query here",
  "Another test query"
]
```

## 🐛 Troubleshooting

### Common Issues

**k6 not found**:
```bash
# Install k6 first
winget install k6  # Windows
brew install k6    # macOS
```

**Python/Locust issues**:
```bash
# Check Python version
python --version  # Should be 3.7+

# Manually install Locust
pip install locust==2.41.2
```

**Service connection errors**:
1. Verify services are running: `curl http://127.0.0.1:8000/health`
2. Check `.env` file URLs
3. Review firewall/network settings

**Permission errors (Linux/macOS)**:
```bash
chmod +x scripts/run_all.sh
```

### Debug Mode

Run individual tests for debugging:

```bash
# k6 with verbose output
k6 run --verbose k6/recommender.k6.js

# Locust with web UI (accessible at http://localhost:8089)
locust -f locust/sql_agent_locust.py --host=http://127.0.0.1:8000
```

## 📈 Performance Best Practices

### Before Testing
1. **Warm-up**: Run services for 5+ minutes before testing
2. **Database**: Ensure adequate test data exists
3. **Resources**: Monitor CPU/memory during tests
4. **Network**: Use reliable network connection

### During Testing
1. **Monitor**: Watch system resources (htop, Task Manager)
2. **Logs**: Check application logs for errors
3. **Metrics**: Track key business metrics alongside performance

### After Testing
1. **Analyze**: Review all output files in `logs/`
2. **Compare**: Establish baseline metrics for future comparison
3. **Optimize**: Address performance bottlenecks identified
4. **Document**: Record findings and improvements

## 🔗 Additional Resources

- [k6 Documentation](https://grafana.com/docs/k6/)
- [Locust Documentation](https://docs.locust.io/)
- [Performance Testing Best Practices](https://grafana.com/docs/k6/latest/testing-guides/)

---

**Need Help?** 
- Check the `logs/` directory for detailed error messages
- Verify your services are running and accessible
- Ensure all dependencies are installed correctly