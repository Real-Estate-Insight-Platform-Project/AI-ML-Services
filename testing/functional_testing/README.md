# Backend Functional Testing Suite

Comprehensive functional testing for all AI-ML-Services backend components.

## Overview

This testing suite provides complete functional testing for:
- **Agent Recommender System** (30 test cases)
- **Risk Map Service** (23 test cases) 
- **SQL Agent** (31 test cases)

**Total: 84 comprehensive backend functional tests**

## Quick Start

```bash
# Run all functional tests
cd testing/functional_testing
python run_functional_tests.py

# Run specific service tests
python run_functional_tests.py --service agent
python run_functional_tests.py --service risk
python run_functional_tests.py --service sql

# Run with coverage report
python run_functional_tests.py --coverage

# Verbose output
python run_functional_tests.py --verbose
```

## Structure

```
functional_testing/
├── run_functional_tests.py          # Main test runner
├── pyproject.toml                   # Pytest configuration
├── README.md                        # This file
├── agent_recommender/               # Agent Recommender tests
│   ├── conftest.py                  # Test fixtures & mocks
│   └── test_agent_recommender_api.py # API tests (30 cases)
├── risk_map/                        # Risk Map tests  
│   ├── conftest.py                  # Test fixtures & mocks
│   └── test_risk_map_api.py         # API tests (23 cases)
└── sql_agent/                       # SQL Agent tests
    ├── conftest.py                  # Test fixtures & mocks
    └── test_sql_agent_api.py        # API tests (31 cases)
```

## Test Coverage

### Agent Recommender System (30 tests)
- Health check endpoint
- Location standardization 
- Debug/count endpoints
- Agent recommendation algorithm
- Filtering (location, price, property type)
- Pagination and sorting
- Error handling and validation
- Performance scoring system

### Risk Map Service (23 tests)  
- Health endpoint
- Counties vector tiles (MVT)
- Properties vector tiles (MVT)
- Property details lookup
- UUID validation
- Empty tile handling
- CORS middleware
- Error responses

### SQL Agent (31 tests)
- Health endpoint  
- Natural language query processing
- SQL injection prevention
- Response formatting (properties, states)
- Error handling
- Database safety measures
- Query result pagination
- Security validation

## Features

- **Service Isolation**: Each service tested independently with mocked dependencies
- **Comprehensive Mocking**: Supabase, AsyncPG, LangChain, SQLAlchemy mocks
- **Test Count Reporting**: Shows passed/failed counts for each service
- **Environment Setup**: Automatic Python path and dependency management
- **Cross-Platform**: Works on Windows, Linux, macOS
- **Coverage Support**: Optional code coverage reporting
- **Verbose Mode**: Detailed test output for debugging

## Dependencies

All testing dependencies are included in the main `requirements.txt`:

- pytest (test framework)
- pytest-asyncio (async test support)  
- httpx (HTTP client for API testing)
- pytest-mock (mocking utilities)
- pytest-cov (coverage reporting)

## Mocking Strategy

Each service uses comprehensive mocking to ensure isolation:

- **Agent Recommender**: Mocks Supabase client and scoring algorithms
- **Risk Map**: Mocks AsyncPG connection pools and database queries  
- **SQL Agent**: Mocks LangChain agents and SQL execution

This ensures tests run fast, don't require external dependencies, and test the application logic in isolation.

## Exit Codes

- `0`: All tests passed
- `1`: Some tests failed or setup error

## Examples

```bash
# Check setup only
python run_functional_tests.py --setup-only

# Run with full output
python run_functional_tests.py --verbose --coverage

# Test specific service with coverage
python run_functional_tests.py --service agent --coverage --verbose
```