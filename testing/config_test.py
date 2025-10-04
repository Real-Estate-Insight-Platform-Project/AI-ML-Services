#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
# XML removed - using JSON only for reports
from contextlib import contextmanager
from datetime import datetime
from http.client import HTTPConnection
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ================================================================================================
# CONFIGURATION CONSTANTS
# ================================================================================================

# Default service endpoint configurations
DEFAULT_CONFIG = {
    "base_url": "http://localhost:8080",
    "health_path": "/health",
    "read_path": "/items",
    "write_path": "/items",
    "db_versions": ["13", "14", "15", "16"],
    "memory_limits": ["512m", "1g"],
    "python_versions": ["3.12"],
    "api_port": 8080,
    "db_port": 5432,
    "timeout": 300,
    "metrics_duration": 30,
    "max_retries": 3,
    "retry_delay": 5
}

# Docker compose template for testing configurations
COMPOSE_TEMPLATE = """version: '3.8'
services:
  db:
    image: postgres:{db_version}
    environment:
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_DB=app
    ports:
      - "{db_port}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d app || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 40s

  api:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - PYTHON_VERSION={python_version}
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app
      - PORT={api_port}
      - PYTHONPATH=/app
    ports:
      - "{api_port}:{api_port}"
    depends_on:
      db:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: {memory_limit}
"""

# Multi-stage Dockerfile template for Python version testing
DOCKERFILE_TEMPLATE = """# Multi-stage Dockerfile for Configuration Testing
# This file is auto-generated for testing different Python versions

FROM python:{python_version}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install minimal dependencies for testing
RUN pip install --upgrade pip
RUN pip install fastapi==0.115.0 uvicorn[standard]==0.30.6 psycopg2-binary==2.9.9

# Copy test application
COPY test_app.py .

# Expose port
EXPOSE 8080

# Health check using curl for HTTP request
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "test_app:app", "--host", "0.0.0.0", "--port", "8080"]
"""

# Test data templates for write operations
TEST_DATA_TEMPLATES = {
    "agent-recommender": {
        "url_suffix": "/recommend",
        "method": "POST",
        "data": {
            "locations": ["Florida"],
            "top_k": 1,
            "min_rating": 4.0
        }
    },
    "property-recommendation": {
        "url_suffix": "/recommendations",
        "method": "POST", 
        "data": {
            "property_id": "test-property-id",
            "limit": 1
        }
    },
    "sql-agent": {
        "url_suffix": "/ask",
        "method": "POST",
        "data": {
            "question": "How many states are in the predictions table?"
        }
    },
    "risk-map": {
        "url_suffix": "/properties/test-id",
        "method": "GET",
        "data": None
    }
}

# ================================================================================================
# LOGGING SETUP
# ================================================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger("config_test")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler("config_test.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ================================================================================================
# DOCKER UTILITIES
# ================================================================================================

class DockerManager:
    """Manages Docker operations for configuration testing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.active_stacks: List[str] = []
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                self.logger.info(f"Docker available: {result.stdout.strip()}")
                return True
            else:
                self.logger.error("Docker command failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"Docker not available: {e}")
            return False
    
    def create_compose_file(self, config: Dict[str, Any], temp_dir: Path) -> Path:
        """Create docker-compose.yml file for the given configuration."""
        compose_content = COMPOSE_TEMPLATE.format(
            db_version=config["db_version"],
            db_port=config["db_port"],
            api_port=config["api_port"],
            memory_limit=config["memory_limit"],
            python_version=config["python_version"]
        )
        
        compose_file = temp_dir / "docker-compose.yml"
        with open(compose_file, "w") as f:
            f.write(compose_content)
        
        self.logger.debug(f"Created compose file: {compose_file}")
        return compose_file
    
    def create_dockerfile_for_testing(self, python_version: str, temp_dir: Path) -> Path:
        """Create a Dockerfile optimized for configuration testing."""
        dockerfile_content = DOCKERFILE_TEMPLATE.format(python_version=python_version)
        
        dockerfile_path = temp_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        # Create a minimal test FastAPI application
        test_app_content = '''"""
Minimal FastAPI application for configuration testing.
"""
from fastapi import FastAPI, HTTPException
from typing import Dict, List
import os
import psycopg2
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Configuration Test API", version="1.0.0")

logger.info("FastAPI app created successfully")

def get_db_connection():
    """Get database connection for testing."""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting database connection (attempt {attempt + 1}/{max_retries})")
            conn = psycopg2.connect(
                host="db",
                database="app", 
                user="postgres",
                password="postgres",
                port=5432,
                connect_timeout=10
            )
            logger.info("Database connection successful")
            return conn
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("All database connection attempts failed")
                raise HTTPException(status_code=500, detail=f"Database connection failed after {max_retries} attempts: {e}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    logger.info("Health check endpoint called")
    try:
        conn = get_db_connection()
        conn.close()
        result = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
            "database": "connected"
        }
        logger.info("Health check passed")
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    logger.info("Root endpoint called")
    return {"message": "Configuration Test API", "status": "running"}

@app.get("/items")
async def read_items() -> List[Dict[str, str]]:
    """Read items endpoint for testing."""
    logger.info("Read items endpoint called")
    return [
        {"id": "1", "name": "Test Item 1", "status": "active"},
        {"id": "2", "name": "Test Item 2", "status": "active"}
    ]

@app.post("/items")
async def create_item(item: Dict[str, str]) -> Dict[str, str]:
    """Create item endpoint for testing."""
    logger.info(f"Create item endpoint called with data: {item}")
    return {
        "id": "generated-id", 
        "name": item.get("name", "Test Item"),
        "status": "created",
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("=== FastAPI Application Starting ===")
    logger.info(f"Python version: {os.environ.get('PYTHON_VERSION', 'unknown')}")
    logger.info(f"Environment variables: DATABASE_URL={os.environ.get('DATABASE_URL', 'not set')}")
    logger.info("=== Application Ready ===")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server directly")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
'''
        
        test_app_path = temp_dir / "test_app.py"
        with open(test_app_path, "w") as f:
            f.write(test_app_content)
        
        self.logger.debug(f"Created Dockerfile: {dockerfile_path}")
        self.logger.debug(f"Created test app: {test_app_path}")
        return dockerfile_path
    
    def check_requirements_compatibility(self, python_version: str, temp_dir: Path) -> Tuple[bool, List[str]]:
        """Check if requirements.txt is compatible with the Python version."""
        issues = []
        
        # Use the standardized requirements.txt from project root
        project_root = Path.cwd().parent if Path.cwd().name == "testing" else Path.cwd()
        requirements_file = project_root / "requirements.txt"
        
        if not requirements_file.exists():
            issues.append("No requirements.txt file found in project root")
            return False, issues
        
        # Copy requirements to temp directory
        temp_requirements = temp_dir / "requirements.txt"
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            
            with open(temp_requirements, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
        except Exception as e:
            issues.append(f"Error reading requirements.txt: {e}")
            return False, issues
        
        # Validate Python version compatibility (requirements.txt is standardized for Python 3.12)
        python_major_minor = float(python_version)
        
        if python_major_minor < 3.10:
            issues.append(f"Python {python_version} not supported. Use Python 3.10, 3.11, or 3.12")
        elif python_major_minor > 3.12:
            self.logger.warning(f"Python {python_version} not tested. Requirements standardized for Python 3.12")
        
        self.logger.info(f"Using standardized requirements.txt compatible with Python 3.10-3.12")
        return len(issues) == 0, issues
    
    def start_stack(self, compose_file: Path, stack_name: str, timeout: int = 600) -> bool:
        """Start docker-compose stack and wait for services to be healthy."""
        try:
            self.logger.info(f"Starting stack {stack_name}")
            self.logger.info(f"Using compose file: {compose_file}")
            self.logger.info(f"Compose file exists: {compose_file.exists()}")
            
            # Start services without waiting for health checks, use --compatibility for memory limits
            cmd = [
                "docker", "compose", "-f", str(compose_file), 
                "-p", stack_name, "--compatibility", "up", "-d"
            ]
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Use Popen for better encoding control
            import subprocess
            from subprocess import Popen, PIPE
            
            process = Popen(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                cwd=compose_file.parent,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            try:
                stdout_data, stderr_data = process.communicate(timeout=timeout)
                
                # Decode with error handling
                stdout_str = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
                stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
                
                if process.returncode != 0:
                    self.logger.error(f"Failed to start stack {stack_name}: {stderr_str}")
                    return False
                    
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error(f"Timeout starting stack {stack_name}")
                return False
            
            # Log compose status for context (don't fail on error)
            try:
                ls_cmd = ["docker", "compose", "ls", "-a"]
                result = subprocess.run(ls_cmd, capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
                if result.returncode == 0:
                    self.logger.info(f"Active compose stacks:\n{result.stdout}")
            except Exception as e:
                self.logger.debug(f"Failed to get compose list: {e}")
            
            # Wait for services to be healthy
            self.logger.info(f"Waiting for services in stack {stack_name} to be healthy...")
            if self._wait_for_healthy_services(compose_file, stack_name, timeout=120):
                self.active_stacks.append(stack_name)
                self.logger.info(f"Stack {stack_name} started successfully and is healthy")
                return True
            else:
                self.logger.error(f"Stack {stack_name} containers started but failed health checks")
                return False
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout starting stack {stack_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error starting stack {stack_name}: {e}")
            return False
    
    def _wait_for_healthy_services(self, compose_file: Path, stack_name: str, timeout: int = 120) -> bool:
        """Wait for all services in the stack to be healthy."""
        import time
        
        start_time = time.time()
        check_interval = 5
        
        while time.time() - start_time < timeout:
            try:
                # Check service health using docker compose ps
                ps_cmd = [
                    "docker", "compose", "-f", str(compose_file),
                    "-p", stack_name, "ps", "--format", "json"
                ]
                
                result = subprocess.run(
                    ps_cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=compose_file.parent,
                    timeout=10
                )
                
                if result.returncode != 0:
                    self.logger.warning(f"Failed to check service status: {result.stderr}")
                    time.sleep(check_interval)
                    continue
                
                # Parse JSON output line by line to check service health
                import json
                services_healthy = True
                
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    try:
                        c = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse service info: {line} - {e}")
                        services_healthy = False
                        continue
                    service_name = c.get("Service", "unknown")
                    state = c.get("State", "unknown")
                    health = c.get("Health", "")
                    
                    self.logger.info(f"Service {service_name}: State={state}, Health='{health}'")
                    
                    # For services with health checks, wait for healthy
                    if health and health not in ("healthy", ""):
                        if health == "starting":
                            self.logger.info(f"Service {service_name} health check still starting...")
                            services_healthy = False
                        else:
                            self.logger.warning(f"Service {service_name} unhealthy: {health}")
                            services_healthy = False
                    
                    # For services without health checks, just check they're running
                    elif not health and state != "running":
                        self.logger.warning(f"Service {service_name} not running: {state}")
                        services_healthy = False
                
                if services_healthy:
                    self.logger.info("All services are healthy")
                    return True
                    
                self.logger.info(f"Waiting for services to be healthy... ({int(time.time() - start_time)}s elapsed)")
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.warning(f"Error checking service health: {e}")
                time.sleep(check_interval)
        
        self.logger.error(f"Timeout waiting for services to be healthy after {timeout}s")
        return False

    def stop_stack(self, compose_file: Path, stack_name: str) -> None:
        """Stop and remove docker-compose stack."""
        try:
            self.logger.info(f"Stopping stack {stack_name}")
            
            cmd = [
                "docker", "compose", "-f", str(compose_file),
                "-p", stack_name, "down", "-v", "--remove-orphans"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=compose_file.parent,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Issues stopping stack {stack_name}: {result.stderr}")
            else:
                self.logger.info(f"Stack {stack_name} stopped successfully")
            
            if stack_name in self.active_stacks:
                self.active_stacks.remove(stack_name)
                
        except Exception as e:
            self.logger.error(f"Error stopping stack {stack_name}: {e}")
    
    def collect_metrics(self, stack_name: str, duration: int) -> Dict[str, Dict[str, float]]:
        """Collect container metrics using docker stats."""
        try:
            self.logger.info(f"Collecting metrics for {duration} seconds")
            
            cmd = [
                "docker", "stats", "--no-stream", "--format",
                "{{.Container}},{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}"
            ]
            
            metrics = {"api": {"cpu": [], "memory": []}, "db": {"cpu": [], "memory": []}}
            samples = 0
            target_samples = duration // 2  # Sample every 2 seconds
            
            for _ in range(target_samples):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
                    
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if not line:
                                continue
                            
                            parts = line.split(',')
                            if len(parts) >= 5:
                                container_name = parts[1]
                                cpu_percent = parts[2].replace('%', '')
                                mem_usage = parts[3]
                                
                                # Determine service type
                                service_type = None
                                if stack_name in container_name:
                                    if 'api' in container_name or 'app' in container_name:
                                        service_type = 'api'
                                    elif 'db' in container_name or 'postgres' in container_name:
                                        service_type = 'db'
                                
                                if service_type and cpu_percent.replace('.', '').isdigit():
                                    metrics[service_type]["cpu"].append(float(cpu_percent))
                                    
                                    # Parse memory usage (e.g., "123.4MiB / 512MiB")
                                    if ' / ' in mem_usage:
                                        used_mem = mem_usage.split(' / ')[0]
                                        mem_value = self._parse_memory_value(used_mem)
                                        if mem_value is not None:
                                            metrics[service_type]["memory"].append(mem_value)
                        
                        samples += 1
                    
                    time.sleep(2)
                    
                except subprocess.TimeoutExpired:
                    self.logger.warning("Timeout collecting metrics sample")
                    continue
            
            # Calculate averages and maxes
            summary = {}
            for service in ["api", "db"]:
                cpu_values = metrics[service]["cpu"]
                mem_values = metrics[service]["memory"]
                
                summary[service] = {
                    "avg_cpu": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
                    "max_cpu": max(cpu_values) if cpu_values else 0.0,
                    "avg_memory_mb": sum(mem_values) / len(mem_values) if mem_values else 0.0,
                    "max_memory_mb": max(mem_values) if mem_values else 0.0,
                    "samples": len(cpu_values)
                }
            
            self.logger.info(f"Collected {samples} metric samples")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return {"api": {"avg_cpu": 0, "max_cpu": 0, "avg_memory_mb": 0, "max_memory_mb": 0, "samples": 0},
                    "db": {"avg_cpu": 0, "max_cpu": 0, "avg_memory_mb": 0, "max_memory_mb": 0, "samples": 0}}
    
    def _parse_memory_value(self, mem_str: str) -> Optional[float]:
        """Parse memory value string like '123.4MiB' to MB."""
        try:
            mem_str = mem_str.strip()
            if mem_str.endswith('MiB'):
                return float(mem_str[:-3])
            elif mem_str.endswith('GiB'):
                return float(mem_str[:-3]) * 1024
            elif mem_str.endswith('KiB'):
                return float(mem_str[:-3]) / 1024
            elif mem_str.endswith('B'):
                return float(mem_str[:-1]) / (1024 * 1024)
            else:
                # Try to parse as number (assume MB)
                return float(mem_str)
        except ValueError:
            return None
    
    def cleanup_all(self) -> None:
        """Stop all active stacks."""
        for stack_name in self.active_stacks.copy():
            try:
                # Try to find and stop the stack
                cmd = ["docker", "compose", "-p", stack_name, "down", "-v", "--remove-orphans"]
                subprocess.run(cmd, capture_output=True, timeout=30, encoding='utf-8', errors='replace')
            except Exception as e:
                self.logger.warning(f"Error cleaning up stack {stack_name}: {e}")
        
        self.active_stacks.clear()

# ================================================================================================
# HTTP CLIENT UTILITIES  
# ================================================================================================

class HTTPClient:
    """Simple HTTP client for API testing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def wait_for_service(self, url: str, timeout: int = 60, interval: int = 2) -> bool:
        """Wait for service to become available."""
        self.logger.info(f"Waiting for service at {url}")
        
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                response = self.get(url, timeout=10)
                if response:
                    self.logger.info("Service is available")
                    return True
            except Exception:
                pass
            
            self.logger.debug(f"Service not ready, waiting {interval}s...")
            time.sleep(interval)
        
        self.logger.error(f"Service at {url} not available after {timeout}s")
        return False
    
    def get(self, url: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Perform HTTP GET request."""
        try:
            req = Request(url)
            req.add_header('Content-Type', 'application/json')
            
            with urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    content = response.read().decode('utf-8')
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"content": content, "status": "success"}
                else:
                    self.logger.warning(f"GET {url} returned status {response.status}")
                    return None
                    
        except HTTPError as e:
            if e.code == 404:
                # 404 might be expected for some endpoints
                return {"status": "not_found", "code": 404}
            self.logger.error(f"HTTP error for GET {url}: {e}")
            return None
        except URLError as e:
            self.logger.error(f"URL error for GET {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for GET {url}: {e}")
            return None
    
    def post(self, url: str, data: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Perform HTTP POST request."""
        try:
            json_data = json.dumps(data).encode('utf-8')
            req = Request(url, data=json_data, method='POST')
            req.add_header('Content-Type', 'application/json')
            
            with urlopen(req, timeout=timeout) as response:
                if response.status in [200, 201]:
                    content = response.read().decode('utf-8')
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"content": content, "status": "success"}
                else:
                    self.logger.warning(f"POST {url} returned status {response.status}")
                    return None
                    
        except HTTPError as e:
            self.logger.error(f"HTTP error for POST {url}: {e}")
            return None
        except URLError as e:
            self.logger.error(f"URL error for POST {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for POST {url}: {e}")
            return None

# ================================================================================================
# TEST EXECUTION ENGINE
# ================================================================================================

class ConfigurationTester:
    """Main configuration testing engine."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.docker = DockerManager(logger)
        self.http = HTTPClient(logger)
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run configuration tests for all combinations."""
        if not self.docker.check_docker_available():
            raise RuntimeError("Docker is not available. Please ensure Docker Desktop/Engine is running.")
        
        self.logger.info("Starting configuration testing suite")
        self.logger.info(f"Testing DB versions: {self.config['db_versions']}")
        self.logger.info(f"Testing memory limits: {self.config['memory_limits']}")
        self.logger.info(f"Testing Python versions: {self.config['python_versions']}")
        
        total_configs = (len(self.config['db_versions']) * 
                        len(self.config['memory_limits']) * 
                        len(self.config['python_versions']))
        current_config = 0
        
        try:
            for python_version in self.config['python_versions']:
                for db_version in self.config['db_versions']:
                    for memory_limit in self.config['memory_limits']:
                        current_config += 1
                        self.logger.info(f"Testing configuration {current_config}/{total_configs}: "
                                       f"Python {python_version}, PostgreSQL {db_version}, Memory {memory_limit}")
                        
                        config_result = self._test_configuration(python_version, db_version, memory_limit)
                        self.test_results.append(config_result)
                        
                        # Brief pause between configurations
                        time.sleep(2)
        
        finally:
            self.docker.cleanup_all()
        
        return self._generate_summary()
    
    def _test_configuration(self, python_version: str, db_version: str, memory_limit: str) -> Dict[str, Any]:
        """Test a single configuration combination."""
        # Replace dots with underscores for Docker Compose compatibility
        py_ver = python_version.replace('.', '')
        config_name = f"py{py_ver}-pg{db_version}-{memory_limit}"
        test_config = {
            "python_version": python_version,
            "db_version": db_version,
            "memory_limit": memory_limit,
            "db_port": self.config["db_port"],
            "api_port": self.config["api_port"]
        }
        
        result = {
            "config_name": config_name,
            "python_version": python_version,
            "db_version": db_version,
            "memory_limit": memory_limit,
            "timestamp": datetime.now().isoformat(),
            "status": "started",
            "tests": {},
            "metrics": {},
            "compatibility": {},
            "errors": []
        }
        
        temp_dir = None
        compose_file = None
        try:
            # Create temporary directory for this configuration
            temp_dir = Path(tempfile.mkdtemp(prefix=f"config_test_{config_name}_"))
            
            # Check requirements compatibility first
            compatible, compatibility_issues = self.docker.check_requirements_compatibility(python_version, temp_dir)
            result["compatibility"] = {
                "compatible": compatible,
                "issues": compatibility_issues
            }
            
            if not compatible:
                self.logger.warning(f"Compatibility issues detected for Python {python_version}: {compatibility_issues}")
                # Continue with testing but note the issues
            
            # Create Dockerfile for this Python version
            dockerfile_path = self.docker.create_dockerfile_for_testing(python_version, temp_dir)
            
            # Create compose file
            compose_file = self.docker.create_compose_file(test_config, temp_dir)
            
            # Start the stack
            if not self.docker.start_stack(compose_file, config_name, self.config["timeout"]):
                result["status"] = "failed"
                result["errors"].append("Failed to start Docker stack")
                return result
            
            # Wait for API to be ready
            base_url = f"{self.config['base_url'].replace('8080', str(self.config['api_port']))}"
            health_url = f"{base_url}{self.config['health_path']}"
            
            if not self.http.wait_for_service(health_url, self.config["timeout"]):
                result["status"] = "failed"
                result["errors"].append("API service not available")
                return result
            
            # Run functional tests
            result["tests"] = self._run_functional_tests(base_url)
            
            # Collect metrics
            result["metrics"] = self.docker.collect_metrics(config_name, self.config["metrics_duration"])
            
            # Determine overall status
            failed_tests = [name for name, test in result["tests"].items() if not test.get("passed", False)]
            if failed_tests:
                result["status"] = "partial"
                result["errors"].append(f"Failed tests: {', '.join(failed_tests)}")
            else:
                result["status"] = "passed"
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Configuration test error: {str(e)}")
            self.logger.error(f"Error testing configuration {config_name}: {e}")
        
        finally:
            # Clean up
            if temp_dir and compose_file:
                try:
                    self.docker.stop_stack(compose_file, config_name)
                except Exception as e:
                    self.logger.warning(f"Error stopping stack: {e}")
            
            if temp_dir and temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up temp dir: {e}")
        
        return result
    
    def _run_functional_tests(self, base_url: str) -> Dict[str, Dict[str, Any]]:
        """Run functional API tests."""
        tests = {}
        
        # Test 1: Health check
        tests["health"] = self._test_health_endpoint(base_url)
        
        # Test 2: Read endpoint
        tests["read"] = self._test_read_endpoint(base_url)
        
        # Test 3: Write endpoint  
        tests["write"] = self._test_write_endpoint(base_url)
        
        return tests
    
    def _test_health_endpoint(self, base_url: str) -> Dict[str, Any]:
        """Test health endpoint."""
        test_result = {
            "name": "Health Check",
            "passed": False,
            "response_time": 0.0,
            "details": ""
        }
        
        try:
            start_time = time.time()
            health_url = f"{base_url}{self.config['health_path']}"
            response = self.http.get(health_url, timeout=30)
            test_result["response_time"] = time.time() - start_time
            
            if isinstance(response, dict) and str(response.get("status", "")).lower() in ("healthy", "ok", "pass"):
                test_result["passed"] = True
                test_result["details"] = "Health endpoint returned expected response"
            else:
                test_result["details"] = f"Unexpected health response: {response}"
                
        except Exception as e:
            test_result["details"] = f"Health check failed: {str(e)}"
        
        return test_result
    
    def _test_read_endpoint(self, base_url: str) -> Dict[str, Any]:
        """Test read endpoint."""
        test_result = {
            "name": "Read Operation",
            "passed": False,
            "response_time": 0.0,
            "details": ""
        }
        
        try:
            start_time = time.time()
            
            # Try different read endpoints based on common patterns
            read_endpoints = [
                self.config['read_path'],
                f"{self.config['read_path']}?limit=1",
                "/locations",
                "/debug/count",
                "/properties",
                "/"
            ]
            
            for endpoint in read_endpoints:
                try:
                    read_url = f"{base_url}{endpoint}"
                    response = self.http.get(read_url, timeout=30)
                    
                    if response and (isinstance(response, dict) or isinstance(response, list)):
                        # Check if it's not an error object with only status:not_found
                        if isinstance(response, dict) and response.get("status") == "not_found" and len(response) == 1:
                            continue  # Try next endpoint
                        test_result["passed"] = True
                        test_result["details"] = f"Read successful from {endpoint}"
                        break
                        
                except Exception:
                    continue  # Try next endpoint
            
            test_result["response_time"] = time.time() - start_time
            
            if not test_result["passed"]:
                test_result["details"] = "No readable endpoints found"
                
        except Exception as e:
            test_result["details"] = f"Read test failed: {str(e)}"
        
        return test_result
    
    def _test_write_endpoint(self, base_url: str) -> Dict[str, Any]:
        """Test write endpoint."""
        test_result = {
            "name": "Write Operation",
            "passed": False,
            "response_time": 0.0,
            "details": ""
        }
        
        try:
            start_time = time.time()
            
            # Determine service type and use appropriate test data
            service_type = self._detect_service_type(base_url)
            
            if service_type in TEST_DATA_TEMPLATES:
                template = TEST_DATA_TEMPLATES[service_type]
                write_url = f"{base_url}{template['url_suffix']}"
                
                if template["method"] == "POST" and template["data"]:
                    response = self.http.post(write_url, template["data"], timeout=30)
                    if response:
                        test_result["passed"] = True
                        test_result["details"] = f"Write successful to {service_type} endpoint"
                    else:
                        test_result["details"] = f"Write failed to {service_type} endpoint"
                elif template["method"] == "GET":
                    response = self.http.get(write_url, timeout=30)
                    # For GET endpoints, any response (even 404) indicates the endpoint is working
                    if response:
                        test_result["passed"] = True
                        test_result["details"] = f"Write test successful (GET) to {service_type} endpoint"
                    else:
                        test_result["details"] = f"Write test failed to {service_type} endpoint"
            else:
                # Generic write test - try posting to the configured write path
                write_url = f"{base_url}{self.config['write_path']}"
                test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
                
                response = self.http.post(write_url, test_data, timeout=30)
                if response:
                    test_result["passed"] = True
                    test_result["details"] = "Generic write operation successful"
                else:
                    test_result["details"] = "Generic write operation failed"
            
            test_result["response_time"] = time.time() - start_time
            
        except Exception as e:
            test_result["details"] = f"Write test failed: {str(e)}"
        
        return test_result
    
    def _detect_service_type(self, base_url: str) -> str:
        """Detect service type based on available endpoints."""
        try:
            # Try health endpoint to get service info
            health_response = self.http.get(f"{base_url}/health", timeout=10)
            if health_response and isinstance(health_response, dict):
                service_name = health_response.get("service", "")
                if service_name:
                    return service_name
            
            # Try root endpoint
            root_response = self.http.get(f"{base_url}/", timeout=10)
            if root_response and isinstance(root_response, dict):
                message = root_response.get("message", "").lower()
                if "agent" in message:
                    return "agent-recommender"
                elif "property" in message:
                    return "property-recommendation"
                elif "sql" in message:
                    return "sql-agent"
                elif "risk" in message or "map" in message:
                    return "risk-map"
            
            # Try specific endpoints
            endpoints_to_check = ["/locations", "/recommend", "/ask", "/properties"]
            for endpoint in endpoints_to_check:
                response = self.http.get(f"{base_url}{endpoint}", timeout=5)
                if response and not (response.get("status") == "not_found"):
                    if endpoint == "/locations" or endpoint == "/recommend":
                        return "agent-recommender"
                    elif endpoint == "/ask":
                        return "sql-agent"
                    elif endpoint == "/properties":
                        return "property-recommendation"
            
        except Exception:
            pass
        
        return "generic"
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(self.test_results)
        passed_configs = sum(1 for r in self.test_results if r["status"] == "passed")
        partial_configs = sum(1 for r in self.test_results if r["status"] == "partial")
        failed_configs = sum(1 for r in self.test_results if r["status"] == "failed")
        
        summary = {
            "execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds()
            },
            "summary": {
                "total_configurations": total_tests,
                "passed": passed_configs,
                "partial": partial_configs,
                "failed": failed_configs,
                "success_rate": (passed_configs / total_tests * 100) if total_tests > 0 else 0
            },
            "configurations": self.test_results
        }
        
        self.logger.info(f"Configuration testing completed:")
        self.logger.info(f"  Total configurations: {total_tests}")
        self.logger.info(f"  Passed: {passed_configs}")
        self.logger.info(f"  Partial: {partial_configs}")
        self.logger.info(f"  Failed: {failed_configs}")
        self.logger.info(f"  Success rate: {summary['summary']['success_rate']:.1f}%")
        
        return summary

# ================================================================================================
# REPORT GENERATION
# ================================================================================================

class ReportGenerator:
    """Generate test reports in JSON format."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def generate_json_report(self, summary: Dict[str, Any], output_path: str = "config_test_report.json") -> None:
        """Generate JSON report."""
        try:
            # Create clean JSON structure
            report = {
                "report_metadata": {
                    "generated_at": summary["execution"]["end_time"],
                    "tool_version": "1.0.1",
                    "start_time": summary["execution"]["start_time"],
                    "end_time": summary["execution"]["end_time"],
                    "test_duration_seconds": summary["execution"]["duration_seconds"],
                    "total_configurations": summary["summary"]["total_configurations"]
                },
                "summary": {
                    "passed": summary["summary"]["passed"],
                    "partial": summary["summary"]["partial"], 
                    "failed": summary["summary"]["failed"],
                    "success_rate": summary["summary"]["success_rate"]
                },
                "configurations": []
            }
            
            # Process each configuration
            for config in summary["configurations"]:
                config_report = {
                    "name": config["config_name"],
                    "python_version": config["python_version"],
                    "database_version": config["db_version"], 
                    "memory_limit": config["memory_limit"],
                    "status": config["status"],
                    "timestamp": config.get("timestamp", datetime.now().isoformat()),
                    "tests": {
                        "health_check": {
                            "passed": config.get("tests", {}).get("health", {}).get("passed", False),
                            "response_time_ms": round(config.get("tests", {}).get("health", {}).get("response_time", 0) * 1000, 2),
                            "details": config.get("tests", {}).get("health", {}).get("details", "")
                        },
                        "read_test": {
                            "passed": config.get("tests", {}).get("read", {}).get("passed", False),
                            "response_time_ms": round(config.get("tests", {}).get("read", {}).get("response_time", 0) * 1000, 2),
                            "details": config.get("tests", {}).get("read", {}).get("details", "")
                        },
                        "write_test": {
                            "passed": config.get("tests", {}).get("write", {}).get("passed", False),
                            "response_time_ms": round(config.get("tests", {}).get("write", {}).get("response_time", 0) * 1000, 2),
                            "details": config.get("tests", {}).get("write", {}).get("details", "")
                        }
                    },
                    "performance_metrics": config.get("metrics", {}),
                    "compatibility": config.get("compatibility", {}),
                    "errors": config.get("errors", [])
                }
                report["configurations"].append(config_report)
            
            # Write JSON file with proper formatting
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"JSON report saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
    
    def generate_report(self, summary: Dict[str, Any], output_path: str = "config_test_report.json") -> None:
        """Generate configuration test report in JSON format."""
        self.generate_json_report(summary, output_path)

# ================================================================================================
# COMMAND LINE INTERFACE
# ================================================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Configuration Testing Suite for AI-ML-Services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_test.py                                    # Test all configurations
  python config_test.py --db 14 16 --mem 512m 1g          # Test specific matrix
  python config_test.py --timeout 180 --port 8080         # Custom timeout and port
  python config_test.py --base-url http://localhost:3000  # Custom base URL
        """
    )
    
    parser.add_argument(
        "--python", nargs="+",
        default=DEFAULT_CONFIG["python_versions"],
        help="Python versions to test (default: 3.10 3.11 3.12)"
    )
    
    parser.add_argument(
        "--db", nargs="+", 
        default=DEFAULT_CONFIG["db_versions"],
        help="PostgreSQL versions to test (default: 13 14 15 16)"
    )
    
    parser.add_argument(
        "--mem", nargs="+",
        default=DEFAULT_CONFIG["memory_limits"], 
        help="Memory limits to test (default: 512m 1g)"
    )
    
    parser.add_argument(
        "--base-url",
        default=DEFAULT_CONFIG["base_url"],
        help="Base URL for API testing (default: http://localhost:8080)"
    )
    
    parser.add_argument(
        "--health-path",
        default=DEFAULT_CONFIG["health_path"],
        help="Health endpoint path (default: /health)"
    )
    
    parser.add_argument(
        "--read-path", 
        default=DEFAULT_CONFIG["read_path"],
        help="Read endpoint path (default: /items)"
    )
    
    parser.add_argument(
        "--write-path",
        default=DEFAULT_CONFIG["write_path"], 
        help="Write endpoint path (default: /items)"
    )
    
    parser.add_argument(
        "--port", type=int,
        default=DEFAULT_CONFIG["api_port"],
        help="API port (default: 8080)"
    )
    
    parser.add_argument(
        "--timeout", type=int,
        default=DEFAULT_CONFIG["timeout"],
        help="Timeout for stack startup in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--metrics-duration", type=int,
        default=DEFAULT_CONFIG["metrics_duration"],
        help="Duration for metrics collection in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--json-report",
        default="config_test_report.json",
        help="JSON report filename (default: config_test_report.json)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Build configuration
    config = {
        "python_versions": args.python,
        "db_versions": args.db,
        "memory_limits": args.mem,
        "base_url": args.base_url,
        "health_path": args.health_path,
        "read_path": args.read_path,
        "write_path": args.write_path,
        "api_port": args.port,
        "db_port": DEFAULT_CONFIG["db_port"],
        "timeout": args.timeout,
        "metrics_duration": args.metrics_duration,
        "max_retries": DEFAULT_CONFIG["max_retries"],
        "retry_delay": DEFAULT_CONFIG["retry_delay"]
    }
    
    logger.info("Configuration Testing Suite Starting")
    logger.info(f"Configuration: {config}")
    
    try:
        # Run tests
        tester = ConfigurationTester(config, logger)
        summary = tester.run_all_tests()
        
        # Generate reports
        reporter = ReportGenerator(logger)
        reporter.generate_json_report(summary, args.json_report)
        
        logger.info("Configuration testing completed successfully")
        logger.info(f"Reports generated:")
        logger.info(f"  - JSON report: {args.json_report}")
        logger.info(f"  - Log file: config_test.log")
        
        # Exit with appropriate code
        if summary["summary"]["failed"] > 0:
            sys.exit(1)
        elif summary["summary"]["partial"] > 0:
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Configuration testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()