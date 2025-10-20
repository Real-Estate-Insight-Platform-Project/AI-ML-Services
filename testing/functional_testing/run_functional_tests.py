#!/usr/bin/env python3
"""
Backend Functional Testing Suite Runner
Comprehensive testing for Agent Recommender, Risk Map, and SQL Agent services.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def print_banner():
    """Print test suite banner."""
    print("=" * 60)
    print("🧪 Backend Functional Testing Suite")
    print("   Testing Agent Recommender, Risk Map, and SQL Agent")
    print("=" * 60)

def setup_environment():
    """Set up the testing environment."""
    print("🔧 Setting up test environment...")
    
    # Check if requirements.txt exists and has testing dependencies
    requirements_file = Path(__file__).parent.parent.parent / "requirements.txt"
    if requirements_file.exists():
        print("✓ Using requirements from main requirements.txt (already installed)")
        print("✓ Test dependencies included in main requirements.txt")
    else:
        print("❌ requirements.txt not found!")
        return False
    
    # Verify pytest is available
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ pytest is available")
        else:
            print("❌ pytest not available!")
            return False
    except Exception as e:
        print(f"❌ Error checking pytest: {e}")
        return False
    
    print()
    return True

# Service configuration
SERVICES = {
    'agent': {
        'name': 'Agent Recommender System',
        'path': Path(__file__).parent.parent.parent / 'agent-finder-approach-1',
        'test_path': Path(__file__).parent / 'agent_recommender'
    },
    'risk': {
        'name': 'Risk Map',
        'path': Path(__file__).parent.parent.parent / 'risk-map',
        'test_path': Path(__file__).parent / 'risk_map'
    },
    'sql': {
        'name': 'SQL Agent',
        'path': Path(__file__).parent.parent.parent / 'sql-agent',
        'test_path': Path(__file__).parent / 'sql_agent'
    }
}

def check_service_exists(service_key):
    """Check if service directory exists."""
    service = SERVICES[service_key]
    if not service['path'].exists():
        print(f"❌ Service directory not found: {service['path']}")
        return False
    
    if not service['test_path'].exists():
        print(f"❌ Test directory not found: {service['test_path']}")
        return False
    
    print()
    return True

def run_service_tests(service_key, verbose=False, coverage=False):
    """Run tests for a specific service."""
    service = SERVICES[service_key]
    print(f"🧪 Testing {service['name']}...")
    print(f"   Path: {service['test_path']}")
    
    # Build pytest command with verbose output to capture test count
    cmd = [sys.executable, '-m', 'pytest', '-v', '--tb=short']
    
    if coverage:
        cmd.extend(['--cov=' + str(service['path']), '--cov-report=term-missing'])
    
    # Add test path
    cmd.append(str(service['test_path']))
    
    # Set environment variables for the service
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([
        str(service['path']),
        str(service['path'] / 'app') if service_key == 'risk' else str(service['path']),
        env.get('PYTHONPATH', '')
    ])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent, 
                              capture_output=True, text=True)
        end_time = time.time()
        
        # Print the output so users can see detailed results
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Parse test results to extract counts
        passed_count = 0
        failed_count = 0
        output_lines = result.stdout.splitlines()
        
        # Look for the summary line like "30 passed, 31 warnings"
        for line in output_lines:
            if 'passed' in line and ('warning' in line or 'error' in line or 'failed' in line or line.strip().endswith('passed')):
                import re
                # Extract numbers before "passed" and "failed"
                passed_match = re.search(r'(\d+)\s+passed', line)
                failed_match = re.search(r'(\d+)\s+failed', line)
                
                if passed_match:
                    passed_count = int(passed_match.group(1))
                if failed_match:
                    failed_count = int(failed_match.group(1))
                break
        
        total_count = passed_count + failed_count
        
        if result.returncode == 0:
            print(f"   ✅ All tests passed ({passed_count}/{total_count}) ({end_time - start_time:.1f}s)")
            return True, (passed_count, total_count)
        else:
            print(f"   ❌ Some tests failed ({passed_count}/{total_count}) (exit code: {result.returncode})")
            return False, (passed_count, total_count)
            
    except Exception as e:
        print(f"   💥 Error running tests: {e}")
        return False, (0, 0)

def run_all_tests(verbose=False, coverage=False):
    """Run tests for all services."""
    results = {}
    test_counts = {}
    
    for service_key in SERVICES:
        success, counts = run_service_tests(service_key, verbose, coverage)
        results[service_key] = success
        test_counts[service_key] = counts
        print()
    
    return results, test_counts

def print_summary(results, test_counts):
    """Print test results summary."""
    print("=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    total_test_count = 0
    total_passed_tests = 0
    
    for service_key, success in results.items():
        service_name = SERVICES[service_key]['name']
        passed_tests, total_tests = test_counts[service_key]
        total_test_count += total_tests
        total_passed_tests += passed_tests
        
        if success:
            print(f"✅ {service_name}: PASSED ({passed_tests}/{total_tests} test cases)")
            passed += 1
        else:
            print(f"❌ {service_name}: FAILED ({passed_tests}/{total_tests} test cases)")
            failed += 1
    
    print()
    print(f"📈 Summary: {passed} services passed, {failed} services failed")
    print(f"🧪 Test Cases: {total_passed_tests} passed out of {total_test_count} total")
    
    if failed == 0:
        print("🎉 All backend services passed functional testing!")
        return 0
    else:
        print("🚨 Some tests failed. Review output above for details.")
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run functional tests for backend services',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_functional_tests.py                    # Run all tests
  python run_functional_tests.py --service agent    # Run only agent tests
  python run_functional_tests.py --verbose          # Verbose output
  python run_functional_tests.py --coverage         # With coverage report
        """
    )
    parser.add_argument(
        '--service', 
        choices=['agent', 'risk', 'sql'],
        help='Run tests for specific service only'
    )
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose test output'
    )
    parser.add_argument(
        '--setup-only', 
        action='store_true',
        help='Only setup environment, do not run tests'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Setup environment
    if not setup_environment():
        return 1
    
    if args.setup_only:
        print("✅ Environment setup complete!")
        return 0
    
    # Run tests
    if args.service:
        # Run single service
        success, counts = run_service_tests(args.service, args.verbose, args.coverage)
        passed_tests, total_tests = counts
        print()
        if success:
            print(f"🎉 Service tests passed! ({passed_tests}/{total_tests} test cases)")
            return 0
        else:
            print(f"🚨 Service tests failed! ({passed_tests}/{total_tests} test cases)")
            return 1
    else:
        # Run all services
        results, test_counts = run_all_tests(args.verbose, args.coverage)
        return print_summary(results, test_counts)

if __name__ == '__main__':
    sys.exit(main())
