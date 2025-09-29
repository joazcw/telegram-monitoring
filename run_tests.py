#!/usr/bin/env python3
"""
Test Runner for Fraud Monitoring System

This script runs all tests (unit and integration) and provides
a comprehensive report of the system's health and functionality.
"""
import sys
import os
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_banner():
    """Print test runner banner"""
    print("=" * 70)
    print("🧪 FRAUD MONITORING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()


def print_section(title):
    """Print section header"""
    print("\n" + "─" * 50)
    print(f"📋 {title}")
    print("─" * 50)


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        print(f"✅ {description} - PASSED")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False, e.stderr


def check_prerequisites():
    """Check system prerequisites"""
    print_section("CHECKING PREREQUISITES")

    checks = []

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"❌ Python version too old: {python_version}")
        checks.append(False)

    # Check essential directories
    essential_dirs = ['src', 'tests', 'scripts', 'migrations']
    for dir_name in essential_dirs:
        if (project_root / dir_name).exists():
            print(f"✅ Directory exists: {dir_name}")
            checks.append(True)
        else:
            print(f"❌ Missing directory: {dir_name}")
            checks.append(False)

    # Check essential files
    essential_files = [
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.env.example',
        'src/main.py',
        'src/config.py'
    ]

    for file_name in essential_files:
        if (project_root / file_name).exists():
            print(f"✅ File exists: {file_name}")
            checks.append(True)
        else:
            print(f"❌ Missing file: {file_name}")
            checks.append(False)

    return all(checks)


def run_unit_tests():
    """Run unit tests"""
    print_section("RUNNING UNIT TESTS")

    if not (project_root / "tests" / "test_unit.py").exists():
        print("❌ Unit test file not found")
        return False

    success, output = run_command(
        f"python -m pytest tests/test_unit.py -v --tb=short",
        "Unit Tests"
    )

    if success:
        print("📊 Unit Test Summary:")
        # Parse pytest output for summary
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                print(f"   {line}")
            elif line.strip().startswith('PASSED') or line.strip().startswith('FAILED'):
                print(f"   {line.strip()}")

    return success


def run_integration_tests():
    """Run integration tests"""
    print_section("RUNNING INTEGRATION TESTS")

    if not (project_root / "tests" / "test_integration.py").exists():
        print("❌ Integration test file not found")
        return False

    success, output = run_command(
        f"python -m pytest tests/test_integration.py -v --tb=short",
        "Integration Tests"
    )

    if success:
        print("📊 Integration Test Summary:")
        # Parse pytest output for summary
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and ('failed' in line or 'error' in line):
                print(f"   {line}")

    return success


def run_component_health_checks():
    """Run individual component health checks"""
    print_section("COMPONENT HEALTH CHECKS")

    # Test imports
    components = [
        ("OCR Engine", "from src.ocr_engine import OCREngine; OCREngine()"),
        ("Brand Matcher", "from src.brand_matcher import BrandMatcher; BrandMatcher()"),
        ("Alert Sender", "from src.alert_sender import AlertSender; AlertSender()"),
        ("Telegram Collector", "from src.telegram_collector import TelegramCollector; TelegramCollector()"),
        ("Processor", "from src.processor import FraudProcessor; FraudProcessor()"),
        ("Main App", "from src.main import FraudMonitorApp; FraudMonitorApp()"),
        ("Database", "from src.database import SessionLocal; SessionLocal()"),
        ("Config", "from src.config import get_config_summary; get_config_summary()")
    ]

    results = []
    for component_name, test_code in components:
        success, output = run_command(
            f"python -c \"{test_code}\"",
            f"{component_name} Import Test"
        )
        results.append(success)

    return all(results)


def run_linting_checks():
    """Run code quality checks"""
    print_section("CODE QUALITY CHECKS")

    # Check if files can be imported without syntax errors
    python_files = [
        "src/config.py",
        "src/models.py",
        "src/database.py",
        "src/ocr_engine.py",
        "src/brand_matcher.py",
        "src/alert_sender.py",
        "src/telegram_collector.py",
        "src/processor.py",
        "src/main.py"
    ]

    results = []
    for file_path in python_files:
        if (project_root / file_path).exists():
            success, output = run_command(
                f"python -m py_compile {file_path}",
                f"Syntax Check: {file_path}"
            )
            results.append(success)
        else:
            print(f"⚠️  File not found: {file_path}")
            results.append(False)

    return all(results)


def test_docker_configuration():
    """Test Docker configuration"""
    print_section("DOCKER CONFIGURATION TESTS")

    results = []

    # Check Docker Compose syntax
    if (project_root / "docker-compose.yml").exists():
        success, output = run_command(
            "docker-compose config",
            "Docker Compose Syntax Check"
        )
        results.append(success)
    else:
        print("❌ docker-compose.yml not found")
        results.append(False)

    # Check Dockerfile exists
    if (project_root / "Dockerfile").exists():
        print("✅ Dockerfile exists")
        results.append(True)
    else:
        print("❌ Dockerfile not found")
        results.append(False)

    return all(results)


def generate_test_report(results):
    """Generate comprehensive test report"""
    print_section("TEST REPORT SUMMARY")

    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests

    print(f"📊 Overall Test Results:")
    print(f"   Total Test Categories: {total_tests}")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ❌ Failed: {failed_tests}")
    print(f"   📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()

    print("📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")

    print()

    if all(results.values()):
        print("🎉 ALL TESTS PASSED!")
        print("✅ The Fraud Monitoring System is ready for deployment!")
        print()
        print("Next Steps:")
        print("1. Set up your .env file with Telegram credentials")
        print("2. Run: ./scripts/auth.py to authenticate Telegram")
        print("3. Deploy: ./scripts/deploy.sh")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please review the failed tests above before deployment.")
        print()
        print("Common issues:")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Missing external tools (Tesseract, Docker)")
        print("- Configuration issues (.env file)")

    return all(results.values())


def main():
    """Main test runner"""
    print_banner()

    start_time = time.time()

    # Run all test categories
    test_results = {}

    test_results["Prerequisites"] = check_prerequisites()
    test_results["Code Quality"] = run_linting_checks()
    test_results["Component Health"] = run_component_health_checks()
    test_results["Unit Tests"] = run_unit_tests()
    test_results["Integration Tests"] = run_integration_tests()
    test_results["Docker Configuration"] = test_docker_configuration()

    # Generate final report
    end_time = time.time()
    duration = end_time - start_time

    print(f"\n⏱️  Total test duration: {duration:.2f} seconds")

    success = generate_test_report(test_results)

    print("\n" + "=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)