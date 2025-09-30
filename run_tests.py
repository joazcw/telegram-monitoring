#!/usr/bin/env python3
"""
Simple Test Runner for Fraud Monitoring System
Run basic tests to verify core functionality works.
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Run basic tests"""
    print("🧪 Running Fraud Monitoring System Tests")
    print("=" * 50)

    project_root = Path(__file__).parent

    # Run pytest with basic options
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return 1

if __name__ == "__main__":
    sys.exit(main())