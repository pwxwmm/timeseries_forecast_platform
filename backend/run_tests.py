#!/usr/bin/env python3
"""
Test Runner Script
Runs all tests to verify the new directory structure works correctly

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file: str, description: str):
    """Run a test file and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main():
    """Main test runner"""
    print("🧪 Backend Structure Test Runner")
    print("Testing the new directory structure...")

    backend_dir = Path(__file__).parent
    tests_dir = backend_dir / "tests"

    # Test files to run
    test_files = [
        (tests_dir / "test_models.py", "Model Testing"),
        (tests_dir / "quick_start.py", "Quick Start Demo"),
        (tests_dir / "example.py", "Usage Examples"),
    ]

    results = []

    for test_file, description in test_files:
        if test_file.exists():
            success = run_test(str(test_file), description)
            results.append((description, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((description, False))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    total = len(results)

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:20}: {status}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The new structure is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
