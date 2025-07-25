#!/usr/bin/env python
"""Script to run schema unit tests."""

import subprocess
import sys


def main():
    """Run unit tests for schema models."""
    print("Running schema unit tests...")
    print("-" * 60)

    # Run pytest for schema tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/schema/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ All schema tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
