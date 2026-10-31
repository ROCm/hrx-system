# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks that CTS cleanup failures survive the real process exit boundary."""

import subprocess
import sys
from pathlib import Path


def main():
    executable, library = (str(Path(argument).resolve()) for argument in sys.argv[1:])
    cases = [
        ("Runtime version is valid", 0, ""),
        (
            "Fixture cleanup error reaches process exit",
            1,
            "fixture cleanup error witness",
        ),
        (
            "Accelerator cleanup error reaches process exit",
            1,
            "accelerator cleanup error witness",
        ),
    ]
    for name, expected_exit, diagnostic in cases:
        result = subprocess.run(
            [executable, "--hrx-library", library, "--hrx-device", "cpu:0", name],
            capture_output=True,
            text=True,
        )
        if result.returncode != expected_exit or diagnostic not in result.stderr:
            raise AssertionError(
                f"{name}: expected exit {expected_exit} and {diagnostic!r}; "
                f"got exit {result.returncode}\n{result.stdout}\n{result.stderr}"
            )
        if "All tests passed" not in result.stdout:
            raise AssertionError(
                f"{name}: test body failed or did not run\n{result.stdout}"
            )


if __name__ == "__main__":
    main()
