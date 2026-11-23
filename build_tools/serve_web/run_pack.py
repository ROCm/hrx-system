# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launches the configured host-side JavaScript generator for CMake."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.exit(
        subprocess.call(["node", Path(__file__).with_name("pack.mjs"), *sys.argv[1:]])
    )
