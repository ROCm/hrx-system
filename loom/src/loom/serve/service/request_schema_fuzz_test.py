# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs bounded mutations through the native loom-serve request schemas."""

import subprocess
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    executable = Path(sys.argv[1]).resolve()
    seeds = sorted(Path(seed).resolve() for seed in sys.argv[2:])
    with tempfile.TemporaryDirectory(prefix="request-schema-fuzz-") as corpus:
        subprocess.run(
            [
                str(executable),
                "-runs=4096",
                "-max_len=65536",
                "-seed_inputs=" + ",".join(map(str, seeds)),
                corpus,
            ],
            cwd=corpus,
            check=True,
        )
