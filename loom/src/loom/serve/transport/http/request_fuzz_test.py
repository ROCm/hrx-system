# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Mutates raw and generated HTTP requests across receive fragmentation."""

import subprocess
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="http-request-fuzz-") as temporary:
        corpus = Path(temporary)
        (corpus / "empty").write_bytes(b"")
        (corpus / "headers").write_bytes(
            b"GET /v1/models HTTP/1.1\r\nHost: local\r\n\r\n"
        )
        subprocess.run(
            [str(executable), "-runs=4096", "-max_len=65536", str(corpus)],
            cwd=corpus,
            check=True,
        )
