# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Guarded fixture for one immutable worker-local data load."""

import struct
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(struct.pack("<I", 0x11111111) + guard)
    (directory / "output.bin").write_bytes(struct.pack("<I", 0xDEADBEEF) + guard)
    (directory / "expected.bin").write_bytes(struct.pack("<I", 0x23456789) + guard)


if __name__ == "__main__":
    main()
