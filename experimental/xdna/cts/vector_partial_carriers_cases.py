# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Raw-byte oracle for captured and partial native vector carriers."""

import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    inputs = bytearray()
    expected = bytearray()
    for record in range(16):
        source = bytearray(
            ((record + 1) * 37 + position * 13) & 0xFF for position in range(256)
        )
        source[:4] = (1).to_bytes(4, "little")
        payload = source[64:192]
        result = bytearray([0xA5] * 512)
        result[0:128] = payload
        result[128:192] = payload[63:127]
        result[192:288] = payload[1:97]
        result[288:416] = payload[80:81] + payload[0:80] + payload[81:128]
        result[416:481] = payload[0:65]
        inputs += source
        expected += result
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xCD]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
