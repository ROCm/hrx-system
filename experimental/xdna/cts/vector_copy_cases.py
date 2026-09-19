# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete scalar oracles for the native C++ vector-copy packet corpus."""

import struct
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    inputs = []
    expected = []
    configurations = [
        (0, 1, 1, 0),
        (1, 3, 2, 1),
        (4, 2, 1, 1),
        (20, 1, 1, 0),
        (20, 3, 2, 1),
        (19, 2, 2, 0),
    ]
    for record, (count, source_stride, destination_stride, origin) in enumerate(
        configurations
    ):
        words = [
            ((record + 1) * 0x10203041 + word * 0x110001) & 0xFFFFFFFF
            for word in range(1024)
        ]
        words[:4] = [count, source_stride, destination_stride, origin]
        result = [0x6BADCAFE] * 1024
        for word in range(count * 16):
            block, lane = divmod(word, 16)
            result[(origin + block * destination_stride) * 16 + lane] = words[
                16 + block * source_stride * 16 + lane
            ]
        inputs.extend(words)
        expected.extend(result)
    # Extra host-binding bytes also expose writes beyond the pipeline's shape.
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(struct.pack(f"<{len(inputs)}I", *inputs))
    (directory / "output.bin").write_bytes(bytes([0xA5]) * (len(expected) * 4) + guard)
    (directory / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected) + guard
    )


if __name__ == "__main__":
    main()
