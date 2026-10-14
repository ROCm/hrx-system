#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Refreshes independent text oracles using CPython's Unicode 15.0 behavior."""

import struct
import unicodedata
from pathlib import Path


def main() -> None:
    if unicodedata.unidata_version != "15.0.0":
        raise SystemExit("The oracle requires Unicode 15.0.0 (CPython 3.12)")
    points = set()
    for edge in (0, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF):
        points.update((edge - 1, edge, edge + 1))
    for point in range(0x110000):
        character = chr(point)
        if character.upper() != character or character.isspace() or character.isdigit():
            points.update((point - 1, point, point + 1))
    rows = []
    for point in sorted(points):
        if not 0 <= point <= 0x10FFFF or 0xD800 <= point <= 0xDFFF:
            continue
        character = chr(point)
        uppercase = list(map(ord, character.upper()))
        rows.append(
            (
                point,
                *uppercase,
                *([0] * (3 - len(uppercase))),
                int(character.isspace()),
                int(character.isdigit()),
            )
        )
    # Each literal stays within the text format's 16-bit source columns.
    blocks = [rows[:1280], rows[1280:2560], rows[2560:]]
    declarations = []
    for name, block in zip(("low", "middle", "high"), blocks):
        data = b"".join(struct.pack("<6I", *row) for row in block)
        declarations.extend(
            [
                f"global.constant @oracle_{name}_count : index = {len(block)}",
                f'global.rodata.def @oracle_{name}_rows = align(4) bytes("{data.hex()}")',
            ]
        )
    path = Path(__file__).with_name("unicode.loom")
    prefix, body = path.read_text().split("// BEGIN GENERATED UNICODE ORACLE\n", 1)
    _, suffix = body.split("// END GENERATED UNICODE ORACLE", 1)
    path.write_text(
        prefix
        + "// BEGIN GENERATED UNICODE ORACLE\n"
        + "\n\n".join(declarations)
        + "\n\n// END GENERATED UNICODE ORACLE"
        + suffix
    )


if __name__ == "__main__":
    main()
