#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Refreshes Unicode data rows without generating interpreter logic.

Inputs are the Unicode 15.0.0 UCD files UnicodeData.txt and SpecialCasing.txt
from https://www.unicode.org/Public/15.0.0/ucd/. Their pinned hashes keep
normalization independent of the generating Python interpreter's UCD version.
"""

import argparse
import hashlib
import struct
from pathlib import Path

HASHES = {
    "UnicodeData.txt": "806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73",
    "SpecialCasing.txt": "78b29c64b5840d25c11a9f31b665ee551b8a499eca6c70d770fcad7dd710f494",
}
BEGIN = "// BEGIN GENERATED UNICODE DATA\n"
END = "// END GENERATED UNICODE DATA"


def read_input(directory: Path, name: str) -> str:
    data = (directory / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError(f"{name} does not match Unicode 15.0.0")
    return data.decode("utf-8")


def ranges(points: set[int]) -> list[tuple[int, int]]:
    result = []
    for point in sorted(points):
        if result and point == result[-1][1] + 1:
            result[-1] = (result[-1][0], point)
        else:
            result.append((point, point))
    return result


def tables(directory: Path) -> dict[str, list[tuple[int, ...]]]:
    uppercase = {}
    spaces = set()
    digits = set()
    for line in read_input(directory, "UnicodeData.txt").splitlines():
        fields = line.split(";")
        point = int(fields[0], 16)
        if fields[12]:
            uppercase[point] = (int(fields[12], 16),)
        if fields[2] == "Zs" or fields[4] in ("WS", "B", "S"):
            spaces.add(point)
        if fields[7]:
            digits.add(point)
    for line in read_input(directory, "SpecialCasing.txt").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(";")]
        if not fields[4]:
            uppercase[int(fields[0], 16)] = tuple(
                int(value, 16) for value in fields[3].split()
            )
    uppercase = {key: value for key, value in uppercase.items() if value != (key,)}
    if (len(uppercase), len(spaces), len(digits)) != (1525, 29, 808):
        raise ValueError("Unicode family extent does not match the pinned UCD")
    if any(not 1 <= len(value) <= 3 or 0 in value for value in uppercase.values()):
        raise ValueError("uppercase rows require 1–3 nonzero codepoints")
    return {
        "uppercase": [
            (key, *value, *([0] * (3 - len(value))))
            for key, value in sorted(uppercase.items())
        ],
        "space": ranges(spaces),
        "digit": ranges(digits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ucd_directory", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    declarations = []
    for name, rows in tables(args.ucd_directory).items():
        data = b"".join(struct.pack("<" + "I" * len(row), *row) for row in rows)
        declarations.extend(
            [
                f"global.constant @text.{name}_count : index = {len(rows)}",
                f'global.rodata.def @text.{name}_rows = align(4) bytes("{data.hex()}")',
            ]
        )
    path = Path(__file__).with_name("unicode.loom")
    source = path.read_text()
    prefix, body = source.split(BEGIN, 1)
    _, suffix = body.split(END, 1)
    generated = prefix + BEGIN + "\n\n".join(declarations) + "\n\n" + END + suffix
    if args.check:
        if generated != source:
            raise SystemExit("Unicode data rows require regeneration")
    else:
        path.write_text(generated)


if __name__ == "__main__":
    main()
