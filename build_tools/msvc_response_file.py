# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_ASCII_WHITESPACE = frozenset(b" \t\r\n\v\f")


def one_argument_per_line(contents: bytes) -> bytes:
    """Rewraps an MSVC response file without changing its arguments."""
    output = bytearray()
    in_quotes = False
    pending_separator = False
    preceding_backslash_count = 0

    for value in contents:
        if value in _ASCII_WHITESPACE and not in_quotes:
            pending_separator = bool(output)
            preceding_backslash_count = 0
            continue

        if pending_separator:
            output.extend(b"\r\n")
            pending_separator = False
        output.append(value)

        if value == ord("\\"):
            preceding_backslash_count += 1
        else:
            if value == ord('"') and preceding_backslash_count % 2 == 0:
                in_quotes = not in_quotes
            preceding_backslash_count = 0

    if in_quotes:
        raise ValueError("unterminated quote in MSVC response file")
    if output:
        output.extend(b"\r\n")
    return bytes(output)


def rewrap_response_file(path: Path) -> None:
    """Atomically rewrites |path| with one argument per physical line."""
    original_contents = path.read_bytes()
    rewrapped_contents = one_argument_per_line(original_contents)
    if rewrapped_contents == original_contents:
        return

    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_bytes(rewrapped_contents)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewraps an MSVC response file to one argument per line."
    )
    parser.add_argument("response_file", type=Path)
    args = parser.parse_args()
    rewrap_response_file(args.response_file)


if __name__ == "__main__":
    main()
