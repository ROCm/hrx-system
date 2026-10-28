# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exports cleaned Loom High IR as readable Loom C++."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, TextIO

from loom.diagnostics import LoomDiagnosticError
from loom.format.bytecode.reader import BytecodeError
from loom.format.cxx import CxxRenderError, load_module, render_module
from loom.format.text.tokenizer import ParseError


def run(
    input_path: str,
    output_path: str,
    *,
    roots: Sequence[str] = (),
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: TextIO,
) -> int:
    """Writes UTF-8 C++ source and returns its public process exit code."""

    try:
        if input_path == "-":
            data = stdin.read()
            filename = "<stdin>"
        else:
            path = Path(input_path)
            data = path.read_bytes()
            filename = str(path)
        result = render_module(
            load_module(data, filename=filename), roots=roots
        ).encode("utf-8")
        if output_path == "-":
            stdout.write(result)
        else:
            Path(output_path).write_bytes(result)
    except (
        BytecodeError,
        CxxRenderError,
        LoomDiagnosticError,
        OSError,
        ParseError,
        ValueError,
    ) as exc:
        stderr.write(f"loom-export-cxx: error: {exc}\n")
        return 2
    return 0


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-export-cxx",
        description="Exports cleaned Loom High IR as readable Loom C++.",
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="A .loom or .loombc module, or '-' for standard input.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="OUTPUT",
        help="Output .cxx path, or '-' for standard output (the default).",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="@SYMBOL",
        help="Kernel root to render. Repeat to select multiple kernels.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _create_argument_parser().parse_args(argv)
    return run(
        args.input,
        args.output,
        roots=args.root,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
        stderr=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
