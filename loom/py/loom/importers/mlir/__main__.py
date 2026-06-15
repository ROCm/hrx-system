# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line wrapper for the MLIR-to-Loom importer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loom.diagnostics import LoomDiagnosticError
from loom.importers.core import kernel_module_ops, print_loom_module
from loom.importers.mlir.importer import (
    MlirImportOptions,
    format_import_report,
    import_mlir_file,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    options = MlirImportOptions(
        kernel=args.kernel,
        include_report=True,
        prefer_abi3_extensions=args.prefer_abi3_extensions,
    )
    try:
        result = import_mlir_file(args.input, options=options)
    except LoomDiagnosticError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    output_text = print_loom_module(
        result.module,
        ops=kernel_module_ops(
            getattr(result.report, "target_format", None) or "unknown"
        ),
        print_locations=args.print_locations,
    )
    if args.output is None:
        sys.stdout.write(output_text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
    if args.report is not None:
        if result.report is None:
            raise AssertionError("include_report option did not produce a report")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            format_import_report(
                result.report,
                source_path=args.input,
                output_path=args.output,
            )
        )
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HAL/codegen-shaped MLIR input")
    parser.add_argument("-o", "--output", type=Path, help="Loom text output")
    parser.add_argument("--kernel", help="Kernel/export name to import")
    parser.add_argument("--report", type=Path, help="Optional Markdown import report")
    parser.add_argument(
        "--print-locations",
        action="store_true",
        help="Print Loom loc(...) annotations for converted MLIR operations",
    )
    parser.add_argument(
        "--prefer-abi3-extensions",
        action="store_true",
        help=(
            "Prefer local .abi3.so MLIR binding extensions over "
            "CPython-specific siblings"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
