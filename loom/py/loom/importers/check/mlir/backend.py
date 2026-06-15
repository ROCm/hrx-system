# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR importer check backend descriptor."""

from __future__ import annotations

import argparse
from pathlib import Path

from loom.importers.check.loom_verifier import LoomOutputVerifier
from loom.importers.check.registry import Availability, has_module
from loom.importers.check.results import CheckResult


class MlirBackend:
    name: str = "mlir"
    help: str = "check MLIR kernel imports"
    extras: tuple[str, ...] = ("mlir",)
    aliases: tuple[str, ...] = ()

    def probe(self) -> Availability:
        if has_module("iree.compiler"):
            return Availability.yes()
        return Availability.unavailable(
            reason="Python package `iree.compiler` is not importable",
            install_hint="install iree-loom[mlir] or iree-base-compiler",
        )

    def prepare(self, args: argparse.Namespace) -> Availability:
        from loom.diagnostics import LoomDiagnosticError
        from loom.importers.mlir.api import import_iree_ir

        try:
            import_iree_ir(prefer_abi3_extensions=args.prefer_abi3_extensions)
        except LoomDiagnosticError as exc:
            return Availability.unavailable(reason=str(exc).strip())
        return Availability.yes()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="+", type=Path)
        parser.add_argument("--kernel")
        parser.add_argument(
            "--update",
            action="store_true",
            help="update inline expected output sections",
        )
        parser.add_argument(
            "--prefer-abi3-extensions",
            action="store_true",
            help="prefer local .abi3.so IREE compiler bindings",
        )

    def run(self, args: argparse.Namespace) -> tuple[CheckResult, ...]:
        from loom.importers.check.mlir.runner import MlirCheckOptions, run_mlir_check

        options = MlirCheckOptions(
            kernel=args.kernel,
            prefer_abi3_extensions=args.prefer_abi3_extensions,
            update=args.update,
            case_filter=args.case_filter,
            output_verifier=LoomOutputVerifier.resolve(args.loom_opt),
        )
        return tuple(
            result
            for path in args.paths
            for result in run_mlir_check(path, options=options)
        )
