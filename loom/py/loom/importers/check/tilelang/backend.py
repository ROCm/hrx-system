# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang importer check backend descriptor."""

from __future__ import annotations

import argparse
from pathlib import Path

from loom.importers.check.loom_verifier import LoomOutputVerifier
from loom.importers.check.registry import Availability, has_module
from loom.importers.check.results import CheckResult
from loom.importers.check.tilelang.runner import TILELANG_ORACLE_MODES


class TileLangBackend:
    name: str = "tilelang"
    help: str = "check TileLang kernel imports"
    extras: tuple[str, ...] = ("tilelang",)
    aliases: tuple[str, ...] = ()

    def probe(self) -> Availability:
        if not has_module("tilelang"):
            return Availability.unavailable(
                reason="Python package `tilelang` is not importable",
                install_hint="install iree-loom[tilelang]",
            )
        return Availability.yes()

    def prepare(self, _args: argparse.Namespace) -> Availability:
        from loom.importers.check.tilelang.harness import (
            TileLangHarness,
            TileLangHarnessError,
        )

        try:
            TileLangHarness()
        except TileLangHarnessError as exc:
            return Availability.unavailable(reason=str(exc))
        return Availability.yes()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="+", type=Path)
        parser.add_argument(
            "--target-preset",
            help="override the target preset imported into the Loom module",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help="update inline expected output sections",
        )
        parser.add_argument(
            "--oracle",
            choices=TILELANG_ORACLE_MODES,
            default="off",
            help=(
                "optionally capture TileLang generated source or code-object "
                "oracle artifacts when the local toolchain supports them"
            ),
        )
        parser.add_argument(
            "--oracle-output-dir",
            type=Path,
            help=(
                "directory for TileLang oracle artifacts; code-object capture "
                "requires this or --dump-temp-dir"
            ),
        )

    def run(self, args: argparse.Namespace) -> tuple[CheckResult, ...]:
        from loom.importers.check.tilelang.runner import (
            TileLangCheckOptions,
            TileLangOracleCheckOptions,
            run_tilelang_check,
        )

        options = TileLangCheckOptions(
            update=args.update,
            target_preset=args.target_preset,
            case_filter=args.case_filter,
            output_verifier=LoomOutputVerifier.resolve(args.loom_opt),
            oracle=TileLangOracleCheckOptions(
                mode=args.oracle,
                output_directory=args.oracle_output_dir or args.dump_temp_dir,
            ),
        )
        return tuple(
            result
            for path in args.paths
            for result in run_tilelang_check(path, options=options)
        )
