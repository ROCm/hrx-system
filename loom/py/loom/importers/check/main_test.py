# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import contextlib
import io
from pathlib import Path

from loom.importers.check.main import main
from loom.importers.check.registry import Availability, BackendRegistry
from loom.importers.check.results import CheckResult


class _UnavailableBackend:
    name: str = "missing"
    help: str = "missing importer"
    extras: tuple[str, ...] = ("missing",)
    aliases: tuple[str, ...] = ()

    def probe(self) -> Availability:
        return Availability.unavailable(
            reason="package missing",
            install_hint="install extra",
        )

    def prepare(self, _args: argparse.Namespace) -> Availability:
        raise AssertionError("unavailable backend should not be prepared")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", type=Path)

    def run(self, _args: argparse.Namespace) -> tuple[CheckResult, ...]:
        raise AssertionError("unavailable backend should not run")


class _EmptyBackend:
    name: str = "empty"
    help: str = "empty importer"
    extras: tuple[str, ...] = ("empty",)
    aliases: tuple[str, ...] = ()

    def probe(self) -> Availability:
        return Availability.yes()

    def prepare(self, _args: argparse.Namespace) -> Availability:
        return Availability.yes()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", type=Path)

    def run(self, _args: argparse.Namespace) -> tuple[CheckResult, ...]:
        return ()


def test_main_skips_unavailable_importer_by_default() -> None:
    registry = BackendRegistry()
    registry.register(_UnavailableBackend())
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        exit_code = main(["missing", "case.py"], registry=registry)

    assert exit_code == 0
    assert "skipped: case.py" in stdout.getvalue()
    assert "package missing; install extra" in stdout.getvalue()
    assert "summary: 0 passed, 0 updated, 1 skipped, 0 failed, 0 crashed" in (
        stdout.getvalue()
    )


def test_main_fail_on_skip_makes_unavailable_importer_fail() -> None:
    registry = BackendRegistry()
    registry.register(_UnavailableBackend())
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        exit_code = main(
            ["missing", "--fail-on-skip", "case.py"],
            registry=registry,
        )

    assert exit_code == 1
    assert "skipped: case.py" in stdout.getvalue()


def test_main_reports_empty_filtered_result_as_failure() -> None:
    registry = BackendRegistry()
    registry.register(_EmptyBackend())
    stderr = io.StringIO()

    with contextlib.redirect_stderr(stderr):
        exit_code = main(["empty", "--filter", "missing"], registry=registry)

    assert exit_code == 1
    assert stderr.getvalue() == "error: no importer check cases matched\n"


def test_main_prints_agent_markdown_without_importer() -> None:
    registry = BackendRegistry()
    registry.register(_EmptyBackend())
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        exit_code = main(["--agents_md"], registry=registry)

    assert exit_code == 0
    output = stdout.getvalue()
    assert "## loom-import-check" in output
    assert "loom_import_check -- \\" in output
    assert "tilelang --update path/to/case.py" in output
    assert "enabled  empty" in output
