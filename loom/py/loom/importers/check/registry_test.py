# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path

from loom.importers.check.registry import (
    Availability,
    BackendRegistry,
    RegisteredBackend,
    skip_unavailable_backend,
)
from loom.importers.check.results import CheckResult, summarize_results


class _Backend:
    name: str = "test"
    help: str = "check test imports"
    extras: tuple[str, ...] = ("test",)
    aliases: tuple[str, ...] = ("alias",)

    def probe(self) -> Availability:
        return Availability.yes()

    def prepare(self, args: argparse.Namespace) -> Availability:
        return Availability.yes()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", type=Path)

    def run(self, args: argparse.Namespace) -> tuple[CheckResult, ...]:
        return ()


def test_registry_resolves_backend_alias() -> None:
    registry = BackendRegistry()
    backend = _Backend()

    registry.register(backend)

    by_name = registry.by_name("test")
    by_alias = registry.by_name("alias")

    assert by_name is not None
    assert by_alias is not None
    assert by_name.backend is backend
    assert by_alias.backend is backend


def test_registry_rejects_duplicate_alias() -> None:
    registry = BackendRegistry()
    registry.register(_Backend())

    message = ""
    try:
        registry.register(_Backend())
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected duplicate backend registration to fail")
    assert "duplicate check backend name" in message


def test_skip_unavailable_backend_reports_skipped_results() -> None:
    registered = RegisteredBackend(
        backend=_Backend(),
        availability=Availability.unavailable(reason="package missing"),
    )

    results = skip_unavailable_backend(
        registered,
        argparse.Namespace(paths=[Path("case.py")]),
        Availability.unavailable(
            reason="package missing", install_hint="install extra"
        ),
    )

    assert results == (
        CheckResult(
            path=Path("case.py"),
            case_index=-1,
            returncode=0,
            stdout="",
            stderr="package missing; install extra\n",
            skipped=True,
        ),
    )
    assert summarize_results(results) == {
        "passed": 0,
        "updated": 0,
        "skipped": 1,
        "failed": 0,
        "crashed": 0,
    }
