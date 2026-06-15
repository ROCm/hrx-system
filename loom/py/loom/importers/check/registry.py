# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional importer check backend registry."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loom.importers.check.results import CheckResult


@dataclass(frozen=True, slots=True)
class Availability:
    """Availability of an optional importer check backend."""

    available: bool
    reason: str = ""
    install_hint: str = ""

    @staticmethod
    def yes() -> Availability:
        return Availability(True)

    @staticmethod
    def unavailable(*, reason: str, install_hint: str = "") -> Availability:
        return Availability(False, reason=reason, install_hint=install_hint)

    def message(self) -> str:
        if self.install_hint:
            return f"{self.reason}; {self.install_hint}"
        return self.reason


class CheckBackend(Protocol):
    """Lightweight descriptor for one importer check backend."""

    name: str
    help: str
    extras: tuple[str, ...]
    aliases: tuple[str, ...]

    def probe(self) -> Availability: ...

    def prepare(self, args: argparse.Namespace) -> Availability: ...

    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    def run(self, args: argparse.Namespace) -> tuple[CheckResult, ...]: ...


@dataclass(frozen=True, slots=True)
class RegisteredBackend:
    """One backend descriptor plus its cheap probe result."""

    backend: CheckBackend
    availability: Availability


class BackendRegistry:
    """Registry of built-in and extension importer check backends."""

    def __init__(self) -> None:
        self._backends: dict[str, RegisteredBackend] = {}
        self._aliases: dict[str, str] = {}

    def register(self, backend: CheckBackend) -> None:
        if backend.name in self._backends or backend.name in self._aliases:
            raise ValueError(f"duplicate check backend name {backend.name!r}")
        registered = RegisteredBackend(
            backend=backend,
            availability=backend.probe(),
        )
        self._backends[backend.name] = registered
        for alias in backend.aliases:
            if alias in self._backends or alias in self._aliases:
                raise ValueError(f"duplicate check backend alias {alias!r}")
            self._aliases[alias] = backend.name

    def all(self) -> tuple[RegisteredBackend, ...]:
        return tuple(self._backends.values())

    def enabled(self) -> tuple[RegisteredBackend, ...]:
        return tuple(
            registered
            for registered in self._backends.values()
            if registered.availability.available
        )

    def disabled(self) -> tuple[RegisteredBackend, ...]:
        return tuple(
            registered
            for registered in self._backends.values()
            if not registered.availability.available
        )

    def by_name(self, name: str) -> RegisteredBackend | None:
        return self._backends.get(self._aliases.get(name, name))


def has_module(module_name: str) -> bool:
    """Returns whether a Python module can be found without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


class DeferredPackageBackend:
    """Descriptor for an importer family whose check runner is not wired yet."""

    def __init__(
        self,
        *,
        name: str,
        help: str,
        package: str,
        extras: tuple[str, ...],
    ) -> None:
        self.name: str = name
        self.help: str = help
        self.package: str = package
        self.extras: tuple[str, ...] = extras
        self.aliases: tuple[str, ...] = ()

    def probe(self) -> Availability:
        if not has_module(self.package):
            return Availability.unavailable(
                reason=f"Python package `{self.package}` is not importable",
                install_hint=f"install iree-loom[{self.extras[0]}]",
            )
        return Availability.unavailable(
            reason=f"{self.name} importer checks are not wired yet",
        )

    def prepare(self, _args: argparse.Namespace) -> Availability:
        return self.probe()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", type=Path)

    def run(self, _args: argparse.Namespace) -> tuple[CheckResult, ...]:
        raise AssertionError(f"disabled backend {self.name!r} cannot run")


def register_builtin_backends(registry: BackendRegistry) -> None:
    from loom.importers.check.mlir.backend import MlirBackend
    from loom.importers.check.tilelang.backend import TileLangBackend

    registry.register(MlirBackend())
    registry.register(TileLangBackend())
    registry.register(
        DeferredPackageBackend(
            name="triton",
            help="check Triton kernel imports",
            package="triton",
            extras=("triton",),
        )
    )


def skip_unavailable_backend(
    registered: RegisteredBackend,
    args: argparse.Namespace,
    availability: Availability,
) -> tuple[CheckResult, ...]:
    paths = tuple(getattr(args, "paths", ()))
    if not paths:
        paths = (Path(f"<{registered.backend.name}>"),)
    return tuple(
        CheckResult(
            path=path,
            case_index=-1,
            returncode=0,
            stdout="",
            stderr=f"{availability.message()}\n",
            skipped=True,
        )
        for path in paths
    )


def print_importers(registry: BackendRegistry) -> str:
    lines: list[str] = []
    for registered in registry.enabled():
        extras = ",".join(registered.backend.extras)
        lines.append(f"enabled  {registered.backend.name:<8} extras=[{extras}]")
    for registered in registry.disabled():
        extras = ",".join(registered.backend.extras)
        lines.append(f"disabled {registered.backend.name:<8} extras=[{extras}]")
        lines.append(f"  {registered.availability.reason}")
        if registered.availability.install_hint:
            lines.append(f"  {registered.availability.install_hint}")
    return "\n".join(lines)


def make_default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    register_builtin_backends(registry)
    return registry
