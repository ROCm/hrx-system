# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deterministic Loom SSA/symbol naming for foreign importers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


def source_name(value: object) -> str:
    """Return the best source-level name for a foreign value object."""
    get_name = getattr(value, "get_name", None)
    if get_name is not None:
        name = get_name()
        if name:
            return str(name)
    return str(value)


def sanitize_identifier(value: str, *, fallback: str = "v") -> str:
    """Sanitize a source spelling into a Loom identifier fragment."""
    text = value.removeprefix("%")
    chars: list[str] = []
    previous_was_underscore = False
    for char in text:
        if char.isalnum() or char == "_":
            chars.append(char)
            previous_was_underscore = False
        elif not previous_was_underscore:
            chars.append("_")
            previous_was_underscore = True
    sanitized = "".join(chars).strip("_")
    if not sanitized:
        sanitized = fallback
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def sanitize_symbol(value: str | None, *, fallback: str = "symbol") -> str:
    """Sanitize a source spelling into a Loom symbol name."""
    if value is None:
        value = fallback
    chars: list[str] = []
    previous_was_underscore = False
    for char in value:
        if char.isalnum() or char == "_":
            chars.append(char)
            previous_was_underscore = False
        elif not previous_was_underscore:
            chars.append("_")
            previous_was_underscore = True
    sanitized = "".join(chars).strip("_")
    if not sanitized:
        sanitized = fallback
    if sanitized[0].isdigit():
        sanitized = f"{fallback}_{sanitized}"
    return sanitized


@dataclass(slots=True)
class NameAllocator:
    """Allocates stable names while preserving meaningful source names."""

    _used: set[str] = field(default_factory=set)

    def __init__(self, reserved: Iterable[str] = ()) -> None:
        self._used = set()
        for name in reserved:
            self._used.add(sanitize_identifier(name))

    @property
    def used(self) -> frozenset[str]:
        return frozenset(self._used)

    def capture(self, name: str) -> None:
        self._used.add(sanitize_identifier(name))

    def fresh(self, base: str, *, fallback: str = "v") -> str:
        root = sanitize_identifier(base, fallback=fallback)
        candidate = root
        suffix = 1
        while candidate in self._used:
            suffix += 1
            candidate = f"{root}_{suffix}"
        self._used.add(candidate)
        return candidate

    def fresh_from_source(self, source: object, *, fallback: str = "v") -> str:
        return self.fresh(source_name(source), fallback=fallback)

    def reserve_or_fresh(self, name: str, *, fallback: str = "v") -> str:
        sanitized = sanitize_identifier(name, fallback=fallback)
        if sanitized not in self._used:
            self._used.add(sanitized)
            return sanitized
        return self.fresh(sanitized, fallback=fallback)
