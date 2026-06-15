# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Data model for TileLang importer results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from loom.importers.core import ImportBodyReport
from loom.ir import Type


@dataclass(frozen=True, slots=True)
class TileLangImportInput:
    """Structured TileLang source object consumed by importer tooling."""

    source: object
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)
    target: str | None = None
    name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class TileLangBinding:
    """One TileLang kernel parameter imported as a Loom kernel argument."""

    ordinal: int
    name: str
    source: object
    type: Type
    buffer: object | None = None
    aliases: tuple[object, ...] = ()
    noalias: bool = False


@dataclass(frozen=True, slots=True)
class TileLangKernelFacts:
    """Facts extracted from one TileLang kernel source object."""

    function_name: str
    target_preset: str
    workgroup_size: tuple[int, int, int]
    bindings: tuple[TileLangBinding, ...]
    operation_counts: Counter[str]
    converted_body: ImportBodyReport


def normalize_tilelang_input(source: object) -> TileLangImportInput:
    """Normalizes a supported source object into TileLang import input."""

    if isinstance(source, TileLangImportInput):
        return source
    if hasattr(source, "source") and type(source).__name__ == "TileLangImportInput":
        return TileLangImportInput(
            source=source.source,
            args=tuple(getattr(source, "args", ())),
            kwargs=dict(getattr(source, "kwargs", {})),
            target=getattr(source, "target", None),
            name=getattr(source, "name", None),
            metadata=dict(getattr(source, "metadata", {})),
        )
    return TileLangImportInput(source=source)


def resolve_tilelang_input(source: object) -> TileLangImportInput:
    """Elaborates JIT-style inputs with `get_tir` while preserving metadata."""

    normalized = normalize_tilelang_input(source)
    get_tir = getattr(normalized.source, "get_tir", None)
    if get_tir is None:
        return normalized
    return TileLangImportInput(
        source=get_tir(*normalized.args, **dict(normalized.kwargs)),
        target=normalized.target,
        name=normalized.name,
        metadata=normalized.metadata,
    )
