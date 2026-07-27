# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang importer check fixture declarations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, overload

TILELANG_CASE_ATTR = "__loom_tilelang_case__"


class TileLangT(Protocol):
    """Structural type for TileLang's language module commonly imported as T."""

    def __getattr__(self, name: str) -> Any: ...

    def prim_func(self, func: Callable[..., Any]) -> Callable[..., Any]: ...


@dataclass(frozen=True, slots=True)
class TileLangCaseMetadata:
    """Metadata attached to one TileLang importer check case function."""

    name: str | None = None
    category: str = "composition"
    tags: tuple[str, ...] = ()


@overload
def tilelang_case[**P, R](
    func: Callable[P, R],
    *,
    name: str | None = None,
    category: str = "composition",
    tags: Sequence[str] = (),
) -> Callable[P, R]: ...


@overload
def tilelang_case[**P, R](
    func: None = None,
    *,
    name: str | None = None,
    category: str = "composition",
    tags: Sequence[str] = (),
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def tilelang_case[**P, R](
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    category: str = "composition",
    tags: Sequence[str] = (),
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Marks a function as a TileLang importer check case."""

    metadata = TileLangCaseMetadata(
        name=name,
        category=category,
        tags=tuple(tags),
    )

    def decorate(case_func: Callable[P, R]) -> Callable[P, R]:
        setattr(case_func, TILELANG_CASE_ATTR, metadata)
        return case_func

    if func is not None:
        return decorate(func)
    return decorate


def is_tilelang_case(value: object) -> bool:
    return isinstance(getattr(value, TILELANG_CASE_ATTR, None), TileLangCaseMetadata)


def get_tilelang_case_metadata(value: object) -> TileLangCaseMetadata | None:
    metadata = getattr(value, TILELANG_CASE_ATTR, None)
    if isinstance(metadata, TileLangCaseMetadata):
        return metadata
    return None
