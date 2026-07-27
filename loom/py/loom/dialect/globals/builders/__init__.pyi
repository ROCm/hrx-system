# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Type

class GlobalBuilder(DialectBuilder):
    def constant(
        self,
        *,
        symbol: str,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        initializer: Any | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def variable(
        self,
        *,
        symbol: str,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        initializer: Any | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def rodata(
        self,
        *,
        symbol: str,
        contents: Any,
        alignment: int | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def load(
        self,
        *,
        global_: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def store(
        self,
        *,
        value: ValueRef,
        global_: str,
        location_id: int | None = ...,
    ) -> None: ...
