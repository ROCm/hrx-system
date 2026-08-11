# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Type

class SanitizerBuilder(DialectBuilder):
    def assert_access(
        self,
        *,
        kind: str,
        view: ValueRef,
        indices: list[int | ValueRef],
        static_extents: list[int] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def value(
        self,
        *,
        values: list[ValueRef] = ...,
        predicates: list[Predicate],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def op(
        self,
        *,
        values: list[ValueRef] = ...,
        predicates: list[Predicate],
        location_id: int | None = ...,
    ) -> None: ...
    def layout(
        self,
        *,
        view: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def race_access(
        self,
        *,
        kind: str,
        view: ValueRef,
        indices: list[int | ValueRef],
        atomic: bool,
        ordering: str | None = ...,
        scope: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def sync(
        self,
        *,
        memory_space: str,
        scope: str,
        ordering: str,
        location_id: int | None = ...,
    ) -> None: ...
    def accesses(
        self,
        *,
        kind: str,
        view: ValueRef,
        indices: list[int | ValueRef],
        static_extents: list[int],
        static_strides: list[int],
        static_count: int,
        location_id: int | None = ...,
    ) -> None: ...
