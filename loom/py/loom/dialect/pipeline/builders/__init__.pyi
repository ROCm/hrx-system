# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Region, Type

class PipelineBuilder(DialectBuilder):
    def def_(
        self,
        *,
        scope: str | None = ...,
        visibility: str | None = ...,
        retain: str | None = ...,
        target: str | None = ...,
        callee: str,
        specializations: list[ValueRef] = ...,
        bindings: list[ValueRef] = ...,
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def finish(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def strand(
        self,
        *,
        participation: str | None = ...,
        target: str | None = ...,
        origins: list[int | ValueRef],
        counts: list[int | ValueRef],
        strides: list[int | ValueRef],
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def end(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def compose(
        self,
        *,
        callee: str,
        specializations: list[ValueRef] = ...,
        bindings: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def memory(
        self,
        *,
        memory_space: str,
        coordinates: list[int | ValueRef],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def decl(
        self,
        *,
        scope: str | None = ...,
        visibility: str | None = ...,
        retain: str | None = ...,
        target: str | None = ...,
        callee: str,
        specializations: list[ValueRef] = ...,
        bindings: list[ValueRef] = ...,
        predicates: list[Predicate] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def serial(
        self,
        *,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def yield_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def concurrent(
        self,
        *,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
