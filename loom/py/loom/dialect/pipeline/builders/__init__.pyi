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
    def scatter(
        self,
        *,
        source: ValueRef,
        group: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def read(
        self,
        *,
        source: ValueRef,
        group: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def stage(
        self,
        *,
        entry: str,
        group: ValueRef,
        inputs: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def buffer(
        self,
        *,
        source: ValueRef,
        capacity: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def reduce(
        self,
        *,
        entry: str,
        source_group: ValueRef,
        source_inputs: list[ValueRef] = ...,
        target_group: ValueRef,
        target_inputs: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def write(
        self,
        *,
        source: ValueRef,
        target: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def return_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
