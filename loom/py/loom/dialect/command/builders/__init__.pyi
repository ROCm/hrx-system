# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Region, Type

class CommandBuilder(DialectBuilder):
    def def_(
        self,
        *,
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
    def decl(
        self,
        *,
        visibility: str | None = ...,
        retain: str | None = ...,
        target: str | None = ...,
        callee: str,
        specializations: list[ValueRef] = ...,
        bindings: list[ValueRef] = ...,
        predicates: list[Predicate] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def launch(
        self,
        *,
        callee: str,
        specializations: list[ValueRef] = ...,
        bindings: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def return_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def yield_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def serial(
        self,
        *,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def concurrent(
        self,
        *,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def parameter(
        self,
        *,
        source: ValueRef,
        pattern: str,
        substitutions: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
