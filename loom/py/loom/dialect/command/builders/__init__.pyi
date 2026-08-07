# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from loom.builder import ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Region

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
