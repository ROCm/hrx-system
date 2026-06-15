# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Region, Type

class FuncBuilder(DialectBuilder):
    def def_(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        temperature: str | None = ...,
        inline_policy: str | None = ...,
        target: str | None = ...,
        abi: str | None = ...,
        abi_attrs: Mapping[str, Any] | None = ...,
        export_symbol: str | None = ...,
        export_attrs: Mapping[str, Any] | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def decl(
        self,
        *,
        visibility: str | None = ...,
        import_module: str | None = ...,
        import_symbol: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        temperature: str | None = ...,
        inline_policy: str | None = ...,
        target: str | None = ...,
        abi: str | None = ...,
        abi_attrs: Mapping[str, Any] | None = ...,
        export_symbol: str | None = ...,
        export_attrs: Mapping[str, Any] | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def template(
        self,
        *,
        implements: str,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        temperature: str | None = ...,
        inline_policy: str | None = ...,
        target: str | None = ...,
        priority: int | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def ukernel(
        self,
        *,
        implements: str,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        temperature: str | None = ...,
        inline_policy: str | None = ...,
        target: str | None = ...,
        priority: int | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def call(
        self,
        *,
        purity: str | None = ...,
        temperature: str | None = ...,
        inline_policy: str | None = ...,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def apply(
        self,
        *,
        contract: str,
        operands: list[ValueRef] = ...,
        purity: str | None = ...,
        temperature: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def return_(
        self,
        *,
        operands: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
