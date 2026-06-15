# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Block, Predicate, Region, Type

class LowBuilder(DialectBuilder):
    def func_def(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        allocation: str | None = ...,
        schedule: str | None = ...,
        target: str,
        abi: str | None = ...,
        abi_attrs: Mapping[str, Any] | None = ...,
        abi_layout: Mapping[str, Any] | None = ...,
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
    def kernel_def(
        self,
        *,
        allocation: str | None = ...,
        schedule: str | None = ...,
        target: str,
        abi_layout: Mapping[str, Any] | None = ...,
        export_symbol: str | None = ...,
        export_linkage: str | None = ...,
        workgroup_size_x: int | None = ...,
        workgroup_size_y: int | None = ...,
        workgroup_size_z: int | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def decl(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        allocation: str | None = ...,
        schedule: str | None = ...,
        import_kind: str | None = ...,
        code_symbol: str | None = ...,
        target: str,
        abi: str | None = ...,
        abi_attrs: Mapping[str, Any] | None = ...,
        abi_layout: Mapping[str, Any] | None = ...,
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
    def return_(
        self,
        *,
        values: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def call(
        self,
        *,
        purity: str | None = ...,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def op(
        self,
        *,
        opcode: str,
        operands: list[ValueRef] = ...,
        attrs: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def const(
        self,
        *,
        opcode: str,
        attrs: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def copy(
        self,
        *,
        source: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def slice(
        self,
        *,
        source: ValueRef,
        offset: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def concat(
        self,
        *,
        sources: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def invoke(
        self,
        *,
        purity: str | None = ...,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def reserve(
        self,
        *,
        byte_length: int,
        byte_alignment: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def view(
        self,
        *,
        source: ValueRef,
        offset: int,
        byte_length: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def spill(
        self,
        *,
        value: ValueRef,
        storage: ValueRef,
        offset: int,
        location_id: int | None = ...,
    ) -> None: ...
    def reload(
        self,
        *,
        storage: ValueRef,
        offset: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def address(
        self,
        *,
        storage: ValueRef,
        offset: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def br(
        self,
        *,
        dest: Block,
        args: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def cond_br(
        self,
        *,
        condition: ValueRef,
        true_dest: Block,
        false_dest: Block,
        location_id: int | None = ...,
    ) -> None: ...
    def resource(
        self,
        *,
        import_kind: str,
        extent_value: ValueRef | None = ...,
        index: int,
        source_type: Type,
        extent: int | None = ...,
        cache_swizzle_stride: int | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def live_in(
        self,
        *,
        source: str,
        attrs: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def yield_(
        self,
        *,
        values: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def if_(
        self,
        *,
        condition: ValueRef,
        results: list[Type | TiedResultSpec],
        then_region: Region | None = ...,
        else_region: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def for_(
        self,
        *,
        lower_bound: ValueRef,
        upper_bound: ValueRef,
        step: ValueRef,
        iter_args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        unroll_factor: ValueRef | None = ...,
        unroll_policy: str | None = ...,
        body: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
