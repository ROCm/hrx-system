# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Type

class BufferBuilder(DialectBuilder):
    def alloca(
        self,
        *,
        memory_space: str,
        base_alignment: int,
        byte_length: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def alignment(
        self,
        *,
        buffers: list[ValueRef] = ...,
        minimum_alignment: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def memory_space(
        self,
        *,
        memory_space: str,
        buffer: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def noalias(
        self,
        *,
        buffers: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def same_root(
        self,
        *,
        buffer: ValueRef,
        root: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def view(
        self,
        *,
        buffer: ValueRef,
        byte_offset: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def pack(
        self,
        *,
        byte_lengths: list[ValueRef] = ...,
        minimum_alignments: list[int],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def length(
        self,
        *,
        buffer: ValueRef,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def load_i8_u(
        self,
        *,
        source: ValueRef,
        byte_offset: ValueRef,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def store_i8(
        self,
        *,
        value: ValueRef,
        target: ValueRef,
        byte_offset: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def copy(
        self,
        *,
        source: ValueRef,
        source_offset: ValueRef,
        target: ValueRef,
        target_offset: ValueRef,
        byte_length: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def fill(
        self,
        *,
        pattern: ValueRef,
        target: ValueRef,
        target_offset: ValueRef,
        byte_length: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def compare(
        self,
        *,
        lhs: ValueRef,
        lhs_offset: ValueRef,
        rhs: ValueRef,
        rhs_offset: ValueRef,
        byte_length: ValueRef,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
