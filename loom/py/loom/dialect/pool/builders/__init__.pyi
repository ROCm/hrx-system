# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Type

class PoolBuilder(DialectBuilder):
    def load(
        self,
        *,
        pool: ValueRef,
        page_id: ValueRef,
        page_bytes: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def store(
        self,
        *,
        pool: ValueRef,
        page_id: ValueRef,
        page_bytes: ValueRef,
        offset_in_page: ValueRef,
        data: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def pin(
        self,
        *,
        pool: ValueRef,
        block_id: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def unpin(
        self,
        *,
        pool: ValueRef,
        block_id: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def buffer(
        self,
        *,
        pool: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
