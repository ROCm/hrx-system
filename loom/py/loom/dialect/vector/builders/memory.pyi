# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Type

class VectorMemoryMixin:
    def fragment_load(
        self,
        *,
        role: str,
        view: ValueRef,
        indices: list[int | ValueRef],
        rows: ValueRef,
        columns: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fragment_store(
        self,
        *,
        role: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        rows: ValueRef,
        columns: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def load(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def store(
        self,
        *,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def load_mask(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        mask: ValueRef,
        passthrough: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def store_mask(
        self,
        *,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        mask: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def expand(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        mask: ValueRef,
        passthrough: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def compress(
        self,
        *,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        mask: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def gather(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def scatter(
        self,
        *,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def gather_mask(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        passthrough: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def scatter_mask(
        self,
        *,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
