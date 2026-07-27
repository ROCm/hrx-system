# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Type

class VectorAtomicMixin:
    def atomic_reduce(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def atomic_reduce_mask(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def rmw(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def atomic_rmw_mask(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        passthrough: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def cmpxchg(
        self,
        *,
        expected: ValueRef,
        replacement: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        success_ordering: str,
        failure_ordering: str,
        scope: str,
        cache_scope: str | None = ...,
        cache_temporal: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
