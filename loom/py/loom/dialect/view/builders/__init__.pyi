# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Type

class ViewBuilder(DialectBuilder):
    def subview(
        self,
        *,
        source: ValueRef,
        offsets: list[int | ValueRef],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def refine(
        self,
        *,
        source: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
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
    def reduce(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
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
    def prefetch(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        intent: str,
        locality: str,
        location_id: int | None = ...,
    ) -> None: ...
