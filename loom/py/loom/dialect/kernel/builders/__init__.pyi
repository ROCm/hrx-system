# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Predicate, Region, Type

class KernelBuilder(DialectBuilder):
    def def_(
        self,
        *,
        target: str | None = ...,
        export_symbol: str | None = ...,
        export_linkage: str | None = ...,
        callee: str,
        config_args: Sequence[tuple[str, Type]] = ...,
        config: Region | None = ...,
        args: list[ValueRef] = ...,
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def config(
        self,
        *,
        workgroup_count_x: ValueRef,
        workgroup_count_y: ValueRef,
        workgroup_count_z: ValueRef,
        workgroup_size_x: ValueRef,
        workgroup_size_y: ValueRef,
        workgroup_size_z: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def return_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def exit(
        self,
        *,
        condition: ValueRef,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def barrier(
        self,
        *,
        memory_space: str,
        ordering: str,
        scope: str,
        location_id: int | None = ...,
    ) -> None: ...
    def copy(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        direction: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_copy_mask(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        predicate: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        direction: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_gather(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_gather_mask(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        predicate: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def group(
        self,
        *,
        tokens: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def wait(
        self,
        *,
        group: ValueRef,
        newer_groups: int,
        location_id: int | None = ...,
    ) -> None: ...
    def descriptor(
        self,
        *,
        dgroups: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_tensor_load_to_lds(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        descriptor: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_tensor_store_from_lds(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        descriptor: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_cluster_gather(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cluster_mask: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def async_cluster_gather_mask(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cluster_mask: ValueRef,
        predicate: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workitem_id(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_id(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_size(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_count(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workitem_dispatch_id(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_id(
        self,
        *,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_count(
        self,
        *,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_size(
        self,
        *,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_lane_id(
        self,
        *,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def shuffle(
        self,
        *,
        mode: str,
        value: ValueRef,
        offset: ValueRef,
        width: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def broadcast(
        self,
        *,
        value: ValueRef,
        lane: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def first(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_reduce(
        self,
        *,
        kind: str,
        value: ValueRef,
        cluster_size: int | None = ...,
        cluster_stride: int | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_scan(
        self,
        *,
        kind: str,
        value: ValueRef,
        cluster_size: int | None = ...,
        cluster_stride: int | None = ...,
        mode: str,
        direction: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_vote_any(
        self,
        *,
        predicate: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_vote_all(
        self,
        *,
        predicate: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def ballot(
        self,
        *,
        predicate: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_active_mask(
        self,
        *,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_match_any(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subgroup_match_all(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def workgroup_reduce(
        self,
        *,
        kind: str,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_scan(
        self,
        *,
        kind: str,
        value: ValueRef,
        mode: str,
        direction: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_vote_any(
        self,
        *,
        predicate: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_vote_all(
        self,
        *,
        predicate: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def workgroup_vote_count(
        self,
        *,
        predicate: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def assert_(
        self,
        *,
        condition: ValueRef,
        message: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
