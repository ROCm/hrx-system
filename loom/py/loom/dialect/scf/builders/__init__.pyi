# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Region, Type

class ScfBuilder(DialectBuilder):
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
    def switch(
        self,
        *,
        selector: ValueRef,
        results: list[Type | TiedResultSpec],
        case_keys: list[int],
        default_region: Region,
        case_regions: list[Region],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def yield_(
        self,
        *,
        values: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def select(
        self,
        *,
        condition: ValueRef,
        true_value: ValueRef,
        false_value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def lookup(
        self,
        *,
        selector: ValueRef,
        case_keys: list[int],
        values: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def condition(
        self,
        *,
        condition: ValueRef,
        forwarded: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def while_(
        self,
        *,
        iter_args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        before: Region | None = ...,
        after: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
