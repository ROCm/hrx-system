# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Region, Type

class CheckBuilder(DialectBuilder):
    def case(
        self,
        *,
        visibility: str | None = ...,
        case_symbol: str,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def return_(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def requires(
        self,
        *,
        provider: str,
        attrs: Mapping[str, Any],
        location_id: int | None = ...,
    ) -> None: ...
    def skip_if(
        self,
        *,
        provider: str,
        attrs: Mapping[str, Any],
        reason: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def range(
        self,
        *,
        policy: str,
        lower: Any,
        upper: Any,
        step: Any | None = ...,
        param_name: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def choice(
        self,
        *,
        values: list[int],
        param_name: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def seed(
        self,
        *,
        base: int,
        count: int,
        param_name: str | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def literal(
        self,
        *,
        value: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def iota(
        self,
        *,
        offset: Any,
        step: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fill(
        self,
        *,
        value: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def uniform(
        self,
        *,
        seed: ValueRef,
        lower: Any,
        upper: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def file_read_npy(
        self,
        *,
        path: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def file_write_npy(
        self,
        *,
        value: ValueRef,
        path: str,
        mode: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def call(
        self,
        *,
        provider: str,
        attrs: Mapping[str, Any] | None = ...,
        callee: str,
        inputs: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def equal(
        self,
        *,
        actual: ValueRef,
        expected: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def bitwise(
        self,
        *,
        actual: ValueRef,
        expected: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def close(
        self,
        *,
        actual: ValueRef,
        expected: ValueRef,
        atol: float,
        rtol: float,
        nan: str,
        location_id: int | None = ...,
    ) -> None: ...
    def shape(
        self,
        *,
        value: ValueRef,
        dims: list[int | ValueRef],
        location_id: int | None = ...,
    ) -> None: ...
    def expect(
        self,
        *,
        provider: str,
        actual: ValueRef,
        expected: ValueRef,
        attrs: Mapping[str, Any] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def event(
        self,
        *,
        provider: str,
        attrs: Mapping[str, Any] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def benchmark(
        self,
        *,
        case_ref: str,
        benchmark: str | None = ...,
        attrs: Mapping[str, Any] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
