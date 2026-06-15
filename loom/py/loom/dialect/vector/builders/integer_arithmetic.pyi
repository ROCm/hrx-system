# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Sequence

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Type

class VectorIntegerArithmeticMixin:
    def addi(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def subi(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def muli(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def divsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def divui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def remsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def remui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def ceildivsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def ceildivui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def floordivsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def negi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def absi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def minsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def maxsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def minui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def maxui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fmai(
        self,
        *,
        overflow: str = ...,
        a: ValueRef,
        b: ValueRef,
        c: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def andi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def ori(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def xori(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def shli(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def shrsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def shrui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def rotli(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def rotri(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def ctlzi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def cttzi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def ctpopi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
