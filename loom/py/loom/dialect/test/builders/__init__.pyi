# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.builders import DialectBuilder
from loom.ir import Block, Predicate, Region, Type

class TestBuilder(DialectBuilder):
    def addi(
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
    def neg(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def cast(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def constant(
        self,
        *,
        value: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def use(
        self,
        *,
        values: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def convergent(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def cmp(
        self,
        *,
        predicate: str,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def map(
        self,
        *,
        inputs: list[ValueRef] = ...,
        body: Region | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def update(
        self,
        *,
        source: ValueRef,
        target: ValueRef,
        offsets: list[int | ValueRef],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def invoke(
        self,
        *,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def low_call(
        self,
        *,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def low_invoke(
        self,
        *,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def slice(
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
    def loop(
        self,
        *,
        lower_bound: ValueRef,
        upper_bound: ValueRef,
        step: ValueRef,
        iter_args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        body: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def block_args(
        self,
        *,
        inputs: list[ValueRef] = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def branch(
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
    def optional_region(
        self,
        *,
        condition: ValueRef,
        body: Region | None = ...,
        else_region: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def implicit_yield(
        self,
        *,
        location_id: int | None = ...,
    ) -> None: ...
    def yield_(
        self,
        *,
        values: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def br(
        self,
        *,
        dest: Block,
        location_id: int | None = ...,
    ) -> None: ...
    def func(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
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
    def split_func(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        config: Region | None = ...,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def decl(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def record(
        self,
        *,
        kind: str | None = ...,
        symbol: str,
        dict: Mapping[str, Any] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def attrs(
        self,
        *,
        input: ValueRef,
        dict: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def operand_dict(
        self,
        *,
        input: ValueRef,
        params: dict[str, ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def attr_table(
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
    def region_table(
        self,
        *,
        selector: ValueRef,
        case_keys: list[int],
        default_region: Region,
        case_regions: list[Region],
        location_id: int | None = ...,
    ) -> None: ...
    def deflate(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def assume(
        self,
        *,
        values: list[ValueRef] = ...,
        predicates: list[Predicate],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def convert(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def reduce(
        self,
        *,
        inputs: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def read_resource(
        self,
        *,
        source: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def write_resource(
        self,
        *,
        target: ValueRef,
        data: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def mutate_resource(
        self,
        *,
        target: ValueRef,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def alloc(
        self,
        *,
        size: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def isolated_region(
        self,
        *,
        results: list[Type | TiedResultSpec],
        body: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def counter(
        self,
        *,
        value: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def dim(
        self,
        *,
        source: ValueRef,
        dim_index: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_range_lo(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_range_hi(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_all_equal_range_lo(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_all_equal_range_hi(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_divisor(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_non_negative(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_non_zero(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_positive(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_power_of_two(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_uniform(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_lane_varying(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_lane_predicate(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_subgroup_lane_mask(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_is_vector_iota(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_is_vector_prefix_mask(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_encoding_layout_kind(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_encoding_layout_stride_hi(
        self,
        *,
        value: ValueRef,
        axis: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_encoding_matrix_field(
        self,
        *,
        value: ValueRef,
        field: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_is_buffer_reference(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_is_view_reference(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_buffer_memory_space(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_memory_space(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_root_matches(
        self,
        *,
        view: ValueRef,
        root: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_alias_scope_known(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_alias_scope_matches(
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
    def fact_view_byte_offset_lo(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_byte_offset_hi(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_byte_length_lo(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_byte_length_hi(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_min_alignment(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_buffer_min_alignment(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_root_min_alignment(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_view_element_bytes(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_is_storage_reference(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_same_backing(
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
    def fact_storage_byte_offset_lo(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_byte_offset_hi(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_byte_offset_divisor(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_byte_length_lo(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_byte_length_hi(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_min_alignment(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def fact_storage_space(
        self,
        *,
        value: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def region_syntax(
        self,
        *,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def low_asm_region(
        self,
        *,
        body: Region | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def clause_constant(
        self,
        *,
        value: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def clause_copy(
        self,
        *,
        source: ValueRef,
        target: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def typed_use(
        self,
        *,
        values: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def shape(
        self,
        *,
        value: ValueRef,
        dims: list[int | ValueRef],
        location_id: int | None = ...,
    ) -> None: ...
    def target(
        self,
        *,
        kind: str,
        symbol: str,
        codegen_format: str | None = ...,
        artifact_format: str | None = ...,
        default_pointer_bitwidth: int | None = ...,
        index_bitwidth: int | None = ...,
        offset_bitwidth: int | None = ...,
        max_workgroup_size_x: int | None = ...,
        max_workgroup_size_y: int | None = ...,
        max_workgroup_size_z: int | None = ...,
        max_flat_workgroup_size: int | None = ...,
        subgroup_size: int | None = ...,
        max_grid_size_x: int | None = ...,
        max_grid_size_y: int | None = ...,
        max_grid_size_z: int | None = ...,
        max_flat_grid_size: int | None = ...,
        max_workgroup_count_x: int | None = ...,
        max_workgroup_count_y: int | None = ...,
        max_workgroup_count_z: int | None = ...,
        memory_space_generic: int | None = ...,
        memory_space_global: int | None = ...,
        memory_space_workgroup: int | None = ...,
        memory_space_constant: int | None = ...,
        memory_space_private: int | None = ...,
        memory_space_host: int | None = ...,
        memory_space_descriptor: int | None = ...,
        abi: str | None = ...,
        export_symbol: str | None = ...,
        linkage: str | None = ...,
        hal_buffer_resource_flags: int | None = ...,
        contract_set_key: str | None = ...,
        contract_feature_bits: int | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def resource_alloc(
        self,
        *,
        size: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def borrow(
        self,
        *,
        resource: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def borrow_ref(
        self,
        *,
        resource: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def consume(
        self,
        *,
        resource: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def retain(
        self,
        *,
        resource: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def release(
        self,
        *,
        resource: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def discard(
        self,
        *,
        resource: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def escape(
        self,
        *,
        resource: ValueRef,
        location_id: int | None = ...,
    ) -> None: ...
    def alias(
        self,
        *,
        resource: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def borrowed(
        self,
        *,
        resource: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def segmented(
        self,
        *,
        root: ValueRef,
        guard: ValueRef | None = ...,
        lhs: list[ValueRef] = ...,
        rhs: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def template_param_symbol(
        self,
        *,
        target: str,
        location_id: int | None = ...,
    ) -> None: ...
    def template_param_symbol_flags(
        self,
        *,
        target: str,
        flags: str = ...,
        location_id: int | None = ...,
    ) -> None: ...
