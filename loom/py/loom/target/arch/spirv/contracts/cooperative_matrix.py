# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative matrix memory contracts over retained source geometry."""

from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.contracts.memory import (
    WORKGROUP_MEMORY_SPACES,
    source_memory_address_feature_guards,
    storage_buffer_address_materializer,
    storage_buffer_alignment_diagnostic,
    storage_buffer_memory_spaces,
    storage_element_format_guards,
    workgroup_address_materializer,
    workgroup_carrier_guard,
)
from loom.target.arch.spirv.cooperative_matrix import (
    COOPERATIVE_MATRIX_CASES,
    CooperativeMatrixCase,
)
from loom.target.arch.spirv.scalar_memory import StorageBufferScalar
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    Guard,
    SourceMemoryAddressCoordinateType,
    SourceMemoryAddressLayout,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    ValueRef,
    Vector,
    View,
)


def _memory_rule(
    case: CooperativeMatrixCase,
    role: str,
    scalar: StorageBufferScalar,
    element: str,
    rows: int,
    columns: int,
    memory_space: str,
    leading_axis: int,
) -> DescriptorRule:
    is_load = role != "result"
    operation = SourceMemoryOperation.LOAD if is_load else SourceMemoryOperation.STORE
    descriptor = logical_core_descriptor(
        case.descriptor_key(
            "op_cooperative_matrix_load_khr"
            if is_load
            else "op_cooperative_matrix_store_khr",
            role=role,
            memory_space=memory_space,
        )
    )
    is_workgroup = memory_space == "workgroup"
    address_materializer = (
        workgroup_address_materializer(scalar, SourceMemoryAddressCoordinateType.OFFSET)
        if is_workgroup
        else storage_buffer_address_materializer(scalar)
    )
    source_memory = SourceMemoryConstraint(
        operation=operation,
        root_kind=(
            SourceMemoryRootKind.ALLOCA if is_workgroup else SourceMemoryRootKind.ANY
        ),
        address_layout=(
            SourceMemoryAddressLayout.ROW_MAJOR
            if leading_axis == 0
            else SourceMemoryAddressLayout.COLUMN_MAJOR
        ),
        memory_spaces=(
            WORKGROUP_MEMORY_SPACES
            if is_workgroup
            else storage_buffer_memory_spaces(operation)
        ),
        element_byte_count=scalar.byte_width,
        vector_lane_count=1,
        vector_lane_byte_stride=scalar.byte_width,
        static_byte_offset_minimum=0,
        static_byte_offset_maximum=(2**63) - 1,
        minimum_alignment=16,
        element_stride_unsigned_bit_count=32,
        dynamic_term_count=None,
        dynamic_index_source=SourceMemoryDynamicIndexSource.NONE,
        require_subgroup_uniform=True,
        alignment_diagnostic=storage_buffer_alignment_diagnostic(16),
    )
    stride_descriptor = logical_core_descriptor("spirv.op_uconvert.offset64.u32")
    operands = {
        "ptr": ValueRef.source_memory_address(),
        "stride": ValueRef.temporary("stride"),
    }
    if not is_load:
        operands["value"] = ValueRef.operand("value")
    return DescriptorRule(
        source_op=vector.vector_fragment_load
        if is_load
        else vector.vector_fragment_store,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("role", role),
            Guard.value_type("view", View(scalar.source_type)),
            *storage_element_format_guards("view", scalar),
            # The opaque cooperative type owns component distribution. Source
            # carrier widths do not describe the driver's per-invocation data.
            Guard.value_type(
                "result" if is_load else "value",
                Vector(
                    element,
                    minimum_lanes=1,
                    maximum_lanes="LOOM_DIM_MAX_STATIC_SIZE",
                ),
            ),
            Guard.value_i64_range("rows", rows, rows),
            Guard.value_i64_range("columns", columns, columns),
            *((workgroup_carrier_guard(scalar.suffix),) if is_workgroup else ()),
            *source_memory_address_feature_guards(address_materializer),
            *descriptor_feature_guards(descriptor, stride_descriptor),
        ),
        emit=(
            emit_descriptor_op(
                descriptor=stride_descriptor,
                operands={
                    "input": ValueRef.source_memory_element_stride(axis=leading_axis)
                },
                results={"dst": ValueRef.temporary("stride")},
                result_types={"dst": DescriptorResultType()},
                source_memory=source_memory,
                source_memory_address_materializer=address_materializer,
            ),
            emit_descriptor_op(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")} if is_load else {},
                immediates={"memory_layout": leading_axis},
                source_memory=source_memory,
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


COOPERATIVE_MATRIX_MEMORY_RULES = tuple(
    _memory_rule(case, role, scalar, element, rows, columns, memory_space, leading_axis)
    for case in COOPERATIVE_MATRIX_CASES
    if case.source_rule_enabled
    for role, scalar, element, rows, columns in (
        (
            "lhs",
            case.lhs_scalar,
            case.lhs_source_type,
            case.lhs_rows,
            case.lhs_columns,
        ),
        (
            "rhs",
            case.rhs_scalar,
            case.rhs_source_type,
            case.rhs_rows,
            case.rhs_columns,
        ),
        (
            "init",
            case.accumulator_scalar,
            case.accumulator_source_type,
            case.accumulator_rows,
            case.accumulator_columns,
        ),
        (
            "result",
            case.result_scalar,
            case.result_source_type,
            case.accumulator_rows,
            case.accumulator_columns,
        ),
    )
    for memory_space in ("storage_buffer", "workgroup")
    for leading_axis in (0, 1)
)
