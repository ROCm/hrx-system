# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V logical memory constraints and address materializers."""

from loom.error.spirv import ERR_SPIRV_028, ERR_SPIRV_029
from loom.error.target import ERR_TARGET_050
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    logical_core_descriptor,
)
from loom.target.arch.spirv.scalar_memory import StorageBufferScalar
from loom.target.contracts import (
    Guard,
    GuardDiagnostic,
    SourceMemoryAddressBase,
    SourceMemoryAddressCoordinateType,
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    i64_param,
    source_memory_minimum_alignment_param,
    string_param,
    target_diagnostic,
    u32_param,
    value_type_param,
)

STORAGE_BUFFER_MEMORY_SPACES = ("unknown", "generic", "global", "descriptor")
WORKGROUP_MEMORY_SPACES = ("workgroup",)


def storage_buffer_alignment_diagnostic(required_alignment: int) -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_TARGET_050,
            value_type_param("value_type", "view"),
            u32_param("required_alignment", required_alignment),
            source_memory_minimum_alignment_param("known_alignment"),
        )
    )


def _storage_buffer_address_diagnostic() -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_SPIRV_028,
            i64_param("required_range_lo", 0),
            i64_param("required_range_hi", (2**31) - 1),
            string_param(
                "constraint_key",
                "source_memory.address_index_non_negative_s32",
            ),
        )
    )


def storage_buffer_address_materializer(
    scalar: StorageBufferScalar,
) -> SourceMemoryAddressMaterializer:
    return SourceMemoryAddressMaterializer(
        const_coordinate=logical_core_descriptor("spirv.op_constant.offset64"),
        add_coordinate=logical_core_descriptor("spirv.op_iadd.offset64"),
        mul_coordinate=logical_core_descriptor("spirv.op_imul.offset64"),
        shl_coordinate=None,
        index_to_coordinate_input=logical_core_descriptor("spirv.op_bitcast.i32.u32"),
        index_to_coordinate=logical_core_descriptor("spirv.op_uconvert.u32.offset64"),
        address=logical_core_descriptor(
            f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset"
        ),
        base=SourceMemoryAddressBase.ROOT,
        coordinate_type=SourceMemoryAddressCoordinateType.OFFSET,
        const_coordinate_immediate="offset64_value",
        diagnostic=_storage_buffer_address_diagnostic(),
    )


def source_memory_address_feature_guards(
    materializer: SourceMemoryAddressMaterializer,
) -> tuple[Guard, ...]:
    descriptors = (
        materializer.const_coordinate,
        materializer.add_coordinate,
        materializer.mul_coordinate,
        materializer.shl_coordinate,
        materializer.index_to_coordinate_input,
        materializer.index_to_coordinate,
        materializer.address,
    )
    return tuple(
        guard
        for descriptor in descriptors
        if descriptor is not None
        for guard in descriptor_feature_guards(descriptor)
    )


def storage_buffer_source_memory(
    operation: SourceMemoryOperation,
    scalar: StorageBufferScalar,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=STORAGE_BUFFER_MEMORY_SPACES,
        element_byte_count=scalar.byte_width,
        vector_lane_count=1,
        vector_lane_byte_stride=scalar.byte_width,
        static_byte_offset_minimum=0,
        static_byte_offset_maximum=(2**63) - 1,
        minimum_alignment=scalar.byte_width,
        dynamic_term_count=None,
        dynamic_index_source=SourceMemoryDynamicIndexSource.NONE,
        diagnostic=storage_buffer_alignment_diagnostic(scalar.byte_width),
    )


def workgroup_source_memory(
    operation: SourceMemoryOperation,
    scalar: StorageBufferScalar,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ALLOCA,
        memory_spaces=WORKGROUP_MEMORY_SPACES,
        element_byte_count=scalar.byte_width,
        vector_lane_count=1,
        vector_lane_byte_stride=scalar.byte_width,
        static_byte_offset_minimum=0,
        static_byte_offset_maximum=(2**63) - 1,
        minimum_alignment=scalar.byte_width,
        dynamic_term_count=None,
        dynamic_index_source=SourceMemoryDynamicIndexSource.NONE,
        diagnostic=storage_buffer_alignment_diagnostic(scalar.byte_width),
    )


def _workgroup_address_diagnostic(
    scalar: StorageBufferScalar,
) -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_SPIRV_029,
            u32_param("element_byte_count", scalar.byte_width),
            i64_param("required_range_lo", 0),
            i64_param("required_range_hi", (2**31) - 1),
            string_param(
                "constraint_key",
                "source_memory.workgroup_element_index_non_negative_s32",
            ),
        )
    )


def workgroup_address_materializer(
    scalar: StorageBufferScalar,
) -> SourceMemoryAddressMaterializer:
    return SourceMemoryAddressMaterializer(
        const_coordinate=logical_core_descriptor("spirv.op_constant.i32"),
        add_coordinate=logical_core_descriptor("spirv.op_iadd.i32"),
        mul_coordinate=logical_core_descriptor("spirv.op_imul.i32"),
        shl_coordinate=logical_core_descriptor("spirv.op_shift_left_logical.i32"),
        address=logical_core_descriptor(
            f"spirv.op_access_chain.workgroup.{scalar.suffix}.element_index"
        ),
        base=SourceMemoryAddressBase.BASE_VIEW,
        coordinate_type=SourceMemoryAddressCoordinateType.INDEX,
        coordinate_unit_byte_count=scalar.byte_width,
        coordinate_minimum=0,
        coordinate_maximum=(2**31) - 1,
        const_coordinate_immediate="i32_value",
        diagnostic=_workgroup_address_diagnostic(scalar),
    )


def workgroup_carrier_guard(scalar_suffix: str) -> Guard:
    return Guard.low_value_register_class(
        "view", f"spirv.ptr.workgroup.array.{scalar_suffix}"
    )
