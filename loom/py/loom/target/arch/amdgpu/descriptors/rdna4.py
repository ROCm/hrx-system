# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU RDNA4 descriptor-set base data."""

from __future__ import annotations

from .common import *
from .control import *

_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=106,
            full_register_part_mask=_REG_PART_SGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_AMDGPU_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_SWMMAC, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            smem_store_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SWMMAC,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_SWMMAC),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_LDS_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_SMEM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_SMEM_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)


def _encoded_operand(operand: Operand, field_name: str) -> Operand:
    return replace(operand, encoding_field_id=amdgpu_encoding_field_id(field_name))


def _encoded_immediate(
    immediate: Immediate,
    field_name: str,
    *,
    bit_width: int | None = None,
    unsigned_max: int | None = None,
    default_value: int | None = None,
) -> Immediate:
    flags = immediate.flags
    if default_value is not None and ImmediateFlag.DEFAULT_VALUE not in flags:
        flags = (*flags, ImmediateFlag.DEFAULT_VALUE)
    return replace(
        immediate,
        flags=flags,
        encoding_field_id=amdgpu_encoding_field_id(field_name),
        bit_width=immediate.bit_width if bit_width is None else bit_width,
        unsigned_max=immediate.unsigned_max if unsigned_max is None else unsigned_max,
        default_value=immediate.default_value
        if default_value is None
        else default_value,
    )


def _gfx1250_wmma_scale_immediates() -> tuple[Immediate, ...]:
    return (
        _encoded_immediate(
            _MATRIX_A_FORMAT_IMMEDIATE, "MATRIX_A_FMT", bit_width=3, unsigned_max=7
        ),
        _encoded_immediate(
            _MATRIX_B_FORMAT_IMMEDIATE, "MATRIX_B_FMT", bit_width=3, unsigned_max=7
        ),
        _encoded_immediate(
            _MATRIX_A_SCALE_IMMEDIATE, "MATRIX_A_SCALE", bit_width=1, unsigned_max=1
        ),
        _encoded_immediate(
            _MATRIX_B_SCALE_IMMEDIATE, "MATRIX_B_SCALE", bit_width=1, unsigned_max=1
        ),
        _encoded_immediate(
            _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
            "MATRIX_A_SCALE_FMT",
            bit_width=2,
            unsigned_max=3,
        ),
        _encoded_immediate(
            _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
            "MATRIX_B_SCALE_FMT",
            bit_width=2,
            unsigned_max=3,
        ),
        _encoded_immediate(_MATRIX_A_REUSE_IMMEDIATE, "MATRIX_A_REUSE"),
        _encoded_immediate(_MATRIX_B_REUSE_IMMEDIATE, "MATRIX_B_REUSE"),
    )


def _gfx1250_matrix_reuse_immediates() -> tuple[Immediate, ...]:
    return (
        _encoded_immediate(
            _MATRIX_A_REUSE_IMMEDIATE, "MATRIX_A_REUSE", default_value=0
        ),
        _encoded_immediate(
            _MATRIX_B_REUSE_IMMEDIATE, "MATRIX_B_REUSE", default_value=0
        ),
    )


def _gfx1250_swmmac_index_immediate(field_name: str) -> Immediate:
    # gfx1250 SWMMAC index-key variants share encoding bit 11.
    return _encoded_immediate(
        Immediate(
            field_name,
            ImmediateKind.UNSIGNED,
            bit_width=32,
            unsigned_max=(2**32) - 1,
        ),
        "INDEX_KEY_16BIT",
        bit_width=1,
        unsigned_max=1,
        default_value=0,
    )


_GFX1250_MATRIX_REUSE_IMMEDIATE_FIELDS = ("matrix_a_reuse", "matrix_b_reuse")

_GFX1250_WMMA_ROWS = (
    ("f32.16x16x4.f32", 0x5D, 2, 2, 8, 8, True),
    ("f32.16x16x32.f16", 0x60, 8, 8, 8, 8, True),
    ("f16.16x16x32.f16", 0x61, 8, 8, 4, 4, True),
    ("f32.16x16x32.bf16", 0x62, 8, 8, 8, 8, True),
    ("bf16.16x16x32.bf16", 0x63, 8, 8, 4, 4, True),
    ("bf16f32.16x16x32.bf16", 0x64, 8, 8, 8, 4, True),
    ("f32.16x16x64.fp8.fp8", 0x6A, 8, 8, 8, 8, True),
    ("f32.16x16x64.fp8.bf8", 0x6B, 8, 8, 8, 8, True),
    ("f32.16x16x64.bf8.fp8", 0x6C, 8, 8, 8, 8, True),
    ("f32.16x16x64.bf8.bf8", 0x6D, 8, 8, 8, 8, True),
    ("f16.16x16x64.fp8.fp8", 0x6E, 8, 8, 4, 4, True),
    ("f16.16x16x64.fp8.bf8", 0x6F, 8, 8, 4, 4, True),
    ("f16.16x16x64.bf8.fp8", 0x70, 8, 8, 4, 4, True),
    ("f16.16x16x64.bf8.bf8", 0x71, 8, 8, 4, 4, True),
    ("i32.16x16x64.iu8", 0x72, 8, 8, 8, 8, True),
    ("f32.16x16x128.fp8.fp8", 0x80, 16, 16, 8, 8, True),
    ("f32.16x16x128.fp8.bf8", 0x81, 16, 16, 8, 8, True),
    ("f32.16x16x128.bf8.fp8", 0x82, 16, 16, 8, 8, True),
    ("f32.16x16x128.bf8.bf8", 0x83, 16, 16, 8, 8, True),
    ("f16.16x16x128.fp8.fp8", 0x84, 16, 16, 4, 4, True),
    ("f16.16x16x128.fp8.bf8", 0x85, 16, 16, 4, 4, True),
    ("f16.16x16x128.bf8.fp8", 0x86, 16, 16, 4, 4, True),
    ("f16.16x16x128.bf8.bf8", 0x87, 16, 16, 4, 4, True),
    ("f32.32x16x128.f4", 0x88, 16, 8, 16, 16, False),
)


def _gfx1250_wmma_descriptor(
    name: str,
    encoding_id: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    has_reuse_immediates: bool,
) -> Descriptor:
    suffix = name.replace(".", "_")
    immediates = _gfx1250_matrix_reuse_immediates() if has_reuse_immediates else ()
    immediate_fields = (
        _GFX1250_MATRIX_REUSE_IMMEDIATE_FIELDS if has_reuse_immediates else ()
    )
    return Descriptor(
        key=f"amdgpu.v_wmma_{suffix}",
        mnemonic=f"v_wmma_{suffix}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("a", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("b", units=rhs_units), "SRC1"),
            _encoded_operand(
                _vgpr_const_operand("acc", units=accumulator_units), "SRC2"
            ),
        ),
        immediates=immediates,
        constraints=_destructive_accumulator_constraints(3),
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 3),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b", "acc"),
            immediates=immediate_fields,
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_WMMA,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx1250_wmma_descriptors() -> tuple[Descriptor, ...]:
    return tuple(_gfx1250_wmma_descriptor(*row) for row in _GFX1250_WMMA_ROWS)


_GFX1250_WMMA_SCALE_ROWS = (
    ("scale.f32.16x16x128.f8f6f4.f8.f8", 0x33, 0x35, 16, 16, 8, 8, 1),
    ("scale16.f32.16x16x128.f8f6f4.f8.f8", 0x33, 0x3A, 16, 16, 8, 8, 2),
    ("scale.f32.32x16x128.f4", 0x88, 0x35, 16, 8, 16, 16, 1),
    ("scale16.f32.32x16x128.f4", 0x88, 0x3A, 16, 8, 16, 16, 2),
)


def _gfx1250_wmma_scale_descriptor(
    name: str,
    encoding_id: int,
    x2_encoding: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    scale_units: int,
) -> Descriptor:
    suffix = name.replace(".", "_")
    return Descriptor(
        key=f"amdgpu.v_wmma_{suffix}",
        mnemonic=f"v_wmma_{suffix}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("a", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("b", units=rhs_units), "SRC1"),
            _encoded_operand(
                _vgpr_const_operand("acc", units=accumulator_units), "SRC2"
            ),
            _encoded_operand(
                _sgpr_vgpr_operand("scale_src0", units=scale_units), "SCALE_SRC0"
            ),
            _encoded_operand(
                _sgpr_vgpr_operand("scale_src1", units=scale_units), "SCALE_SRC1"
            ),
        ),
        immediates=_gfx1250_wmma_scale_immediates(),
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("X2ENCODING"), x2_encoding),
        ),
        constraints=_destructive_accumulator_constraints(3),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b", "acc", "scale_src0", "scale_src1"),
            immediates=(
                "matrix_a_fmt",
                "matrix_b_fmt",
                "matrix_a_scale",
                "matrix_b_scale",
                "matrix_a_scale_fmt",
                "matrix_b_scale_fmt",
                "matrix_a_reuse",
                "matrix_b_reuse",
            ),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_WMMA_SCALE,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3PX2,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx1250_wmma_scale_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _gfx1250_wmma_scale_descriptor(*row) for row in _GFX1250_WMMA_SCALE_ROWS
    )


_GFX1250_SWMMAC_ROWS = (
    ("f32.16x16x64.f16", 0x65, 8, 16, 8, 8, "index_key_16bit", 1),
    ("f32.16x16x64.bf16", 0x66, 8, 16, 8, 8, "index_key_16bit", 1),
    ("f16.16x16x64.f16", 0x67, 8, 16, 4, 4, "index_key_16bit", 1),
    ("bf16.16x16x64.bf16", 0x68, 8, 16, 4, 4, "index_key_16bit", 1),
    ("bf16f32.16x16x64.bf16", 0x69, 8, 16, 8, 8, "index_key_16bit", 1),
    ("f32.16x16x128.fp8.fp8", 0x73, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f32.16x16x128.fp8.bf8", 0x74, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f32.16x16x128.bf8.fp8", 0x75, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f32.16x16x128.bf8.bf8", 0x76, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f16.16x16x128.fp8.fp8", 0x77, 8, 16, 4, 4, "index_key_32bit", 2),
    ("f16.16x16x128.fp8.bf8", 0x78, 8, 16, 4, 4, "index_key_32bit", 2),
    ("f16.16x16x128.bf8.fp8", 0x79, 8, 16, 4, 4, "index_key_32bit", 2),
    ("f16.16x16x128.bf8.bf8", 0x7A, 8, 16, 4, 4, "index_key_32bit", 2),
    ("i32.16x16x128.iu8", 0x7B, 8, 16, 8, 8, "index_key_32bit", 2),
)


def _gfx1250_swmmac_descriptor(
    name: str,
    encoding_id: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    index_immediate: str,
    index_units: int,
) -> Descriptor:
    suffix = name.replace(".", "_")
    return Descriptor(
        key=f"amdgpu.v_swmmac_{suffix}",
        mnemonic=f"v_swmmac_{suffix}",
        semantic_tag=f"matrix.swmmac.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("acc", units=accumulator_units), "VDST"),
            _encoded_operand(_vgpr_operand("a", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("b", units=rhs_units), "SRC1"),
            _encoded_operand(_sgpr_vgpr_operand("index", units=index_units), "SRC2"),
        ),
        immediates=(
            _gfx1250_swmmac_index_immediate(index_immediate),
            *_gfx1250_matrix_reuse_immediates(),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 3),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("acc", "a", "b", "index"),
            immediates=(index_immediate, *_GFX1250_MATRIX_REUSE_IMMEDIATE_FIELDS),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_SWMMAC,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx1250_swmmac_descriptors() -> tuple[Descriptor, ...]:
    return tuple(_gfx1250_swmmac_descriptor(*row) for row in _GFX1250_SWMMAC_ROWS)


def _gfx125x_reg_classes() -> tuple[RegClass, ...]:
    return (
        *(
            replace(reg_class, allocatable_count=1024)
            if reg_class.name == _REG_VGPR
            else reg_class
            for reg_class in _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.reg_classes
        ),
        RegClass(
            _REG_MODE,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    )


_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.gfx125x.core",
    reg_classes=_gfx125x_reg_classes(),
    register_parts=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.register_parts,
    resources=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
        ScheduleClass(
            _SCHEDULE_WMMA_SCALE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MODE_CONTROL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=(
        _s_set_vgpr_msb_descriptor(),
        *_gfx1250_wmma_descriptors(),
        *_gfx1250_wmma_scale_descriptors(),
        *_gfx1250_swmmac_descriptors(),
    ),
)


__all__ = (
    "_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE",
    "_gfx125x_reg_classes",
)
