# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU RDNA4 descriptor-set base data."""

from __future__ import annotations

from ..matrix_formats import AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
from .common import *
from .control import *

_RESOURCE_GFX125X_XDL = "amdgpu.gfx125x.xdl"

_SCHEDULE_MATRIX_VALU_EXACT_16 = f"{_SCHEDULE_MATRIX}.valu.exact.16"
_SCHEDULE_MATRIX_XDL_EXACT_4 = f"{_SCHEDULE_MATRIX}.xdl.exact.4"
_SCHEDULE_MATRIX_XDL_EXACT_8 = f"{_SCHEDULE_MATRIX}.xdl.exact.8"
_SCHEDULE_MATRIX_XDL_EXACT_16 = f"{_SCHEDULE_MATRIX}.xdl.exact.16"
_SCHEDULE_MATRIX_XDL_EXACT_32 = f"{_SCHEDULE_MATRIX}.xdl.exact.32"
_SCHEDULE_MATRIX_XDL_ESTIMATED_16 = f"{_SCHEDULE_MATRIX}.xdl.estimated.16"
_SCHEDULE_MATRIX_XDL_ESTIMATED_32 = f"{_SCHEDULE_MATRIX}.xdl.estimated.32"

_GFX125X_MATRIX_FORMAT_ENUM_DOMAIN_NAMES = {
    physical_format.token: (f"amdgpu.gfx125x.matrix_format.{physical_format.token}")
    for physical_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
}
_GFX125X_MATRIX_SCALE_FORMAT_ENUM_DOMAIN_NAME = "amdgpu.gfx125x.matrix_scale_format"
_GFX125X_MATRIX_ENUM_DOMAINS = (
    *(
        EnumDomain(
            _GFX125X_MATRIX_FORMAT_ENUM_DOMAIN_NAMES[physical_format.token],
            values=tuple(
                EnumValue(token, value)
                for token, value in physical_format.selector_values
            ),
        )
        for physical_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
    ),
    EnumDomain(
        _GFX125X_MATRIX_SCALE_FORMAT_ENUM_DOMAIN_NAME,
        values=(EnumValue("e8m0", 0), EnumValue("fp8_e4m3", 2)),
    ),
)


def _s_getreg_b32_cluster_workgroup_flat_id_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_getreg_b32.cluster_workgroup_flat_id",
        instruction_name="S_GETREG_B32",
        mnemonic="s_getreg_b32_cluster_workgroup_flat_id",
        encoding_name="ENC_SOPK",
        semantic_tag="kernel.cluster.workgroup.flat_id",
        schedule_class=_SCHEDULE_SALU,
        operands=(AmdgpuOperandOverlay("SDST", _sgpr_result()),),
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "SIMM16",
                ignore_reason="fixed-gfx125x-cluster-local-rank-hwreg",
                fixed_encoding_value=0x1D5C,
            ),
        ),
        asm_forms=_asm(
            results=("dst",),
            native_assembly_mnemonic="s_getreg_b32",
            native_assembly_values=(
                _native_result("dst"),
                _native_literal("hwreg(HW_REG_IB_STS2, 21, 4)"),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=106,
            fixed_location_base=108,
            fixed_location_count=16,
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
            _REG_VCC,
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
        Resource(
            _RESOURCE_WMMA,
            capacity_per_cycle=1,
            kind=ResourceKind.MATRIX,
            flags=(ResourceFlag.VECTOR_ISSUE,),
        ),
        Resource(
            _RESOURCE_SWMMAC,
            capacity_per_cycle=1,
            kind=ResourceKind.MATRIX,
            flags=(ResourceFlag.VECTOR_ISSUE,),
        ),
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
            latency_kind=LatencyKind.EXACT,
            latency_cycles=5,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_SWMMAC,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=5,
            issue_uses=(IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_SWMMAC),
            model_quality=ModelQuality.EXACT,
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


def _gfx125x_matrix_format_immediate(
    field_name: str, encoding_field_name: str, physical_format: str
) -> Immediate:
    return _encoded_immediate(
        Immediate(
            field_name,
            ImmediateKind.ENUM,
            bit_width=3,
            enum_domain=_GFX125X_MATRIX_FORMAT_ENUM_DOMAIN_NAMES[physical_format],
        ),
        encoding_field_name,
    )


def _gfx125x_wmma_scale_immediates(
    matrix_physical_formats: tuple[str, str] | None,
) -> tuple[Immediate, ...]:
    lhs_physical_format, rhs_physical_format = matrix_physical_formats or ("", "")
    format_immediates = (
        (
            _gfx125x_matrix_format_immediate(
                "matrix_a_fmt", "MATRIX_A_FMT", lhs_physical_format
            ),
            _gfx125x_matrix_format_immediate(
                "matrix_b_fmt", "MATRIX_B_FMT", rhs_physical_format
            ),
        )
        if matrix_physical_formats is not None
        else ()
    )
    return (
        *format_immediates,
        _encoded_immediate(
            _MATRIX_A_SCALE_IMMEDIATE,
            "MATRIX_A_SCALE",
            bit_width=1,
            unsigned_max=1,
            default_value=0,
        ),
        _encoded_immediate(
            _MATRIX_B_SCALE_IMMEDIATE,
            "MATRIX_B_SCALE",
            bit_width=1,
            unsigned_max=1,
            default_value=0,
        ),
        _encoded_immediate(
            replace(
                _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
                kind=ImmediateKind.ENUM,
                enum_domain=_GFX125X_MATRIX_SCALE_FORMAT_ENUM_DOMAIN_NAME,
            ),
            "MATRIX_A_SCALE_FMT",
            bit_width=2,
            default_value=0,
        ),
        _encoded_immediate(
            replace(
                _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
                kind=ImmediateKind.ENUM,
                enum_domain=_GFX125X_MATRIX_SCALE_FORMAT_ENUM_DOMAIN_NAME,
            ),
            "MATRIX_B_SCALE_FMT",
            bit_width=2,
            default_value=0,
        ),
        *_gfx125x_matrix_reuse_immediates(),
    )


def _gfx125x_matrix_format_native_values() -> tuple[NativeAsmValue, ...]:
    return (
        _native_amdgpu_required_named_i64_immediate("matrix_a_fmt"),
        _native_amdgpu_required_named_i64_immediate("matrix_b_fmt"),
    )


def _gfx125x_wmma_scale_native_values(
    has_matrix_format_selectors: bool,
) -> tuple[NativeAsmValue, ...]:
    return (
        *(
            _gfx125x_matrix_format_native_values()
            if has_matrix_format_selectors
            else ()
        ),
        _native_amdgpu_named_i64_immediate("matrix_a_scale"),
        _native_amdgpu_named_i64_immediate("matrix_b_scale"),
        _native_amdgpu_named_i64_immediate("matrix_a_scale_fmt"),
        _native_amdgpu_named_i64_immediate("matrix_b_scale_fmt"),
        *_gfx125x_matrix_reuse_native_values(),
    )


def _gfx125x_matrix_reuse_immediates() -> tuple[Immediate, ...]:
    return (
        _encoded_immediate(
            _MATRIX_A_REUSE_IMMEDIATE, "MATRIX_A_REUSE", default_value=0
        ),
        _encoded_immediate(
            _MATRIX_B_REUSE_IMMEDIATE, "MATRIX_B_REUSE", default_value=0
        ),
    )


def _gfx125x_matrix_reuse_native_values() -> tuple[NativeAsmValue, ...]:
    return tuple(
        _native_amdgpu_named_flag_immediate(field_name)
        for field_name in _GFX125X_MATRIX_REUSE_IMMEDIATE_FIELDS
    )


def _gfx125x_matrix_integer_control_immediates() -> tuple[Immediate, ...]:
    return (
        _encoded_immediate(_MATRIX_SIGN_SELECT_IMMEDIATE, "NEG"),
        _encoded_immediate(_MATRIX_CLAMP_IMMEDIATE, "CLAMP"),
    )


def _gfx125x_matrix_integer_control_native_values() -> tuple[NativeAsmValue, ...]:
    return (
        _native_amdgpu_named_bit_list_immediate("neg_lo", 3),
        _native_amdgpu_named_flag_immediate("clamp"),
    )


def _gfx125x_swmmac_index_immediate(field_name: str) -> Immediate:
    # gfx125x SWMMAC index-key variants share encoding bit 11.
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


_GFX125X_MATRIX_REUSE_IMMEDIATE_FIELDS = ("matrix_a_reuse", "matrix_b_reuse")

_GFX125X_WMMA_ROW_GROUPS = (
    (
        _SCHEDULE_MATRIX_VALU_EXACT_16,
        (("f32.16x16x4.f32", 0x5D, 2, 2, 8, 8, True),),
    ),
    (
        _SCHEDULE_MATRIX_XDL_EXACT_4,
        (
            ("f32.16x16x64.fp8.fp8", 0x6A, 8, 8, 8, 8, True),
            ("f32.16x16x64.fp8.bf8", 0x6B, 8, 8, 8, 8, True),
            ("f32.16x16x64.bf8.fp8", 0x6C, 8, 8, 8, 8, True),
            ("f32.16x16x64.bf8.bf8", 0x6D, 8, 8, 8, 8, True),
            ("f16.16x16x64.fp8.fp8", 0x6E, 8, 8, 4, 4, True),
            ("f16.16x16x64.fp8.bf8", 0x6F, 8, 8, 4, 4, True),
            ("f16.16x16x64.bf8.fp8", 0x70, 8, 8, 4, 4, True),
            ("f16.16x16x64.bf8.bf8", 0x71, 8, 8, 4, 4, True),
        ),
    ),
    (
        _SCHEDULE_MATRIX_XDL_EXACT_8,
        (
            ("f32.16x16x32.f16", 0x60, 8, 8, 8, 8, True),
            ("f16.16x16x32.f16", 0x61, 8, 8, 4, 4, True),
            ("f32.16x16x32.bf16", 0x62, 8, 8, 8, 8, True),
            ("bf16.16x16x32.bf16", 0x63, 8, 8, 4, 4, True),
            ("bf16f32.16x16x32.bf16", 0x64, 8, 8, 8, 4, True),
            ("f32.16x16x128.fp8.fp8", 0x80, 16, 16, 8, 8, True),
            ("f32.16x16x128.fp8.bf8", 0x81, 16, 16, 8, 8, True),
            ("f32.16x16x128.bf8.fp8", 0x82, 16, 16, 8, 8, True),
            ("f32.16x16x128.bf8.bf8", 0x83, 16, 16, 8, 8, True),
            ("f16.16x16x128.fp8.fp8", 0x84, 16, 16, 4, 4, True),
            ("f16.16x16x128.fp8.bf8", 0x85, 16, 16, 4, 4, True),
            ("f16.16x16x128.bf8.fp8", 0x86, 16, 16, 4, 4, True),
            ("f16.16x16x128.bf8.bf8", 0x87, 16, 16, 4, 4, True),
            ("f32.32x16x128.f4", 0x88, 16, 8, 16, 16, False),
        ),
    ),
    (
        _SCHEDULE_MATRIX_XDL_EXACT_16,
        (("i32.16x16x64.iu8", 0x72, 8, 8, 8, 8, True),),
    ),
)


def _gfx125x_wmma_descriptor(
    name: str,
    encoding_id: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    has_reuse_immediates: bool,
    *,
    schedule_class: str,
) -> Descriptor:
    suffix = name.replace(".", "_")
    has_integer_controls = name == "i32.16x16x64.iu8"
    has_distinct_result = name == "bf16f32.16x16x32.bf16"
    integer_immediates = (
        _gfx125x_matrix_integer_control_immediates() if has_integer_controls else ()
    )
    reuse_immediates = (
        _gfx125x_matrix_reuse_immediates() if has_reuse_immediates else ()
    )
    immediates = (*integer_immediates, *reuse_immediates)
    immediate_fields = (("neg_lo", "clamp") if has_integer_controls else ()) + (
        _GFX125X_MATRIX_REUSE_IMMEDIATE_FIELDS if has_reuse_immediates else ()
    )
    return Descriptor(
        key=f"amdgpu.v_wmma_{suffix}",
        mnemonic=f"v_wmma_{suffix}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("lhs", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("rhs", units=rhs_units), "SRC1"),
            _encoded_operand(
                _vgpr_const_operand("acc", units=accumulator_units), "SRC2"
            ),
        ),
        immediates=immediates,
        constraints=(
            _EARLY_CLOBBER_RESULT_CONSTRAINTS
            if has_distinct_result
            else _destructive_accumulator_constraints(3)
        ),
        encoding_field_values=(
            EncodingFieldValue(
                amdgpu_encoding_field_id("OPSEL_HI"),
                3 if has_reuse_immediates else 7,
            ),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("lhs", "rhs", "acc"),
            immediates=immediate_fields,
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
                _native_operand("rhs"),
                _native_operand("acc"),
                *(
                    _gfx125x_matrix_reuse_native_values()
                    if has_reuse_immediates
                    else ()
                ),
                *(
                    _gfx125x_matrix_integer_control_native_values()
                    if has_integer_controls
                    else ()
                ),
            ),
        ),
        schedule_class=schedule_class,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx125x_wmma_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _gfx125x_wmma_descriptor(*row, schedule_class=schedule_class)
        for schedule_class, rows in _GFX125X_WMMA_ROW_GROUPS
        for row in rows
    )


def _gfx125x_wmma_f8f6f4_descriptor(
    lhs_physical_format: str,
    lhs_units: int,
    rhs_physical_format: str,
    rhs_units: int,
) -> Descriptor:
    name = f"f32.16x16x128.f8f6f4.{lhs_physical_format}.{rhs_physical_format}"
    return Descriptor(
        key=f"amdgpu.v_wmma_{name.replace('.', '_')}",
        mnemonic=f"v_wmma_{name.replace('.', '_')}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=8), "VDST"),
            _encoded_operand(_vgpr_operand("lhs", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("rhs", units=rhs_units), "SRC1"),
            _encoded_operand(_vgpr_const_operand("acc", units=8), "SRC2"),
        ),
        immediates=(
            _gfx125x_matrix_format_immediate(
                "matrix_a_fmt", "MATRIX_A_FMT", lhs_physical_format
            ),
            _gfx125x_matrix_format_immediate(
                "matrix_b_fmt", "MATRIX_B_FMT", rhs_physical_format
            ),
        ),
        constraints=_destructive_accumulator_constraints(3),
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 3),
        ),
        asm_forms=_asm(
            native_assembly_mnemonic="v_wmma_f32_16x16x128_f8f6f4",
            results=("dst",),
            operands=("lhs", "rhs", "acc"),
            immediates=("matrix_a_fmt", "matrix_b_fmt"),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
                _native_operand("rhs"),
                _native_operand("acc"),
                *_gfx125x_matrix_format_native_values(),
            ),
        ),
        schedule_class=(
            _SCHEDULE_MATRIX_XDL_EXACT_4
            if lhs_physical_format == "f4" and rhs_physical_format == "f4"
            else _SCHEDULE_MATRIX_XDL_EXACT_8
        ),
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=0x33,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx125x_wmma_f8f6f4_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _gfx125x_wmma_f8f6f4_descriptor(
            lhs_format.token,
            lhs_format.register_count_for(64),
            rhs_format.token,
            rhs_format.register_count_for(64),
        )
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
        for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
    )


_GFX125X_WMMA_SCALE_ROW_GROUPS = (
    (
        _SCHEDULE_MATRIX_XDL_EXACT_8,
        tuple(
            (
                f"{scale_kind}.f32.16x16x128.f8f6f4."
                f"{lhs_format.token}.{rhs_format.token}",
                0x33,
                x2_encoding,
                lhs_format.register_count_for(64),
                rhs_format.register_count_for(64),
                8,
                8,
                scale_units,
                (lhs_format.token, rhs_format.token),
                f"v_wmma_{scale_kind}_f32_16x16x128_f8f6f4",
            )
            for scale_kind, x2_encoding, scale_units in (
                ("scale", 0x35, 1),
                ("scale16", 0x3A, 2),
            )
            for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
            for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
        ),
    ),
    (
        _SCHEDULE_MATRIX_XDL_EXACT_8,
        (
            (
                "scale.f32.32x16x128.f4",
                0x88,
                0x35,
                16,
                8,
                16,
                16,
                1,
                None,
                None,
            ),
            (
                "scale16.f32.32x16x128.f4",
                0x88,
                0x3A,
                16,
                8,
                16,
                16,
                2,
                None,
                None,
            ),
        ),
    ),
)


def _gfx125x_wmma_scale_descriptor(
    name: str,
    encoding_id: int,
    x2_encoding: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    scale_units: int,
    matrix_physical_formats: tuple[str, str] | None,
    native_assembly_mnemonic: str | None,
    *,
    schedule_class: str,
) -> Descriptor:
    suffix = name.replace(".", "_")
    has_matrix_format_selectors = matrix_physical_formats is not None
    effective_schedule_class = (
        _SCHEDULE_MATRIX_XDL_EXACT_4
        if matrix_physical_formats == ("f4", "f4")
        else schedule_class
    )
    immediate_fields = (
        ("matrix_a_fmt", "matrix_b_fmt") if has_matrix_format_selectors else ()
    ) + (
        "matrix_a_scale",
        "matrix_b_scale",
        "matrix_a_scale_fmt",
        "matrix_b_scale_fmt",
        "matrix_a_reuse",
        "matrix_b_reuse",
    )
    return Descriptor(
        key=f"amdgpu.v_wmma_{suffix}",
        mnemonic=f"v_wmma_{suffix}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("lhs", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("rhs", units=rhs_units), "SRC1"),
            _encoded_operand(
                _vgpr_const_operand("acc", units=accumulator_units), "SRC2"
            ),
            _encoded_operand(
                _sgpr_vgpr_operand("lhs_scale", units=scale_units), "SCALE_SRC0"
            ),
            _encoded_operand(
                _sgpr_vgpr_operand("rhs_scale", units=scale_units), "SCALE_SRC1"
            ),
        ),
        immediates=_gfx125x_wmma_scale_immediates(matrix_physical_formats),
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("X2ENCODING"), x2_encoding),
            *(
                ()
                if has_matrix_format_selectors
                else (EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 7),)
            ),
        ),
        constraints=_destructive_accumulator_constraints(3),
        asm_forms=_asm(
            native_assembly_mnemonic=native_assembly_mnemonic,
            results=("dst",),
            operands=("lhs", "rhs", "acc", "lhs_scale", "rhs_scale"),
            immediates=immediate_fields,
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
                _native_operand("rhs"),
                _native_operand("acc"),
                _native_operand("lhs_scale"),
                _native_operand("rhs_scale"),
                *_gfx125x_wmma_scale_native_values(has_matrix_format_selectors),
            ),
        ),
        schedule_class=effective_schedule_class,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3PX2,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx125x_wmma_scale_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _gfx125x_wmma_scale_descriptor(*row, schedule_class=schedule_class)
        for schedule_class, rows in _GFX125X_WMMA_SCALE_ROW_GROUPS
        for row in rows
    )


_GFX125X_SWMMAC_ROW_GROUPS = (
    (
        _SCHEDULE_MATRIX_XDL_EXACT_8,
        (
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
        ),
    ),
    (
        _SCHEDULE_MATRIX_XDL_EXACT_16,
        (
            (
                "i32.16x16x128.iu8",
                0x7B,
                8,
                16,
                8,
                8,
                "index_key_32bit",
                2,
            ),
        ),
    ),
)


def _gfx125x_swmmac_descriptor(
    name: str,
    encoding_id: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    index_immediate: str,
    index_units: int,
    *,
    schedule_class: str,
) -> Descriptor:
    suffix = name.replace(".", "_")
    has_integer_controls = name == "i32.16x16x128.iu8"
    integer_immediates = (
        _gfx125x_matrix_integer_control_immediates() if has_integer_controls else ()
    )
    integer_immediate_fields = ("neg_lo", "clamp") if has_integer_controls else ()
    return Descriptor(
        key=f"amdgpu.v_swmmac_{suffix}",
        mnemonic=f"v_swmmac_{suffix}",
        semantic_tag=f"matrix.swmmac.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("acc", units=accumulator_units), "VDST"),
            _encoded_operand(_vgpr_operand("lhs", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("rhs", units=rhs_units), "SRC1"),
            _encoded_operand(
                _sgpr_vgpr_operand("sparse_metadata", units=index_units), "SRC2"
            ),
        ),
        immediates=(
            _gfx125x_swmmac_index_immediate(index_immediate),
            *integer_immediates,
            *_gfx125x_matrix_reuse_immediates(),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 3),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("acc", "lhs", "rhs", "sparse_metadata"),
            immediates=(
                index_immediate,
                *integer_immediate_fields,
                *_GFX125X_MATRIX_REUSE_IMMEDIATE_FIELDS,
            ),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
                _native_operand("rhs"),
                _native_operand("sparse_metadata"),
                _native_amdgpu_named_i64_immediate(index_immediate, name="index_key"),
                *_gfx125x_matrix_reuse_native_values(),
                *(
                    _gfx125x_matrix_integer_control_native_values()
                    if has_integer_controls
                    else ()
                ),
            ),
        ),
        schedule_class=schedule_class,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx125x_swmmac_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _gfx125x_swmmac_descriptor(*row, schedule_class=schedule_class)
        for schedule_class, rows in _GFX125X_SWMMAC_ROW_GROUPS
        for row in rows
    )


def _with_gfx125x_inherited_matrix_schedules(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    # Gfx125x retains the gfx12 SWMMAC encodings but executes them on XDL.
    # Integer forms occupy the pipe twice as long as floating-point forms.
    result: list[Descriptor] = []
    for descriptor in descriptors:
        if descriptor.schedule_class != _SCHEDULE_SWMMAC:
            result.append(descriptor)
            continue

        semantic_tag = descriptor.semantic_tag or ""
        input_type = semantic_tag.rpartition(".")[2]
        if input_type in ("iu8", "iu4"):
            schedule_class = _SCHEDULE_MATRIX_XDL_EXACT_16
        elif input_type in ("f16", "bf16", "fp8", "bf8"):
            schedule_class = _SCHEDULE_MATRIX_XDL_EXACT_8
        else:
            raise ValueError(
                f"gfx125x matrix descriptor '{descriptor.key}' with semantic "
                f"tag '{semantic_tag}' has no pipeline schedule"
            )
        result.append(replace(descriptor, schedule_class=schedule_class))
    return tuple(result)


def _with_gfx1250_a0_matrix_schedules(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    result: list[Descriptor] = []
    for descriptor in descriptors:
        schedule_class = descriptor.schedule_class
        if schedule_class == _SCHEDULE_MATRIX_XDL_EXACT_4:
            schedule_class = _SCHEDULE_MATRIX_XDL_EXACT_8

        semantic_components = (descriptor.semantic_tag or "").split(".")
        if (
            len(semantic_components) >= 2
            and "f8f6f4" in semantic_components
            and "f8" in semantic_components[-2:]
        ):
            schedule_class = _SCHEDULE_MATRIX_XDL_EXACT_16

        result.append(
            descriptor
            if schedule_class == descriptor.schedule_class
            else replace(descriptor, schedule_class=schedule_class)
        )
    return tuple(result)


def _with_xdl_latency_tiers(
    descriptors: tuple[Descriptor, ...],
    *,
    regular_schedule_class: str = _SCHEDULE_MATRIX_XDL_EXACT_16,
    slow_schedule_class: str = _SCHEDULE_MATRIX_XDL_EXACT_32,
) -> tuple[Descriptor, ...]:
    result: list[Descriptor] = []
    for descriptor in descriptors:
        semantic_tag = descriptor.semantic_tag or ""
        if not semantic_tag.startswith(("matrix.wmma.", "matrix.swmmac.")):
            result.append(descriptor)
            continue

        if descriptor.schedule_class == _SCHEDULE_MATRIX_VALU_EXACT_16:
            result.append(descriptor)
            continue
        if descriptor.schedule_class not in (
            _SCHEDULE_MATRIX_XDL_EXACT_4,
            _SCHEDULE_MATRIX_XDL_EXACT_8,
            _SCHEDULE_MATRIX_XDL_EXACT_16,
        ):
            raise ValueError(
                f"two-tier XDL matrix descriptor '{descriptor.key}' has unexpected "
                f"schedule class '{descriptor.schedule_class}'"
            )

        is_slow = descriptor.schedule_class == _SCHEDULE_MATRIX_XDL_EXACT_16
        if semantic_tag.startswith("matrix.wmma."):
            is_slow = is_slow or (
                ".16x16x128.fp8." in semantic_tag
                or ".16x16x128.bf8." in semantic_tag
                or ".32x16x128.f4" in semantic_tag
            )
            if ".16x16x128.f8f6f4." in semantic_tag:
                is_slow = not semantic_tag.endswith(".f4.f4")
        result.append(
            replace(
                descriptor,
                schedule_class=(
                    slow_schedule_class if is_slow else regular_schedule_class
                ),
            )
        )
    return tuple(result)


_XDL_ESTIMATED_MATRIX_SCHEDULE_CLASSES = (
    ScheduleClass(
        _SCHEDULE_MATRIX_XDL_ESTIMATED_16,
        latency_kind=LatencyKind.ESTIMATE,
        latency_cycles=16,
        issue_uses=(IssueUse(_RESOURCE_GFX125X_XDL, cycles=16, units=1),),
        hazards=_matrix_hazards(_RESOURCE_GFX125X_XDL),
        model_quality=ModelQuality.ESTIMATED,
    ),
    ScheduleClass(
        _SCHEDULE_MATRIX_XDL_ESTIMATED_32,
        latency_kind=LatencyKind.ESTIMATE,
        latency_cycles=32,
        issue_uses=(IssueUse(_RESOURCE_GFX125X_XDL, cycles=32, units=1),),
        hazards=_matrix_hazards(_RESOURCE_GFX125X_XDL),
        model_quality=ModelQuality.ESTIMATED,
    ),
)


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
    enum_domains=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.enum_domains,
        *_GFX125X_MATRIX_ENUM_DOMAINS,
    ),
    resources=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
        Resource(
            _RESOURCE_GFX125X_XDL,
            capacity_per_cycle=1,
            kind=ResourceKind.MATRIX,
            flags=(
                ResourceFlag.VECTOR_ISSUE,
                ResourceFlag.MATRIX_COEXECUTION_SOURCE,
            ),
        ),
        Resource(
            _RESOURCE_TENSOR,
            capacity_per_cycle=1,
            kind=ResourceKind.LOAD,
        ),
    ),
    schedule_classes=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
        ScheduleClass(
            _SCHEDULE_MATRIX_VALU_EXACT_16,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VALU, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_MATRIX_XDL_EXACT_4,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_GFX125X_XDL, cycles=4, units=1),),
            hazards=_matrix_hazards(_RESOURCE_GFX125X_XDL),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_MATRIX_XDL_EXACT_8,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_GFX125X_XDL, cycles=8, units=1),),
            hazards=_matrix_hazards(_RESOURCE_GFX125X_XDL),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_MATRIX_XDL_EXACT_16,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_GFX125X_XDL, cycles=16, units=1),),
            hazards=_matrix_hazards(_RESOURCE_GFX125X_XDL),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_MATRIX_XDL_EXACT_32,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_GFX125X_XDL, cycles=32, units=1),),
            hazards=_matrix_hazards(_RESOURCE_GFX125X_XDL),
            model_quality=ModelQuality.EXACT,
        ),
        *_XDL_ESTIMATED_MATRIX_SCHEDULE_CLASSES,
        ScheduleClass(
            _SCHEDULE_TENSOR_LOAD_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_TENSOR, cycles=1, units=1),),
            hazards=_TENSOR_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CLUSTER_LOAD_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),),
            hazards=_ASYNC_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_TENSOR,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_TENSOR_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ASYNC,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ASYNC_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_X,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_X_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=(
        _s_set_vgpr_msb_descriptor(),
        *_gfx125x_wmma_descriptors(),
        *_gfx125x_wmma_f8f6f4_descriptors(),
        *_gfx125x_wmma_scale_descriptors(),
        *_gfx125x_swmmac_descriptors(),
    ),
)

_AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.gfx1250_a0.core",
    reg_classes=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.reg_classes,
    register_parts=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.register_parts,
    enum_domains=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.enum_domains,
    resources=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=(_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.schedule_classes),
    descriptors=_with_gfx1250_a0_matrix_schedules(
        _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.descriptors
    ),
    categories=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.categories,
)

_AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.gfx1251.core",
    reg_classes=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.reg_classes,
    register_parts=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.register_parts,
    enum_domains=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.enum_domains,
    resources=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
    descriptors=_with_xdl_latency_tiers(
        _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.descriptors
    ),
    categories=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.categories,
)

_AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE = (
    _amdgpu_core_descriptor_set_intersection(
        key="amdgpu.gfx12.generic.core",
        members=(_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,),
    )
)

_AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE = (
    _amdgpu_core_descriptor_set_intersection(
        key="amdgpu.gfx12_5.generic.core",
        members=(_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,),
    )
)
_AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE = replace(
    _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    schedule_classes=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
    descriptors=_with_xdl_latency_tiers(
        tuple(
            descriptor
            for descriptor in (
                _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE.descriptors
            )
            if not (descriptor.semantic_tag or "").startswith("matrix.swmmac.")
        ),
        regular_schedule_class=_SCHEDULE_MATRIX_XDL_ESTIMATED_16,
        slow_schedule_class=_SCHEDULE_MATRIX_XDL_ESTIMATED_32,
    ),
)


__all__ = (
    "_AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE",
    "_SCHEDULE_MATRIX_XDL_ESTIMATED_16",
    "_SCHEDULE_MATRIX_XDL_ESTIMATED_32",
    "_gfx125x_reg_classes",
    "_s_getreg_b32_cluster_workgroup_flat_id_overlay",
    "_with_gfx1250_a0_matrix_schedules",
    "_with_gfx125x_inherited_matrix_schedules",
    "_with_xdl_latency_tiers",
)
