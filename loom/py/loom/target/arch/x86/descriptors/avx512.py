# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX512 descriptor rows and view metadata."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.vector_encoding import (
    VectorEncodingPrefix,
    VectorMachineInstruction,
)
from loom.target.arch.x86.vector_families import (
    AVX512_BITWISE_FAMILIES,
    AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS,
    AVX512_FLOAT_BINARY_FAMILIES,
    AVX512_FLOAT_COMPARE_MNEMONICS,
    AVX512_FLOAT_FMA_MNEMONICS,
    AVX512_FLOAT_ROOT_FAMILIES,
    AVX512_INTEGER_BINARY_FAMILIES,
    AVX512_INTEGER_COMPARE_MNEMONICS,
    AVX512_SCALAR_FLOAT_ROOT_FAMILIES,
    AVX512_SELECT_MNEMONICS,
    AVX512_UNIFORM_SHIFT_FAMILIES,
    AVX512VL_INTEGER_BINARY_FAMILIES,
    AVX512VL_UNIFORM_SHIFT_FAMILIES,
    AVX512VL_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    FLOAT_EXTREMA_MNEMONICS,
    INTEGER_ELEMENTS,
    X86_VECTOR_INTEGER_NARROWING_FAMILIES,
    X86_VECTOR_INTEGER_WIDENING_FAMILIES,
    VectorIntegerNarrowingFamily,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Immediate,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    ModelQuality,
    RegClass,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

from .avx2 import X86_AVX2_DESCRIPTORS
from .common import (
    _LANE_I32X4_IMMEDIATE,
    _READ_ONLY_DATA_IMMEDIATE,
    _REG_K,
    _REG_XMM,
    _REG_YMM,
    _REG_ZMM,
    _RESOURCE_ADDRESS,
    _RESOURCE_CONTROL,
    _RESOURCE_DIVISION,
    _RESOURCE_DOT,
    _RESOURCE_LOAD,
    _RESOURCE_MASK,
    _RESOURCE_SCALAR,
    _RESOURCE_STORE,
    _RESOURCE_VECTOR,
    _SCHEDULE_MASK,
    _SCHEDULE_MEMORY_LOAD_XMM,
    _SCHEDULE_MEMORY_LOAD_YMM,
    _SCHEDULE_MEMORY_LOAD_ZMM,
    _SCHEDULE_MEMORY_STORE_XMM,
    _SCHEDULE_MEMORY_STORE_YMM,
    _SCHEDULE_MEMORY_STORE_ZMM,
    _SCHEDULE_VECTOR_COMPARE_XMM,
    _SCHEDULE_VECTOR_COMPARE_YMM,
    _SCHEDULE_VECTOR_COMPARE_ZMM,
    _SCHEDULE_VECTOR_DOT_ZMM,
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _SCHEDULE_VECTOR_F32_ZMM,
    _SCHEDULE_VECTOR_FMA_F32_XMM,
    _SCHEDULE_VECTOR_FMA_F32_YMM,
    _SCHEDULE_VECTOR_FMA_F32_ZMM,
    _SCHEDULE_VECTOR_I32_XMM,
    _SCHEDULE_VECTOR_I32_YMM,
    _SCHEDULE_VECTOR_I32_ZMM,
    _TWO_LANE_IMMEDIATE,
    _asm,
    _gpr32_operand,
    _gpr64_operand,
    _gpr64_result,
    _k_operand,
    _k_result,
    _scalar_float_unary_descriptor,
    _vector_f32_binary_descriptor,
    _vector_float_unary_descriptor,
    _vector_fma_descriptor,
    _vector_i32_binary_descriptor,
    _vector_i32_schedule_class,
    _vector_lane_units,
    _vector_mask_compare_descriptor,
    _vector_mask_select_descriptor,
    _vector_operand,
    _vector_result,
    _vector_splat_descriptor,
    _vector_zero_descriptor,
    _xmm_operand,
    _xmm_result,
    _zmm_operand,
    _zmm_result,
)
from .float_conversion import float_conversion_descriptors
from .half_conversion import half_conversion_descriptors
from .integer_float import integer_float_descriptors
from .memory import memory_descriptors
from .scalar import X86_SCALAR_DESCRIPTOR_SET

_REGISTER_SUFFIXES = {64: "xmm", 128: "xmm", 256: "ymm", 512: "zmm"}


def _evex_descriptor(
    descriptor: Descriptor, instruction: VectorMachineInstruction
) -> Descriptor:
    return instruction.bind(descriptor, VectorEncodingPrefix.EVEX)


def _vex_descriptor(
    descriptor: Descriptor, instruction: VectorMachineInstruction
) -> Descriptor:
    return instruction.bind(descriptor, VectorEncodingPrefix.VEX)


def _direct_broadcast_asm_mnemonic(
    mnemonic: str,
    element_bit_width: int,
    vector_bit_width: int,
) -> str | None:
    if vector_bit_width == 256 or (
        vector_bit_width == 128 and element_bit_width in (8, 16)
    ):
        return f"avx512.{mnemonic}.{_REGISTER_SUFFIXES[vector_bit_width]}"
    return None


def _zmm_widen_descriptor(
    mnemonic: str,
    source_bit_width: int,
    source_element_bit_width: int,
    result_element_bit_width: int,
    signedness: str,
    instruction: VectorMachineInstruction,
) -> Descriptor:
    source_suffix = _REGISTER_SUFFIXES[source_bit_width]
    return _evex_descriptor(
        Descriptor(
            key=f"x86.avx512.{mnemonic}.zmm.{source_suffix}",
            mnemonic=mnemonic,
            semantic_tag=(
                f"integer.extend.{signedness}."
                f"i{source_element_bit_width}x"
                f"{512 // result_element_bit_width}."
                f"i{result_element_bit_width}x{512 // result_element_bit_width}"
            ),
            operands=(_zmm_result(), _vector_operand(source_bit_width, "source")),
            asm_forms=_asm(
                mnemonic=f"avx512.{mnemonic}.zmm.{source_suffix}",
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _vector_narrow_descriptor(
    family: VectorIntegerNarrowingFamily,
    source_vector_bit_width: int,
    instruction: VectorMachineInstruction,
) -> Descriptor:
    result_vector_bit_width = max(
        128,
        source_vector_bit_width * family.result.bit_width // family.source.bit_width,
    )
    source_suffix = _REGISTER_SUFFIXES[source_vector_bit_width]
    result_suffix = _REGISTER_SUFFIXES[result_vector_bit_width]
    lane_count = source_vector_bit_width // family.source.bit_width
    return _evex_descriptor(
        Descriptor(
            key=(f"x86.avx512.{family.mnemonic}.{result_suffix}.{source_suffix}"),
            mnemonic=family.mnemonic,
            semantic_tag=(
                f"integer.truncate.{family.source.name}x{lane_count}."
                f"{family.result.name}x{lane_count}"
            ),
            operands=(
                _vector_result(result_vector_bit_width),
                _vector_operand(source_vector_bit_width, "source"),
            ),
            asm_forms=_asm(
                mnemonic=(f"avx512.{family.mnemonic}.{result_suffix}.{source_suffix}"),
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_vector_i32_schedule_class(source_vector_bit_width),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _predicate_conversion_descriptor(
    element_bit_width: int,
    vector_bit_width: int,
    instruction: VectorMachineInstruction,
    *,
    mask_to_carrier: bool,
) -> Descriptor:
    element_suffix = {8: "b", 16: "w", 32: "d", 64: "q"}[element_bit_width]
    lane_count = vector_bit_width // element_bit_width
    register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
    mnemonic = (
        f"vpmovm2{element_suffix}" if mask_to_carrier else f"vpmov{element_suffix}2m"
    )
    return _evex_descriptor(
        Descriptor(
            key=(
                f"x86.avx512.{mnemonic}.{register_suffix}.k"
                if mask_to_carrier
                else f"x86.avx512.{mnemonic}.k.{register_suffix}"
            ),
            mnemonic=mnemonic,
            semantic_tag=(
                f"mask.expand.i1x{lane_count}.i{element_bit_width}x{lane_count}"
                if mask_to_carrier
                else f"mask.compress.i{element_bit_width}x{lane_count}.i1x{lane_count}"
            ),
            operands=(
                (_vector_result(vector_bit_width), _k_operand("source"))
                if mask_to_carrier
                else (_k_result(), _vector_operand(vector_bit_width, "source"))
            ),
            asm_forms=_asm(
                mnemonic=(
                    f"avx512.{mnemonic}.{register_suffix}.k"
                    if mask_to_carrier
                    else f"avx512.{mnemonic}.k.{register_suffix}"
                ),
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_SCHEDULE_MASK,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _mask_move_descriptor(
    instruction: VectorMachineInstruction, *, mask_to_gpr: bool
) -> Descriptor:
    asm_mnemonic = "avx512.kmovq.gpr64.k" if mask_to_gpr else "avx512.kmovq.k.gpr64"
    descriptor = Descriptor(
        key=("x86.avx512.kmovq.gpr64.k" if mask_to_gpr else "x86.avx512.kmovq.k.gpr64"),
        mnemonic="kmovq",
        semantic_tag="mask.move.i64",
        operands=(
            (_gpr64_result(), _k_operand("source"))
            if mask_to_gpr
            else (_k_result(), _gpr64_operand("source"))
        ),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            results=("dst",),
            operands=("source",),
        ),
        schedule_class=_SCHEDULE_MASK,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return instruction.bind(descriptor, VectorEncodingPrefix.VEX)


X86_AVX512_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx512.core",
    target_key="x86",
    feature_key="x86.avx512.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/x86/descriptors/avx512_descriptors.h"
    ),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx512_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX512_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/descriptors/avx512_descriptors.h",
    function_name="loom_x86_avx512_core_descriptor_set",
    c_table_prefix="X86Avx512Core",
    c_enum_prefix="X86_AVX512_CORE",
    generator_version=1,
    supports_native_scheduling=True,
    physical_registers=X86_SCALAR_DESCRIPTOR_SET.physical_registers,
    reg_classes=(
        *X86_SCALAR_DESCRIPTOR_SET.reg_classes,
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=32,
            alias_set_id=1,
        ),
        RegClass(
            _REG_YMM,
            256,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=32,
            alias_set_id=1,
        ),
        RegClass(
            _REG_ZMM,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=32,
            alias_set_id=1,
        ),
        RegClass(
            _REG_K,
            64,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=8,
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(
            _RESOURCE_DIVISION,
            capacity_per_cycle=1,
            kind=ResourceKind.PIPELINE,
        ),
        Resource(
            _RESOURCE_VECTOR,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
        Resource(
            _RESOURCE_DOT,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
        Resource(_RESOURCE_MASK, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(
            _RESOURCE_LOAD,
            capacity_per_cycle=4,
            kind=ResourceKind.LOAD,
            contention_group_id=2,
        ),
        Resource(
            _RESOURCE_STORE,
            capacity_per_cycle=4,
            kind=ResourceKind.STORE,
            contention_group_id=3,
        ),
        Resource(
            _RESOURCE_ADDRESS,
            capacity_per_cycle=1,
            kind=ResourceKind.ADDRESS,
            contention_group_id=4,
        ),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *X86_SCALAR_DESCRIPTOR_SET.schedule_classes,
        ScheduleClass(
            _SCHEDULE_VECTOR_I32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_I32_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(256)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_I32_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_F32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_F32_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(256)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_F32_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_FMA_F32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_FMA_F32_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(256)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_FMA_F32_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_COMPARE_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_COMPARE_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(256)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_COMPARE_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MASK,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_MASK, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_ZMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=_vector_lane_units(512)),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_XMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=_vector_lane_units(128)),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_YMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=_vector_lane_units(256)),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE_ZMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_STORE, cycles=1, units=_vector_lane_units(512)),
            ),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE_XMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_STORE, cycles=1, units=_vector_lane_units(128)),
            ),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE_YMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_STORE, cycles=1, units=_vector_lane_units(256)),
            ),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    enum_domains=X86_SCALAR_DESCRIPTOR_SET.enum_domains,
    descriptors=(
        *X86_AVX2_DESCRIPTORS,
        *integer_float_descriptors(unsigned=True),
        *float_conversion_descriptors(evex=True),
        *half_conversion_descriptors(evex=True),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vextractf32x4.xmm.zmm",
                mnemonic="vextractf32x4",
                semantic_tag="float.extract.f32x16.quarter",
                operands=(_xmm_result(), _zmm_operand("source")),
                immediates=(_LANE_I32X4_IMMEDIATE,),
                asm_forms=_asm(
                    mnemonic="vextractf32x4.xmm",
                    results=("dst",),
                    operands=("source",),
                    immediates=("lane",),
                ),
                schedule_class=_SCHEDULE_VECTOR_F32_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VEXTRACTF32X4,
        ),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vinsertf32x4.zmm.xmm",
                mnemonic="vinsertf32x4",
                semantic_tag="bits.insert.128.zmm",
                operands=(
                    _zmm_result(),
                    _zmm_operand("dest"),
                    _xmm_operand("value"),
                ),
                immediates=(_LANE_I32X4_IMMEDIATE,),
                asm_forms=_asm(
                    mnemonic="vinsertf32x4.zmm.xmm",
                    results=("dst",),
                    operands=("dest", "value"),
                    immediates=("lane",),
                ),
                schedule_class=_SCHEDULE_VECTOR_F32_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VINSERTF32X4,
        ),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vextracti64x4.ymm.zmm",
                mnemonic="vextracti64x4",
                semantic_tag="bits.extract.256.zmm",
                operands=(_vector_result(256), _zmm_operand("source")),
                immediates=(_TWO_LANE_IMMEDIATE,),
                asm_forms=_asm(
                    mnemonic="vextracti64x4.ymm",
                    results=("dst",),
                    operands=("source",),
                    immediates=("lane",),
                ),
                schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VEXTRACTI64X4,
        ),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vinserti64x4.zmm.ymm",
                mnemonic="vinserti64x4",
                semantic_tag="bits.insert.256.zmm",
                operands=(
                    _zmm_result(),
                    _zmm_operand("dest"),
                    _vector_operand(256, "value"),
                ),
                immediates=(_TWO_LANE_IMMEDIATE,),
                asm_forms=_asm(
                    mnemonic="vinserti64x4.zmm.ymm",
                    results=("dst",),
                    operands=("dest", "value"),
                    immediates=("lane",),
                ),
                schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VINSERTI64X4,
        ),
        *(
            _vector_narrow_descriptor(family, source_vector_bit_width, instruction)
            for family, instruction in zip(
                X86_VECTOR_INTEGER_NARROWING_FAMILIES,
                native.PACKED_INTEGER_NARROW,
                strict=True,
            )
            for source_vector_bit_width in (128, 256, 512)
        ),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vshufi64x2.zmm",
                mnemonic="vshufi64x2",
                semantic_tag="bits.shuffle.i64x8",
                operands=(
                    _zmm_result(),
                    _zmm_operand("lhs"),
                    _zmm_operand("rhs"),
                ),
                immediates=(
                    Immediate(
                        "control",
                        ImmediateKind.UNSIGNED,
                        bit_width=8,
                        unsigned_max=255,
                    ),
                ),
                asm_forms=_asm(
                    results=("dst",),
                    operands=("lhs", "rhs"),
                    immediates=("control",),
                ),
                schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VSHUFI64X2,
        ),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vpsrldq.zmm",
                mnemonic="vpsrldq",
                semantic_tag="bits.shift_right_bytes.v512x128",
                operands=(_zmm_result(), _zmm_operand("source")),
                immediates=(
                    Immediate(
                        "bytes",
                        ImmediateKind.UNSIGNED,
                        bit_width=8,
                        unsigned_max=255,
                    ),
                ),
                asm_forms=_asm(
                    mnemonic="vpsrldq.zmm",
                    results=("dst",),
                    operands=("source",),
                    immediates=("bytes",),
                ),
                schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VPSRLDQ,
        ),
        *(
            _evex_descriptor(
                _vector_splat_descriptor(
                    vector_bit_width=vector_bit_width,
                    key=(
                        f"x86.avx512.{mnemonic}.{_REGISTER_SUFFIXES[vector_bit_width]}"
                    ),
                    mnemonic=mnemonic,
                    semantic_tag=(
                        f"bits.splat.{element_bit_width}.v{vector_bit_width}"
                    ),
                    operand=(
                        _gpr64_operand("value")
                        if element_bit_width == 64
                        else _gpr32_operand("value")
                    ),
                    schedule_class=_vector_i32_schedule_class(vector_bit_width),
                    asm_mnemonic=_direct_broadcast_asm_mnemonic(
                        mnemonic,
                        element_bit_width,
                        vector_bit_width,
                    ),
                ),
                instruction,
            )
            for (element_bit_width, mnemonic), instruction in zip(
                (
                    (8, "vpbroadcastb"),
                    (16, "vpbroadcastw"),
                    (32, "vpbroadcastd"),
                    (64, "vpbroadcastq"),
                ),
                native.AVX512_GPR_BROADCAST,
                strict=True,
            )
            for vector_bit_width in AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS
        ),
        *(
            _evex_descriptor(
                _vector_splat_descriptor(
                    vector_bit_width=512,
                    key=f"x86.avx512.{mnemonic}.zmm",
                    mnemonic=mnemonic,
                    semantic_tag=(
                        f"float.splat.{element.name}x{element.lane_count(512)}"
                    ),
                    operand=_xmm_operand("value"),
                    schedule_class=_SCHEDULE_VECTOR_F32_ZMM,
                ),
                instruction,
            )
            for element, mnemonic, instruction in zip(
                FLOAT_ELEMENTS,
                ("vbroadcastss", "vbroadcastsd"),
                (native.VBROADCASTSS, native.VBROADCASTSD),
                strict=True,
            )
        ),
        _evex_descriptor(
            _vector_zero_descriptor(
                vector_bit_width=512,
                key="x86.avx512.vxorps.zero.zmm",
            ),
            native.VXORPS,
        ),
        *(
            _zmm_widen_descriptor(
                mnemonic,
                512 * family.source.bit_width // family.result.bit_width,
                family.source.bit_width,
                family.result.bit_width,
                signedness,
                instruction,
            )
            for family, instructions in zip(
                X86_VECTOR_INTEGER_WIDENING_FAMILIES,
                native.PACKED_INTEGER_WIDEN,
                strict=True,
            )
            for mnemonic, signedness, instruction in zip(
                (family.signed_mnemonic, family.unsigned_mnemonic),
                ("signed", "unsigned"),
                instructions,
                strict=True,
            )
        ),
        *(
            _evex_descriptor(
                _vector_i32_binary_descriptor(
                    vector_bit_width=vector_bit_width,
                    key=(
                        f"x86.avx512.{family.mnemonic}."
                        f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                    ),
                    mnemonic=family.mnemonic,
                    semantic_tag=(
                        f"{family.semantic}.{family.element.name}x"
                        f"{family.element.lane_count(vector_bit_width)}"
                    ),
                ),
                instruction,
            )
            for family, instruction in zip(
                AVX512VL_INTEGER_BINARY_FAMILIES,
                native.AVX512VL_INTEGER_BINARY,
                strict=True,
            )
            for vector_bit_width in AVX512VL_VECTOR_BIT_WIDTHS
        ),
        *(
            _evex_descriptor(
                _vector_i32_binary_descriptor(
                    vector_bit_width=512,
                    key=f"x86.avx512.{family.mnemonic}.zmm",
                    mnemonic=family.mnemonic,
                    semantic_tag=(
                        f"{family.semantic}.{family.element.name}x"
                        f"{family.element.lane_count(512)}"
                    ),
                ),
                instruction,
            )
            for family, instruction in zip(
                AVX512_INTEGER_BINARY_FAMILIES,
                native.AVX512_INTEGER_BINARY,
                strict=True,
            )
        ),
        *(
            _evex_descriptor(
                _vector_i32_binary_descriptor(
                    vector_bit_width=512,
                    key=f"x86.avx512.{mnemonic}.zmm",
                    mnemonic=mnemonic,
                    semantic_tag=f"{semantic}.v512",
                ),
                instruction,
            )
            for (_, mnemonic, semantic), instruction in zip(
                AVX512_BITWISE_FAMILIES, native.AVX512_BITWISE, strict=True
            )
        ),
        _evex_descriptor(
            _vector_i32_binary_descriptor(
                vector_bit_width=512,
                key="x86.avx512.vpshufb.zmm",
                mnemonic="vpshufb",
                semantic_tag="bits.shuffle.bytes.v512",
            ),
            native.VPSHUFB,
        ),
        *(
            _evex_descriptor(
                Descriptor(
                    key=f"x86.avx512.{mnemonic}.zmm",
                    mnemonic=mnemonic,
                    semantic_tag=f"bits.shuffle.i{element_bit_width}x{lane_count}",
                    operands=(
                        _zmm_result(),
                        _zmm_operand("source"),
                        _zmm_operand("control"),
                    ),
                    asm_forms=_asm(
                        mnemonic=f"avx512.{mnemonic}.zmm",
                        results=("dst",),
                        operands=("source", "control"),
                    ),
                    schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
                    flags=(DescriptorFlag.DEAD_REMOVABLE,),
                ),
                instruction,
            )
            for mnemonic, element_bit_width, lane_count, instruction in (
                ("vpermw", 16, 32, native.VPERMW),
                ("vpermd", 32, 16, native.VPERMD),
                ("vpermq", 64, 8, native.VPERMQ_VECTOR_BINARY),
            )
        ),
        *(
            _evex_descriptor(
                Descriptor(
                    key=f"x86.avx512.{family.mnemonic}.{_REGISTER_SUFFIXES[width]}",
                    mnemonic=family.mnemonic,
                    semantic_tag=f"{family.semantic}.{family.element.name}x{family.element.lane_count(width)}",
                    operands=(_vector_result(width), _vector_operand(width, "source")),
                    immediates=(
                        Immediate(
                            "shift",
                            ImmediateKind.UNSIGNED,
                            bit_width=8,
                            unsigned_max=255,
                        ),
                    ),
                    asm_forms=_asm(
                        mnemonic=f"avx512.{family.mnemonic}.{_REGISTER_SUFFIXES[width]}",
                        results=("dst",),
                        operands=("source",),
                        immediates=("shift",),
                    ),
                    schedule_class=_vector_i32_schedule_class(width),
                    flags=(DescriptorFlag.DEAD_REMOVABLE,),
                ),
                instruction,
            )
            for family, instruction in zip(
                AVX512_UNIFORM_SHIFT_FAMILIES,
                native.AVX512_UNIFORM_SHIFT_IMMEDIATE,
                strict=True,
            )
            for width in (
                (128, 256, 512) if family in AVX512VL_UNIFORM_SHIFT_FAMILIES else (512,)
            )
        ),
        _evex_descriptor(
            Descriptor(
                key="x86.avx512.vmovdqu64.rodata.zmm",
                mnemonic="vmovdqu64",
                semantic_tag="memory.load.rodata.v512",
                operands=(_zmm_result(),),
                immediates=(_READ_ONLY_DATA_IMMEDIATE,),
                asm_forms=_asm(
                    mnemonic="avx512.vmovdqu64.rodata.zmm",
                    results=("dst",),
                    immediates=("data",),
                ),
                schedule_class=_SCHEDULE_MEMORY_LOAD_ZMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VMOVDQU64,
        ),
        *(
            _evex_descriptor(
                _vector_f32_binary_descriptor(
                    vector_bit_width=512,
                    key=f"x86.avx512.{family.mnemonic}.zmm",
                    mnemonic=family.mnemonic,
                    semantic_tag=(
                        f"{family.semantic}.{family.element.name}x"
                        f"{family.element.lane_count(512)}"
                    ),
                ),
                instruction,
            )
            for family, instruction in zip(
                AVX512_FLOAT_BINARY_FAMILIES,
                native.AVX2_FLOAT_BINARY,
                strict=True,
            )
        ),
        *(
            _evex_descriptor(
                _vector_float_unary_descriptor(
                    vector_bit_width=vector_bit_width,
                    key=(
                        f"x86.avx512.{family.mnemonic}."
                        f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                    ),
                    mnemonic=family.mnemonic,
                    semantic_tag=(
                        f"{family.semantic}.{family.element.name}x"
                        f"{family.element.lane_count(vector_bit_width)}"
                    ),
                ),
                instruction,
            )
            for family, instruction in zip(
                AVX512_FLOAT_ROOT_FAMILIES,
                native.AVX512_FLOAT_ROOT,
                strict=True,
            )
            for vector_bit_width in ((128, 256, 512) if family.approximate else (512,))
        ),
        *(
            _evex_descriptor(
                _scalar_float_unary_descriptor(
                    key=f"x86.avx512.{family.mnemonic}.xmm",
                    mnemonic=family.mnemonic,
                    semantic_tag=f"{family.semantic}.{family.element.name}",
                ),
                instruction,
            )
            for family, instruction in zip(
                AVX512_SCALAR_FLOAT_ROOT_FAMILIES,
                native.AVX512_SCALAR_FLOAT_ROOT,
                strict=True,
            )
        ),
        *(
            _evex_descriptor(
                _vector_f32_binary_descriptor(
                    vector_bit_width=512,
                    key=f"x86.avx512.{mnemonic}.zmm",
                    mnemonic=mnemonic,
                    semantic_tag=(
                        f"float.fast_extrema.{element.name}x{element.lane_count(512)}"
                    ),
                ),
                instruction,
            )
            for element, instructions in zip(
                FLOAT_ELEMENTS,
                (native.AVX2_FLOAT_EXTREMA[:2], native.AVX2_FLOAT_EXTREMA[2:]),
                strict=True,
            )
            for mnemonic, instruction in zip(
                (
                    FLOAT_EXTREMA_MNEMONICS["minimumf"][element.name],
                    FLOAT_EXTREMA_MNEMONICS["maximumf"][element.name],
                ),
                instructions,
                strict=True,
            )
        ),
        *(
            _evex_descriptor(
                _vector_fma_descriptor(
                    vector_bit_width=512,
                    key=(f"x86.avx512.{AVX512_FLOAT_FMA_MNEMONICS[element.name]}.zmm"),
                    mnemonic=AVX512_FLOAT_FMA_MNEMONICS[element.name],
                    semantic_tag=(
                        f"float.fma.{element.name}x{element.lane_count(512)}"
                    ),
                ),
                instruction,
            )
            for element, instruction in zip(
                FLOAT_ELEMENTS, native.AVX2_FLOAT_FMA, strict=True
            )
        ),
        *(
            _evex_descriptor(
                _vector_mask_compare_descriptor(
                    vector_bit_width=vector_bit_width,
                    key=(
                        f"x86.avx512.{mnemonic}.{_REGISTER_SUFFIXES[vector_bit_width]}"
                    ),
                    mnemonic=mnemonic,
                    semantic_tag=(
                        f"integer.cmp.{signedness}.{element.name}x"
                        f"{element.lane_count(vector_bit_width)}"
                    ),
                ),
                instruction,
            )
            for vector_bit_width in (128, 256, 512)
            for element, instructions in zip(
                INTEGER_ELEMENTS, native.AVX512_INTEGER_COMPARE, strict=True
            )
            for mnemonic, signedness, instruction in zip(
                AVX512_INTEGER_COMPARE_MNEMONICS[element.name],
                ("signed", "unsigned"),
                instructions,
                strict=True,
            )
        ),
        *(
            _evex_descriptor(
                _vector_mask_compare_descriptor(
                    vector_bit_width=vector_bit_width,
                    key=(
                        f"x86.avx512."
                        f"{AVX512_FLOAT_COMPARE_MNEMONICS[element.name]}."
                        f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                    ),
                    mnemonic=AVX512_FLOAT_COMPARE_MNEMONICS[element.name],
                    semantic_tag=(
                        f"float.cmp.{element.name}x"
                        f"{element.lane_count(vector_bit_width)}"
                    ),
                ),
                instruction,
            )
            for vector_bit_width in (128, 256, 512)
            for element, instruction in zip(
                FLOAT_ELEMENTS, native.AVX512_FLOAT_COMPARE, strict=True
            )
        ),
        *(
            _evex_descriptor(
                _vector_mask_select_descriptor(
                    vector_bit_width=vector_bit_width,
                    key=(
                        f"x86.avx512.{mnemonic}.{_REGISTER_SUFFIXES[vector_bit_width]}"
                    ),
                    mnemonic=mnemonic,
                    semantic_tag=f"bits.select.{mnemonic}.v{vector_bit_width}",
                    schedule_class=(
                        {
                            128: _SCHEDULE_VECTOR_F32_XMM,
                            256: _SCHEDULE_VECTOR_F32_YMM,
                            512: _SCHEDULE_VECTOR_F32_ZMM,
                        }[vector_bit_width]
                        if mnemonic.startswith("vblend")
                        else {
                            128: _SCHEDULE_VECTOR_I32_XMM,
                            256: _SCHEDULE_VECTOR_I32_YMM,
                            512: _SCHEDULE_VECTOR_I32_ZMM,
                        }[vector_bit_width]
                    ),
                ),
                instruction,
            )
            for vector_bit_width in (128, 256, 512)
            for mnemonic, instruction in zip(
                dict.fromkeys(AVX512_SELECT_MNEMONICS.values()),
                native.AVX512_SELECT,
                strict=True,
            )
        ),
        *memory_descriptors(
            key_prefix="x86.avx512",
            load_mnemonic="vmovdqu32",
            store_mnemonic="vmovdqu32",
            register_class=_REG_ZMM,
            register_suffix="zmm",
            semantic_type="v512",
            width_bits=512,
            load_schedule_class=_SCHEDULE_MEMORY_LOAD_ZMM,
            store_schedule_class=_SCHEDULE_MEMORY_STORE_ZMM,
            assembly_suffix="",
            vector_instructions=native.VECTOR_MEMORY,
            vector_prefix=VectorEncodingPrefix.EVEX,
        ),
        *(
            _predicate_conversion_descriptor(
                element_bit_width,
                vector_bit_width,
                instruction,
                mask_to_carrier=mask_to_carrier,
            )
            for element_bit_width, instructions in zip(
                (8, 16, 32, 64),
                native.AVX512_PREDICATE_CONVERSION,
                strict=True,
            )
            for vector_bit_width in (128, 256, 512)
            for mask_to_carrier, instruction in zip(
                (False, True), instructions, strict=True
            )
        ),
        _mask_move_descriptor(native.KMOVQ_FROM_GPR64, mask_to_gpr=False),
        _mask_move_descriptor(native.KMOVQ_TO_GPR64, mask_to_gpr=True),
        _vex_descriptor(
            Descriptor(
                key="x86.avx512.kandq",
                mnemonic="kandq",
                semantic_tag="mask.and.i64",
                operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
                asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
                schedule_class=_SCHEDULE_MASK,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.KANDQ,
        ),
        _vex_descriptor(
            Descriptor(
                key="x86.avx512.korq",
                mnemonic="korq",
                semantic_tag="mask.or.i64",
                operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
                asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
                schedule_class=_SCHEDULE_MASK,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.KORQ,
        ),
        _vex_descriptor(
            Descriptor(
                key="x86.avx512.kxorq",
                mnemonic="kxorq",
                semantic_tag="mask.xor.i64",
                operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
                asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
                schedule_class=_SCHEDULE_MASK,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.KXORQ,
        ),
    ),
)
