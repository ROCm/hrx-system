# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX2 descriptor rows and view metadata."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    RegClass,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

from .common import (
    _ADDRESS_SCALE_IMMEDIATE,
    _DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
    _DISP32_IMMEDIATE,
    _INSERTPS_CONTROL_IMMEDIATE,
    _LANE_I32X4_IMMEDIATE,
    _REG_XMM,
    _REG_YMM,
    _RESOURCE_ADDRESS,
    _RESOURCE_LOAD,
    _RESOURCE_STORE,
    _RESOURCE_VECTOR,
    _SCHEDULE_MEMORY_LOAD_XMM,
    _SCHEDULE_MEMORY_STORE_XMM,
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_FMA_F32_XMM,
    _SCHEDULE_VECTOR_I32_XMM,
    _SHUFFLE_4X2_CONTROL_IMMEDIATE,
    _asm,
    _gpr32_operand,
    _gpr32_result,
    _gpr64_resource,
    _load_effect,
    _low_subset_operand,
    _scalar_f32_binary_descriptor,
    _store_effect,
    _vector_f32_binary_descriptor,
    _vector_i32_binary_descriptor,
    _vector_lane_units,
    _vector_operand,
    _vector_result,
    _vector_splat_descriptor,
    _xmm_operand,
)
from .scalar import (
    X86_SCALAR_DESCRIPTOR_SET,
    X86_SCALAR_PREFIX_DESCRIPTORS,
    X86_SCALAR_SUFFIX_DESCRIPTORS,
)

_X86_VEX_ADDRESSABLE_REGISTER_COUNT = 16


def _vex_operand(operand: Operand) -> Operand:
    if any(reg_alt.reg_class in (_REG_XMM, _REG_YMM) for reg_alt in operand.reg_alts):
        return _low_subset_operand(operand, _X86_VEX_ADDRESSABLE_REGISTER_COUNT)
    return operand


def _vex_descriptor(descriptor: Descriptor) -> Descriptor:
    return replace(
        descriptor,
        operands=tuple(_vex_operand(operand) for operand in descriptor.operands),
    )


_X86_AVX2_XMM_DESCRIPTORS = (
    _vector_splat_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vpbroadcastd.xmm",
        mnemonic="vpbroadcastd",
        semantic_tag="integer.splat.i32x4",
        operand=_gpr32_operand("value"),
        schedule_class=_SCHEDULE_VECTOR_I32_XMM,
    ),
    Descriptor(
        key="x86.avx2.vpextrd.gpr32.xmm",
        mnemonic="vpextrd",
        semantic_tag="integer.extract.i32x4",
        operands=(_gpr32_result(), _xmm_operand("source")),
        immediates=(_LANE_I32X4_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vpextrd.xmm",
            results=("dst",),
            operands=("source",),
            immediates=("lane",),
        ),
        schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.avx2.vpinsrd.xmm",
        mnemonic="vpinsrd",
        semantic_tag="integer.insert.i32x4",
        operands=(
            _vector_result(128),
            _xmm_operand("dest"),
            _gpr32_operand("value"),
        ),
        immediates=(_LANE_I32X4_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vpinsrd.xmm",
            results=("dst",),
            operands=("dest", "value"),
            immediates=("lane",),
        ),
        schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.avx2.vpshufd.xmm",
        mnemonic="vpshufd",
        semantic_tag="integer.shuffle.i32x4",
        operands=(_vector_result(128), _xmm_operand("source")),
        immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vpshufd.xmm",
            results=("dst",),
            operands=("source",),
            immediates=("control",),
        ),
        schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    _vector_splat_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vbroadcastss.xmm",
        mnemonic="vbroadcastss",
        semantic_tag="float.splat.f32x4",
        operand=_xmm_operand("value"),
        schedule_class=_SCHEDULE_VECTOR_F32_XMM,
    ),
    Descriptor(
        key="x86.avx2.vpermilps.xmm",
        mnemonic="vpermilps",
        semantic_tag="float.shuffle.f32x4",
        operands=(_vector_result(128), _xmm_operand("source")),
        immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vpermilps.xmm",
            results=("dst",),
            operands=("source",),
            immediates=("control",),
        ),
        schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.avx2.vinsertps.xmm",
        mnemonic="vinsertps",
        semantic_tag="float.insert.f32x4",
        operands=(
            _vector_result(128),
            _xmm_operand("dest"),
            _xmm_operand("value"),
        ),
        immediates=(_INSERTPS_CONTROL_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vinsertps.xmm",
            results=("dst",),
            operands=("dest", "value"),
            immediates=("control",),
        ),
        schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    _vector_i32_binary_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vpaddd.xmm",
        mnemonic="vpaddd",
        semantic_tag="integer.add.i32x4",
    ),
    _vector_i32_binary_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vpsubd.xmm",
        mnemonic="vpsubd",
        semantic_tag="integer.sub.i32x4",
    ),
    _vector_i32_binary_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vpmulld.xmm",
        mnemonic="vpmulld",
        semantic_tag="integer.mul.i32x4",
    ),
    _vector_f32_binary_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vaddps.xmm",
        mnemonic="vaddps",
        semantic_tag="float.add.f32x4",
    ),
    _scalar_f32_binary_descriptor(
        key="x86.avx2.vaddss.xmm",
        mnemonic="vaddss",
        semantic_tag="float.add.f32",
    ),
    _vector_f32_binary_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vsubps.xmm",
        mnemonic="vsubps",
        semantic_tag="float.sub.f32x4",
    ),
    _scalar_f32_binary_descriptor(
        key="x86.avx2.vsubss.xmm",
        mnemonic="vsubss",
        semantic_tag="float.sub.f32",
    ),
    _vector_f32_binary_descriptor(
        vector_bit_width=128,
        key="x86.avx2.vmulps.xmm",
        mnemonic="vmulps",
        semantic_tag="float.mul.f32x4",
    ),
    _scalar_f32_binary_descriptor(
        key="x86.avx2.vmulss.xmm",
        mnemonic="vmulss",
        semantic_tag="float.mul.f32",
    ),
    Descriptor(
        key="x86.avx2.vfmadd231ss.xmm",
        mnemonic="vfmadd231ss",
        semantic_tag="float.fma.f32",
        operands=(
            _vector_result(128),
            _xmm_operand("acc"),
            _xmm_operand("lhs"),
            _xmm_operand("rhs"),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic="vfmadd231ss.xmm",
            results=("dst",),
            operands=("acc", "lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_VECTOR_FMA_F32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.avx2.vmovdqu32.load.xmm",
        mnemonic="vmovdqu32",
        semantic_tag="memory.load.i32x4",
        operands=(_vector_result(128), _gpr64_resource("base")),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vmovdqu32.load.xmm",
            results=("dst",),
            operands=("base",),
            immediates=("disp32",),
            named_immediates=True,
        ),
        effects=(_load_effect(128),),
        schedule_class=_SCHEDULE_MEMORY_LOAD_XMM,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.avx2.vmovdqu32.load.indexed.xmm",
        mnemonic="vmovdqu32",
        semantic_tag="memory.load.indexed.i32x4",
        operands=(
            _vector_result(128),
            _gpr64_resource("base"),
            _gpr64_resource("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="vmovdqu32.load.indexed.xmm",
            results=("dst",),
            operands=("base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        effects=(_load_effect(128),),
        schedule_class=_SCHEDULE_MEMORY_LOAD_XMM,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.avx2.vmovdqu32.store.xmm",
        mnemonic="vmovdqu32",
        semantic_tag="memory.store.i32x4",
        operands=(_vector_operand(128, "value"), _gpr64_resource("base")),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="vmovdqu32.store.xmm",
            operands=("value", "base"),
            immediates=("disp32",),
            named_immediates=True,
        ),
        effects=(_store_effect(128),),
        schedule_class=_SCHEDULE_MEMORY_STORE_XMM,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.avx2.vmovdqu32.store.indexed.xmm",
        mnemonic="vmovdqu32",
        semantic_tag="memory.store.indexed.i32x4",
        operands=(
            _vector_operand(128, "value"),
            _gpr64_resource("base"),
            _gpr64_resource("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="vmovdqu32.store.indexed.xmm",
            operands=("value", "base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        effects=(_store_effect(128),),
        schedule_class=_SCHEDULE_MEMORY_STORE_XMM,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
)

X86_AVX2_XMM_DESCRIPTORS = tuple(
    _vex_descriptor(descriptor) for descriptor in _X86_AVX2_XMM_DESCRIPTORS
)

X86_AVX2_DESCRIPTORS = (
    *X86_SCALAR_PREFIX_DESCRIPTORS,
    *X86_AVX2_XMM_DESCRIPTORS,
    *X86_SCALAR_SUFFIX_DESCRIPTORS,
)

X86_AVX2_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx2.core",
    target_key="x86",
    feature_key="x86.avx2.v1",
    c_header_path=Path("loom/src/loom/target/arch/x86/descriptors/avx2_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx2_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX2_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/descriptors/avx2_descriptors.h",
    function_name="loom_x86_avx2_core_descriptor_set",
    c_table_prefix="X86Avx2Core",
    c_enum_prefix="X86_AVX2_CORE",
    generator_version=1,
    reg_classes=(
        *X86_SCALAR_DESCRIPTOR_SET.reg_classes,
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=2,
        ),
        RegClass(
            _REG_YMM,
            256,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=2,
        ),
    ),
    resources=(
        *X86_SCALAR_DESCRIPTOR_SET.resources,
        Resource(
            _RESOURCE_VECTOR,
            capacity_per_cycle=2,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=2,
        ),
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
            _SCHEDULE_VECTOR_F32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_FMA_F32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_XMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=_vector_lane_units(128)),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
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
    ),
    enum_domains=X86_SCALAR_DESCRIPTOR_SET.enum_domains,
    descriptors=X86_AVX2_DESCRIPTORS,
)
