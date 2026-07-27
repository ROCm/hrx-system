# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 packed-dot feature descriptor rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loom.target.arch.x86.packed_dot_data import (
    FAMILY_AVX10_2,
    FAMILY_AVX512_BF16,
    FAMILY_AVX512_VNNI,
    FAMILY_AVX_VNNI,
    FAMILY_AVX_VNNI_INT8,
    FAMILY_AVX_VNNI_INT16,
    X86_PACKED_DOT_DESCRIPTORS,
    PackedDotDescriptor,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    IssueUse,
    LatencyKind,
    ModelQuality,
    RegClass,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    SpillSlotSpace,
)

from .common import (
    _REG_XMM,
    _REG_YMM,
    _REG_ZMM,
    _RESOURCE_DOT,
    _SCHEDULE_VECTOR_DOT_XMM,
    _SCHEDULE_VECTOR_DOT_YMM,
    _SCHEDULE_VECTOR_DOT_ZMM,
    _low_subset_operand,
    _packed_dot_descriptor,
    _vector_lane_units,
)

_PACKED_DOT_SOURCE_DIR = Path("loom/src/loom/target/arch/x86")
_PACKED_DOT_PUBLIC_HEADER_DIR = "loom/target/arch/x86"
_PACKED_DOT_VECTOR_WIDTH_TO_REG_CLASS = {
    128: _REG_XMM,
    256: _REG_YMM,
    512: _REG_ZMM,
}
_PACKED_DOT_VECTOR_WIDTH_TO_SCHEDULE = {
    128: _SCHEDULE_VECTOR_DOT_XMM,
    256: _SCHEDULE_VECTOR_DOT_YMM,
    512: _SCHEDULE_VECTOR_DOT_ZMM,
}
_X86_VEX_ADDRESSABLE_REGISTER_COUNT = 16
_X86_VEX_PACKED_DOT_FAMILIES = frozenset(
    (
        FAMILY_AVX_VNNI,
        FAMILY_AVX_VNNI_INT8,
        FAMILY_AVX_VNNI_INT16,
    )
)


def _packed_dot_file_name(stem: str, suffix: str) -> Path:
    return _PACKED_DOT_SOURCE_DIR / f"{stem}_descriptors.{suffix}"


def _packed_dot_public_header(stem: str) -> str:
    return f"{_PACKED_DOT_PUBLIC_HEADER_DIR}/{stem}_descriptors.h"


def _packed_dot_header_guard(stem: str) -> str:
    return f"LOOM_TARGET_ARCH_X86_{stem.upper()}_DESCRIPTORS_H_"


def _packed_dot_reg_classes(
    vector_bit_widths: Sequence[int], *, allocatable_count: int
) -> tuple[RegClass, ...]:
    return tuple(
        RegClass(
            _PACKED_DOT_VECTOR_WIDTH_TO_REG_CLASS[vector_bit_width],
            vector_bit_width,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=allocatable_count,
            alias_set_id=2,
        )
        for vector_bit_width in vector_bit_widths
    )


def _packed_dot_resources() -> tuple[Resource, ...]:
    return (
        Resource(
            _RESOURCE_DOT,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
    )


def _packed_dot_schedule_classes(
    vector_bit_widths: Sequence[int],
) -> tuple[ScheduleClass, ...]:
    return tuple(
        ScheduleClass(
            _PACKED_DOT_VECTOR_WIDTH_TO_SCHEDULE[vector_bit_width],
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(
                    _RESOURCE_DOT,
                    cycles=1,
                    units=_vector_lane_units(vector_bit_width),
                ),
            ),
            model_quality=ModelQuality.ESTIMATED,
        )
        for vector_bit_width in vector_bit_widths
    )


def _packed_dot_descriptor_data(
    families: Sequence[str], vector_bit_widths: Sequence[int]
) -> tuple[PackedDotDescriptor, ...]:
    family_set = frozenset(families)
    vector_bit_width_set = frozenset(vector_bit_widths)
    descriptor_data = tuple(
        descriptor
        for descriptor in X86_PACKED_DOT_DESCRIPTORS
        if descriptor.family in family_set
        and descriptor.vector_bit_width in vector_bit_width_set
    )
    if not descriptor_data:
        family_names = ", ".join(sorted(family_set))
        raise ValueError(f"x86 packed-dot overlay selected no rows for {family_names}")
    return descriptor_data


def _packed_dot_target_descriptor(
    descriptor_data: PackedDotDescriptor,
    *,
    qualify_asm_mnemonic: bool = False,
) -> Descriptor:
    descriptor = _packed_dot_descriptor(
        descriptor_data,
        qualify_asm_mnemonic=qualify_asm_mnemonic,
    )
    if descriptor_data.family not in _X86_VEX_PACKED_DOT_FAMILIES:
        return descriptor
    return replace(
        descriptor,
        operands=tuple(
            _low_subset_operand(operand, _X86_VEX_ADDRESSABLE_REGISTER_COUNT)
            for operand in descriptor.operands
        ),
    )


def _descriptor_set(
    *,
    key: str,
    feature_key: str,
    stem: str,
    function_name: str,
    c_table_prefix: str,
    c_enum_prefix: str,
    descriptor_data: Sequence[PackedDotDescriptor],
    vector_bit_widths: Sequence[int],
    allocatable_count: int,
    qualify_asm_mnemonics: bool = False,
) -> DescriptorSet:
    return DescriptorSet(
        key=key,
        target_key="x86",
        feature_key=feature_key,
        c_header_path=_packed_dot_file_name(stem, "h"),
        c_source_path=_packed_dot_file_name(stem, "c"),
        header_guard=_packed_dot_header_guard(stem),
        public_header=_packed_dot_public_header(stem),
        function_name=function_name,
        c_table_prefix=c_table_prefix,
        c_enum_prefix=c_enum_prefix,
        generator_version=1,
        reg_classes=_packed_dot_reg_classes(
            vector_bit_widths, allocatable_count=allocatable_count
        ),
        resources=_packed_dot_resources(),
        schedule_classes=_packed_dot_schedule_classes(vector_bit_widths),
        descriptors=tuple(
            _packed_dot_target_descriptor(
                descriptor,
                qualify_asm_mnemonic=qualify_asm_mnemonics,
            )
            for descriptor in descriptor_data
        ),
    )


def _overlay_descriptor_set(
    *,
    key: str,
    feature_key: str,
    stem: str,
    function_name: str,
    c_table_prefix: str,
    c_enum_prefix: str,
    family: str,
    vector_bit_widths: Sequence[int],
    allocatable_count: int,
) -> DescriptorSet:
    return _descriptor_set(
        key=key,
        feature_key=feature_key,
        stem=stem,
        function_name=function_name,
        c_table_prefix=c_table_prefix,
        c_enum_prefix=c_enum_prefix,
        descriptor_data=_packed_dot_descriptor_data((family,), vector_bit_widths),
        vector_bit_widths=vector_bit_widths,
        allocatable_count=allocatable_count,
    )


X86_AVX512_VNNI_DESCRIPTOR_SET = _overlay_descriptor_set(
    key="x86.avx512_vnni.core",
    feature_key="x86.avx512_vnni.v1",
    stem="avx512_vnni",
    function_name="loom_x86_avx512_vnni_core_descriptor_set",
    c_table_prefix="X86Avx512VnniCore",
    c_enum_prefix="X86_AVX512_VNNI_CORE",
    family=FAMILY_AVX512_VNNI,
    vector_bit_widths=(128, 256, 512),
    allocatable_count=32,
)

X86_AVX512_BF16_DESCRIPTOR_SET = _overlay_descriptor_set(
    key="x86.avx512_bf16.core",
    feature_key="x86.avx512_bf16.v1",
    stem="avx512_bf16",
    function_name="loom_x86_avx512_bf16_core_descriptor_set",
    c_table_prefix="X86Avx512Bf16Core",
    c_enum_prefix="X86_AVX512_BF16_CORE",
    family=FAMILY_AVX512_BF16,
    vector_bit_widths=(128, 256, 512),
    allocatable_count=32,
)

X86_AVX_VNNI_DESCRIPTOR_SET = _overlay_descriptor_set(
    key="x86.avx_vnni.core",
    feature_key="x86.avx_vnni.v1",
    stem="avx_vnni",
    function_name="loom_x86_avx_vnni_core_descriptor_set",
    c_table_prefix="X86AvxVnniCore",
    c_enum_prefix="X86_AVX_VNNI_CORE",
    family=FAMILY_AVX_VNNI,
    vector_bit_widths=(128, 256),
    allocatable_count=16,
)

X86_AVX_VNNI_INT8_DESCRIPTOR_SET = _overlay_descriptor_set(
    key="x86.avx_vnni_int8.core",
    feature_key="x86.avx_vnni_int8.v1",
    stem="avx_vnni_int8",
    function_name="loom_x86_avx_vnni_int8_core_descriptor_set",
    c_table_prefix="X86AvxVnniInt8Core",
    c_enum_prefix="X86_AVX_VNNI_INT8_CORE",
    family=FAMILY_AVX_VNNI_INT8,
    vector_bit_widths=(128, 256),
    allocatable_count=16,
)

X86_AVX_VNNI_INT16_DESCRIPTOR_SET = _overlay_descriptor_set(
    key="x86.avx_vnni_int16.core",
    feature_key="x86.avx_vnni_int16.v1",
    stem="avx_vnni_int16",
    function_name="loom_x86_avx_vnni_int16_core_descriptor_set",
    c_table_prefix="X86AvxVnniInt16Core",
    c_enum_prefix="X86_AVX_VNNI_INT16_CORE",
    family=FAMILY_AVX_VNNI_INT16,
    vector_bit_widths=(128, 256),
    allocatable_count=16,
)

X86_AVX10_2_DESCRIPTOR_SET = _overlay_descriptor_set(
    key="x86.avx10_2.core",
    feature_key="x86.avx10_2.v1",
    stem="avx10_2",
    function_name="loom_x86_avx10_2_core_descriptor_set",
    c_table_prefix="X86Avx102Core",
    c_enum_prefix="X86_AVX10_2_CORE",
    family=FAMILY_AVX10_2,
    vector_bit_widths=(128, 256, 512),
    allocatable_count=32,
)

X86_PACKED_DOT_FEATURE_DESCRIPTOR_SETS = (
    X86_AVX512_VNNI_DESCRIPTOR_SET,
    X86_AVX512_BF16_DESCRIPTOR_SET,
    X86_AVX_VNNI_DESCRIPTOR_SET,
    X86_AVX_VNNI_INT8_DESCRIPTOR_SET,
    X86_AVX_VNNI_INT16_DESCRIPTOR_SET,
    X86_AVX10_2_DESCRIPTOR_SET,
)

X86_PACKED_DOT_DESCRIPTOR_SET = _descriptor_set(
    key="x86.packed_dot.core",
    feature_key="x86.packed_dot.v1",
    stem="packed_dot",
    function_name="loom_x86_packed_dot_core_descriptor_set",
    c_table_prefix="X86PackedDotCore",
    c_enum_prefix="X86_PACKED_DOT_CORE",
    descriptor_data=X86_PACKED_DOT_DESCRIPTORS,
    vector_bit_widths=(128, 256, 512),
    allocatable_count=32,
    qualify_asm_mnemonics=True,
)
