# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX512-BF16 conversion and packed-dot descriptor rows."""

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.feature_bits import (
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
)
from loom.target.arch.x86.vector_encoding import VectorEncodingPrefix
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    Effect,
    EffectFlag,
    EffectKind,
    MemorySpace,
)

from .avx512 import X86_AVX512_CORE_DESCRIPTOR_SET
from .common import (
    _ADDRESS_SCALE_IMMEDIATE,
    _DISP32_IMMEDIATE,
    _SCHEDULE_MEMORY_LOAD_XMM,
    _SCHEDULE_MEMORY_LOAD_YMM,
    _SCHEDULE_MEMORY_LOAD_ZMM,
    _asm,
    _descriptor_support_tables,
    _evex_writemask_operand,
    _gpr64_resource,
    _merge_named_items,
    _vector_f32_schedule_class,
    _vector_operand,
    _vector_result,
)
from .packed_dot import X86_AVX512_BF16_DOT_DESCRIPTOR_SET

_VECTOR_BIT_WIDTHS = (128, 256, 512)
_REGISTER_SUFFIXES = {64: "xmm", 128: "xmm", 256: "ymm", 512: "zmm"}
_MEMORY_SCHEDULES = {
    128: _SCHEDULE_MEMORY_LOAD_XMM,
    256: _SCHEDULE_MEMORY_LOAD_YMM,
    512: _SCHEDULE_MEMORY_LOAD_ZMM,
}


def _required_feature_bits(vector_bit_width: int) -> int:
    feature_bits = FEATURE_AVX512_BF16
    if vector_bit_width < 512:
        feature_bits |= FEATURE_AVX512_VL
    return feature_bits


def _load_effect(memory_source: native.EvexMemorySource, width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=(
            32 if memory_source is native.EvexMemorySource.BROADCAST_32 else width_bits
        ),
    )


def _conversion_descriptor(
    *,
    source_count: int,
    vector_bit_width: int,
    masking: native.EvexMasking,
    memory_source: native.EvexMemorySource,
    indexed: bool,
) -> Descriptor:
    mnemonic = "vcvtneps2bf16" if source_count == 1 else "vcvtne2ps2bf16"
    result_bit_width = vector_bit_width // 2 if source_count == 1 else vector_bit_width
    result_suffix = _REGISTER_SUFFIXES[result_bit_width]
    source_suffix = _REGISTER_SUFFIXES[vector_bit_width]

    name_parts = [mnemonic]
    if memory_source is not native.EvexMemorySource.NONE:
        name_parts.append(
            "broadcast"
            if memory_source is native.EvexMemorySource.BROADCAST_32
            else "load"
        )
        if indexed:
            name_parts.append("indexed")
    if masking is not native.EvexMasking.NONE:
        name_parts.append(masking.value)
    name_parts.append(result_suffix)
    if source_count == 1:
        name_parts.append(source_suffix)
    form_name = ".".join(name_parts)

    operands = [_vector_result(result_bit_width)]
    asm_operands: list[str] = []
    if masking is not native.EvexMasking.NONE:
        operands.append(_evex_writemask_operand())
        asm_operands.append("mask")
    if source_count == 2:
        operands.append(_vector_operand(vector_bit_width, "high"))
        asm_operands.append("high")
    if memory_source is native.EvexMemorySource.NONE:
        source_name = "input" if source_count == 1 else "low"
        operands.append(_vector_operand(vector_bit_width, source_name))
        asm_operands.append(source_name)
        immediates = ()
        effects = ()
    else:
        operands.append(_gpr64_resource("base"))
        asm_operands.append("base")
        if indexed:
            operands.append(_gpr64_resource("index"))
            asm_operands.append("index")
            immediates = (_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE)
        else:
            immediates = (_DISP32_IMMEDIATE,)
        effects = (_load_effect(memory_source, vector_bit_width),)
    if masking is native.EvexMasking.MERGE:
        operands.append(_vector_operand(result_bit_width, "passthrough"))
        asm_operands.append("passthrough")

    constraints = ()
    if masking is native.EvexMasking.MERGE:
        passthrough_index = len(operands) - 1
        constraints = (
            Constraint(ConstraintKind.TIED, 0, passthrough_index),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, passthrough_index),
        )
    descriptor = Descriptor(
        key=f"x86.avx512_bf16.{form_name}",
        mnemonic=mnemonic,
        semantic_tag=(
            f"float.truncate.concat.f32.bf16x{vector_bit_width // 16}"
            if source_count == 2
            else f"float.truncate.f32.bf16x{vector_bit_width // 32}"
        ),
        operands=tuple(operands),
        immediates=immediates,
        asm_forms=_asm(
            mnemonic=f"avx512_bf16.{form_name}",
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=tuple(immediate.field_name for immediate in immediates),
            named_immediates=True,
        ),
        effects=effects,
        constraints=constraints,
        schedule_class=(
            _MEMORY_SCHEDULES[vector_bit_width]
            if memory_source is not native.EvexMemorySource.NONE
            else _vector_f32_schedule_class(vector_bit_width)
        ),
        feature_mask_words=(_required_feature_bits(vector_bit_width),),
        flags=(
            (DescriptorFlag.SIDE_EFFECTING,)
            if memory_source is not native.EvexMemorySource.NONE
            else (DescriptorFlag.DEAD_REMOVABLE,)
        ),
    )
    instruction = native.avx512_bf16_conversion_instruction(
        source_count=source_count,
        masking=masking,
        memory_source=memory_source,
        indexed=indexed,
    )
    return instruction.bind(
        descriptor,
        VectorEncodingPrefix.EVEX,
        vector_bit_width=vector_bit_width,
    )


def _conversion_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _conversion_descriptor(
            source_count=source_count,
            vector_bit_width=vector_bit_width,
            masking=masking,
            memory_source=memory_source,
            indexed=indexed,
        )
        for source_count in (1, 2)
        for vector_bit_width in _VECTOR_BIT_WIDTHS
        for masking in native.EvexMasking
        for memory_source in native.EvexMemorySource
        for indexed in (
            (False, True)
            if memory_source is not native.EvexMemorySource.NONE
            else (False,)
        )
    )


_CONVERSION_DESCRIPTORS = _conversion_descriptors()
_BF16_DESCRIPTORS = (
    *X86_AVX512_BF16_DOT_DESCRIPTOR_SET.descriptors,
    *_CONVERSION_DESCRIPTORS,
)
_BF16_SUPPORT_SOURCE = replace(
    X86_AVX512_CORE_DESCRIPTOR_SET,
    resources=_merge_named_items(
        (
            X86_AVX512_CORE_DESCRIPTOR_SET.resources,
            X86_AVX512_BF16_DOT_DESCRIPTOR_SET.resources,
        )
    ),
    schedule_classes=_merge_named_items(
        (
            X86_AVX512_CORE_DESCRIPTOR_SET.schedule_classes,
            X86_AVX512_BF16_DOT_DESCRIPTOR_SET.schedule_classes,
        )
    ),
)
(
    _BF16_REG_CLASSES,
    _BF16_RESOURCES,
    _BF16_SCHEDULE_CLASSES,
) = _descriptor_support_tables(_BF16_DESCRIPTORS, _BF16_SUPPORT_SOURCE)

X86_AVX512_BF16_DESCRIPTOR_SET = replace(
    X86_AVX512_BF16_DOT_DESCRIPTOR_SET,
    physical_registers=X86_AVX512_CORE_DESCRIPTOR_SET.physical_registers,
    reg_classes=_BF16_REG_CLASSES,
    resources=_BF16_RESOURCES,
    schedule_classes=_BF16_SCHEDULE_CLASSES,
    descriptors=_BF16_DESCRIPTORS,
)
