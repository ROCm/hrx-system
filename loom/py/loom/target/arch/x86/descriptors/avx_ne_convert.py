# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX-NE-CONVERT packed floating-point conversion descriptors."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.feature_bits import FEATURE_AVX_NE_CONVERT
from loom.target.arch.x86.vector_encoding import (
    VectorEncodingPrefix,
    VectorMachineInstruction,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    MemorySpace,
)

from .avx2 import X86_AVX2_DESCRIPTOR_SET
from .common import (
    _ADDRESS_SCALE_IMMEDIATE,
    _DISP32_IMMEDIATE,
    _SCHEDULE_MEMORY_LOAD_XMM,
    _SCHEDULE_MEMORY_LOAD_YMM,
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _asm,
    _descriptor_support_tables,
    _gpr64_resource,
    _vector_operand,
    _vector_result,
    _vex_descriptor,
)

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm"}
_MEMORY_SCHEDULES = {
    128: _SCHEDULE_MEMORY_LOAD_XMM,
    256: _SCHEDULE_MEMORY_LOAD_YMM,
}
_CONVERSION_SCHEDULES = {
    128: _SCHEDULE_VECTOR_F32_XMM,
    256: _SCHEDULE_VECTOR_F32_YMM,
}

# Mnemonic, source format, lane selection, and fixed memory footprint. A zero
# footprint selects the result width: even/odd conversions consume the full
# narrow source vector while producing half as many f32 lanes.
_MEMORY_ROWS = (
    ("vbcstnebf162ps", "bf16", "broadcast", 16),
    ("vbcstnesh2ps", "f16", "broadcast", 16),
    ("vcvtneebf162ps", "bf16", "even", 0),
    ("vcvtneeph2ps", "f16", "even", 0),
    ("vcvtneobf162ps", "bf16", "odd", 0),
    ("vcvtneoph2ps", "f16", "odd", 0),
)


def _load_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _memory_descriptors(
    row: tuple[str, str, str, int],
    instructions: tuple[VectorMachineInstruction, VectorMachineInstruction],
) -> tuple[Descriptor, ...]:
    mnemonic, source_format, selection, fixed_memory_width = row
    descriptors = []
    for result_bit_width in (128, 256):
        register_suffix = _REGISTER_SUFFIXES[result_bit_width]
        memory_width = fixed_memory_width or result_bit_width
        for address_suffix, address_fields, immediates, instruction in zip(
            ("", ".indexed"),
            (("base",), ("base", "index")),
            (
                (_DISP32_IMMEDIATE,),
                (_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
            ),
            instructions,
            strict=True,
        ):
            operation = f"load{address_suffix}"
            descriptors.append(
                _vex_descriptor(
                    Descriptor(
                        key=(
                            f"x86.avx_ne_convert.{mnemonic}.{operation}."
                            f"{register_suffix}"
                        ),
                        mnemonic=mnemonic,
                        semantic_tag=(
                            f"float.load.{selection}.{source_format}."
                            f"f32x{result_bit_width // 32}"
                        ),
                        operands=(
                            _vector_result(result_bit_width),
                            *(
                                _gpr64_resource(field_name)
                                for field_name in address_fields
                            ),
                        ),
                        immediates=immediates,
                        asm_forms=_asm(
                            mnemonic=f"{mnemonic}.{operation}.{register_suffix}",
                            results=("dst",),
                            operands=address_fields,
                            immediates=tuple(
                                immediate.field_name for immediate in immediates
                            ),
                            named_immediates=True,
                        ),
                        effects=(_load_effect(memory_width),),
                        schedule_class=_MEMORY_SCHEDULES[result_bit_width],
                        feature_mask_words=(FEATURE_AVX_NE_CONVERT,),
                        flags=(DescriptorFlag.SIDE_EFFECTING,),
                    ),
                    instruction,
                )
            )
    return tuple(descriptors)


def _narrowing_descriptor(source_bit_width: int) -> Descriptor:
    source_suffix = _REGISTER_SUFFIXES[source_bit_width]
    result_bit_width = source_bit_width // 2
    descriptor = Descriptor(
        key=f"x86.avx_ne_convert.vcvtneps2bf16.xmm.{source_suffix}",
        mnemonic="vcvtneps2bf16",
        semantic_tag=f"float.truncate.f32.bf16x{source_bit_width // 32}",
        operands=(
            _vector_result(result_bit_width),
            _vector_operand(source_bit_width, "input"),
        ),
        asm_forms=_asm(
            mnemonic=f"vcvtneps2bf16.xmm.{source_suffix}",
            results=("dst",),
            operands=("input",),
        ),
        schedule_class=_CONVERSION_SCHEDULES[source_bit_width],
        feature_mask_words=(FEATURE_AVX_NE_CONVERT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return native.VCVTNEPS2BF16.bind(descriptor, VectorEncodingPrefix.VEX)


_AVX_NE_CONVERT_DESCRIPTORS = (
    *(
        descriptor
        for row, instructions in zip(
            _MEMORY_ROWS, native.AVX_NE_CONVERT_MEMORY, strict=True
        )
        for descriptor in _memory_descriptors(row, instructions)
    ),
    *(_narrowing_descriptor(source_bit_width) for source_bit_width in (128, 256)),
)

(
    _AVX_NE_CONVERT_REG_CLASSES,
    _AVX_NE_CONVERT_RESOURCES,
    _AVX_NE_CONVERT_SCHEDULE_CLASSES,
) = _descriptor_support_tables(_AVX_NE_CONVERT_DESCRIPTORS, X86_AVX2_DESCRIPTOR_SET)

X86_AVX_NE_CONVERT_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx_ne_convert.core",
    target_key="x86",
    feature_key="x86.avx_ne_convert.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/x86/descriptors/avx_ne_convert_descriptors.h"
    ),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx_ne_convert_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX_NE_CONVERT_DESCRIPTORS_H_",
    public_header=("loom/target/arch/x86/descriptors/avx_ne_convert_descriptors.h"),
    function_name="loom_x86_avx_ne_convert_core_descriptor_set",
    c_table_prefix="X86AvxNeConvertCore",
    c_enum_prefix="X86_AVX_NE_CONVERT_CORE",
    generator_version=1,
    supports_native_scheduling=True,
    reg_classes=_AVX_NE_CONVERT_REG_CLASSES,
    resources=_AVX_NE_CONVERT_RESOURCES,
    schedule_classes=_AVX_NE_CONVERT_SCHEDULE_CLASSES,
    descriptors=_AVX_NE_CONVERT_DESCRIPTORS,
)
