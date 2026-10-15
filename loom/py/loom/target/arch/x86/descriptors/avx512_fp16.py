# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX512-FP16 arithmetic and conversion descriptor rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.feature_bits import (
    FEATURE_AVX512_FP16,
    FEATURE_AVX512_VL,
)
from loom.target.arch.x86.vector_encoding import (
    VectorEncodingPrefix,
    VectorMachineInstruction,
)
from loom.target.arch.x86.vector_families import (
    AVX512_FP16_FLOAT_BINARY_FAMILIES,
    AVX512_FP16_FLOAT_COMPARE_MNEMONIC,
    AVX512_FP16_FLOAT_EXTREMA_MNEMONICS,
    AVX512_FP16_FLOAT_FMA_MNEMONIC,
    AVX512_FP16_FLOAT_ROOT_FAMILIES,
    AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX512_FP16_SCALAR_FLOAT_COMPARE_MNEMONIC,
    AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS,
    AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC,
    AVX512_FP16_SCALAR_FLOAT_ROOT_FAMILIES,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
)

from .avx512 import X86_AVX512_CORE_DESCRIPTOR_SET
from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _asm,
    _descriptor_support_tables,
    _scalar_float_binary_descriptor,
    _scalar_float_unary_descriptor,
    _vector_f32_binary_descriptor,
    _vector_f32_schedule_class,
    _vector_float_unary_descriptor,
    _vector_fma_descriptor,
    _vector_mask_compare_descriptor,
    _vector_operand,
    _vector_result,
    _vector_splat_descriptor,
    _xmm_operand,
    _xmm_result,
)

_REGISTER_SUFFIXES = {64: "xmm", 128: "xmm", 256: "ymm", 512: "zmm"}
_VECTOR_BIT_WIDTHS = (128, 256, 512)


def _required_feature_bits(vector_bit_width: int, *, scalar: bool = False) -> int:
    bits = FEATURE_AVX512_FP16
    if not scalar and vector_bit_width < 512:
        bits |= FEATURE_AVX512_VL
    return bits


def _bind(
    descriptor: Descriptor,
    instruction: VectorMachineInstruction,
    *,
    vector_bit_width: int,
    scalar: bool = False,
) -> Descriptor:
    descriptor = replace(
        descriptor,
        feature_mask_words=(_required_feature_bits(vector_bit_width, scalar=scalar),),
    )
    return instruction.bind(descriptor, VectorEncodingPrefix.EVEX)


def _conversion_descriptor(
    mnemonic: str,
    result_bit_width: int,
    input_bit_width: int,
) -> Descriptor:
    result_suffix = _REGISTER_SUFFIXES[result_bit_width]
    input_suffix = _REGISTER_SUFFIXES[input_bit_width]
    return Descriptor(
        key=(f"x86.avx512_fp16.{mnemonic}.{result_suffix}.{input_suffix}"),
        mnemonic=mnemonic,
        semantic_tag=(
            f"float.convert.{mnemonic}.v{input_bit_width}.v{result_bit_width}"
        ),
        operands=(
            _vector_result(result_bit_width),
            _vector_operand(input_bit_width, "input"),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.{result_suffix}.{input_suffix}",
            results=("dst",),
            operands=("input",),
        ),
        schedule_class=_vector_f32_schedule_class(result_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _scalar_conversion_descriptor(mnemonic: str, semantic_tag: str) -> Descriptor:
    return Descriptor(
        key=f"x86.avx512_fp16.{mnemonic}.xmm",
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _xmm_result(),
            _xmm_operand("passthrough"),
            _xmm_operand("input"),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.xmm",
            results=("dst",),
            operands=("passthrough", "input"),
        ),
        schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _descriptors() -> tuple[Descriptor, ...]:
    packed_binary = tuple(
        _bind(
            _vector_f32_binary_descriptor(
                vector_bit_width=vector_bit_width,
                key=(
                    f"x86.avx512_fp16.{family.mnemonic}."
                    f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                ),
                mnemonic=family.mnemonic,
                semantic_tag=(f"{family.semantic}.f16x{vector_bit_width // 16}"),
            ),
            instruction,
            vector_bit_width=vector_bit_width,
        )
        for family, instruction in zip(
            AVX512_FP16_FLOAT_BINARY_FAMILIES,
            native.AVX512_FP16_FLOAT_BINARY,
            strict=True,
        )
        for vector_bit_width in _VECTOR_BIT_WIDTHS
    )
    scalar_binary = tuple(
        _bind(
            _scalar_float_binary_descriptor(
                key=f"x86.avx512_fp16.{family.mnemonic}.xmm",
                mnemonic=family.mnemonic,
                semantic_tag=f"{family.semantic}.f16",
            ),
            instruction,
            vector_bit_width=128,
            scalar=True,
        )
        for family, instruction in zip(
            AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES,
            native.AVX512_FP16_SCALAR_FLOAT_BINARY,
            strict=True,
        )
    )
    packed_root = tuple(
        _bind(
            _vector_float_unary_descriptor(
                vector_bit_width=vector_bit_width,
                key=(
                    f"x86.avx512_fp16.{family.mnemonic}."
                    f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                ),
                mnemonic=family.mnemonic,
                semantic_tag=(f"{family.semantic}.f16x{vector_bit_width // 16}"),
            ),
            instruction,
            vector_bit_width=vector_bit_width,
        )
        for family, instruction in zip(
            AVX512_FP16_FLOAT_ROOT_FAMILIES,
            native.AVX512_FP16_FLOAT_ROOT,
            strict=True,
        )
        for vector_bit_width in _VECTOR_BIT_WIDTHS
    )
    scalar_root = tuple(
        _bind(
            _scalar_float_unary_descriptor(
                key=f"x86.avx512_fp16.{family.mnemonic}.xmm",
                mnemonic=family.mnemonic,
                semantic_tag=f"{family.semantic}.f16",
            ),
            instruction,
            vector_bit_width=128,
            scalar=True,
        )
        for family, instruction in zip(
            AVX512_FP16_SCALAR_FLOAT_ROOT_FAMILIES,
            native.AVX512_FP16_SCALAR_FLOAT_ROOT,
            strict=True,
        )
    )
    packed_fma = tuple(
        _bind(
            _vector_fma_descriptor(
                vector_bit_width=vector_bit_width,
                key=(
                    f"x86.avx512_fp16.{AVX512_FP16_FLOAT_FMA_MNEMONIC}."
                    f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                ),
                mnemonic=AVX512_FP16_FLOAT_FMA_MNEMONIC,
                semantic_tag=f"float.fma.f16x{vector_bit_width // 16}",
            ),
            native.AVX512_FP16_FLOAT_FMA,
            vector_bit_width=vector_bit_width,
        )
        for vector_bit_width in _VECTOR_BIT_WIDTHS
    )
    scalar_fma = _bind(
        _vector_fma_descriptor(
            vector_bit_width=128,
            key=(f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC}.xmm"),
            mnemonic=AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC,
            semantic_tag="float.fma.f16",
        ),
        native.AVX512_FP16_SCALAR_FLOAT_FMA,
        vector_bit_width=128,
        scalar=True,
    )
    packed_compare = tuple(
        _bind(
            _vector_mask_compare_descriptor(
                vector_bit_width=vector_bit_width,
                key=(
                    "x86.avx512_fp16."
                    f"{AVX512_FP16_FLOAT_COMPARE_MNEMONIC}."
                    f"{_REGISTER_SUFFIXES[vector_bit_width]}"
                ),
                mnemonic=AVX512_FP16_FLOAT_COMPARE_MNEMONIC,
                semantic_tag=f"float.cmp.f16x{vector_bit_width // 16}",
            ),
            native.AVX512_FP16_FLOAT_COMPARE,
            vector_bit_width=vector_bit_width,
        )
        for vector_bit_width in _VECTOR_BIT_WIDTHS
    )
    scalar_compare = _bind(
        _vector_mask_compare_descriptor(
            vector_bit_width=128,
            key=(f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_COMPARE_MNEMONIC}.xmm"),
            mnemonic=AVX512_FP16_SCALAR_FLOAT_COMPARE_MNEMONIC,
            semantic_tag="float.cmp.f16",
        ),
        native.AVX512_FP16_SCALAR_FLOAT_COMPARE,
        vector_bit_width=128,
        scalar=True,
    )
    packed_extrema = tuple(
        _bind(
            _vector_f32_binary_descriptor(
                vector_bit_width=vector_bit_width,
                key=(
                    f"x86.avx512_fp16.{mnemonic}.{_REGISTER_SUFFIXES[vector_bit_width]}"
                ),
                mnemonic=mnemonic,
                semantic_tag=f"float.fast_extrema.f16x{vector_bit_width // 16}",
            ),
            instruction,
            vector_bit_width=vector_bit_width,
        )
        for mnemonic, instruction in zip(
            (
                AVX512_FP16_FLOAT_EXTREMA_MNEMONICS["minimumf"],
                AVX512_FP16_FLOAT_EXTREMA_MNEMONICS["maximumf"],
            ),
            native.AVX512_FP16_FLOAT_EXTREMA,
            strict=True,
        )
        for vector_bit_width in _VECTOR_BIT_WIDTHS
    )
    scalar_extrema = tuple(
        _bind(
            _scalar_float_binary_descriptor(
                key=f"x86.avx512_fp16.{mnemonic}.xmm",
                mnemonic=mnemonic,
                semantic_tag="float.fast_extrema.f16",
            ),
            instruction,
            vector_bit_width=128,
            scalar=True,
        )
        for mnemonic, instruction in zip(
            (
                AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS["minimumf"],
                AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS["maximumf"],
            ),
            native.AVX512_FP16_SCALAR_FLOAT_EXTREMA,
            strict=True,
        )
    )
    packed_conversions = (
        *(
            _bind(
                _conversion_descriptor("vcvtph2psx", result_width, input_width),
                native.VCVTPH2PSX,
                vector_bit_width=max(result_width, input_width),
            )
            for result_width, input_width in ((128, 64), (256, 128), (512, 256))
        ),
        *(
            _bind(
                _conversion_descriptor("vcvtps2phx", result_width, input_width),
                native.VCVTPS2PHX,
                vector_bit_width=max(result_width, input_width),
            )
            for result_width, input_width in ((64, 128), (128, 256), (256, 512))
        ),
    )
    scalar_conversions = (
        _bind(
            _scalar_conversion_descriptor("vcvtsh2ss", "float.extend.f16.f32"),
            native.VCVTSH2SS,
            vector_bit_width=128,
            scalar=True,
        ),
        _bind(
            _scalar_conversion_descriptor("vcvtss2sh", "float.truncate.f32.f16"),
            native.VCVTSS2SH,
            vector_bit_width=128,
            scalar=True,
        ),
    )
    broadcast = _bind(
        _vector_splat_descriptor(
            vector_bit_width=512,
            key="x86.avx512_fp16.vpbroadcastw.zmm.xmm",
            mnemonic="vpbroadcastw",
            semantic_tag="bits.splat.16.v512",
            operand=_xmm_operand("value"),
            schedule_class=_vector_f32_schedule_class(512),
            asm_mnemonic="vpbroadcastw.zmm.xmm",
        ),
        native.VPBROADCASTW_VECTOR_UNARY,
        vector_bit_width=512,
    )
    return (
        *packed_binary,
        *scalar_binary,
        *packed_root,
        *scalar_root,
        *packed_fma,
        scalar_fma,
        *packed_compare,
        scalar_compare,
        *packed_extrema,
        *scalar_extrema,
        *packed_conversions,
        *scalar_conversions,
        broadcast,
    )


_FP16_DESCRIPTORS = _descriptors()
(
    _FP16_REG_CLASSES,
    _FP16_RESOURCES,
    _FP16_SCHEDULE_CLASSES,
) = _descriptor_support_tables(_FP16_DESCRIPTORS, X86_AVX512_CORE_DESCRIPTOR_SET)


X86_AVX512_FP16_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx512_fp16.core",
    target_key="x86",
    feature_key="x86.avx512_fp16.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/x86/descriptors/avx512_fp16_descriptors.h"
    ),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx512_fp16_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX512_FP16_DESCRIPTORS_H_",
    public_header=("loom/target/arch/x86/descriptors/avx512_fp16_descriptors.h"),
    function_name="loom_x86_avx512_fp16_core_descriptor_set",
    c_table_prefix="X86Avx512Fp16Core",
    c_enum_prefix="X86_AVX512_FP16_CORE",
    generator_version=1,
    supports_native_scheduling=True,
    reg_classes=_FP16_REG_CLASSES,
    resources=_FP16_RESOURCES,
    schedule_classes=_FP16_SCHEDULE_CLASSES,
    descriptors=_FP16_DESCRIPTORS,
)
