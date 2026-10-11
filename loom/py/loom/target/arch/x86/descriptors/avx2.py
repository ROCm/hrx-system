# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX2 descriptor rows and view metadata."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.vector_encoding import (
    VectorEncodingPrefix,
    VectorMachineInstruction,
)
from loom.target.arch.x86.vector_families import (
    AVX2_BITWISE_FAMILIES,
    AVX2_FLOAT_BINARY_FAMILIES,
    AVX2_FLOAT_COMPARE_MNEMONICS,
    AVX2_FLOAT_FMA_MNEMONICS,
    AVX2_INTEGER_BINARY_FAMILIES,
    AVX2_INTEGER_COMPARE_MNEMONICS,
    AVX2_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX2_SCALAR_FLOAT_EXTREMA_MNEMONICS,
    AVX2_SCALAR_FLOAT_FMA_MNEMONICS,
    AVX2_UNIFORM_SHIFT_FAMILIES,
    AVX2_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    FLOAT_EXTREMA_MNEMONICS,
    INTEGER_ELEMENTS,
    X86_LANE_FAMILIES,
    VectorBinaryFamily,
    VectorLaneFamily,
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

from .common import (
    _DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
    _INSERTPS_CONTROL_IMMEDIATE,
    _READ_ONLY_DATA_IMMEDIATE,
    _REG_XMM,
    _REG_YMM,
    _RESOURCE_ADDRESS,
    _RESOURCE_LOAD,
    _RESOURCE_STORE,
    _RESOURCE_VECTOR,
    _SCHEDULE_MEMORY_LOAD_XMM,
    _SCHEDULE_MEMORY_LOAD_YMM,
    _SCHEDULE_MEMORY_STORE_XMM,
    _SCHEDULE_MEMORY_STORE_YMM,
    _SCHEDULE_VECTOR_COMPARE_XMM,
    _SCHEDULE_VECTOR_COMPARE_YMM,
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _SCHEDULE_VECTOR_FMA_F32_XMM,
    _SCHEDULE_VECTOR_FMA_F32_YMM,
    _SCHEDULE_VECTOR_I32_XMM,
    _SCHEDULE_VECTOR_I32_YMM,
    _SHUFFLE_2X1_CONTROL_IMMEDIATE,
    _SHUFFLE_4X2_CONTROL_IMMEDIATE,
    _TWO_LANE_IMMEDIATE,
    _asm,
    _gpr32_operand,
    _gpr32_result,
    _gpr64_operand,
    _gpr64_result,
    _scalar_float_binary_descriptor,
    _vector_lane_units,
    _vector_operand,
    _vector_result,
    _vector_splat_descriptor,
    _vector_zero_descriptor,
    _vex_descriptor,
    _xmm_operand,
    _xmm_result,
)
from .memory import memory_descriptors
from .scalar import (
    X86_SCALAR_DESCRIPTOR_SET,
    X86_SCALAR_PREFIX_DESCRIPTORS,
    X86_SCALAR_SUFFIX_DESCRIPTORS,
)

_VECTOR_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm"}
_VECTOR_INTEGER_SCHEDULE_CLASSES = {
    128: _SCHEDULE_VECTOR_I32_XMM,
    256: _SCHEDULE_VECTOR_I32_YMM,
}
_VECTOR_FLOAT_SCHEDULE_CLASSES = {
    128: _SCHEDULE_VECTOR_F32_XMM,
    256: _SCHEDULE_VECTOR_F32_YMM,
}
_VECTOR_FMA_SCHEDULE_CLASSES = {
    128: _SCHEDULE_VECTOR_FMA_F32_XMM,
    256: _SCHEDULE_VECTOR_FMA_F32_YMM,
}
_VECTOR_COMPARE_SCHEDULE_CLASSES = {
    128: _SCHEDULE_VECTOR_COMPARE_XMM,
    256: _SCHEDULE_VECTOR_COMPARE_YMM,
}


def _vector_binary_descriptor(
    *,
    vector_bit_width: int,
    mnemonic: str,
    semantic_tag: str,
    schedule_class: str,
    instruction: VectorMachineInstruction,
) -> Descriptor:
    register_suffix = _VECTOR_REGISTER_SUFFIXES[vector_bit_width]
    return _vex_descriptor(
        Descriptor(
            key=f"x86.avx2.{mnemonic}.{register_suffix}",
            mnemonic=mnemonic,
            semantic_tag=semantic_tag,
            operands=(
                _vector_result(vector_bit_width),
                _vector_operand(vector_bit_width, "lhs"),
                _vector_operand(vector_bit_width, "rhs"),
            ),
            asm_forms=_asm(
                mnemonic=f"{mnemonic}.{register_suffix}",
                results=("dst",),
                operands=("lhs", "rhs"),
            ),
            schedule_class=schedule_class,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _vector_binary_family_descriptors(
    rows: tuple[VectorBinaryFamily, ...],
    instructions: tuple[VectorMachineInstruction, ...],
) -> tuple[Descriptor, ...]:
    return tuple(
        _vector_binary_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=row.mnemonic,
            semantic_tag=(
                f"{row.semantic}.{row.element.name}x"
                f"{row.element.lane_count(vector_bit_width)}"
            ),
            schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
            instruction=instruction,
        )
        for row, instruction in zip(rows, instructions, strict=True)
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    )


def _vector_float_binary_family_descriptors(
    rows: tuple[VectorBinaryFamily, ...],
    instructions: tuple[VectorMachineInstruction, ...],
) -> tuple[Descriptor, ...]:
    return tuple(
        _vector_binary_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=row.mnemonic,
            semantic_tag=(
                f"{row.semantic}.{row.element.name}x"
                f"{row.element.lane_count(vector_bit_width)}"
            ),
            schedule_class=_VECTOR_FLOAT_SCHEDULE_CLASSES[vector_bit_width],
            instruction=instruction,
        )
        for row, instruction in zip(rows, instructions, strict=True)
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    )


def _vector_compare_descriptor(
    *,
    vector_bit_width: int,
    mnemonic: str,
    semantic_tag: str,
    instruction: VectorMachineInstruction,
    immediate: bool = False,
) -> Descriptor:
    register_suffix = _VECTOR_REGISTER_SUFFIXES[vector_bit_width]
    return _vex_descriptor(
        Descriptor(
            key=f"x86.avx2.{mnemonic}.{register_suffix}",
            mnemonic=mnemonic,
            semantic_tag=semantic_tag,
            operands=(
                _vector_result(vector_bit_width),
                _vector_operand(vector_bit_width, "lhs"),
                _vector_operand(vector_bit_width, "rhs"),
            ),
            immediates=(
                (
                    Immediate(
                        "predicate",
                        ImmediateKind.UNSIGNED,
                        bit_width=5,
                        unsigned_max=31,
                    ),
                )
                if immediate
                else ()
            ),
            asm_forms=_asm(
                mnemonic=(
                    f"avx2.{mnemonic}.{register_suffix}"
                    if immediate
                    else f"{mnemonic}.{register_suffix}"
                ),
                results=("dst",),
                operands=("lhs", "rhs"),
                immediates=("predicate",) if immediate else (),
            ),
            schedule_class=_VECTOR_COMPARE_SCHEDULE_CLASSES[vector_bit_width],
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _vector_fma_descriptor(
    *,
    vector_bit_width: int,
    mnemonic: str,
    semantic_tag: str,
    instruction: VectorMachineInstruction,
) -> Descriptor:
    register_suffix = _VECTOR_REGISTER_SUFFIXES[vector_bit_width]
    return _vex_descriptor(
        Descriptor(
            key=f"x86.avx2.{mnemonic}.{register_suffix}",
            mnemonic=mnemonic,
            semantic_tag=semantic_tag,
            operands=(
                _vector_result(vector_bit_width),
                _vector_operand(vector_bit_width, "acc"),
                _vector_operand(vector_bit_width, "lhs"),
                _vector_operand(vector_bit_width, "rhs"),
            ),
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
            asm_forms=_asm(
                mnemonic=f"{mnemonic}.{register_suffix}",
                results=("dst",),
                operands=("acc", "lhs", "rhs"),
            ),
            schedule_class=_VECTOR_FMA_SCHEDULE_CLASSES[vector_bit_width],
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _vector_lane_extract_descriptor(
    row: VectorLaneFamily, instruction: VectorMachineInstruction
) -> Descriptor:
    lane_count = 128 // row.element_bit_width
    result = _gpr64_result() if row.element_bit_width == 64 else _gpr32_result()
    return _vex_descriptor(
        Descriptor(
            key=(
                f"x86.avx2.{row.extract_mnemonic}."
                f"gpr{max(32, row.element_bit_width)}.xmm"
            ),
            mnemonic=row.extract_mnemonic,
            semantic_tag=f"bits.extract.{row.element_bit_width}x{lane_count}",
            operands=(result, _xmm_operand("source")),
            immediates=(
                Immediate(
                    "lane",
                    ImmediateKind.UNSIGNED,
                    bit_width=8,
                    unsigned_max=lane_count - 1,
                ),
            ),
            asm_forms=_asm(
                mnemonic=f"{row.extract_mnemonic}.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _vector_lane_insert_descriptor(
    row: VectorLaneFamily, instruction: VectorMachineInstruction
) -> Descriptor:
    lane_count = 128 // row.element_bit_width
    value = (
        _gpr64_operand("value")
        if row.element_bit_width == 64
        else _gpr32_operand("value")
    )
    return _vex_descriptor(
        Descriptor(
            key=f"x86.avx2.{row.insert_mnemonic}.xmm",
            mnemonic=row.insert_mnemonic,
            semantic_tag=f"bits.insert.{row.element_bit_width}x{lane_count}",
            operands=(_vector_result(128), _xmm_operand("dest"), value),
            immediates=(
                Immediate(
                    "lane",
                    ImmediateKind.UNSIGNED,
                    bit_width=8,
                    unsigned_max=lane_count - 1,
                ),
            ),
            asm_forms=_asm(
                mnemonic=f"{row.insert_mnemonic}.xmm",
                results=("dst",),
                operands=("dest", "value"),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


def _vector_widen_descriptor(
    *,
    mnemonic: str,
    result_bit_width: int,
    semantic_tag: str,
    instruction: VectorMachineInstruction,
) -> Descriptor:
    register_suffix = _VECTOR_REGISTER_SUFFIXES[result_bit_width]
    return _vex_descriptor(
        Descriptor(
            key=f"x86.avx2.{mnemonic}.{register_suffix}.xmm",
            mnemonic=mnemonic,
            semantic_tag=semantic_tag,
            operands=(_vector_result(result_bit_width), _xmm_operand("source")),
            asm_forms=_asm(
                mnemonic=f"{mnemonic}.{register_suffix}",
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[result_bit_width],
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        instruction,
    )


_X86_AVX2_VECTOR_DESCRIPTORS = (
    *(
        _vex_descriptor(
            Descriptor(
                key=f"x86.avx2.{family.mnemonic}.{register_suffix}",
                mnemonic=family.mnemonic,
                semantic_tag=f"{family.semantic}.{family.element.name}x{family.element.lane_count(vector_bit_width)}",
                operands=(
                    _vector_result(vector_bit_width),
                    _vector_operand(vector_bit_width, "source"),
                ),
                immediates=(
                    Immediate(
                        "shift", ImmediateKind.UNSIGNED, bit_width=8, unsigned_max=255
                    ),
                ),
                asm_forms=_asm(
                    mnemonic=f"{family.mnemonic}.{register_suffix}",
                    results=("dst",),
                    operands=("source",),
                    immediates=("shift",),
                ),
                schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            instruction,
        )
        for family, instruction in zip(
            AVX2_UNIFORM_SHIFT_FAMILIES,
            native.AVX2_UNIFORM_SHIFT_IMMEDIATE,
            strict=True,
        )
        for vector_bit_width, register_suffix in _VECTOR_REGISTER_SUFFIXES.items()
    ),
    *(
        _vex_descriptor(
            Descriptor(
                key=f"x86.avx2.{family.mnemonic}.count.{register_suffix}",
                mnemonic=family.mnemonic,
                semantic_tag=f"{family.semantic}.{family.element.name}x{family.element.lane_count(vector_bit_width)}",
                operands=(
                    _vector_result(vector_bit_width),
                    _vector_operand(vector_bit_width, "source"),
                    _xmm_operand("count"),
                ),
                asm_forms=_asm(
                    mnemonic=f"{family.mnemonic}.count.{register_suffix}",
                    results=("dst",),
                    operands=("source", "count"),
                ),
                schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            instruction,
        )
        for family, instruction in zip(
            AVX2_UNIFORM_SHIFT_FAMILIES, native.AVX2_UNIFORM_SHIFT_COUNT, strict=True
        )
        for vector_bit_width, register_suffix in _VECTOR_REGISTER_SUFFIXES.items()
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vmovd.gpr32.xmm",
            mnemonic="vmovd",
            semantic_tag="bits.move.xmm.gpr32",
            operands=(_gpr32_result(), _xmm_operand("input")),
            asm_forms=_asm(
                mnemonic="vmovd.gpr32.xmm",
                results=("dst",),
                operands=("input",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VMOVD_TO_GPR32,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vmovd.xmm.gpr32",
            mnemonic="vmovd",
            semantic_tag="bits.move.gpr32.xmm",
            operands=(_xmm_result(), _gpr32_operand("input")),
            asm_forms=_asm(
                mnemonic="vmovd.xmm.gpr32",
                results=("dst",),
                operands=("input",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VMOVD_FROM_GPR32,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vmovq.gpr64.xmm",
            mnemonic="vmovq",
            semantic_tag="bits.move.xmm.gpr64",
            operands=(_gpr64_result(), _xmm_operand("input")),
            asm_forms=_asm(
                mnemonic="vmovq.gpr64.xmm",
                results=("dst",),
                operands=("input",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VMOVQ_TO_GPR64,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vmovq.xmm.gpr64",
            mnemonic="vmovq",
            semantic_tag="bits.move.gpr64.xmm",
            operands=(_xmm_result(), _gpr64_operand("input")),
            asm_forms=_asm(
                mnemonic="vmovq.xmm.gpr64",
                results=("dst",),
                operands=("input",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VMOVQ_FROM_GPR64,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vblendvpd.xmm",
            mnemonic="vblendvpd",
            semantic_tag="float.select.f64x2",
            operands=(
                _xmm_result(),
                _xmm_operand("false_value"),
                _xmm_operand("true_value"),
                _xmm_operand("mask"),
            ),
            asm_forms=_asm(
                mnemonic="vblendvpd.xmm",
                results=("dst",),
                operands=("false_value", "true_value", "mask"),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VBLENDVPD,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vblendvps.ymm",
            mnemonic="vblendvps",
            semantic_tag="float.select.f32x8",
            operands=(
                _vector_result(256),
                _vector_operand(256, "false_value"),
                _vector_operand(256, "true_value"),
                _vector_operand(256, "mask"),
            ),
            asm_forms=_asm(
                mnemonic="vblendvps.ymm",
                results=("dst",),
                operands=("false_value", "true_value", "mask"),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VBLENDVPS,
    ),
    *(
        descriptor
        for row, instructions in zip(X86_LANE_FAMILIES, native.AVX2_LANE, strict=True)
        for descriptor in (
            _vector_lane_extract_descriptor(row, instructions[0]),
            _vector_lane_insert_descriptor(row, instructions[1]),
        )
    ),
    _vex_descriptor(
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
        native.VPSHUFD,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpshuflw.xmm",
            mnemonic="vpshuflw",
            semantic_tag="bits.permute.16x4",
            operands=(_vector_result(128), _xmm_operand("source")),
            immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpshuflw.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPSHUFLW,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpermilpd.xmm",
            mnemonic="vpermilpd",
            semantic_tag="bits.permute.64x2",
            operands=(_vector_result(128), _xmm_operand("source")),
            immediates=(_SHUFFLE_2X1_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpermilpd.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPERMILPD,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpermq.ymm",
            mnemonic="vpermq",
            semantic_tag="bits.permute.64x4",
            operands=(_vector_result(256), _vector_operand(256, "source")),
            immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpermq.ymm",
                results=("dst",),
                operands=("source",),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPERMQ_VECTOR_UNARY,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpermps.ymm",
            mnemonic="vpermps",
            semantic_tag="bits.permute.32x8",
            operands=(
                _vector_result(256),
                _vector_operand(256, "control"),
                _vector_operand(256, "source"),
            ),
            asm_forms=_asm(
                mnemonic="vpermps.ymm",
                results=("dst",),
                operands=("control", "source"),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPERMPS,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vshufpd.xmm",
            mnemonic="vshufpd",
            semantic_tag="float.shuffle2.f64x2",
            operands=(
                _vector_result(128),
                _xmm_operand("lhs"),
                _xmm_operand("rhs"),
            ),
            immediates=(_SHUFFLE_2X1_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vshufpd.xmm",
                results=("dst",),
                operands=("lhs", "rhs"),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VSHUFPD,
    ),
    *(
        _vex_descriptor(
            _vector_splat_descriptor(
                vector_bit_width=vector_bit_width,
                key=f"x86.avx2.{mnemonic}.{_VECTOR_REGISTER_SUFFIXES[vector_bit_width]}",
                mnemonic=mnemonic,
                semantic_tag=f"bits.splat.{element_bit_width}.v{vector_bit_width}",
                operand=_xmm_operand("value"),
                schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
                asm_mnemonic=(
                    f"avx2.{mnemonic}.xmm"
                    if vector_bit_width == 128 and mnemonic == "vpbroadcastq"
                    else None
                ),
            ),
            instruction,
        )
        for (element_bit_width, mnemonic), instruction in zip(
            (
                (8, "vpbroadcastb"),
                (16, "vpbroadcastw"),
                (64, "vpbroadcastq"),
            ),
            native.AVX2_SIMD_BROADCAST,
            strict=True,
        )
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    _vex_descriptor(
        _vector_splat_descriptor(
            vector_bit_width=128,
            key="x86.avx2.vpbroadcastd.xmm",
            mnemonic="vpbroadcastd",
            semantic_tag="bits.splat.32.v128",
            operand=_xmm_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            asm_mnemonic="avx2.vpbroadcastd.xmm",
        ),
        native.VPBROADCASTD_VECTOR_UNARY,
    ),
    _vex_descriptor(
        _vector_splat_descriptor(
            vector_bit_width=128,
            key="x86.avx2.vbroadcastss.xmm",
            mnemonic="vbroadcastss",
            semantic_tag="float.splat.f32x4",
            operand=_xmm_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        ),
        native.VBROADCASTSS,
    ),
    _vex_descriptor(
        _vector_splat_descriptor(
            vector_bit_width=256,
            key="x86.avx2.vbroadcastss.ymm",
            mnemonic="vbroadcastss",
            semantic_tag="float.splat.f32x8",
            operand=_xmm_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_F32_YMM,
        ),
        native.VBROADCASTSS,
    ),
    *(
        _vex_descriptor(
            _vector_splat_descriptor(
                vector_bit_width=vector_bit_width,
                key=f"x86.avx2.vbroadcastsd.{_VECTOR_REGISTER_SUFFIXES[vector_bit_width]}",
                mnemonic="vbroadcastsd",
                semantic_tag=f"float.splat.f64x{vector_bit_width // 64}",
                operand=_xmm_operand("value"),
                schedule_class=_VECTOR_FLOAT_SCHEDULE_CLASSES[vector_bit_width],
            ),
            native.VBROADCASTSD,
        )
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    _vex_descriptor(
        _vector_splat_descriptor(
            vector_bit_width=256,
            key="x86.avx2.vpbroadcastd.ymm",
            mnemonic="vpbroadcastd",
            semantic_tag="integer.splat.i32x8",
            operand=_xmm_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
        ),
        native.VPBROADCASTD_VECTOR_UNARY,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpmovsxdq.ymm.xmm",
            mnemonic="vpmovsxdq",
            semantic_tag="integer.extend.signed.i32x4.i64x4",
            operands=(_vector_result(256), _xmm_operand("source")),
            asm_forms=_asm(
                mnemonic="vpmovsxdq.ymm.xmm",
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPMOVSXDQ,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpmovsxwd.ymm.xmm",
            mnemonic="vpmovsxwd",
            semantic_tag="integer.extend.signed.i16x8.i32x8",
            operands=(_vector_result(256), _xmm_operand("source")),
            asm_forms=_asm(
                mnemonic="vpmovsxwd.ymm.xmm",
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPMOVSXWD,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpmovsxbw.ymm.xmm",
            mnemonic="vpmovsxbw",
            semantic_tag="integer.extend.signed.i8x16.i16x16",
            operands=(_vector_result(256), _xmm_operand("source")),
            asm_forms=_asm(
                mnemonic="vpmovsxbw.ymm.xmm",
                results=("dst",),
                operands=("source",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPMOVSXBW,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpackssdw.xmm",
            mnemonic="vpackssdw",
            semantic_tag="integer.pack.signed.i32x8.i16x8",
            operands=(
                _vector_result(128),
                _xmm_operand("low"),
                _xmm_operand("high"),
            ),
            asm_forms=_asm(
                mnemonic="vpackssdw.xmm",
                results=("dst",),
                operands=("low", "high"),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPACKSSDW,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpacksswb.xmm",
            mnemonic="vpacksswb",
            semantic_tag="integer.pack.signed.i16x16.i8x16",
            operands=(
                _vector_result(128),
                _xmm_operand("low"),
                _xmm_operand("high"),
            ),
            asm_forms=_asm(
                mnemonic="vpacksswb.xmm",
                results=("dst",),
                operands=("low", "high"),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPACKSSWB,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vshufps.xmm",
            mnemonic="vshufps",
            semantic_tag="bits.pack.i64x4.i32x4",
            operands=(
                _vector_result(128),
                _xmm_operand("low"),
                _xmm_operand("high"),
            ),
            immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vshufps.xmm",
                results=("dst",),
                operands=("low", "high"),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VSHUFPS,
    ),
    _vex_descriptor(
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
        native.VPERMILPS,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vextractf128.xmm.ymm",
            mnemonic="vextractf128",
            semantic_tag="float.extract.f32x8.half",
            operands=(_xmm_result(), _vector_operand(256, "source")),
            immediates=(_TWO_LANE_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vextractf128.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VEXTRACTF128,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vinsertf128.ymm.xmm",
            mnemonic="vinsertf128",
            semantic_tag="bits.insert.v128.v256",
            operands=(
                _vector_result(256),
                _vector_operand(256, "dest"),
                _xmm_operand("value"),
            ),
            immediates=(_TWO_LANE_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vinsertf128.ymm.xmm",
                results=("dst",),
                operands=("dest", "value"),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_YMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VINSERTF128,
    ),
    _vex_descriptor(
        _vector_zero_descriptor(
            vector_bit_width=128,
            key="x86.avx2.vxorps.zero.xmm",
        ),
        native.VXORPS,
    ),
    _vex_descriptor(
        _vector_zero_descriptor(
            vector_bit_width=256,
            key="x86.avx2.vxorps.zero.ymm",
        ),
        native.VXORPS,
    ),
    _vex_descriptor(
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
        native.VINSERTPS,
    ),
    *_vector_binary_family_descriptors(
        AVX2_INTEGER_BINARY_FAMILIES, native.AVX2_INTEGER_BINARY
    ),
    *(
        _vector_binary_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=mnemonic,
            semantic_tag=f"{semantic}.v{vector_bit_width}",
            schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
            instruction=instruction,
        )
        for (_, mnemonic, semantic), instruction in zip(
            AVX2_BITWISE_FAMILIES, native.AVX2_BITWISE, strict=True
        )
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    *(
        _vex_descriptor(
            Descriptor(
                key=f"x86.avx2.vmovdqu.rodata.{register_suffix}",
                mnemonic="vmovdqu",
                semantic_tag=f"memory.load.rodata.v{vector_bit_width}",
                operands=(_vector_result(vector_bit_width),),
                immediates=(_READ_ONLY_DATA_IMMEDIATE,),
                asm_forms=_asm(
                    mnemonic=f"avx2.vmovdqu.rodata.{register_suffix}",
                    results=("dst",),
                    immediates=("data",),
                ),
                schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VMOVDQU,
        )
        for vector_bit_width, register_suffix in _VECTOR_REGISTER_SUFFIXES.items()
    ),
    *(
        _vector_compare_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=mnemonic,
            semantic_tag=(
                f"{semantic}.{element.name}x{element.lane_count(vector_bit_width)}"
            ),
            instruction=instruction,
        )
        for element, instructions in zip(
            INTEGER_ELEMENTS, native.AVX2_INTEGER_COMPARE, strict=True
        )
        for mnemonic, semantic, instruction in zip(
            AVX2_INTEGER_COMPARE_MNEMONICS[element.name],
            ("integer.cmp.eq", "integer.cmp.sgt"),
            instructions,
            strict=True,
        )
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    *(
        _vector_widen_descriptor(
            mnemonic=mnemonic,
            result_bit_width=result_bit_width,
            semantic_tag=(
                f"integer.extui.i8x{result_bit_width // result_element_bit_width}."
                f"i{result_element_bit_width}x"
                f"{result_bit_width // result_element_bit_width}"
            ),
            instruction=instruction,
        )
        for mnemonic, result_element_bit_width, instruction in (
            ("vpmovzxbw", 16, native.VPMOVZXBW),
            ("vpmovzxbd", 32, native.VPMOVZXBD),
            ("vpmovzxbq", 64, native.VPMOVZXBQ),
        )
        for result_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    _vector_binary_descriptor(
        vector_bit_width=128,
        mnemonic="vpunpcklqdq",
        semantic_tag="bits.interleave.low.i64x2",
        schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        instruction=native.VPUNPCKLQDQ,
    ),
    *(
        _vector_binary_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic="vpshufb",
            semantic_tag=f"bits.shuffle.bytes.v{vector_bit_width}",
            schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
            instruction=native.VPSHUFB,
        )
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    _vector_binary_descriptor(
        vector_bit_width=128,
        mnemonic="vpackuswb",
        semantic_tag="integer.packu.i16x16.i8x16",
        schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        instruction=native.VPACKUSWB,
    ),
    _vex_descriptor(
        Descriptor(
            key="x86.avx2.vpsrldq.xmm",
            mnemonic="vpsrldq",
            semantic_tag="bits.shift_right_bytes.v128",
            operands=(_vector_result(128), _xmm_operand("source")),
            immediates=(
                Immediate(
                    "bytes",
                    ImmediateKind.UNSIGNED,
                    bit_width=8,
                    unsigned_max=255,
                ),
            ),
            asm_forms=_asm(
                mnemonic="vpsrldq.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("bytes",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        native.VPSRLDQ,
    ),
    *_vector_float_binary_family_descriptors(
        AVX2_FLOAT_BINARY_FAMILIES, native.AVX2_FLOAT_BINARY
    ),
    *(
        _vector_binary_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=mnemonic,
            semantic_tag=(
                f"float.fast_extrema.{element.name}x"
                f"{element.lane_count(vector_bit_width)}"
            ),
            schedule_class=_VECTOR_FLOAT_SCHEDULE_CLASSES[vector_bit_width],
            instruction=instruction,
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
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    *(
        _vex_descriptor(
            _scalar_float_binary_descriptor(
                key=f"x86.avx2.{row.mnemonic}.xmm",
                mnemonic=row.mnemonic,
                semantic_tag=f"{row.semantic}.{row.element.name}",
            ),
            instruction,
        )
        for row, instruction in zip(
            AVX2_SCALAR_FLOAT_BINARY_FAMILIES,
            native.AVX2_SCALAR_FLOAT_BINARY,
            strict=True,
        )
    ),
    *(
        _vex_descriptor(
            _scalar_float_binary_descriptor(
                key=f"x86.avx2.{mnemonic}.xmm",
                mnemonic=mnemonic,
                semantic_tag=f"float.fast_extrema.{element.name}",
            ),
            instruction,
        )
        for element, instructions in zip(
            FLOAT_ELEMENTS,
            (
                native.AVX2_SCALAR_FLOAT_EXTREMA[:2],
                native.AVX2_SCALAR_FLOAT_EXTREMA[2:],
            ),
            strict=True,
        )
        for mnemonic, instruction in zip(
            (
                AVX2_SCALAR_FLOAT_EXTREMA_MNEMONICS["minimumf"][element.name],
                AVX2_SCALAR_FLOAT_EXTREMA_MNEMONICS["maximumf"][element.name],
            ),
            instructions,
            strict=True,
        )
    ),
    *(
        _vector_fma_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=mnemonic,
            semantic_tag=(
                f"float.fma.{element.name}x{element.lane_count(vector_bit_width)}"
            ),
            instruction=instruction,
        )
        for element, instruction in zip(
            FLOAT_ELEMENTS, native.AVX2_FLOAT_FMA, strict=True
        )
        for mnemonic in (AVX2_FLOAT_FMA_MNEMONICS[element.name],)
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    *(
        _vector_compare_descriptor(
            vector_bit_width=vector_bit_width,
            mnemonic=mnemonic,
            semantic_tag=(
                f"float.cmp.{element.name}x{element.lane_count(vector_bit_width)}"
            ),
            instruction=instruction,
            immediate=True,
        )
        for element, instruction in zip(
            FLOAT_ELEMENTS, native.AVX2_FLOAT_COMPARE, strict=True
        )
        for mnemonic in (AVX2_FLOAT_COMPARE_MNEMONICS[element.name],)
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    ),
    *(
        _vex_descriptor(
            Descriptor(
                key=f"x86.avx2.vpblendvb.{register_suffix}",
                mnemonic="vpblendvb",
                semantic_tag=f"bits.select.v{vector_bit_width}",
                operands=(
                    _vector_result(vector_bit_width),
                    _vector_operand(vector_bit_width, "false_value"),
                    _vector_operand(vector_bit_width, "true_value"),
                    _vector_operand(vector_bit_width, "mask"),
                ),
                asm_forms=_asm(
                    mnemonic=f"vpblendvb.{register_suffix}",
                    results=("dst",),
                    operands=("false_value", "true_value", "mask"),
                ),
                schedule_class=_VECTOR_INTEGER_SCHEDULE_CLASSES[vector_bit_width],
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            native.VPBLENDVB,
        )
        for vector_bit_width, register_suffix in _VECTOR_REGISTER_SUFFIXES.items()
    ),
    *(
        _vex_descriptor(
            Descriptor(
                key=f"x86.avx2.{mnemonic}.xmm",
                mnemonic=mnemonic,
                semantic_tag=f"float.fma.{element.name}",
                operands=(
                    _vector_result(128),
                    _xmm_operand("acc"),
                    _xmm_operand("lhs"),
                    _xmm_operand("rhs"),
                ),
                constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
                asm_forms=_asm(
                    mnemonic=f"{mnemonic}.xmm",
                    results=("dst",),
                    operands=("acc", "lhs", "rhs"),
                ),
                schedule_class=_SCHEDULE_VECTOR_FMA_F32_XMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            instruction,
        )
        for element, instruction in zip(
            FLOAT_ELEMENTS, native.AVX2_SCALAR_FLOAT_FMA, strict=True
        )
        for mnemonic in (AVX2_SCALAR_FLOAT_FMA_MNEMONICS[element.name],)
    ),
    *memory_descriptors(
        key_prefix="x86.avx2",
        load_mnemonic="vmovdqu32",
        store_mnemonic="vmovdqu32",
        register_class=_REG_XMM,
        register_suffix="xmm",
        semantic_type="v128",
        width_bits=128,
        load_schedule_class=_SCHEDULE_MEMORY_LOAD_XMM,
        store_schedule_class=_SCHEDULE_MEMORY_STORE_XMM,
        assembly_suffix=".xmm",
        vector_instructions=native.VECTOR_MEMORY,
        vector_prefix=VectorEncodingPrefix.VEX,
    ),
    *(
        descriptor
        for mnemonic, width_bits, instructions in (
            ("vmovss", 32, native.XMM_DWORD_MEMORY),
            ("vmovsd", 64, native.XMM_QWORD_MEMORY),
        )
        for descriptor in memory_descriptors(
            key_prefix="x86.avx2",
            load_mnemonic=mnemonic,
            store_mnemonic=mnemonic,
            register_class=_REG_XMM,
            register_suffix="xmm",
            semantic_type=f"v{width_bits}",
            width_bits=width_bits,
            load_schedule_class=_SCHEDULE_MEMORY_LOAD_XMM,
            store_schedule_class=_SCHEDULE_MEMORY_STORE_XMM,
            assembly_suffix=".xmm",
            vector_instructions=instructions,
            vector_prefix=VectorEncodingPrefix.VEX,
        )
    ),
    *memory_descriptors(
        key_prefix="x86.avx2",
        load_mnemonic="vmovdqu32",
        store_mnemonic="vmovdqu32",
        register_class=_REG_YMM,
        register_suffix="ymm",
        semantic_type="v256",
        width_bits=256,
        load_schedule_class=_SCHEDULE_MEMORY_LOAD_YMM,
        store_schedule_class=_SCHEDULE_MEMORY_STORE_YMM,
        assembly_suffix=".ymm",
        vector_instructions=native.VECTOR_MEMORY,
        vector_prefix=VectorEncodingPrefix.VEX,
    ),
)

X86_AVX2_VECTOR_DESCRIPTORS = _X86_AVX2_VECTOR_DESCRIPTORS

X86_AVX2_DESCRIPTORS = (
    *X86_SCALAR_PREFIX_DESCRIPTORS,
    *X86_AVX2_VECTOR_DESCRIPTORS,
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
    supports_native_scheduling=True,
    physical_registers=X86_SCALAR_DESCRIPTOR_SET.physical_registers,
    reg_classes=(
        *X86_SCALAR_DESCRIPTOR_SET.reg_classes,
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=1,
        ),
        RegClass(
            _REG_YMM,
            256,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=1,
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
            _SCHEDULE_VECTOR_I32_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(256)),
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
    descriptors=X86_AVX2_DESCRIPTORS,
)
