# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bit-preserving unpack and two-source permutation descriptor families."""

from functools import partial

from loom.target.arch.x86.feature_bits import FEATURE_AVX512_VBMI
from loom.target.arch.x86.native_permutation import (
    TWO_SOURCE_PERMUTE_INSTRUCTIONS,
    UNPACK_INSTRUCTIONS,
)
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
    _asm,
    _avx512_descriptor,
    _vector_i32_schedule_class,
    _vector_operand,
    _vector_result,
    _vex_descriptor,
)

_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}


def unpack_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    prefix = "avx512" if evex else "avx2"
    bind = partial(_avx512_descriptor, scalar=False) if evex else _vex_descriptor
    return tuple(
        bind(
            Descriptor(
                key=f"x86.{prefix}.{instruction.descriptor_mnemonic}.{suffix}",
                mnemonic=instruction.descriptor_mnemonic,
                semantic_tag=(
                    f"bits.interleave.{'high' if high else 'low'}."
                    f"i{element_bit_width}x{width // element_bit_width}"
                ),
                operands=(
                    _vector_result(width),
                    _vector_operand(width, "lhs"),
                    _vector_operand(width, "rhs"),
                ),
                asm_forms=_asm(
                    mnemonic=(
                        f"{prefix + '.' if evex else ''}"
                        f"{instruction.descriptor_mnemonic}.{suffix}"
                    ),
                    results=("dst",),
                    operands=("lhs", "rhs"),
                ),
                schedule_class=_vector_i32_schedule_class(width),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            instruction,
        )
        for (element_bit_width, high), instruction in UNPACK_INSTRUCTIONS.items()
        for width in ((128, 256, 512) if evex else (128, 256))
        for suffix in (_SUFFIXES[width],)
    )


def two_source_permute_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _avx512_descriptor(
            Descriptor(
                key=f"x86.avx512.{instruction.descriptor_mnemonic}.{suffix}",
                mnemonic=instruction.descriptor_mnemonic,
                semantic_tag=(
                    f"bits.permute.two.i{element_bit_width}"
                    f"x{width // element_bit_width}"
                ),
                operands=(
                    _vector_result(width),
                    *(_vector_operand(width, name) for name in operands),
                ),
                constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
                asm_forms=_asm(
                    mnemonic=f"{instruction.descriptor_mnemonic}.{suffix}",
                    results=("dst",),
                    operands=operands,
                ),
                schedule_class=_vector_i32_schedule_class(width),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            instruction,
            scalar=False,
            required_feature_bits=FEATURE_AVX512_VBMI if element_bit_width == 8 else 0,
        )
        for (
            element_bit_width,
            overwrite_index,
        ), instruction in TWO_SOURCE_PERMUTE_INSTRUCTIONS.items()
        for operands in (
            ("control", "lhs", "rhs") if overwrite_index else ("lhs", "control", "rhs"),
        )
        for width, suffix in _SUFFIXES.items()
    )
