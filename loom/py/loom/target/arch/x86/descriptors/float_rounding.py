# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Instruction-controlled scalar and packed floating-point rounding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.vector_families import FloatRoundingFamily
from loom.target.low_descriptors import Descriptor, Immediate, ImmediateKind

from .common import (
    _asm,
    _avx512_descriptor,
    _scalar_float_unary_descriptor,
    _vector_asm_mnemonic,
    _vector_float_unary_descriptor,
    _vex_descriptor,
)

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_ROUNDING_CONTROL = Immediate(
    "rounding", ImmediateKind.UNSIGNED, bit_width=8, unsigned_max=255
)


def float_rounding_descriptors(
    *,
    key_prefix: str,
    families: Sequence[FloatRoundingFamily],
    vector_bit_widths: Sequence[int],
    evex: bool,
    required_feature_bits: int = 0,
) -> tuple[Descriptor, ...]:
    descriptors = []
    for family in families:
        for scalar, widths, mnemonic in (
            (True, (128,), family.scalar_mnemonic),
            (False, vector_bit_widths, family.vector_mnemonic),
        ):
            instruction = getattr(native, mnemonic.upper())
            for width in widths:
                suffix = _REGISTER_SUFFIXES[width]
                key = f"{key_prefix}.{mnemonic}.{suffix}"
                semantic = f"float.round.{family.element.name}"
                if scalar:
                    descriptor = _scalar_float_unary_descriptor(
                        key=key, mnemonic=mnemonic, semantic_tag=semantic
                    )
                else:
                    descriptor = _vector_float_unary_descriptor(
                        key=key,
                        mnemonic=mnemonic,
                        semantic_tag=f"{semantic}x{family.element.lane_count(width)}",
                        vector_bit_width=width,
                    )
                descriptor = replace(
                    descriptor,
                    immediates=(_ROUNDING_CONTROL,),
                    asm_forms=_asm(
                        mnemonic=(
                            f"{mnemonic}.xmm"
                            if scalar
                            else _vector_asm_mnemonic(mnemonic, width)
                        ),
                        results=("dst",),
                        operands=("passthrough", "input") if scalar else ("input",),
                        immediates=("rounding",),
                    ),
                )
                descriptors.append(
                    _avx512_descriptor(
                        descriptor,
                        instruction,
                        scalar=scalar,
                        required_feature_bits=required_feature_bits,
                    )
                    if evex
                    else _vex_descriptor(descriptor, instruction)
                )
    return tuple(descriptors)
