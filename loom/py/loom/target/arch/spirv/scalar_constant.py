# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for floating-point SPIR-V scalar constants."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.scalar_alu import FLOAT_SCALAR_ALU_TYPES


@dataclass(frozen=True, slots=True)
class FloatConstantType:
    """One floating-point scalar type accepted by OpConstant."""

    source_type: str
    suffix: str
    scalar_enum: str
    bit_width: int
    feature_atoms: tuple[str, ...] = ()

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)

    @property
    def literal_word_count(self) -> int:
        return (self.bit_width + 31) // 32


_FLOAT_BIT_WIDTHS = {
    "f16": 16,
    "f32": 32,
    "f64": 64,
}

_ARITHMETIC_FLOAT_CONSTANT_TYPES = tuple(
    FloatConstantType(
        source_type=scalar.source_type,
        suffix=scalar.suffix,
        scalar_enum=scalar.scalar_enum,
        bit_width=_FLOAT_BIT_WIDTHS[scalar.source_type],
        feature_atoms=scalar.feature_atoms,
    )
    for scalar in FLOAT_SCALAR_ALU_TYPES
)

BFLOAT16_CONSTANT_TYPE = FloatConstantType(
    source_type="bf16",
    suffix="bf16",
    scalar_enum="LOOM_SPIRV_SCALAR_TYPE_BF16",
    bit_width=16,
    feature_atoms=("bfloat16_type_khr",),
)

FLOAT_CONSTANT_TYPES = (
    _ARITHMETIC_FLOAT_CONSTANT_TYPES[0],
    BFLOAT16_CONSTANT_TYPE,
    *_ARITHMETIC_FLOAT_CONSTANT_TYPES[1:],
)
