# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-independent numeric format definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loom.ir import ScalarTypeKind

NUMERIC_FORMAT_KEYWORDS = (
    "none",
    "f64",
    "f32",
    "tf32",
    "f16",
    "bf16",
    "i32",
    "u32",
    "i16",
    "u16",
    "i8",
    "u8",
    "i6",
    "u6",
    "i5",
    "u5",
    "i4",
    "u4",
    "i3",
    "u3",
    "i2",
    "u2",
    "i1",
    "u1",
    "f8e4m3",
    "f8e5m2",
    "f8e4m3fn",
    "f8e4m3fnuz",
    "f8e5m2fnuz",
    "e8m0",
    "bf8",
    "f6e3m2",
    "f6e2m3",
    "bf6",
    "f4e2m1",
    "ternary",
    "sign_bit",
    "codebook_index",
    "quant_i8",
    "quant_i6",
    "quant_i4",
)


class Fp8SpecialPolicy(Enum):
    """Interpretation of the maximal exponent and signed-zero payloads."""

    IEEE = "ieee"
    FINITE_NAN = "finite_nan"
    FINITE_NAN_UNSIGNED_ZERO = "finite_nan_unsigned_zero"


@dataclass(frozen=True)
class Fp8Format:
    """Exact target-independent semantics for one FP8 numeric format."""

    keyword: str
    carrier_type: ScalarTypeKind
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    special_policy: Fp8SpecialPolicy


FP8_FORMATS = (
    Fp8Format(
        "f8e4m3",
        ScalarTypeKind.F8E4M3,
        exponent_bits=4,
        mantissa_bits=3,
        exponent_bias=7,
        special_policy=Fp8SpecialPolicy.IEEE,
    ),
    Fp8Format(
        "f8e5m2",
        ScalarTypeKind.F8E5M2,
        exponent_bits=5,
        mantissa_bits=2,
        exponent_bias=15,
        special_policy=Fp8SpecialPolicy.IEEE,
    ),
    Fp8Format(
        "f8e4m3fn",
        ScalarTypeKind.F8E4M3,
        exponent_bits=4,
        mantissa_bits=3,
        exponent_bias=7,
        special_policy=Fp8SpecialPolicy.FINITE_NAN,
    ),
    Fp8Format(
        "f8e4m3fnuz",
        ScalarTypeKind.F8E4M3,
        exponent_bits=4,
        mantissa_bits=3,
        exponent_bias=8,
        special_policy=Fp8SpecialPolicy.FINITE_NAN_UNSIGNED_ZERO,
    ),
    Fp8Format(
        "f8e5m2fnuz",
        ScalarTypeKind.F8E5M2,
        exponent_bits=5,
        mantissa_bits=2,
        exponent_bias=16,
        special_policy=Fp8SpecialPolicy.FINITE_NAN_UNSIGNED_ZERO,
    ),
)
