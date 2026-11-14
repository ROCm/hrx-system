# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for ordinary SPIR-V scalar conversions."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.scalar_alu import (
    BFLOAT16_SCALAR_TYPE,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    ScalarAluType,
)

_BIT_WIDTHS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "f64": 64,
}

_FLOAT32_SCALAR_TYPE = next(
    scalar for scalar in FLOAT_SCALAR_ALU_TYPES if scalar.source_type == "f32"
)


@dataclass(frozen=True, slots=True)
class ScalarConversion:
    source_op_key: str
    descriptor_suffix: str
    mnemonic: str
    opcode: str
    source_type: ScalarAluType
    result_type: ScalarAluType
    feature_atoms: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return (
            f"spirv.op_{self.descriptor_suffix}."
            f"{self.source_type.suffix}.{self.result_type.suffix}"
        )

    @property
    def display_mnemonic(self) -> str:
        return f"{self.mnemonic}.{self.source_type.suffix}.{self.result_type.suffix}"

    @property
    def feature_bits(self) -> int:
        return (
            self.source_type.feature_bits
            | self.result_type.feature_bits
            | feature_bits_value(self.feature_atoms)
        )


@dataclass(frozen=True, slots=True)
class IntegerValueViewConversion:
    source_type: ScalarAluType
    result_type: ScalarAluType

    @property
    def key(self) -> str:
        return f"spirv.op_bitcast.{self.source_type.suffix}.{self.result_type.suffix}"

    @property
    def display_mnemonic(self) -> str:
        return f"OpBitcast.{self.source_type.suffix}.{self.result_type.suffix}"

    @property
    def feature_bits(self) -> int:
        return self.source_type.feature_bits | self.result_type.feature_bits


def _bit_width(scalar: ScalarAluType) -> int:
    return _BIT_WIDTHS[scalar.source_type]


def _float_narrowing_feature_atoms(
    source_width: int, result_width: int
) -> tuple[str, ...]:
    if result_width not in (16, 32):
        raise ValueError(f"unsupported native float narrowing width {result_width}")

    # Vulkan assumes NSZ, NotInf, and NotNaN when an entry point omits
    # SignedZeroInfNanPreserve for any operand or result width used by an
    # instruction. Native narrowing therefore needs the promise for both
    # widths even though rounding and destination-denormal behavior are
    # selected by the result format.
    atoms = (
        f"float{result_width}_denorm_preserve",
        f"float{result_width}_rounding_rte",
        f"float{result_width}_signed_zero_inf_nan_preserve",
        f"float{source_width}_signed_zero_inf_nan_preserve",
    )
    if source_width == 64 and result_width == 16:
        atoms = (*atoms, "float64_to_float16_conversion")
    return atoms


def _integer_width_conversions() -> tuple[ScalarConversion, ...]:
    rows: list[ScalarConversion] = []
    for source_type in SIGNED_INTEGER_SCALAR_ALU_TYPES:
        for result_type in SIGNED_INTEGER_SCALAR_ALU_TYPES:
            if source_type == result_type:
                continue
            source_width = _bit_width(source_type)
            result_width = _bit_width(result_type)
            rows.append(
                ScalarConversion(
                    source_op_key="extsi" if source_width < result_width else "trunci",
                    descriptor_suffix="s_convert",
                    mnemonic="OpSConvert",
                    opcode="LOOM_SPIRV_OP_S_CONVERT",
                    source_type=source_type,
                    result_type=result_type,
                )
            )
    return tuple(rows)


def _float_width_conversions() -> tuple[ScalarConversion, ...]:
    rows: list[ScalarConversion] = []
    for source_type in FLOAT_SCALAR_ALU_TYPES:
        for result_type in FLOAT_SCALAR_ALU_TYPES:
            if source_type == result_type:
                continue
            source_width = _bit_width(source_type)
            result_width = _bit_width(result_type)
            feature_atoms: tuple[str, ...] = ()
            if source_width > result_width:
                feature_atoms = _float_narrowing_feature_atoms(
                    source_width, result_width
                )
            else:
                # Nonzero finite widening results are normal in the destination,
                # but OpFConvert may flush a source subnormal. Both widths
                # also need the promise that preserves zeros and nonfinite
                # values. Portable bit-pattern recipes cover other profiles.
                feature_atoms = (
                    f"float{source_width}_denorm_preserve",
                    f"float{source_width}_signed_zero_inf_nan_preserve",
                    f"float{result_width}_signed_zero_inf_nan_preserve",
                )
            rows.append(
                ScalarConversion(
                    source_op_key="extf" if source_width < result_width else "fptrunc",
                    descriptor_suffix="f_convert",
                    mnemonic="OpFConvert",
                    opcode="LOOM_SPIRV_OP_F_CONVERT",
                    source_type=source_type,
                    result_type=result_type,
                    feature_atoms=feature_atoms,
                )
            )
    return tuple(rows)


def _signed_integer_to_float_conversions() -> tuple[ScalarConversion, ...]:
    return tuple(
        ScalarConversion(
            source_op_key="sitofp",
            descriptor_suffix="convert_s_to_f",
            mnemonic="OpConvertSToF",
            opcode="LOOM_SPIRV_OP_CONVERT_S_TO_F",
            source_type=source_type,
            result_type=result_type,
        )
        for source_type in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for result_type in FLOAT_SCALAR_ALU_TYPES
    )


def _float_to_signed_integer_conversions() -> tuple[ScalarConversion, ...]:
    return tuple(
        ScalarConversion(
            source_op_key="fptosi",
            descriptor_suffix="convert_f_to_s",
            mnemonic="OpConvertFToS",
            opcode="LOOM_SPIRV_OP_CONVERT_F_TO_S",
            source_type=source_type,
            result_type=result_type,
        )
        for source_type in FLOAT_SCALAR_ALU_TYPES
        for result_type in SIGNED_INTEGER_SCALAR_ALU_TYPES
    )


def _unsigned_integer_width_conversions() -> tuple[ScalarConversion, ...]:
    rows: list[ScalarConversion] = []
    unsigned_types = tuple(
        scalar_pair.unsigned for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
    )
    for source_type in unsigned_types:
        for result_type in unsigned_types:
            if source_type == result_type:
                continue
            source_width = _bit_width(source_type)
            result_width = _bit_width(result_type)
            if source_width > result_width:
                continue
            rows.append(
                ScalarConversion(
                    source_op_key="extui",
                    descriptor_suffix="u_convert",
                    mnemonic="OpUConvert",
                    opcode="LOOM_SPIRV_OP_U_CONVERT",
                    source_type=source_type,
                    result_type=result_type,
                )
            )
    return tuple(rows)


def _unsigned_integer_to_float_conversions() -> tuple[ScalarConversion, ...]:
    return tuple(
        ScalarConversion(
            source_op_key="uitofp",
            descriptor_suffix="convert_u_to_f",
            mnemonic="OpConvertUToF",
            opcode="LOOM_SPIRV_OP_CONVERT_U_TO_F",
            source_type=scalar_pair.unsigned,
            result_type=result_type,
        )
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        for result_type in FLOAT_SCALAR_ALU_TYPES
    )


def _float_to_unsigned_integer_conversions() -> tuple[ScalarConversion, ...]:
    return tuple(
        ScalarConversion(
            source_op_key="fptoui",
            descriptor_suffix="convert_f_to_u",
            mnemonic="OpConvertFToU",
            opcode="LOOM_SPIRV_OP_CONVERT_F_TO_U",
            source_type=source_type,
            result_type=scalar_pair.unsigned,
        )
        for source_type in FLOAT_SCALAR_ALU_TYPES
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
    )


def _integer_value_view_conversions() -> tuple[IntegerValueViewConversion, ...]:
    rows: list[IntegerValueViewConversion] = []
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        rows.append(
            IntegerValueViewConversion(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
            )
        )
        rows.append(
            IntegerValueViewConversion(
                source_type=scalar_pair.unsigned,
                result_type=scalar_pair.signed,
            )
        )
    return tuple(rows)


def _bitcast_conversions() -> tuple[ScalarConversion, ...]:
    pairs = (
        ("i16", "f16"),
        ("i16", "bf16"),
        ("i32", "f32"),
        ("i64", "f64"),
    )
    scalar_types = {
        scalar.source_type: scalar
        for scalar in (
            *SIGNED_INTEGER_SCALAR_ALU_TYPES,
            *FLOAT_SCALAR_ALU_TYPES,
            BFLOAT16_SCALAR_TYPE,
        )
    }
    rows: list[ScalarConversion] = []
    for lhs_type, rhs_type in pairs:
        for source_type, result_type in (
            (scalar_types[lhs_type], scalar_types[rhs_type]),
            (scalar_types[rhs_type], scalar_types[lhs_type]),
        ):
            rows.append(
                ScalarConversion(
                    source_op_key="bitcast",
                    descriptor_suffix="bitcast",
                    mnemonic="OpBitcast",
                    opcode="LOOM_SPIRV_OP_BITCAST",
                    source_type=source_type,
                    result_type=result_type,
                )
            )
    return tuple(rows)


SIGNED_INTEGER_WIDTH_CONVERSIONS = _integer_width_conversions()
UNSIGNED_INTEGER_WIDTH_CONVERSIONS = _unsigned_integer_width_conversions()
SCALAR_BITCAST_CONVERSIONS = _bitcast_conversions()

DIRECT_SCALAR_CONVERSIONS = (
    *SIGNED_INTEGER_WIDTH_CONVERSIONS,
    *_float_width_conversions(),
    ScalarConversion(
        source_op_key="extf",
        descriptor_suffix="f_convert",
        mnemonic="OpFConvert",
        opcode="LOOM_SPIRV_OP_F_CONVERT",
        source_type=BFLOAT16_SCALAR_TYPE,
        result_type=_FLOAT32_SCALAR_TYPE,
        feature_atoms=("float32_denorm_preserve",),
    ),
    ScalarConversion(
        source_op_key="fptrunc",
        descriptor_suffix="f_convert",
        mnemonic="OpFConvert",
        opcode="LOOM_SPIRV_OP_F_CONVERT",
        source_type=_FLOAT32_SCALAR_TYPE,
        result_type=BFLOAT16_SCALAR_TYPE,
    ),
    *_signed_integer_to_float_conversions(),
    *_float_to_signed_integer_conversions(),
    *SCALAR_BITCAST_CONVERSIONS,
)

UNSIGNED_SCALAR_CONVERSIONS = (
    *UNSIGNED_INTEGER_WIDTH_CONVERSIONS,
    *_unsigned_integer_to_float_conversions(),
    *_float_to_unsigned_integer_conversions(),
)

INTEGER_VALUE_VIEW_CONVERSIONS = _integer_value_view_conversions()

LOW_SCALAR_CONVERSIONS = (
    *DIRECT_SCALAR_CONVERSIONS,
    *UNSIGNED_SCALAR_CONVERSIONS,
)
