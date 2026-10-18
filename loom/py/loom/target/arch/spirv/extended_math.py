# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GLSL.std.450 extended math instructions exposed by the SPIR-V target."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorInstructionType,
    OrdinaryVectorType,
)


@dataclass(frozen=True, slots=True)
class ExtendedMathOperation:
    """One math primitive and its GLSL.std.450 instruction."""

    # Scalar/vector source operation suffix.
    source_op_key: str
    # Source operands consumed by the extended instruction, in wire order.
    operand_names: tuple[str, ...]
    # GLSL.std.450 instruction spelling used in assembly.
    instruction_name: str
    # Wire instruction enumerant owned by extended_instruction.h.
    instruction_c_enum: str
    # Source permissions required by this native approximation.
    required_fastmath_flags: tuple[str, ...] = ()
    # Element formats admitted by this GLSL instruction.
    source_types: tuple[str, ...] = ("f32",)
    # Whether exact selection depends on the operand width's denormal mode.
    preserve_denormals: bool = False


@dataclass(frozen=True, slots=True)
class ExtendedMathInstruction:
    """One typed descriptor and packet emitted through OpExtInst."""

    # Source semantics and extended-instruction identity.
    operation: ExtendedMathOperation
    # Native scalar or vector carrier for this descriptor.
    value_type: OrdinaryVectorInstructionType

    @property
    def descriptor_key(self) -> str:
        return (
            "spirv.op_ext_inst.glsl_std_450."
            f"{self.operation.source_op_key}.{self.value_type.suffix}"
        )

    @property
    def mnemonic(self) -> str:
        return (
            "OpExtInst.GLSL.std.450."
            f"{self.operation.instruction_name}.{self.value_type.suffix}"
        )

    @property
    def feature_bits(self) -> int:
        bits = self.value_type.feature_bits
        if self.operation.preserve_denormals:
            component = (
                self.value_type.component_type
                if isinstance(self.value_type, OrdinaryVectorType)
                else self.value_type
            )
            bits |= feature_bits_value(
                (
                    "float32_denorm_preserve"
                    if component.bit_width == 32
                    else "float16_64_denorm_preserve",
                )
            )
        return bits


EXTENDED_MATH_OPERATIONS = (
    ExtendedMathOperation(
        source_op_key="sinf",
        operand_names=("input",),
        instruction_name="Sin",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_SIN",
        required_fastmath_flags=("afn",),
        source_types=("f16", "f32"),
    ),
    ExtendedMathOperation(
        source_op_key="cosf",
        operand_names=("input",),
        instruction_name="Cos",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_COS",
        required_fastmath_flags=("afn",),
        source_types=("f16", "f32"),
    ),
    ExtendedMathOperation(
        source_op_key="expf",
        operand_names=("input",),
        instruction_name="Exp",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_EXP",
        required_fastmath_flags=("afn",),
    ),
    ExtendedMathOperation(
        source_op_key="logf",
        operand_names=("input",),
        instruction_name="Log",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_LOG",
        required_fastmath_flags=("afn",),
    ),
    ExtendedMathOperation(
        source_op_key="exp2f",
        operand_names=("input",),
        instruction_name="Exp2",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_EXP2",
        required_fastmath_flags=("afn",),
    ),
    ExtendedMathOperation(
        source_op_key="log2f",
        operand_names=("input",),
        instruction_name="Log2",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_LOG2",
        required_fastmath_flags=("afn",),
    ),
    ExtendedMathOperation(
        source_op_key="sqrtf",
        operand_names=("input",),
        instruction_name="Sqrt",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_SQRT",
        required_fastmath_flags=("afn",),
        source_types=("f16", "f32", "f64"),
    ),
    ExtendedMathOperation(
        source_op_key="rsqrtf",
        operand_names=("input",),
        instruction_name="InverseSqrt",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_INVERSE_SQRT",
        required_fastmath_flags=("afn",),
        source_types=("f16", "f32", "f64"),
    ),
)

EXTREMA_MATH_OPERATIONS = (
    ExtendedMathOperation(
        source_op_key="minnumf",
        operand_names=("lhs", "rhs"),
        instruction_name="NMin",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_NMIN",
        required_fastmath_flags=("nnan", "ninf", "nsz"),
        source_types=("f16", "f32", "f64"),
        preserve_denormals=True,
    ),
    ExtendedMathOperation(
        source_op_key="maxnumf",
        operand_names=("lhs", "rhs"),
        instruction_name="NMax",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_NMAX",
        required_fastmath_flags=("nnan", "ninf", "nsz"),
        source_types=("f16", "f32", "f64"),
        preserve_denormals=True,
    ),
)

ROUNDING_MATH_OPERATIONS = tuple(
    ExtendedMathOperation(
        source_op_key=source_op_key,
        operand_names=("input",),
        instruction_name=instruction_name,
        instruction_c_enum=f"LOOM_SPIRV_GLSL_STD_450_{instruction_enum}",
        source_types=("f16", "f32", "f64"),
    )
    for source_op_key, instruction_name, instruction_enum in (
        ("ceilf", "Ceil", "CEIL"),
        ("floorf", "Floor", "FLOOR"),
        ("roundevenf", "RoundEven", "ROUND_EVEN"),
        ("truncf", "Trunc", "TRUNC"),
    )
)

INTEGER_MATH_OPERATIONS = (
    ExtendedMathOperation(
        source_op_key="find_umsb",
        operand_names=("input",),
        instruction_name="FindUMsb",
        instruction_c_enum="LOOM_SPIRV_GLSL_STD_450_FIND_UMSB",
        source_types=("i32",),
    ),
)

EXTENDED_MATH_INSTRUCTIONS = tuple(
    ExtendedMathInstruction(operation=operation, value_type=value_type)
    for operation in (
        *EXTENDED_MATH_OPERATIONS,
        *EXTREMA_MATH_OPERATIONS,
        *ROUNDING_MATH_OPERATIONS,
        *INTEGER_MATH_OPERATIONS,
    )
    for value_type in (*ORDINARY_VECTOR_COMPONENT_TYPES, *ORDINARY_VECTOR_TYPES)
    if (
        value_type.component_type.source_types
        if isinstance(value_type, OrdinaryVectorType)
        else value_type.source_types
    )[0]
    in operation.source_types
)
