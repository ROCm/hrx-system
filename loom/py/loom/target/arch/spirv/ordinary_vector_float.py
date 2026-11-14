# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native SPIR-V ordinary-vector floating-point arithmetic rows."""

from dataclasses import dataclass

from loom.target.arch.spirv.ordinary_vector import (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorInstruction,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.scalar_alu import (
    FLOAT_BINARY_OPERATIONS,
    FLOAT_COMPARE_PREDICATES,
    FLOAT_COMPARE_TYPES,
    FLOAT_SCALAR_ALU_TYPES,
    ORDERED_CLAMP_COMPARE_PREDICATES,
    float_preservation_feature_atoms,
)

_FLOAT_SUFFIXES = frozenset(scalar.suffix for scalar in FLOAT_SCALAR_ALU_TYPES)


@dataclass(frozen=True, slots=True)
class FusedMultiplyAddInstruction(OrdinaryVectorInstruction):
    """One-rounding arithmetic with explicit source permission requirements."""

    # Source permissions needed for this row's floating-point environment.
    required_fastmath_flags: tuple[str, ...] = ()


def _fused_multiply_add_instructions() -> tuple[FusedMultiplyAddInstruction, ...]:
    rows = []
    for value_type in (*ORDINARY_VECTOR_COMPONENT_TYPES, *ORDINARY_VECTOR_TYPES):
        scalar = (
            value_type.component_type
            if isinstance(value_type, OrdinaryVectorType)
            else value_type
        )
        if scalar.suffix not in _FLOAT_SUFFIXES:
            continue
        for suffix, permissions in (
            ("", ()),
            (".afn", ("afn",)),
            (".finite", ("nnan", "ninf", "nsz")),
            (".finite_afn", ("nnan", "ninf", "nsz", "afn")),
        ):
            preserve_special = "nsz" not in permissions
            features = (
                f"fma_float{scalar.bit_width}",
                f"float{scalar.bit_width}_rounding_rte",
            )
            if "afn" not in permissions:
                features += (f"float{scalar.bit_width}_denorm_preserve",)
            if preserve_special:
                # The legacy SignedZeroInfNanPreserve execution mode does not
                # cover OpFmaKHR. Per-instruction controls preserve these values.
                features += (
                    "float_controls2",
                    f"float{scalar.bit_width}_signed_zero_inf_nan_preserve",
                )
            rows.append(
                FusedMultiplyAddInstruction(
                    key=f"spirv.op_fma{suffix}.{value_type.suffix}",
                    mnemonic=f"OpFmaKHR{suffix}.{value_type.suffix}",
                    opcode="LOOM_SPIRV_OP_FMA_KHR",
                    packet_form="LOOM_SPIRV_PACKET_FORM_TYPED",
                    result_type=value_type,
                    operand_names=("a", "b", "c"),
                    operand_types=(value_type, value_type, value_type),
                    feature_atoms=features,
                    no_contraction=not preserve_special,
                    fp_fast_math_none=preserve_special,
                    required_fastmath_flags=permissions,
                )
            )
    return tuple(rows)


FUSED_MULTIPLY_ADD_INSTRUCTIONS = _fused_multiply_add_instructions()

ORDINARY_VECTOR_FLOAT_BINARY_INSTRUCTIONS = tuple(
    OrdinaryVectorInstruction(
        key=f"spirv.op_{operation.descriptor_suffix}.{value_type.suffix}",
        mnemonic=f"{operation.mnemonic}.{value_type.suffix}",
        opcode=operation.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_BINARY_LHS_TYPE",
        result_type=value_type,
        operand_names=("lhs", "rhs"),
        operand_types=(value_type, value_type),
    )
    for value_type in ORDINARY_VECTOR_TYPES
    if value_type.component_type.suffix in _FLOAT_SUFFIXES
    for operation in FLOAT_BINARY_OPERATIONS
)

ORDINARY_VECTOR_FLOAT_SCALE_INSTRUCTIONS = tuple(
    OrdinaryVectorInstruction(
        key=f"spirv.op_vector_times_scalar.{value_type.suffix}",
        mnemonic=f"OpVectorTimesScalar.{value_type.suffix}",
        opcode="LOOM_SPIRV_OP_VECTOR_TIMES_SCALAR",
        packet_form="LOOM_SPIRV_PACKET_FORM_BINARY_LHS_TYPE",
        result_type=value_type,
        operand_names=("vector", "scalar"),
        operand_types=(value_type, value_type.component_type),
    )
    for value_type in ORDINARY_VECTOR_TYPES
    if value_type.component_type.suffix in _FLOAT_SUFFIXES
)

ORDINARY_VECTOR_FLOAT_COMPARE_INSTRUCTIONS = tuple(
    OrdinaryVectorInstruction(
        key=f"spirv.op_{predicate.descriptor_suffix}.{value_type.suffix}",
        mnemonic=f"{predicate.mnemonic}.{value_type.suffix}",
        opcode=predicate.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE",
        result_type=OrdinaryVectorType(
            BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
            value_type.lane_count,
        ),
        operand_names=("lhs", "rhs"),
        operand_types=(value_type, value_type),
    )
    for scalar in FLOAT_COMPARE_TYPES
    for value_type in ORDINARY_VECTOR_TYPES
    if value_type.component_type.suffix == scalar.suffix
    for predicate in FLOAT_COMPARE_PREDICATES
)

ORDINARY_VECTOR_FLOAT_PRESERVE_COMPARE_INSTRUCTIONS = tuple(
    OrdinaryVectorInstruction(
        key=(f"spirv.op_{predicate.descriptor_suffix}.preserve.{value_type.suffix}"),
        mnemonic=f"{predicate.mnemonic}.preserve.{value_type.suffix}",
        opcode=predicate.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE",
        result_type=OrdinaryVectorType(
            BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
            value_type.lane_count,
        ),
        operand_names=("lhs", "rhs"),
        operand_types=(value_type, value_type),
        feature_atoms=float_preservation_feature_atoms(scalar),
    )
    for scalar in FLOAT_COMPARE_TYPES
    for value_type in ORDINARY_VECTOR_TYPES
    if value_type.component_type.suffix == scalar.suffix
    for predicate in ORDERED_CLAMP_COMPARE_PREDICATES
)
