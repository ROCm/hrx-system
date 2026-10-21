# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native SPIR-V ordinary-vector floating-point arithmetic rows."""

from loom.target.arch.spirv.ordinary_vector import (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
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
