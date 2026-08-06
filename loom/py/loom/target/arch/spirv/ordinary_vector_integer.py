# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for native SPIR-V ordinary-vector integer ALU."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.ordinary_vector import (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    SIGNED_INTEGER_ORDINARY_VECTOR_COMPONENT_TYPES,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
    OrdinaryVectorInstruction,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    IntegerAluTypePair,
    IntegerComparePredicate,
    ScalarBinaryOperation,
)


@dataclass(frozen=True, slots=True)
class OrdinaryVectorIntegerComponentTypePair:
    """Signed source representation and target-private unsigned value view."""

    source_type: str
    signed: OrdinaryVectorComponentType
    unsigned: OrdinaryVectorComponentType

    def __post_init__(self) -> None:
        if self.signed.kind != OrdinaryVectorComponentKind.SIGNED_INTEGER:
            raise ValueError("ordinary-vector integer source must be signed")
        if self.unsigned.kind != OrdinaryVectorComponentKind.UNSIGNED_INTEGER:
            raise ValueError("ordinary-vector integer view must be unsigned")
        if self.source_type not in self.signed.source_types:
            raise ValueError("signed vector component does not admit source type")
        if self.unsigned.source_types:
            raise ValueError("unsigned vector view must be source-ineligible")
        if self.signed.bit_width != self.unsigned.bit_width:
            raise ValueError("integer vector value views must have equal widths")
        if self.signed.feature_atoms != self.unsigned.feature_atoms:
            raise ValueError("integer vector value views must require equal features")

    @property
    def feature_bits(self) -> int:
        return self.signed.feature_bits | self.unsigned.feature_bits


def _integer_component_pair(
    scalar_pair: IntegerAluTypePair,
) -> OrdinaryVectorIntegerComponentTypePair:
    matching_signed_components = tuple(
        component
        for component in SIGNED_INTEGER_ORDINARY_VECTOR_COMPONENT_TYPES
        if scalar_pair.source_type in component.source_types
    )
    if len(matching_signed_components) != 1:
        raise ValueError(
            f"{scalar_pair.source_type}: expected one signed vector component"
        )
    return OrdinaryVectorIntegerComponentTypePair(
        source_type=scalar_pair.source_type,
        signed=matching_signed_components[0],
        unsigned=OrdinaryVectorComponentType(
            source_types=(),
            suffix=scalar_pair.unsigned.suffix,
            kind=OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
            scalar_enum=scalar_pair.unsigned.scalar_enum,
            bit_width=scalar_pair.bit_width,
            feature_atoms=scalar_pair.unsigned.feature_atoms,
        ),
    )


ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS = tuple(
    _integer_component_pair(scalar_pair)
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
)


@dataclass(frozen=True, slots=True)
class OrdinaryVectorIntegerTypePair:
    """One native signed vector type and its unsigned target value view."""

    component_pair: OrdinaryVectorIntegerComponentTypePair
    lane_count: int

    def __post_init__(self) -> None:
        if self.lane_count not in NATIVE_ORDINARY_VECTOR_LANE_COUNTS:
            raise ValueError("native integer vectors require 2-4 lanes")

    @property
    def source_type(self) -> str:
        return self.component_pair.source_type

    @property
    def signed(self) -> OrdinaryVectorType:
        return OrdinaryVectorType(self.component_pair.signed, self.lane_count)

    @property
    def unsigned(self) -> OrdinaryVectorType:
        return OrdinaryVectorType(self.component_pair.unsigned, self.lane_count)


ORDINARY_VECTOR_INTEGER_TYPE_PAIRS = tuple(
    OrdinaryVectorIntegerTypePair(
        component_pair=component_pair,
        lane_count=lane_count,
    )
    for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS
    for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
)


def _binary_instruction(
    value_type: OrdinaryVectorType,
    operation: ScalarBinaryOperation,
) -> OrdinaryVectorInstruction:
    return OrdinaryVectorInstruction(
        key=f"spirv.op_{operation.descriptor_suffix}.{value_type.suffix}",
        mnemonic=f"{operation.mnemonic}.{value_type.suffix}",
        opcode=operation.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
        result_type=value_type,
        operand_names=("lhs", "rhs"),
        operand_types=(value_type, value_type),
    )


def _integer_view_instruction(
    source_type: OrdinaryVectorType,
    result_type: OrdinaryVectorType,
) -> OrdinaryVectorInstruction:
    return OrdinaryVectorInstruction(
        key=f"spirv.op_bitcast.{source_type.suffix}.{result_type.suffix}",
        mnemonic=f"OpBitcast.{source_type.suffix}.{result_type.suffix}",
        opcode="LOOM_SPIRV_OP_BITCAST",
        packet_form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
        result_type=result_type,
        operand_names=("input",),
        operand_types=(source_type,),
    )


def _compare_instruction(
    operand_type: OrdinaryVectorType,
    predicate: IntegerComparePredicate,
) -> OrdinaryVectorInstruction:
    result_type = OrdinaryVectorType(
        BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
        operand_type.lane_count,
    )
    return OrdinaryVectorInstruction(
        key=f"spirv.op_{predicate.descriptor_suffix}.{operand_type.suffix}",
        mnemonic=f"{predicate.mnemonic}.{operand_type.suffix}",
        opcode=predicate.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE",
        result_type=result_type,
        operand_names=("lhs", "rhs"),
        operand_types=(operand_type, operand_type),
    )


def _boolean_binary_instruction(
    lane_count: int,
    operation: ScalarBinaryOperation,
) -> OrdinaryVectorInstruction:
    value_type = OrdinaryVectorType(
        BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
        lane_count,
    )
    return OrdinaryVectorInstruction(
        key=f"spirv.op_{operation.descriptor_suffix}.{value_type.suffix}",
        mnemonic=f"{operation.mnemonic}.{value_type.suffix}",
        opcode=operation.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
        result_type=value_type,
        operand_names=("lhs", "rhs"),
        operand_types=(value_type, value_type),
    )


SIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS = tuple(
    _binary_instruction(type_pair.signed, operation)
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    for operation in (
        *SIGNED_INTEGER_BINARY_OPERATIONS,
        *INTEGER_BITWISE_BINARY_OPERATIONS,
    )
)

UNSIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS = tuple(
    _binary_instruction(type_pair.unsigned, operation)
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS
)

ORDINARY_VECTOR_INTEGER_VALUE_VIEW_INSTRUCTIONS = tuple(
    _integer_view_instruction(source_type, result_type)
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    for source_type, result_type in (
        (type_pair.signed, type_pair.unsigned),
        (type_pair.unsigned, type_pair.signed),
    )
)

SIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS = tuple(
    _compare_instruction(type_pair.signed, predicate)
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
)

UNSIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS = tuple(
    _compare_instruction(type_pair.unsigned, predicate)
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
)

ORDINARY_VECTOR_BOOLEAN_BINARY_INSTRUCTIONS = tuple(
    _boolean_binary_instruction(lane_count, operation)
    for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    for operation in BOOLEAN_BINARY_OPERATIONS
)

ORDINARY_VECTOR_INTEGER_INSTRUCTIONS = (
    *SIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS,
    *UNSIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS,
    *ORDINARY_VECTOR_INTEGER_VALUE_VIEW_INSTRUCTIONS,
    *SIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS,
    *UNSIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS,
    *ORDINARY_VECTOR_BOOLEAN_BINARY_INSTRUCTIONS,
)
