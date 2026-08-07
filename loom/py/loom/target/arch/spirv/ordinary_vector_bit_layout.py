# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for exact SPIR-V ordinary-vector bit layout."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.ordinary_vector import (
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorComponentType,
    OrdinaryVectorInstruction,
    OrdinaryVectorInstructionType,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.scalar_conversion import SCALAR_BITCAST_CONVERSIONS

ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES = (
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "f64",
)
ORDINARY_VECTOR_BIT_LAYOUT_LANE_COUNTS = (
    1,
    *NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
)


@dataclass(frozen=True, slots=True)
class OrdinaryVectorBitLayoutType:
    """One logical source vector and its exact SPIR-V value representation."""

    element_type: str
    lane_count: int
    value_type: OrdinaryVectorInstructionType

    def __post_init__(self) -> None:
        if self.element_type not in ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES:
            raise ValueError(
                f"unsupported bit-layout element type {self.element_type!r}"
            )
        if self.lane_count not in ORDINARY_VECTOR_BIT_LAYOUT_LANE_COUNTS:
            raise ValueError("bit-layout vectors require 1-4 static lanes")
        if isinstance(self.value_type, OrdinaryVectorType):
            if self.lane_count == 1 or self.value_type.lane_count != self.lane_count:
                raise ValueError(
                    "native bit-layout vector representation has wrong lanes"
                )
            component_type = self.value_type.component_type
        else:
            if self.lane_count != 1:
                raise ValueError("only v1 may use a scalar bit-layout representation")
            component_type = self.value_type
        if self.element_type not in component_type.source_types:
            raise ValueError("bit-layout element and SPIR-V component must correspond")

    @property
    def component_type(self) -> OrdinaryVectorComponentType:
        if isinstance(self.value_type, OrdinaryVectorType):
            return self.value_type.component_type
        return self.value_type

    @property
    def suffix(self) -> str:
        return self.value_type.suffix

    @property
    def total_bit_width(self) -> int:
        return self.lane_count * self.component_type.bit_width


@dataclass(frozen=True, slots=True)
class OrdinaryVectorBitLayoutCase:
    """One exact equal-total-bit source/result representation pair."""

    source: OrdinaryVectorBitLayoutType
    result: OrdinaryVectorBitLayoutType

    def __post_init__(self) -> None:
        if self.source.total_bit_width != self.result.total_bit_width:
            raise ValueError("bit-layout casts require equal total bit widths")
        smaller_lane_count = min(
            self.source.lane_count,
            self.result.lane_count,
        )
        larger_lane_count = max(
            self.source.lane_count,
            self.result.lane_count,
        )
        if larger_lane_count % smaller_lane_count != 0:
            raise ValueError("bit-layout component counts must divide exactly")

    @property
    def is_identity(self) -> bool:
        return self.source == self.result

    @property
    def key(self) -> str:
        return f"spirv.op_bitcast.{self.source.suffix}.{self.result.suffix}"

    @property
    def feature_bits(self) -> int:
        return self.source.value_type.feature_bits | self.result.value_type.feature_bits

    @property
    def instruction(self) -> OrdinaryVectorInstruction:
        if self.is_identity:
            raise ValueError("identity bit-layout cases alias their input")
        return OrdinaryVectorInstruction(
            key=self.key,
            mnemonic=(f"OpBitcast.{self.source.suffix}.{self.result.suffix}"),
            opcode="LOOM_SPIRV_OP_BITCAST",
            packet_form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
            result_type=self.result.value_type,
            operand_names=("input",),
            operand_types=(self.source.value_type,),
        )


_SELECTED_COMPONENT_TYPE_PAIRS = tuple(
    (element_type, component_type)
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
    for element_type in component_type.source_types
    if element_type in ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES
)
_COMPONENT_TYPES_BY_ELEMENT_TYPE = dict(_SELECTED_COMPONENT_TYPE_PAIRS)
if len(_COMPONENT_TYPES_BY_ELEMENT_TYPE) != len(_SELECTED_COMPONENT_TYPE_PAIRS) or set(
    _COMPONENT_TYPES_BY_ELEMENT_TYPE
) != set(ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES):
    raise ValueError("bit-layout element types require unique SPIR-V components")

_NATIVE_VECTOR_TYPES_BY_COMPONENT_AND_LANE = {
    (vector_type.component_type, vector_type.lane_count): vector_type
    for vector_type in ORDINARY_VECTOR_TYPES
}
if len(_NATIVE_VECTOR_TYPES_BY_COMPONENT_AND_LANE) != len(ORDINARY_VECTOR_TYPES):
    raise ValueError("ordinary-vector component and lane pairs must be unique")


def _bit_layout_type(
    element_type: str,
    lane_count: int,
) -> OrdinaryVectorBitLayoutType:
    component_type = _COMPONENT_TYPES_BY_ELEMENT_TYPE[element_type]
    value_type: OrdinaryVectorInstructionType = component_type
    if lane_count != 1:
        value_type = _NATIVE_VECTOR_TYPES_BY_COMPONENT_AND_LANE[
            (component_type, lane_count)
        ]
    return OrdinaryVectorBitLayoutType(
        element_type=element_type,
        lane_count=lane_count,
        value_type=value_type,
    )


ORDINARY_VECTOR_BIT_LAYOUT_TYPES = tuple(
    _bit_layout_type(element_type, lane_count)
    for element_type in ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES
    for lane_count in ORDINARY_VECTOR_BIT_LAYOUT_LANE_COUNTS
)

ORDINARY_VECTOR_BIT_LAYOUT_CASES = tuple(
    OrdinaryVectorBitLayoutCase(source, result)
    for source in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
    for result in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
    if source.total_bit_width == result.total_bit_width
)

ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES = tuple(
    case for case in ORDINARY_VECTOR_BIT_LAYOUT_CASES if case.is_identity
)

ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES = tuple(
    case for case in ORDINARY_VECTOR_BIT_LAYOUT_CASES if not case.is_identity
)

_SCALAR_BITCASTS_BY_KEY = {row.key: row for row in SCALAR_BITCAST_CONVERSIONS}
if len(_SCALAR_BITCASTS_BY_KEY) != len(SCALAR_BITCAST_CONVERSIONS):
    raise ValueError("scalar bitcast descriptor keys must be unique")

ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES = tuple(
    case
    for case in ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES
    if case.key in _SCALAR_BITCASTS_BY_KEY
)
if {
    case.key for case in ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES
} != _SCALAR_BITCASTS_BY_KEY.keys():
    raise ValueError("scalar bitcast descriptors must match selected v1 cases")
for case in ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES:
    scalar_row = _SCALAR_BITCASTS_BY_KEY[case.key]
    if (
        scalar_row.display_mnemonic
        != f"OpBitcast.{case.source.suffix}.{case.result.suffix}"
        or scalar_row.feature_bits != case.feature_bits
    ):
        raise ValueError("scalar bitcast descriptor must match its v1 case")

ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTION_CASES = tuple(
    case
    for case in ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES
    if case.key not in _SCALAR_BITCASTS_BY_KEY
)

ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS = tuple(
    case.instruction for case in ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTION_CASES
)
