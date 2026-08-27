# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for native SPIR-V vector integer conversions."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.ordinary_vector import (
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    OrdinaryVectorInstruction,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_TYPE_PAIRS,
)
from loom.target.arch.spirv.scalar_alu import ScalarAluType
from loom.target.arch.spirv.scalar_conversion import (
    SIGNED_INTEGER_WIDTH_CONVERSIONS,
    UNSIGNED_INTEGER_WIDTH_CONVERSIONS,
    ScalarConversion,
)


@dataclass(frozen=True, slots=True)
class OrdinaryVectorIntegerConversion:
    """One scalar integer conversion lifted to a native vector type pair."""

    scalar_conversion: ScalarConversion
    source_type: OrdinaryVectorType
    result_type: OrdinaryVectorType

    def __post_init__(self) -> None:
        if self.source_type.lane_count != self.result_type.lane_count:
            raise ValueError("ordinary-vector conversion lane counts must match")
        for scalar_type, vector_type in (
            (self.scalar_conversion.source_type, self.source_type),
            (self.scalar_conversion.result_type, self.result_type),
        ):
            component_type = vector_type.component_type
            if component_type.suffix != scalar_type.suffix:
                raise ValueError("scalar and vector conversion types must correspond")
            if component_type.scalar_enum != scalar_type.scalar_enum:
                raise ValueError("scalar and vector conversion encodings must match")
            if component_type.feature_atoms != scalar_type.feature_atoms:
                raise ValueError("scalar and vector conversion features must match")

    @property
    def instruction(self) -> OrdinaryVectorInstruction:
        scalar_conversion = self.scalar_conversion
        return OrdinaryVectorInstruction(
            key=(
                f"spirv.op_{scalar_conversion.descriptor_suffix}."
                f"{self.source_type.suffix}.{self.result_type.suffix}"
            ),
            mnemonic=(
                f"{scalar_conversion.mnemonic}."
                f"{self.source_type.suffix}.{self.result_type.suffix}"
            ),
            opcode=scalar_conversion.opcode,
            packet_form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
            result_type=self.result_type,
            operand_names=("input",),
            operand_types=(self.source_type,),
        )


_INTEGER_VECTOR_TYPES_BY_SCALAR_SUFFIX_AND_LANE = {
    (vector_type.component_type.suffix, type_pair.lane_count): vector_type
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    for vector_type in (type_pair.signed, type_pair.unsigned)
}
if len(_INTEGER_VECTOR_TYPES_BY_SCALAR_SUFFIX_AND_LANE) != (
    2 * len(ORDINARY_VECTOR_INTEGER_TYPE_PAIRS)
):
    raise ValueError("integer vector scalar suffix and lane pairs must be unique")


def _integer_vector_type(
    scalar_type: ScalarAluType,
    lane_count: int,
) -> OrdinaryVectorType:
    vector_type = _INTEGER_VECTOR_TYPES_BY_SCALAR_SUFFIX_AND_LANE.get(
        (scalar_type.suffix, lane_count)
    )
    if vector_type is None:
        raise ValueError(
            f"{scalar_type.suffix}: missing v{lane_count} integer vector type"
        )
    return vector_type


def _lift_integer_conversion(
    scalar_conversion: ScalarConversion,
) -> tuple[OrdinaryVectorIntegerConversion, ...]:
    return tuple(
        OrdinaryVectorIntegerConversion(
            scalar_conversion=scalar_conversion,
            source_type=_integer_vector_type(
                scalar_conversion.source_type,
                lane_count,
            ),
            result_type=_integer_vector_type(
                scalar_conversion.result_type,
                lane_count,
            ),
        )
        for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    )


SIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS = tuple(
    vector_conversion
    for scalar_conversion in SIGNED_INTEGER_WIDTH_CONVERSIONS
    for vector_conversion in _lift_integer_conversion(scalar_conversion)
)

UNSIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS = tuple(
    vector_conversion
    for scalar_conversion in UNSIGNED_INTEGER_WIDTH_CONVERSIONS
    for vector_conversion in _lift_integer_conversion(scalar_conversion)
)

ORDINARY_VECTOR_INTEGER_CONVERSIONS = (
    *SIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS,
    *UNSIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS,
)

ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS = tuple(
    conversion.instruction for conversion in ORDINARY_VECTOR_INTEGER_CONVERSIONS
)
