# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth types for core Vulkan SPIR-V ordinary vectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.scalar_alu import INTEGER_SCALAR_ALU_TYPE_PAIRS
from loom.target.arch.spirv.scalar_constant import FLOAT_CONSTANT_TYPES


@unique
class OrdinaryVectorComponentKind(Enum):
    """Scalar semantic class at an ordinary-vector component boundary."""

    BOOLEAN = "boolean"
    SIGNED_INTEGER = "signed_integer"
    UNSIGNED_INTEGER = "unsigned_integer"
    FLOAT = "float"
    OFFSET = "offset"


@dataclass(frozen=True, slots=True)
class OrdinaryVectorComponentType:
    """One target component representation used by the vector contract.

    Public component types name the Loom source types that may select them.
    Target-private unsigned views have no source types and remain outside
    ``ORDINARY_VECTOR_COMPONENT_TYPES``.
    """

    source_types: tuple[str, ...]
    suffix: str
    kind: OrdinaryVectorComponentKind
    scalar_enum: str
    bit_width: int
    feature_atoms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.kind == OrdinaryVectorComponentKind.UNSIGNED_INTEGER
            and self.source_types
        ):
            raise ValueError("unsigned vector views must be source-ineligible")
        if (
            self.kind != OrdinaryVectorComponentKind.UNSIGNED_INTEGER
            and not self.source_types
        ):
            raise ValueError("ordinary-vector component requires a source type")
        if self.bit_width <= 0:
            raise ValueError("ordinary-vector component bit width must be positive")
        if self.kind == OrdinaryVectorComponentKind.BOOLEAN:
            if self.scalar_enum != "LOOM_SPIRV_SCALAR_TYPE_UNKNOWN":
                raise ValueError("boolean vector components are not numeric scalars")
        elif self.scalar_enum == "LOOM_SPIRV_SCALAR_TYPE_UNKNOWN":
            raise ValueError("numeric vector component requires a scalar type")

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)

    @property
    def scalar_value_class(self) -> str:
        if self.kind == OrdinaryVectorComponentKind.BOOLEAN:
            return "LOOM_SPIRV_VALUE_CLASS_BOOL"
        if self.kind == OrdinaryVectorComponentKind.OFFSET:
            return "LOOM_SPIRV_VALUE_CLASS_OFFSET64"
        return "LOOM_SPIRV_VALUE_CLASS_SCALAR"

    @property
    def vector_value_class(self) -> str:
        if self.kind == OrdinaryVectorComponentKind.BOOLEAN:
            return "LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR"
        return "LOOM_SPIRV_VALUE_CLASS_VECTOR"

    @property
    def signed_minimum(self) -> int:
        if self.kind != OrdinaryVectorComponentKind.SIGNED_INTEGER:
            raise ValueError(f"{self.suffix} is not a signed integer component")
        return -(2 ** (self.bit_width - 1))

    @property
    def signed_maximum(self) -> int:
        if self.kind != OrdinaryVectorComponentKind.SIGNED_INTEGER:
            raise ValueError(f"{self.suffix} is not a signed integer component")
        return (2 ** (self.bit_width - 1)) - 1


SIGNED_INTEGER_ORDINARY_VECTOR_COMPONENT_TYPES = tuple(
    OrdinaryVectorComponentType(
        source_types=(
            (scalar_pair.source_type, "index")
            if scalar_pair.source_type == "i32"
            else (scalar_pair.source_type,)
        ),
        suffix=scalar_pair.signed.suffix,
        kind=OrdinaryVectorComponentKind.SIGNED_INTEGER,
        scalar_enum=scalar_pair.signed.scalar_enum,
        bit_width=scalar_pair.bit_width,
        feature_atoms=scalar_pair.signed.feature_atoms,
    )
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
)

INDEX_ORDINARY_VECTOR_COMPONENT_TYPE = next(
    component_type
    for component_type in SIGNED_INTEGER_ORDINARY_VECTOR_COMPONENT_TYPES
    if "index" in component_type.source_types
)

_FLOAT_COMPONENTS = tuple(
    OrdinaryVectorComponentType(
        source_types=(scalar.source_type,),
        suffix=scalar.suffix,
        kind=OrdinaryVectorComponentKind.FLOAT,
        scalar_enum=scalar.scalar_enum,
        bit_width=scalar.bit_width,
        feature_atoms=scalar.feature_atoms,
    )
    for scalar in FLOAT_CONSTANT_TYPES
)

BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE = OrdinaryVectorComponentType(
    source_types=("i1",),
    suffix="bool",
    kind=OrdinaryVectorComponentKind.BOOLEAN,
    scalar_enum="LOOM_SPIRV_SCALAR_TYPE_UNKNOWN",
    bit_width=1,
)

ORDINARY_VECTOR_COMPONENT_TYPES = (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
    *SIGNED_INTEGER_ORDINARY_VECTOR_COMPONENT_TYPES,
    OrdinaryVectorComponentType(
        source_types=("offset",),
        suffix="offset64",
        kind=OrdinaryVectorComponentKind.OFFSET,
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U64",
        bit_width=64,
        feature_atoms=("int64",),
    ),
    *_FLOAT_COMPONENTS,
)

NATIVE_ORDINARY_VECTOR_LANE_COUNTS = (2, 3, 4)


@dataclass(frozen=True, slots=True)
class OrdinaryVectorType:
    """A native ordinary-vector type admitted by the Vulkan core contract."""

    component_type: OrdinaryVectorComponentType
    lane_count: int

    def __post_init__(self) -> None:
        if self.lane_count not in NATIVE_ORDINARY_VECTOR_LANE_COUNTS:
            raise ValueError("core Vulkan SPIR-V ordinary vectors require 2-4 lanes")

    @property
    def suffix(self) -> str:
        return f"v{self.lane_count}{self.component_type.suffix}"

    @property
    def feature_bits(self) -> int:
        return self.component_type.feature_bits


ORDINARY_VECTOR_TYPES = tuple(
    OrdinaryVectorType(component_type=component_type, lane_count=lane_count)
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
    for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
)

type OrdinaryVectorInstructionType = OrdinaryVectorComponentType | OrdinaryVectorType


@dataclass(frozen=True, slots=True)
class OrdinaryVectorInstruction:
    """One descriptor and packet row crossing an ordinary-vector boundary."""

    key: str
    mnemonic: str
    opcode: str
    packet_form: str
    result_type: OrdinaryVectorInstructionType
    operand_names: tuple[str, ...]
    operand_types: tuple[OrdinaryVectorInstructionType, ...]
    feature_atoms: tuple[str, ...] = ()
    component_index_maximum: int | None = None
    # Arithmetic decorations are mutually exclusive under FloatControls2.
    no_contraction: bool = False
    # Preserve special values and evaluation order with FPFastMathMode None.
    fp_fast_math_none: bool = False

    def __post_init__(self) -> None:
        if len(self.operand_names) != len(self.operand_types):
            raise ValueError(
                f"{self.key}: operand names and types must have equal length"
            )
        if self.no_contraction and self.fp_fast_math_none:
            raise ValueError(f"{self.key}: incompatible floating-point decorations")

    @property
    def feature_bits(self) -> int:
        feature_bits = self.result_type.feature_bits | feature_bits_value(
            self.feature_atoms
        )
        for operand_type in self.operand_types:
            feature_bits |= operand_type.feature_bits
        return feature_bits


@dataclass(frozen=True, slots=True)
class OrdinaryVectorShuffleInstruction:
    """One native SPIR-V vector-shuffle shape.

    SPIR-V accepts two vectors with the same component type and permits the
    result to have a different native lane count. Source ``vector.shuffle``
    uses the equal-shape subset and binds the same value to both inputs.
    """

    source0_type: OrdinaryVectorType
    source1_type: OrdinaryVectorType
    result_type: OrdinaryVectorType

    def __post_init__(self) -> None:
        if not (
            self.source0_type.component_type
            == self.source1_type.component_type
            == self.result_type.component_type
        ):
            raise ValueError("vector shuffle component types must match")

    @property
    def key(self) -> str:
        source_suffix = self.source0_type.suffix
        if self.source1_type != self.source0_type:
            source_suffix = f"{source_suffix}.{self.source1_type.suffix}"
        return f"spirv.op_vector_shuffle.{source_suffix}.{self.result_type.suffix}"

    @property
    def mnemonic(self) -> str:
        source_suffix = self.source0_type.suffix
        if self.source1_type != self.source0_type:
            source_suffix = f"{source_suffix}.{self.source1_type.suffix}"
        return f"OpVectorShuffle.{source_suffix}.{self.result_type.suffix}"

    @property
    def opcode(self) -> str:
        return "LOOM_SPIRV_OP_VECTOR_SHUFFLE"

    @property
    def packet_form(self) -> str:
        return "LOOM_SPIRV_PACKET_FORM_VECTOR_SHUFFLE"

    @property
    def selector_names(self) -> tuple[str, ...]:
        return tuple(
            f"selector{lane_index}" for lane_index in range(self.result_type.lane_count)
        )

    @property
    def selector_maximum(self) -> int:
        return self.source0_type.lane_count + self.source1_type.lane_count - 1

    @property
    def feature_bits(self) -> int:
        return (
            self.source0_type.feature_bits
            | self.source1_type.feature_bits
            | self.result_type.feature_bits
        )


@dataclass(frozen=True, slots=True)
class OrdinaryVectorConcatShape:
    """One complete binary partition of a native ordinary-vector result."""

    left_lane_count: int
    right_lane_count: int

    def __post_init__(self) -> None:
        if self.left_lane_count <= 0 or self.right_lane_count <= 0:
            raise ValueError("vector concat inputs must be nonempty")
        if self.result_lane_count not in NATIVE_ORDINARY_VECTOR_LANE_COUNTS:
            raise ValueError("vector concat result must have 2-4 lanes")

    @property
    def result_lane_count(self) -> int:
        return self.left_lane_count + self.right_lane_count

    @property
    def source0_lane_count(self) -> int:
        return max(2, self.left_lane_count)

    @property
    def source1_lane_count(self) -> int:
        return max(2, self.right_lane_count)

    @property
    def selectors(self) -> tuple[int, ...]:
        return (
            *range(self.left_lane_count),
            *(self.source0_lane_count + lane for lane in range(self.right_lane_count)),
        )


def _structural_instructions(
    vector_type: OrdinaryVectorType,
) -> tuple[OrdinaryVectorInstruction, ...]:
    component_type = vector_type.component_type
    component_operands = tuple(
        f"component{lane_index}" for lane_index in range(vector_type.lane_count)
    )
    return (
        OrdinaryVectorInstruction(
            key=f"spirv.op_composite_construct.{vector_type.suffix}",
            mnemonic=f"OpCompositeConstruct.{vector_type.suffix}",
            opcode="LOOM_SPIRV_OP_COMPOSITE_CONSTRUCT",
            packet_form="LOOM_SPIRV_PACKET_FORM_COMPOSITE_CONSTRUCT",
            result_type=vector_type,
            operand_names=component_operands,
            operand_types=(component_type,) * vector_type.lane_count,
        ),
        OrdinaryVectorInstruction(
            key=(
                f"spirv.op_composite_extract.{vector_type.suffix}."
                f"{component_type.suffix}"
            ),
            mnemonic=(
                f"OpCompositeExtract.{vector_type.suffix}.{component_type.suffix}"
            ),
            opcode="LOOM_SPIRV_OP_COMPOSITE_EXTRACT",
            packet_form="LOOM_SPIRV_PACKET_FORM_COMPOSITE_EXTRACT",
            result_type=component_type,
            operand_names=("composite",),
            operand_types=(vector_type,),
            component_index_maximum=vector_type.lane_count - 1,
        ),
        OrdinaryVectorInstruction(
            key=(
                f"spirv.op_composite_insert.{component_type.suffix}."
                f"{vector_type.suffix}"
            ),
            mnemonic=(
                f"OpCompositeInsert.{component_type.suffix}.{vector_type.suffix}"
            ),
            opcode="LOOM_SPIRV_OP_COMPOSITE_INSERT",
            packet_form="LOOM_SPIRV_PACKET_FORM_COMPOSITE_INSERT",
            result_type=vector_type,
            operand_names=("component", "composite"),
            operand_types=(component_type, vector_type),
            component_index_maximum=vector_type.lane_count - 1,
        ),
        OrdinaryVectorInstruction(
            key=(
                f"spirv.op_vector_extract_dynamic.{vector_type.suffix}."
                f"{component_type.suffix}"
            ),
            mnemonic=(
                f"OpVectorExtractDynamic.{vector_type.suffix}.{component_type.suffix}"
            ),
            opcode="LOOM_SPIRV_OP_VECTOR_EXTRACT_DYNAMIC",
            packet_form="LOOM_SPIRV_PACKET_FORM_TYPED",
            result_type=component_type,
            operand_names=("vector", "index"),
            operand_types=(vector_type, INDEX_ORDINARY_VECTOR_COMPONENT_TYPE),
        ),
        OrdinaryVectorInstruction(
            key=(
                f"spirv.op_vector_insert_dynamic.{component_type.suffix}."
                f"{vector_type.suffix}"
            ),
            mnemonic=(
                f"OpVectorInsertDynamic.{component_type.suffix}.{vector_type.suffix}"
            ),
            opcode="LOOM_SPIRV_OP_VECTOR_INSERT_DYNAMIC",
            packet_form="LOOM_SPIRV_PACKET_FORM_TYPED",
            result_type=vector_type,
            operand_names=("vector", "component", "index"),
            operand_types=(
                vector_type,
                component_type,
                INDEX_ORDINARY_VECTOR_COMPONENT_TYPE,
            ),
        ),
        OrdinaryVectorInstruction(
            key=f"spirv.op_select.{vector_type.suffix}",
            mnemonic=f"OpSelect.{vector_type.suffix}",
            opcode="LOOM_SPIRV_OP_SELECT",
            packet_form="LOOM_SPIRV_PACKET_FORM_SELECT",
            result_type=vector_type,
            operand_names=("condition", "true_value", "false_value"),
            operand_types=(
                OrdinaryVectorType(
                    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
                    vector_type.lane_count,
                ),
                vector_type,
                vector_type,
            ),
        ),
    )


ORDINARY_VECTOR_INSTRUCTIONS = tuple(
    instruction
    for vector_type in ORDINARY_VECTOR_TYPES
    for instruction in _structural_instructions(vector_type)
)

ORDINARY_VECTOR_SOURCE_SHUFFLE_INSTRUCTIONS = tuple(
    OrdinaryVectorShuffleInstruction(
        source0_type=vector_type,
        source1_type=vector_type,
        result_type=vector_type,
    )
    for vector_type in ORDINARY_VECTOR_TYPES
)

ORDINARY_VECTOR_CONCAT_SHAPES = tuple(
    OrdinaryVectorConcatShape(left_lane_count, result_lane_count - left_lane_count)
    for result_lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    for left_lane_count in range(1, result_lane_count)
)

_ORDINARY_VECTOR_CONCAT_SHUFFLE_LANE_SHAPES = tuple(
    sorted(
        {
            (
                shape.source0_lane_count,
                shape.source1_lane_count,
                shape.result_lane_count,
            )
            for shape in ORDINARY_VECTOR_CONCAT_SHAPES
            if shape.result_lane_count > 2
        }
    )
)

ORDINARY_VECTOR_CONCAT_SHUFFLE_INSTRUCTIONS = tuple(
    OrdinaryVectorShuffleInstruction(
        source0_type=OrdinaryVectorType(component_type, source0_lane_count),
        source1_type=OrdinaryVectorType(component_type, source1_lane_count),
        result_type=OrdinaryVectorType(component_type, result_lane_count),
    )
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
    for source0_lane_count, source1_lane_count, result_lane_count in (
        _ORDINARY_VECTOR_CONCAT_SHUFFLE_LANE_SHAPES
    )
)

ORDINARY_VECTOR_DEINTERLEAVE_SHUFFLE_INSTRUCTIONS = tuple(
    OrdinaryVectorShuffleInstruction(
        source0_type=OrdinaryVectorType(component_type, 4),
        source1_type=OrdinaryVectorType(component_type, 4),
        result_type=OrdinaryVectorType(component_type, 2),
    )
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
)

ORDINARY_VECTOR_SHUFFLE_INSTRUCTIONS = (
    *ORDINARY_VECTOR_SOURCE_SHUFFLE_INSTRUCTIONS,
    *ORDINARY_VECTOR_CONCAT_SHUFFLE_INSTRUCTIONS,
    *ORDINARY_VECTOR_DEINTERLEAVE_SHUFFLE_INSTRUCTIONS,
)
