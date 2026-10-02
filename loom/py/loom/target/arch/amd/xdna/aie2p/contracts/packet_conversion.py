# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native AIE2P packet-conversion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from loom.dialect.encoding import defs as encoding
from loom.dialect.vector import defs as vector
from loom.dsl import EncodingOperandSummaryDef, Op
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
    I8_INTERLEAVE_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    SourceNode,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

# Source-visible lane counts supported by native BF16/F32 packet conversion.
FLOAT_PACKET_LANE_COUNTS = (16, 32)

# Packed i4 byte counts consumed by native VUNPACK forms. Each input byte
# produces two sign- or zero-extended i8 lanes.
I4_UNPACK_SOURCE_LANE_COUNTS = (32, 64)


@dataclass(frozen=True, slots=True)
class IntegerWidenInstruction:
    """One physical vector shape consumed and produced by a VUPS form."""

    # Source vector element type.
    input_element: str
    # Result vector element type.
    result_element: str
    # Native lane count converted by the instruction.
    native_lane_count: int

    @property
    def memory_width_bits(self) -> int:
        """Number of source bits consumed by a fused widening load."""

        return self.native_lane_count * int(self.input_element[1:])

    @property
    def result_width_bits(self) -> int:
        """Number of result bits produced by the widening operation."""

        return self.native_lane_count * int(self.result_element[1:])

    @property
    def accumulator_unit_count(self) -> int:
        """Number of 512-bit accumulator units produced by VUPS."""

        return self.result_width_bits // 512

    @property
    def physical_shape(self) -> str:
        """Physical VUPS width relation encoded by the instruction."""

        widening_factor = self.result_width_bits // self.memory_width_bits
        source_carrier = "w" if self.memory_width_bits == 256 else "x"
        result_carrier = {1: "b", 2: "c", 4: "d"}[self.accumulator_unit_count]
        return f"{widening_factor}x.{source_carrier}-to-{result_carrier}"

    @property
    def ups_mode(self) -> int:
        """AIE2P crUpsMode value selecting the result element width."""

        return int(self.result_element[1:]) // 32 - 1

    @property
    def slice_input(self) -> bool:
        """Whether the W source is sliced from its ordinary X carrier."""

        return self.memory_width_bits == 256

    @property
    def direct_accumulator_result(self) -> bool:
        """Whether the result remains natively in the accumulator file."""

        return self.accumulator_unit_count == 4


@dataclass(frozen=True, slots=True)
class IntegerWidenRuleShape:
    """Logical lane interval realized by one physical VUPS form."""

    # Physical instruction shape used for the conversion.
    instruction: IntegerWidenInstruction
    # First logical lane count realized by this rule.
    minimum_lane_count: int
    # Last logical lane count realized by this rule.
    maximum_lane_count: int
    # Number of 512-bit result units containing logical lanes.
    result_accumulator_unit_count: int

    def __post_init__(self) -> None:
        if not (
            1
            <= self.minimum_lane_count
            <= self.maximum_lane_count
            <= self.instruction.native_lane_count
        ):
            raise ValueError("integer widening logical lane interval is invalid")
        if not (
            1
            <= self.result_accumulator_unit_count
            <= self.instruction.accumulator_unit_count
        ):
            raise ValueError("integer widening result unit count is invalid")
        result_lanes_per_unit = 512 // int(self.instruction.result_element[1:])
        if (
            self.minimum_lane_count
            <= (self.result_accumulator_unit_count - 1) * result_lanes_per_unit
            or self.maximum_lane_count
            > self.result_accumulator_unit_count * result_lanes_per_unit
        ):
            raise ValueError(
                "integer widening logical interval crosses a result carrier boundary"
            )

    @property
    def input_type(self) -> Vector:
        """Source-visible input type interval."""

        return Vector(
            self.instruction.input_element,
            minimum_lanes=self.minimum_lane_count,
            maximum_lanes=self.maximum_lane_count,
        )

    @property
    def result_type(self) -> Vector:
        """Source-visible result type interval."""

        return Vector(
            self.instruction.result_element,
            minimum_lanes=self.minimum_lane_count,
            maximum_lanes=self.maximum_lane_count,
        )

    def report_key(self, signedness: str) -> str:
        """Stable compile-report key for this logical interval."""

        lane_range = (
            str(self.minimum_lane_count)
            if self.minimum_lane_count == self.maximum_lane_count
            else f"{self.minimum_lane_count}-{self.maximum_lane_count}"
        )
        return (
            f"native_{signedness}_{self.instruction.input_element}x{lane_range}_to_"
            f"{self.instruction.result_element}x{lane_range}"
        )


@dataclass(frozen=True, slots=True)
class FloatPacketRuleShape:
    """Logical lane interval realized by one physical float packet."""

    # Native lane count converted by the physical instruction.
    native_lane_count: int
    # First logical lane count realized by this rule.
    minimum_lane_count: int
    # Last logical lane count realized by this rule.
    maximum_lane_count: int

    def __post_init__(self) -> None:
        if not (
            self.native_lane_count in FLOAT_PACKET_LANE_COUNTS
            and 1
            <= self.minimum_lane_count
            <= self.maximum_lane_count
            <= self.native_lane_count
        ):
            raise ValueError("float packet logical lane interval is invalid")

    def vector_type(self, element: str) -> Vector:
        """Source-visible type interval for one side of the conversion."""

        return Vector(
            element,
            minimum_lanes=self.minimum_lane_count,
            maximum_lanes=self.maximum_lane_count,
        )

    @property
    def report_lane_range(self) -> str:
        """Stable logical lane spelling used by compile reports."""

        if self.minimum_lane_count == self.maximum_lane_count:
            return str(self.minimum_lane_count)
        return f"{self.minimum_lane_count}-{self.maximum_lane_count}"


@dataclass(frozen=True, slots=True)
class Float8PacketFormat:
    """One float8 encoding handled through native integer packets."""

    # Source vector element type.
    element: str
    # Number of explicit source significand bits.
    mantissa_bits: int
    # Source exponent bias.
    exponent_bias: int
    # Whether the maximum exponent with a zero mantissa encodes infinity.
    has_infinity: bool

    @property
    def normal_shift(self) -> int:
        """Left shift positioning a normal payload as BF16 bits."""

        return 7 - self.mantissa_bits

    @property
    def normal_base(self) -> int:
        """BF16 exponent contribution independent of the source payload."""

        return (127 - self.exponent_bias) << 7

    @property
    def special_payload(self) -> int:
        """First unsigned payload carrying the maximum source exponent."""

        exponent_bits = 7 - self.mantissa_bits
        return ((1 << exponent_bits) - 1) << self.mantissa_bits

    @property
    def report_name(self) -> str:
        """Stable lowercase type spelling used by compile reports."""

        return self.element.lower()

    @property
    def minimum_normal_exponent(self) -> int:
        """Smallest unbiased exponent represented as a normal value."""

        return 1 - self.exponent_bias

    @property
    def minimum_rounding_exponent(self) -> int:
        """Smallest source exponent that can round to a nonzero value."""

        return self.minimum_normal_exponent - self.mantissa_bits - 1

    @property
    def nan_payload(self) -> int:
        """Canonical destination NaN magnitude."""

        return 0x7F

    @property
    def finite_clamp(self) -> int:
        """Largest magnitude produced by finite-source rounding."""

        return self.special_payload if self.has_infinity else self.nan_payload - 1


@dataclass(frozen=True, slots=True)
class _Float8WidenProgram:
    """Reusable values emitted while widening one FP8 packet."""

    # Widened BF16 packet.
    result: ValueRef
    # Canonical BF16 NaN packet.
    canonical_nan: ValueRef
    # E4M3FN NaN predicate, or None for formats with infinity encodings.
    is_nan: ValueRef | None
    # Ordered descriptor program defining the reusable values.
    emits: tuple[ContractEmit, ...]


@dataclass(frozen=True, slots=True)
class FloatPacketSourceFormat:
    """One floating-point encoding narrowed through native i32 packets."""

    # Source vector element type.
    element: str
    # Number of explicit source significand bits.
    mantissa_bits: int
    # Source exponent bias.
    exponent_bias: int
    # Number of source exponent bits.
    exponent_bits: int

    @property
    def bit_width(self) -> int:
        """Total source element width."""

        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def sign_bit(self) -> int:
        """Source sign bit."""

        return 1 << (self.bit_width - 1)

    @property
    def nonsign_mask(self) -> int:
        """Mask retaining the source exponent and fraction."""

        return self.sign_bit - 1

    @property
    def fraction_mask(self) -> int:
        """Mask retaining the explicit source significand bits."""

        return (1 << self.mantissa_bits) - 1

    @property
    def hidden_bit(self) -> int:
        """Implicit leading bit of a normal source significand."""

        return 1 << self.mantissa_bits

    @property
    def infinity_bits(self) -> int:
        """Unsigned source infinity encoding."""

        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def report_name(self) -> str:
        """Stable source type spelling used by compile reports."""

        return {
            "f16": "binary16",
            "bf16": "bfloat16",
            "f32": "binary32",
        }[self.element]

    def exponent_bits_for(self, unbiased_exponent: int) -> int:
        """Returns the normal source encoding for an unbiased exponent."""

        return (unbiased_exponent + self.exponent_bias) << self.mantissa_bits


@dataclass(frozen=True, slots=True)
class IntegerShiftRuleShape:
    """Logical lane interval realized by one physical i32 shift packet."""

    # Native lane count shifted by the physical instruction sequence.
    native_lane_count: int
    # First logical lane count realized by this rule.
    minimum_lane_count: int
    # Last logical lane count realized by this rule.
    maximum_lane_count: int

    def __post_init__(self) -> None:
        if not (
            self.native_lane_count == 16
            and 1
            <= self.minimum_lane_count
            <= self.maximum_lane_count
            <= self.native_lane_count
        ):
            raise ValueError("integer shift logical lane interval is invalid")

    @property
    def vector_type(self) -> Vector:
        """Source-visible packet type interval."""

        if self.minimum_lane_count == self.maximum_lane_count:
            return Vector("i32", lanes=self.minimum_lane_count)
        return Vector(
            "i32",
            minimum_lanes=self.minimum_lane_count,
            maximum_lanes=self.maximum_lane_count,
        )

    @property
    def report_lane_range(self) -> str:
        """Stable logical lane spelling used by compile reports."""

        if self.minimum_lane_count == self.maximum_lane_count:
            return str(self.minimum_lane_count)
        return f"{self.minimum_lane_count}-{self.maximum_lane_count}"


@dataclass(frozen=True, slots=True)
class IntegerPackInstruction:
    """One exact physical shape supported by a native VPACK form."""

    # Source vector element type.
    input_element: str
    # Number of source lanes packed by the conversion.
    native_lane_count: int
    # Source-visible result vector element type.
    result_element: str
    # Required vector.bitpack width, or None for vector.trunci.
    bit_width: int | None

    def __post_init__(self) -> None:
        input_bits = int(self.input_element[1:])
        result_bits = int(self.result_element[1:])
        if input_bits != self.output_element_bits * 2:
            raise ValueError("integer pack must narrow each source lane by two")
        if self.native_lane_count * self.output_element_bits % result_bits:
            raise ValueError("integer pack result does not fill whole storage lanes")
        if self.memory_width_bits not in (256, 512):
            raise ValueError("integer pack must produce one native W or X carrier")
        if self.pack_size not in (0, 1):
            raise ValueError("integer pack size must fit the one-bit crPackSize field")

    @property
    def source_op(self) -> Op:
        """Source conversion operation selecting the pack semantics."""

        return vector.vector_trunci if self.bit_width is None else vector.vector_bitpack

    @property
    def source_field(self) -> str:
        """Source operand field consumed by the conversion."""

        return "input" if self.bit_width is None else "source"

    @property
    def output_element_bits(self) -> int:
        """Logical result element width packed by VPACK."""

        return (
            int(self.result_element[1:]) if self.bit_width is None else self.bit_width
        )

    @property
    def result_lanes(self) -> int:
        """Number of source-visible storage lanes carrying the packed result."""

        return (
            self.native_lane_count
            * self.output_element_bits
            // int(self.result_element[1:])
        )

    @property
    def memory_width_bits(self) -> int:
        """Number of result bits written by a fused packing store."""

        return self.result_lanes * int(self.result_element[1:])

    @property
    def physical_width(self) -> str:
        """Physical VPACK result carrier width."""

        return {256: "w", 512: "x"}[self.memory_width_bits]

    @property
    def pack_size(self) -> int:
        """AIE2P crPackSize value selecting the result element width."""

        return self.output_element_bits.bit_length() - 3

    @property
    def report_key(self) -> str:
        """Stable compile-report key for the standalone conversion."""

        if self.bit_width is None:
            return (
                f"native_trunc_{self.input_element}x{self.native_lane_count}_to_"
                f"{self.result_element}x{self.result_lanes}"
            )
        return (
            f"native_bitpack_{self.input_element}x{self.native_lane_count}_to_"
            f"i{self.bit_width}x{self.native_lane_count}"
        )

    @property
    def pad_result(self) -> bool:
        """Whether the result preserves an unused X-carrier half."""

        return self.memory_width_bits == 256


@dataclass(frozen=True, slots=True)
class IntegerPackRuleShape:
    """Logical lane interval realized by one physical VPACK form."""

    # Physical instruction shape used for the conversion.
    instruction: IntegerPackInstruction
    # First logical source lane count realized by this rule.
    minimum_lane_count: int
    # Last logical source lane count realized by this rule.
    maximum_lane_count: int

    def __post_init__(self) -> None:
        if not (
            1
            <= self.minimum_lane_count
            <= self.maximum_lane_count
            <= self.instruction.native_lane_count
        ):
            raise ValueError("integer pack logical lane interval is invalid")
        if self.instruction.bit_width is not None and (
            self.minimum_lane_count != self.instruction.native_lane_count
            or self.maximum_lane_count != self.instruction.native_lane_count
        ):
            raise ValueError("packed sub-byte results require an exact lane count")

    @staticmethod
    def _vector_type(element: str, minimum_lanes: int, maximum_lanes: int) -> Vector:
        if minimum_lanes == maximum_lanes:
            return Vector(element, lanes=minimum_lanes)
        return Vector(
            element,
            minimum_lanes=minimum_lanes,
            maximum_lanes=maximum_lanes,
        )

    @property
    def input_type(self) -> Vector:
        """Source-visible input type interval."""

        return self._vector_type(
            self.instruction.input_element,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )

    @property
    def result_type(self) -> Vector:
        """Source-visible result type interval."""

        if self.instruction.bit_width is not None:
            return Vector(
                self.instruction.result_element,
                lanes=self.instruction.result_lanes,
            )
        return self._vector_type(
            self.instruction.result_element,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )

    @property
    def report_key(self) -> str:
        """Stable compile-report key for this logical interval."""

        if (
            self.minimum_lane_count == self.instruction.native_lane_count
            and self.maximum_lane_count == self.instruction.native_lane_count
        ):
            return self.instruction.report_key
        lane_range = f"{self.minimum_lane_count}-{self.maximum_lane_count}"
        return (
            f"native_trunc_{self.instruction.input_element}x{lane_range}_to_"
            f"{self.instruction.result_element}x{lane_range}"
        )


@dataclass(frozen=True, slots=True)
class IntegerTruncationInstruction:
    """One integer lane-width relation implemented by native VSHUFFLEs."""

    # Source vector element type.
    input_element: str
    # Result vector element type.
    result_element: str

    def __post_init__(self) -> None:
        input_bits = int(self.input_element[1:])
        result_bits = int(self.result_element[1:])
        if (
            input_bits not in (32, 64)
            or result_bits not in (8, 16, 32)
            or input_bits <= result_bits
        ):
            raise ValueError(
                "shuffle truncation requires a supported narrowing width relation"
            )

    @property
    def shuffle_controls(self) -> tuple[int, ...]:
        """Even-sublane filters applied from widest to narrowest."""

        result_bits = int(self.result_element[1:])
        sublane_bits = int(self.input_element[1:]) // 2
        controls: list[int] = []
        while sublane_bits >= result_bits:
            controls.append({8: 0, 16: 2, 32: 4}[sublane_bits])
            sublane_bits //= 2
        return tuple(controls)

    @property
    def native_lane_count(self) -> int:
        """Number of source lanes carried by one native X register."""

        return 512 // int(self.input_element[1:])


@dataclass(frozen=True, slots=True)
class IntegerTruncationRuleShape:
    """Logical lane interval realized by one VSHUFFLE truncation program."""

    # Lane-width relation and shuffle sequence used by the conversion.
    instruction: IntegerTruncationInstruction
    # First logical lane count realized by this rule.
    minimum_lane_count: int
    # Last logical lane count realized by this rule.
    maximum_lane_count: int

    def __post_init__(self) -> None:
        native_lane_count = self.instruction.native_lane_count
        if not (
            1
            <= self.minimum_lane_count
            <= self.maximum_lane_count
            <= native_lane_count * 2
        ):
            raise ValueError("integer truncation logical lane interval is invalid")
        if self.minimum_lane_count <= native_lane_count < self.maximum_lane_count:
            raise ValueError("integer truncation interval crosses a carrier boundary")

    @staticmethod
    def _vector_type(element: str, minimum_lanes: int, maximum_lanes: int) -> Vector:
        if minimum_lanes == maximum_lanes:
            return Vector(element, lanes=minimum_lanes)
        return Vector(
            element,
            minimum_lanes=minimum_lanes,
            maximum_lanes=maximum_lanes,
        )

    @property
    def input_type(self) -> Vector:
        """Source-visible input type interval."""

        return self._vector_type(
            self.instruction.input_element,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )

    @property
    def result_type(self) -> Vector:
        """Source-visible result type interval."""

        return self._vector_type(
            self.instruction.result_element,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )

    @property
    def source_carrier_count(self) -> int:
        """Number of source X registers consumed by the first shuffle."""

        return int(self.maximum_lane_count > self.instruction.native_lane_count) + 1

    @property
    def report_key(self) -> str:
        """Stable compile-report key for this logical interval."""

        lane_range = (
            str(self.minimum_lane_count)
            if self.minimum_lane_count == self.maximum_lane_count
            else f"{self.minimum_lane_count}-{self.maximum_lane_count}"
        )
        return (
            f"native_trunc_{self.instruction.input_element}x{lane_range}_to_"
            f"{self.instruction.result_element}x{lane_range}"
        )


_I16_TO_I32_W = IntegerWidenInstruction("i16", "i32", 16)
_I32_TO_I64_W = IntegerWidenInstruction("i32", "i64", 8)
_I8_TO_I32_W = IntegerWidenInstruction("i8", "i32", 32)
_I16_TO_I64_W = IntegerWidenInstruction("i16", "i64", 16)
_I16_TO_I32_X = IntegerWidenInstruction("i16", "i32", 32)
_I32_TO_I64_X = IntegerWidenInstruction("i32", "i64", 16)
_I8_TO_I32_X = IntegerWidenInstruction("i8", "i32", 64)
_I16_TO_I64_X = IntegerWidenInstruction("i16", "i64", 32)

# Exact physical shapes also own fused memory rules, whose access width cannot
# exceed the source value's logical footprint.
INTEGER_WIDEN_INSTRUCTIONS = (
    _I16_TO_I32_W,
    _I32_TO_I64_W,
    _I8_TO_I32_W,
    _I16_TO_I64_W,
    _I16_TO_I32_X,
    _I32_TO_I64_X,
    _I8_TO_I32_X,
    _I16_TO_I64_X,
)

# Standalone conversions can consume every physical lane in the source's X
# carrier because integer VUPS is nontrapping and lanes outside the logical
# value domain remain unobservable. Keep each interval within one result
# carrier count so emission retains exactly the units containing logical lanes.
INTEGER_WIDEN_RULE_SHAPES = (
    *(
        IntegerWidenRuleShape(
            instruction,
            instruction.native_lane_count,
            instruction.native_lane_count,
            instruction.accumulator_unit_count,
        )
        for instruction in INTEGER_WIDEN_INSTRUCTIONS
    ),
    IntegerWidenRuleShape(_I16_TO_I32_W, 1, 15, 1),
    IntegerWidenRuleShape(_I32_TO_I64_W, 1, 7, 1),
    IntegerWidenRuleShape(_I8_TO_I32_W, 1, 16, 1),
    IntegerWidenRuleShape(_I8_TO_I32_W, 17, 31, 2),
    IntegerWidenRuleShape(_I16_TO_I64_W, 1, 8, 1),
    IntegerWidenRuleShape(_I16_TO_I64_W, 9, 15, 2),
    IntegerWidenRuleShape(_I16_TO_I32_X, 17, 31, 2),
    IntegerWidenRuleShape(_I32_TO_I64_X, 9, 15, 2),
)

# BF16 packet conversion is nontrapping, so a partial logical vector can use
# every lane of its physical X carrier. The unused native results remain beyond
# the logical value domain. Keep exact shapes for fused memory rules and stable
# report identities while admitting the partial interval independently.
FLOAT_PACKET_RULE_SHAPES = (
    *(
        FloatPacketRuleShape(lane_count, lane_count, lane_count)
        for lane_count in FLOAT_PACKET_LANE_COUNTS
    ),
    FloatPacketRuleShape(16, 1, 15),
    FloatPacketRuleShape(32, 17, 31),
)

# FP8 narrowing has no fused-memory variants, so one rule can cover the exact
# sixteen-lane shape and its partial carrier. Sixteen-bit sources also retain
# the same X carrier through lane 32. Binary32 needs a distinct exact-32 rule
# because that source is carried in an accumulator while widths 17-31 use
# ordinary vector registers.
_FLOAT8_NARROW_16BIT_RULE_SHAPES = (
    FloatPacketRuleShape(16, 1, 16),
    FloatPacketRuleShape(32, 17, 32),
)
_FLOAT8_NARROW_F32_RULE_SHAPES = (
    FloatPacketRuleShape(16, 1, 16),
    FloatPacketRuleShape(32, 17, 31),
    FloatPacketRuleShape(32, 32, 32),
)

_F8E4M3_PACKET_FORMAT = Float8PacketFormat("f8E4M3", 3, 7, False)
_F8E5M2_PACKET_FORMAT = Float8PacketFormat("f8E5M2", 2, 15, True)
FLOAT8_PACKET_FORMATS = (_F8E4M3_PACKET_FORMAT, _F8E5M2_PACKET_FORMAT)

FLOAT_PACKET_SOURCE_FORMATS = (
    FloatPacketSourceFormat("f16", 10, 15, 5),
    FloatPacketSourceFormat("bf16", 7, 127, 8),
    FloatPacketSourceFormat("f32", 23, 127, 8),
)

_FP8_PAYLOAD_AND_SIGN_MASK = 0x807F
_FP8_SUBNORMAL_THRESHOLD = 0x0080
_CANONICAL_BF16_NAN = 0x7FC0


def _mxfp8_e4m3fn_e8m0_schema(lane_count: int) -> EncodingOperandSummaryDef:
    """Builds one exact dense MXFP8 group schema."""

    return EncodingOperandSummaryDef(
        element_format=encoding.enum_fact(encoding.NumericFormat, "f8e4m3fn"),
        scale_format=encoding.enum_fact(encoding.NumericFormat, "e8m0"),
        payload_packing=encoding.enum_fact(encoding.PayloadPacking, "dense_lanes"),
        scale_topology=encoding.enum_fact(encoding.ScaleTopology, "block_1d"),
        affine_policy=encoding.enum_fact(encoding.AffinePolicy, "scale_only"),
        payload_element_count=lane_count,
        scale_group_element_count=lane_count,
        scale_group_shape=(lane_count,),
        scale_operand_count=1,
    )


MXFP8_E4M3FN_E8M0_X8_SCHEMA = _mxfp8_e4m3fn_e8m0_schema(8)
MXFP8_E4M3FN_E8M0_X32_SCHEMA = _mxfp8_e4m3fn_e8m0_schema(32)

_MXFP8_E4M3FN_E8M0_RULE_SHAPES = (
    (8, MXFP8_E4M3FN_E8M0_X8_SCHEMA),
    (32, MXFP8_E4M3FN_E8M0_X32_SCHEMA),
)

_I16_TO_I8_W_PACK = IntegerPackInstruction("i16", 32, "i8", None)
_I16_TO_I8_X_PACK = IntegerPackInstruction("i16", 64, "i8", None)
_I8_TO_I4_W_PACK = IntegerPackInstruction("i8", 64, "i8", 4)
_I8_TO_I4_X_PACK = IntegerPackInstruction("i8", 128, "i8", 4)

# Exact physical shapes also own fused memory rules, whose access width cannot
# exceed the source value's logical footprint.
INTEGER_PACK_INSTRUCTIONS = (
    _I16_TO_I8_W_PACK,
    _I16_TO_I8_X_PACK,
    _I8_TO_I4_W_PACK,
    _I8_TO_I4_X_PACK,
)

# Standalone truncation can consume every physical lane in the source carrier
# because VPACK is nontrapping and packed lanes beyond the logical value remain
# unobservable. Sub-byte bitpack keeps exact shapes: a partial source interval
# does not map linearly to an unrestricted interval of byte storage lanes.
INTEGER_PACK_RULE_SHAPES = (
    *(
        IntegerPackRuleShape(
            instruction,
            instruction.native_lane_count,
            instruction.native_lane_count,
        )
        for instruction in INTEGER_PACK_INSTRUCTIONS
    ),
    IntegerPackRuleShape(_I16_TO_I8_W_PACK, 1, 31),
    IntegerPackRuleShape(_I16_TO_I8_X_PACK, 33, 63),
)

# VPACK only has a one-bit pack-size control and therefore cannot express
# integer truncation above i16->i8. Wider lanes are reinterpreted as their
# result-width sublanes and each shuffle selects the even (low) half. Repeating
# that primitive covers every integer width relation admitted by the shared
# AMDGPU vector conversion contract.
INTEGER_TRUNCATION_INSTRUCTIONS = (
    IntegerTruncationInstruction("i32", "i16"),
    IntegerTruncationInstruction("i32", "i8"),
    IntegerTruncationInstruction("i64", "i32"),
    IntegerTruncationInstruction("i64", "i16"),
    IntegerTruncationInstruction("i64", "i8"),
)


def _integer_truncation_rule_shapes(
    instruction: IntegerTruncationInstruction,
) -> tuple[IntegerTruncationRuleShape, ...]:
    native_lane_count = instruction.native_lane_count
    return (
        IntegerTruncationRuleShape(instruction, 1, native_lane_count),
        IntegerTruncationRuleShape(
            instruction, native_lane_count + 1, native_lane_count * 2
        ),
    )


INTEGER_TRUNCATION_RULE_SHAPES = tuple(
    rule_shape
    for instruction in INTEGER_TRUNCATION_INSTRUCTIONS
    for rule_shape in _integer_truncation_rule_shapes(instruction)
)

# Uniform i32 shifts widen through one physical sixteen-lane accumulator
# packet. Lanes beyond a partial logical vector remain unobservable.
INTEGER_SHIFT_RULE_SHAPES = (
    IntegerShiftRuleShape(16, 16, 16),
    IntegerShiftRuleShape(16, 1, 15),
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _exact_vector(element: str, element_count: int) -> Vector:
    return Vector(element, lanes=element_count)


class _PacketProgram:
    """Builds lane-wise descriptor programs over one X carrier."""

    def __init__(self, element_bits: int, temporary_prefix: str = "") -> None:
        if element_bits not in (16, 32):
            raise ValueError("packet program element width must be 16 or 32")
        self.emits: list[ContractEmit] = []
        self.element_bits = element_bits
        self.temporary_prefix = temporary_prefix
        self.shift_values: dict[int, ValueRef] = {}

    def temporary(self, name: str) -> ValueRef:
        return ValueRef.temporary(f"{self.temporary_prefix}{name}")

    def constant(
        self,
        name: str,
        value: int,
        *,
        descriptor_key: str | None = None,
    ) -> ValueRef:
        if descriptor_key is None:
            descriptor_key = (
                "amd.xdna.aie2p.constant.i32.short"
                if -1024 <= value <= 1023
                else "amd.xdna.aie2p.constant.i32"
            )
        result = self.temporary(name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(descriptor_key),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def operation(
        self,
        name: str | None,
        descriptor_key: str,
        result_field: str,
        *,
        immediates: dict[str, int] | None = None,
        copy_operands: tuple[str, ...] = (),
        **operands: ValueRef,
    ) -> ValueRef:
        result = ValueRef.result("result") if name is None else self.temporary(name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(f"amd.xdna.aie2p.{descriptor_key}"),
                operands=operands,
                results={result_field: result},
                result_types=(
                    None if name is None else {result_field: DescriptorResultType()}
                ),
                immediates={} if immediates is None else immediates,
                form=DescriptorEmitForm.OP,
                copy_operands=copy_operands,
            )
        )
        return result

    def state(self, descriptor_key: str, value: int) -> None:
        """Sets one instruction-state field consumed by later descriptors."""

        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(
                    f"amd.xdna.aie2p.state.{descriptor_key}.immediate"
                ),
                immediates={"i": value},
                form=DescriptorEmitForm.OP,
            )
        )

    def splat(
        self,
        name: str,
        value: int,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        scalar = self.constant(f"{name}_scalar", value)
        return self.operation(
            name,
            f"splat.i{element_bits}x{512 // element_bits}",
            "dst",
            src=scalar,
        )

    def shift(self, value: int) -> ValueRef:
        """Returns one shared shift-register constant for |value|."""

        shift = self.shift_values.get(value)
        if shift is None:
            shift = self.constant(
                f"shift_{value}",
                value,
                descriptor_key="amd.xdna.aie2p.constant.i32.shift",
            )
            self.shift_values[value] = shift
        return shift

    def binary(
        self,
        name: str,
        descriptor_key: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(name, descriptor_key, "d", s1=lhs, s2=rhs)

    def _complete_comparison(
        self,
        name: str,
        descriptor_key: str,
        **operands: ValueRef,
    ) -> ValueRef:
        """Completes one low-half comparison into an X-sized predicate."""

        low = self.operation(f"{name}_low", descriptor_key, "cmp", **operands)
        result = self.temporary(name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.predicate.complete.zero.high32"),
                operands={"storage": low},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": 0},
                form=DescriptorEmitForm.OP,
            )
        )
        return result

    def compare_zero(
        self,
        name: str,
        value: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        return self._complete_comparison(
            name,
            f"cmp.eqz.i{element_bits}x{512 // element_bits}.el.low32",
            s2=value,
        )

    def compare_unsigned_less_than(
        self,
        name: str,
        lhs: ValueRef,
        rhs: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        return self._complete_comparison(
            name,
            f"cmp.lt.unsigned.i{element_bits}x{512 // element_bits}.el.low32",
            s1=lhs,
            s2=rhs,
        )

    def compare_unsigned_greater_equal(
        self,
        name: str,
        lhs: ValueRef,
        rhs: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        return self._complete_comparison(
            name,
            f"cmp.ge.unsigned.i{element_bits}x{512 // element_bits}.el.low32",
            s1=lhs,
            s2=rhs,
        )

    def select(
        self,
        name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        return self.operation(
            name,
            f"select.i{element_bits}x{512 // element_bits}.mask64",
            "d",
            s1=false_value,
            s2=true_value,
            sel=condition,
        )


def integer_widen_state_emits(
    ups_mode: int,
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Builds the explicit configured state consumed by one VUPS form."""

    shift = ValueRef.temporary("shift")
    return shift, (
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.constant.i32.shift"),
            results={"dst": shift},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": 0},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.saturation.immediate"),
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.ups-mode.immediate"),
            immediates={"i": ups_mode},
            form=DescriptorEmitForm.OP,
        ),
    )


def integer_widen_result_emits(
    instruction: IntegerWidenInstruction,
    result: ValueRef,
    result_accumulator_unit_count: int | None = None,
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Bridges a native accumulator result to its source-visible carrier."""

    if result_accumulator_unit_count is None:
        result_accumulator_unit_count = instruction.accumulator_unit_count
    if (
        instruction.direct_accumulator_result
        and result_accumulator_unit_count == instruction.accumulator_unit_count
    ):
        return result, ()

    native_result = ValueRef.temporary("wide_result")
    output_emits: list[ContractEmit] = []
    vector_units: list[ValueRef] = []
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )
    for unit in range(result_accumulator_unit_count):
        accumulator_unit = native_result
        if instruction.accumulator_unit_count > 1:
            accumulator_unit = ValueRef.temporary(f"accumulator_unit_{unit}")
            output_emits.append(
                EmitRegisterSlice(
                    source=native_result,
                    result=accumulator_unit,
                    unit_offset=unit,
                    unit_count=1,
                )
            )
        vector_unit = (
            result
            if result_accumulator_unit_count == 1
            else ValueRef.temporary(f"vector_unit_{unit}")
        )
        output_emits.append(
            EmitDescriptorOp(
                descriptor=move_from_accumulator,
                operands={"src": accumulator_unit},
                results={"dst": vector_unit},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result_accumulator_unit_count > 1
                    else None
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        vector_units.append(vector_unit)
    if result_accumulator_unit_count > 1:
        output_emits.append(
            EmitRegisterConcat(
                sources=vector_units,
                result=result,
            )
        )
    return native_result, tuple(output_emits)


def integer_pack_state_emits(
    pack_size: int, *, saturation: int = 0
) -> tuple[ContractEmit, ...]:
    """Builds the explicit configured state consumed by one VPACK form."""

    return (
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.saturation.immediate"),
            immediates={"i": saturation},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.pack-size.immediate"),
            immediates={"i": pack_size},
            form=DescriptorEmitForm.OP,
        ),
    )


def integer_unpack_state_emits() -> tuple[ContractEmit, ...]:
    """Builds the configured state consumed by one four-bit VUNPACK form."""

    return (
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.unpack-size.immediate"),
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        ),
    )


def _integer_bitunpack_rule(
    source_op: Op,
    source_kind: str,
    source_lane_count: int,
) -> DescriptorRule:
    result_lane_count = source_lane_count * 2
    unpack = _descriptor(
        f"amd.xdna.aie2p.unpack.{source_kind}4x{result_lane_count}.to."
        f"{source_kind}8x{result_lane_count}.configured"
    )
    source = ValueRef.operand("source")
    input_emits: tuple[ContractEmit, ...] = ()
    if source_lane_count == 32:
        source = ValueRef.temporary("packed_source")
        input_emits = (
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=source,
                unit_count=1,
            ),
        )
    signedness = "unsigned" if source_kind == "u" else "signed"
    return DescriptorRule(
        source_op=source_op,
        descriptor=unpack,
        guards=(
            Guard.value_type("source", _exact_vector("i8", source_lane_count)),
            Guard.value_type("result", _exact_vector("i8", result_lane_count)),
            Guard.attr_kind("width", "i64"),
            Guard.i64_range("width", 4, 4),
        ),
        emit=(
            *input_emits,
            *integer_unpack_state_emits(),
            EmitDescriptorOp(
                descriptor=unpack,
                operands={"src": source},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key=(
            f"native_{signedness}_i4x{result_lane_count}_to_i8x{result_lane_count}"
        ),
    )


def _integer_widen_rule(
    source_op: Op,
    signedness: str,
    rule_shape: IntegerWidenRuleShape,
) -> DescriptorRule:
    instruction = rule_shape.instruction
    source = ValueRef.operand("input")
    input_emits: tuple[ContractEmit, ...] = ()
    if instruction.slice_input:
        source = ValueRef.temporary("source_w")
        input_emits = (
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
                result=source,
                unit_count=1,
            ),
        )
    shift, state_emits = integer_widen_state_emits(instruction.ups_mode)
    result = ValueRef.result("result")
    native_result, output_emits = integer_widen_result_emits(
        instruction,
        result,
        rule_shape.result_accumulator_unit_count,
    )
    widen = _descriptor(
        f"amd.xdna.aie2p.widen.{instruction.physical_shape}.{signedness}.configured"
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=widen,
        guards=(
            Guard.value_type("input", rule_shape.input_type),
            Guard.value_type("result", rule_shape.result_type),
        ),
        emit=(
            *input_emits,
            *state_emits,
            EmitDescriptorOp(
                descriptor=widen,
                operands={"src": source, "su": shift},
                results={"dst": native_result},
                result_types=(
                    None
                    if (
                        instruction.direct_accumulator_result
                        and rule_shape.result_accumulator_unit_count
                        == instruction.accumulator_unit_count
                    )
                    else {"dst": DescriptorResultType()}
                ),
                form=DescriptorEmitForm.OP,
            ),
            *output_emits,
        ),
        report_key=rule_shape.report_key(signedness),
    )


def _integer_shift_rule(
    source_op: Op, rule_shape: IntegerShiftRuleShape
) -> DescriptorRule:
    """Shifts uniform i32 packets through one accumulator widening."""

    packet = rule_shape.vector_type
    signedness = "signed" if source_op is vector.vector_shrsi else "unsigned"
    widen = _descriptor(f"amd.xdna.aie2p.widen.2x.x-to-c.{signedness}.configured")
    narrow = _descriptor(f"amd.xdna.aie2p.narrow.2x.c-to-x.{signedness}.configured")
    shift_left = source_op is vector.vector_shli
    distance = ValueProject.exact_i64("rhs")
    return DescriptorRule(
        source_op=source_op,
        descriptor=widen,
        guards=(
            *(Guard.value_type(field, packet) for field in ("lhs", "rhs", "result")),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", 0, 31),
        ),
        emit=(
            *(
                EmitDescriptorOp(
                    descriptor=_descriptor("amd.xdna.aie2p.constant.i32.shift"),
                    results={"dst": ValueRef.temporary(name)},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": amount},
                    form=DescriptorEmitForm.CONST,
                )
                for name, amount in (
                    ("upshift", distance if shift_left else 0),
                    ("downshift", 0 if shift_left else distance),
                )
            ),
            # Widen to i64 before shifting. Unsaturated SRS then selects the
            # low i32 bits; floor rounding preserves arithmetic right shift.
            *(
                EmitDescriptorOp(
                    descriptor=_descriptor(f"amd.xdna.aie2p.state.{name}.immediate"),
                    immediates={"i": value},
                    form=DescriptorEmitForm.OP,
                )
                for name, value in (
                    ("saturation", 0),
                    ("ups-mode", 1),
                    ("srs-mode", 1),
                    ("rounding", 0),
                )
            ),
            EmitDescriptorOp(
                descriptor=widen,
                operands={
                    "src": ValueRef.operand("lhs"),
                    "su": ValueRef.temporary("upshift"),
                },
                results={"dst": ValueRef.temporary("wide")},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=narrow,
                operands={
                    "src": ValueRef.temporary("wide"),
                    "su": ValueRef.temporary("downshift"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key="native_"
        + source_op.name.removeprefix("vector.")
        + f"_i32x{rule_shape.report_lane_range}_uniform",
    )


def _f32_to_bf16_vector_rule(
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    native_lane_count = rule_shape.native_lane_count
    set_rounding = _descriptor("amd.xdna.aie2p.state.rounding.immediate")
    convert = _descriptor(
        f"amd.xdna.aie2p.convert.f32x{native_lane_count}.to.bf16x{native_lane_count}"
    )
    source_type = rule_shape.vector_type("f32")
    result_type = rule_shape.vector_type("bf16")
    native_source = ValueRef.operand("input")
    input_emits: tuple[ContractEmit, ...] = ()
    native_result = ValueRef.result("result")
    result_types = None
    output_emits: tuple[ContractEmit, ...] = ()
    if native_lane_count == 16:
        native_source = ValueRef.temporary("source_accumulator")
        native_result = ValueRef.temporary("converted_w")
        result_types = {"dst": DescriptorResultType()}
        input_emits = (
            EmitDescriptorOp(
                descriptor=_descriptor(
                    "amd.xdna.aie2p.move.vector512.to.accumulator512"
                ),
                operands={"src": ValueRef.operand("input")},
                results={"dst": native_source},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
        )
        output_emits = (
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
                result=ValueRef.temporary("unused_w"),
                unit_offset=1,
                unit_count=1,
            ),
            EmitRegisterConcat(
                sources=(native_result, ValueRef.temporary("unused_w")),
                result=ValueRef.result("result"),
            ),
        )
    elif rule_shape.maximum_lane_count < native_lane_count:
        accumulator_units = []
        input_emit_list: list[ContractEmit] = []
        native_source = ValueRef.temporary("source_accumulator")
        for unit_index in range(2):
            vector_unit = ValueRef.temporary(f"source_vector_unit_{unit_index}")
            accumulator_unit = ValueRef.temporary(
                f"source_accumulator_unit_{unit_index}"
            )
            input_emit_list.extend(
                (
                    EmitRegisterSlice(
                        source=ValueRef.operand("input"),
                        result=vector_unit,
                        unit_offset=2 * unit_index,
                        unit_count=2,
                    ),
                    EmitDescriptorOp(
                        descriptor=_descriptor(
                            "amd.xdna.aie2p.move.vector512.to.accumulator512"
                        ),
                        operands={"src": vector_unit},
                        results={"dst": accumulator_unit},
                        result_types={"dst": DescriptorResultType()},
                        form=DescriptorEmitForm.OP,
                    ),
                )
            )
            accumulator_units.append(accumulator_unit)
        input_emit_list.append(
            EmitRegisterConcat(
                sources=accumulator_units,
                result=native_source,
                result_type=_exact_vector("f32", native_lane_count),
            )
        )
        input_emits = tuple(input_emit_list)
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=convert,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            *input_emits,
            EmitDescriptorOp(
                descriptor=set_rounding,
                immediates={"i": BF16_CONVERSION_ROUNDING},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=convert,
                operands={"src": native_source},
                results={"dst": native_result},
                result_types=result_types,
                form=DescriptorEmitForm.OP,
            ),
            *output_emits,
        ),
        report_key=(
            f"native_binary32x{rule_shape.report_lane_range}_to_"
            f"bfloat16x{rule_shape.report_lane_range}"
        ),
    )


def _bf16_to_f32_emits(
    rule_shape: FloatPacketRuleShape,
    source: ValueRef,
    *,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    """Converts one source-visible BF16 packet to its F32 carrier."""

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    native_lane_count = rule_shape.native_lane_count
    convert = _descriptor(
        f"amd.xdna.aie2p.convert.bf16x{native_lane_count}.to.f32x{native_lane_count}"
    )
    native_source = source
    input_emits: tuple[ContractEmit, ...] = ()
    native_result = ValueRef.result("result")
    result_types = None
    output_emits: tuple[ContractEmit, ...] = ()
    if native_lane_count == 16:
        native_source = temporary("source_w")
        native_result = temporary("converted_accumulator")
        result_types = {"dst": DescriptorResultType()}
        input_emits = (
            EmitRegisterSlice(
                source=source,
                result=native_source,
                unit_count=1,
            ),
        )
        output_emits = (
            EmitDescriptorOp(
                descriptor=_descriptor(
                    "amd.xdna.aie2p.move.accumulator512.to.vector512"
                ),
                operands={"src": native_result},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        )
    elif rule_shape.maximum_lane_count < native_lane_count:
        native_result = temporary("converted_accumulator")
        result_types = {"dst": DescriptorResultType()}
        vector_units = []
        output_emit_list: list[ContractEmit] = []
        for unit_index in range(2):
            accumulator_unit = temporary(f"result_accumulator_unit_{unit_index}")
            vector_unit = temporary(f"result_vector_unit_{unit_index}")
            output_emit_list.extend(
                (
                    EmitRegisterSlice(
                        source=native_result,
                        result=accumulator_unit,
                        unit_offset=unit_index,
                        unit_count=1,
                    ),
                    EmitDescriptorOp(
                        descriptor=_descriptor(
                            "amd.xdna.aie2p.move.accumulator512.to.vector512"
                        ),
                        operands={"src": accumulator_unit},
                        results={"dst": vector_unit},
                        result_types={"dst": DescriptorResultType()},
                        form=DescriptorEmitForm.OP,
                    ),
                )
            )
            vector_units.append(vector_unit)
        output_emit_list.append(
            EmitRegisterConcat(
                sources=vector_units,
                result=ValueRef.result("result"),
            )
        )
        output_emits = tuple(output_emit_list)
    return (
        *input_emits,
        EmitDescriptorOp(
            descriptor=convert,
            operands={"src": native_source},
            results={"dst": native_result},
            result_types=result_types,
            form=DescriptorEmitForm.OP,
        ),
        *output_emits,
    )


def _bf16_to_f32_vector_rule(
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    convert = _descriptor(
        f"amd.xdna.aie2p.convert.bf16x{rule_shape.native_lane_count}.to."
        f"f32x{rule_shape.native_lane_count}"
    )
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=convert,
        guards=(
            Guard.value_type("input", rule_shape.vector_type("bf16")),
            Guard.value_type("result", rule_shape.vector_type("f32")),
        ),
        emit=_bf16_to_f32_emits(rule_shape, ValueRef.operand("input")),
        report_key=(
            f"native_bfloat16x{rule_shape.report_lane_range}_to_"
            f"binary32x{rule_shape.report_lane_range}"
        ),
    )


def _fp8_to_bf16_emits(
    fp8_format: Float8PacketFormat,
    *,
    result_name: str | None,
    source: ValueRef | None = None,
) -> _Float8WidenProgram:
    """Widens up to thirty-two FP8 lanes into exact BF16 bit patterns."""

    program = _PacketProgram(16, "fp8_")
    if source is None:
        source = ValueRef.operand("input")
    zero = program.operation("zero", "sub.i8x64", "d", s1=source, s2=source)
    interleave_control = program.constant(
        "interleave_control",
        I8_INTERLEAVE_CONTROL,
        descriptor_key="amd.xdna.aie2p.constant.i32.mova",
    )
    low_bytes = program.operation(
        "low_bytes",
        "shuffle.x.configured",
        "dst",
        s1=source,
        s2=zero,
        mod=interleave_control,
    )
    high_bytes = program.operation(
        "high_bytes",
        "shuffle.x.configured",
        "dst",
        s1=zero,
        s2=source,
        mod=interleave_control,
    )
    payload_and_sign_mask = program.splat(
        "payload_and_sign_mask", _FP8_PAYLOAD_AND_SIGN_MASK
    )
    payload = program.binary("payload", "and.bits512", low_bytes, payload_and_sign_mask)
    sign = program.binary("sign", "and.bits512", high_bytes, payload_and_sign_mask)

    scaled_payloads = [payload]
    for shift in range(1, 7):
        scaled_payloads.append(
            program.binary(
                f"payload_x{1 << shift}",
                "add.i16x32",
                scaled_payloads[-1],
                scaled_payloads[-1],
            )
        )

    threshold = program.splat("subnormal_threshold", _FP8_SUBNORMAL_THRESHOLD)
    normal_base = program.splat("normal_base", fp8_format.normal_base)
    normal = program.binary(
        "normal",
        "add.i16x32",
        scaled_payloads[fp8_format.normal_shift],
        normal_base,
    )

    # Each source-subnormal range with the same leading one is affine. The
    # range base advances by one BF16 exponent bit (0x80), while its payload
    # shift decreases by one. Build the few ranges from smallest to largest.
    single_base = normal_base
    for index in range(fp8_format.mantissa_bits - 1):
        single_base = program.binary(
            f"single_base_{index}",
            "sub.i16x32",
            single_base,
            threshold,
        )
    is_zero = program.compare_zero("is_zero", payload)
    subnormal = program.select("subnormal_1", zero, single_base, is_zero)
    range_base = single_base
    for leading_bit in range(1, fp8_format.mantissa_bits):
        payload_shift = 7 - leading_bit
        candidate = program.binary(
            f"subnormal_{1 << leading_bit}",
            "add.i16x32",
            scaled_payloads[payload_shift],
            range_base,
        )
        below_range = program.compare_unsigned_less_than(
            f"below_{1 << leading_bit}",
            scaled_payloads[payload_shift],
            threshold,
        )
        subnormal = program.select(
            f"subnormal_through_{(1 << (leading_bit + 1)) - 1}",
            subnormal,
            candidate,
            below_range,
        )
        if leading_bit + 1 < fp8_format.mantissa_bits:
            range_base = program.binary(
                f"range_base_{leading_bit + 1}",
                "add.i16x32",
                range_base,
                threshold,
            )

    exponent_is_zero = program.compare_unsigned_less_than(
        "exponent_is_zero",
        scaled_payloads[fp8_format.normal_shift],
        threshold,
    )
    finite_unsigned = program.select(
        "finite_unsigned", subnormal, normal, exponent_is_zero
    )
    finite = program.binary("finite", "or.bits512", finite_unsigned, sign)
    canonical_nan = program.splat("canonical_nan", _CANONICAL_BF16_NAN)

    is_nan = None
    if fp8_format.has_infinity:
        special_payload = program.splat("special_payload", fp8_format.special_payload)
        is_finite = program.compare_unsigned_less_than(
            "is_finite", payload, special_payload
        )
        special_delta = program.binary(
            "special_delta", "sub.i16x32", payload, special_payload
        )
        is_infinity = program.compare_zero("is_infinity", special_delta)
        infinity_unsigned = program.binary(
            "infinity_unsigned", "add.i16x32", normal, normal_base
        )
        infinity = program.binary("infinity", "or.bits512", infinity_unsigned, sign)
        special = program.select("special", infinity, canonical_nan, is_infinity)
        result = program.select(result_name, finite, special, is_finite)
    else:
        nan_payload = program.splat("nan_payload", 0x7F)
        nan_delta = program.binary("nan_delta", "sub.i16x32", payload, nan_payload)
        is_nan = program.compare_zero("is_nan", nan_delta)
        result = program.select(result_name, canonical_nan, finite, is_nan)

    return _Float8WidenProgram(
        result=result,
        canonical_nan=canonical_nan,
        is_nan=is_nan,
        emits=tuple(program.emits),
    )


def _mxfp8_e4m3fn_e8m0_to_bf16_rule(
    lane_count: int,
    schema: EncodingOperandSummaryDef,
) -> DescriptorRule:
    """Decodes one MXFP8 block without scalarizing its payload lanes."""

    payload_program = _fp8_to_bf16_emits(
        _F8E4M3_PACKET_FORMAT,
        result_name="payload_bf16",
        source=ValueRef.operand("payload"),
    )
    if payload_program.is_nan is None:
        raise ValueError("MXFP8 E4M3FN widening must produce a NaN predicate")
    payload_bf16 = payload_program.result
    program = _PacketProgram(16, "mxfp8_")
    scale = ValueRef.operand("auxiliary", element=0)

    scale_word = program.operation(
        "scale_word",
        "extract.i32.immediate",
        "dst",
        immediates={"idx": 0},
        s1=scale,
    )
    byte_mask = program.constant("byte_mask", 0xFF)
    scale_byte = program.operation(
        "scale_byte", "and.i32", "d0", s0=scale_word, s1=byte_mask
    )
    zero_scalar = program.constant("zero", 0)
    one_scalar = program.constant("one", 1)
    two_scalar = program.constant("two", 2)
    three_scalar = program.constant("three", 3)
    seven_scalar = program.constant("seven", 7)
    nine_scalar = program.constant("nine", 9)
    exponent_bias = program.constant("exponent_bias", 127)
    normal_origin = program.constant("normal_origin", 128)
    overflow_origin = program.constant("overflow_origin", 382)

    sign_mask = program.splat("sign_mask", 0x8000)
    absolute_mask = program.splat("absolute_mask", 0x7FFF)
    infinity = program.splat("infinity", 0x7F80)
    sign = program.binary("sign", "and.bits512", payload_bf16, sign_mask)
    magnitude = program.binary("magnitude", "and.bits512", payload_bf16, absolute_mask)

    exponent_delta = program.operation(
        "exponent_delta",
        "sub.i32",
        "d0",
        s0=scale_byte,
        s1=exponent_bias,
    )
    bit_delta = program.operation(
        "bit_delta", "lshl.i32", "d0", s0=exponent_delta, s1=seven_scalar
    )
    bit_delta_vector = program.operation(
        "bit_delta_vector", "splat.i16x32", "dst", src=bit_delta
    )
    adjusted_magnitude = program.binary(
        "adjusted_magnitude", "add.i16x32", magnitude, bit_delta_vector
    )

    normal_threshold_exponent = program.operation(
        "normal_threshold_exponent",
        "sub.i32",
        "d0",
        s0=normal_origin,
        s1=scale_byte,
    )
    normal_threshold_is_negative = program.operation(
        "normal_threshold_is_negative",
        "cmp.slt.i32",
        "d0",
        s0=normal_threshold_exponent,
        s1=zero_scalar,
    )
    normal_threshold_clamped = program.operation(
        "normal_threshold_clamped",
        "select.nonzero.i32",
        "d0",
        copy_operands=("s2",),
        s0=zero_scalar,
        s1=normal_threshold_exponent,
        s2=normal_threshold_is_negative,
    )
    normal_threshold_bits = program.operation(
        "normal_threshold_bits",
        "lshl.i32",
        "d0",
        s0=normal_threshold_clamped,
        s1=seven_scalar,
    )
    normal_threshold = program.operation(
        "normal_threshold", "splat.i16x32", "dst", src=normal_threshold_bits
    )
    has_normal = program.compare_unsigned_greater_equal(
        "has_normal", magnitude, normal_threshold
    )

    overflow_threshold_exponent = program.operation(
        "overflow_threshold_exponent",
        "sub.i32",
        "d0",
        s0=overflow_origin,
        s1=scale_byte,
    )
    overflow_threshold_above_limit = program.operation(
        "overflow_threshold_above_limit",
        "cmp.ult.i32",
        "d0",
        s0=byte_mask,
        s1=overflow_threshold_exponent,
    )
    overflow_threshold_clamped = program.operation(
        "overflow_threshold_clamped",
        "select.nonzero.i32",
        "d0",
        copy_operands=("s2",),
        s0=byte_mask,
        s1=overflow_threshold_exponent,
        s2=overflow_threshold_above_limit,
    )
    overflow_threshold_bits = program.operation(
        "overflow_threshold_bits",
        "lshl.i32",
        "d0",
        s0=overflow_threshold_clamped,
        s1=seven_scalar,
    )
    overflow_threshold = program.operation(
        "overflow_threshold", "splat.i16x32", "dst", src=overflow_threshold_bits
    )
    is_overflow = program.compare_unsigned_greater_equal(
        "is_overflow", magnitude, overflow_threshold
    )

    magnitude_low = program.temporary("magnitude_low")
    program.emits.append(
        EmitRegisterSlice(
            source=magnitude,
            result=magnitude_low,
            unit_count=1,
        )
    )
    subnormal_units_i32 = program.operation(
        "subnormal_units_i32",
        "convert.floor.bf16x16.to.i32x16",
        "dst",
        src=magnitude_low,
        shft=program.shift(9),
    )
    subnormal_units_accumulator = program.operation(
        "subnormal_units_accumulator",
        "move.vector512.to.accumulator512",
        "dst",
        src=subnormal_units_i32,
    )
    shift_zero = program.shift(0)
    shift_one = program.shift(1)
    shift_two = program.shift(2)
    shift_three = program.shift(3)
    program.state("rounding", 12)
    program.state("srs-mode", 0)
    program.state("saturation", 0)
    units_low = program.operation(
        "units_low",
        "narrow.2x.b-to-w.unsigned.configured",
        "dst",
        src=subnormal_units_accumulator,
        su=shift_zero,
    )
    scale_two_low = program.operation(
        "scale_two_low",
        "narrow.2x.b-to-w.unsigned.configured",
        "dst",
        src=subnormal_units_accumulator,
        su=shift_one,
    )
    scale_one_low = program.operation(
        "scale_one_low",
        "narrow.2x.b-to-w.unsigned.configured",
        "dst",
        src=subnormal_units_accumulator,
        su=shift_two,
    )
    scale_zero_low = program.operation(
        "scale_zero_low",
        "narrow.2x.b-to-w.unsigned.configured",
        "dst",
        src=subnormal_units_accumulator,
        su=shift_three,
    )
    unused_high = program.temporary("unused_high")
    program.emits.append(
        EmitRegisterSlice(
            source=payload_bf16,
            result=unused_high,
            unit_offset=1,
            unit_count=1,
        )
    )

    def concat_low(name: str, low: ValueRef) -> ValueRef:
        result = program.temporary(name)
        program.emits.append(
            EmitRegisterConcat(
                sources=(low, unused_high),
                result=result,
                result_type=_exact_vector("i16", 32),
            )
        )
        return result

    units = concat_low("units", units_low)
    scale_two = concat_low("scale_two", scale_two_low)
    scale_one = concat_low("scale_one", scale_one_low)
    scale_zero = concat_low("scale_zero", scale_zero_low)

    scale_below_three = program.operation(
        "scale_below_three",
        "cmp.ult.i32",
        "d0",
        s0=scale_byte,
        s1=three_scalar,
    )
    scale_at_least_three = program.operation(
        "scale_at_least_three",
        "select.nonzero.i32",
        "d0",
        copy_operands=("s2",),
        s0=three_scalar,
        s1=scale_byte,
        s2=scale_below_three,
    )
    scale_above_nine = program.operation(
        "scale_above_nine",
        "cmp.ult.i32",
        "d0",
        s0=nine_scalar,
        s1=scale_at_least_three,
    )
    scale_clamped = program.operation(
        "scale_clamped",
        "select.nonzero.i32",
        "d0",
        copy_operands=("s2",),
        s0=nine_scalar,
        s1=scale_at_least_three,
        s2=scale_above_nine,
    )
    factor_shift = program.operation(
        "factor_shift",
        "sub.i32",
        "d0",
        s0=scale_clamped,
        s1=three_scalar,
    )
    factor = program.operation(
        "factor", "lshl.i32", "d0", s0=one_scalar, s1=factor_shift
    )
    factor_vector = program.operation(
        "factor_vector", "splat.i16x32", "dst", src=factor
    )
    multiply_control = program.constant(
        "multiply_control",
        858,
        descriptor_key="amd.xdna.aie2p.constant.i32.mova",
    )
    scaled_units_accumulator = program.operation(
        "scaled_units_accumulator",
        "multiply.i16x32.configured",
        "dst",
        s1=units,
        s2=factor_vector,
        acc=multiply_control,
    )
    program.state("rounding", 0)
    program.state("srs-mode", 1)
    program.state("saturation", 0)
    scaled_units = program.operation(
        "scaled_units",
        "narrow.trunc.signed.i16x32",
        "dst",
        src=scaled_units_accumulator,
        su=shift_zero,
    )

    def scalar_mask(name: str, condition: ValueRef) -> ValueRef:
        return program.operation(
            name,
            "select.mask.i32",
            "d0",
            immediates={"imm": -1},
            s0=condition,
        )

    scale_is_two = program.operation(
        "scale_is_two", "cmp.eq.i32", "d0", s0=scale_byte, s1=two_scalar
    )
    through_two = program.operation(
        "through_two",
        "select.i32x16",
        "d",
        s1=scale_two,
        s2=scaled_units,
        sel=scalar_mask("scale_two_mask", scale_is_two),
    )
    scale_is_one = program.operation(
        "scale_is_one", "cmp.eq.i32", "d0", s0=scale_byte, s1=one_scalar
    )
    through_one = program.operation(
        "through_one",
        "select.i32x16",
        "d",
        s1=scale_one,
        s2=through_two,
        sel=scalar_mask("scale_one_mask", scale_is_one),
    )
    scale_is_zero = program.operation(
        "scale_is_zero", "cmp.eqz.i32", "d0", s0=scale_byte
    )
    subnormal_magnitude = program.operation(
        "subnormal_magnitude",
        "select.i32x16",
        "d",
        s1=scale_zero,
        s2=through_one,
        sel=scalar_mask("scale_zero_mask", scale_is_zero),
    )

    finite_result_magnitude = program.select(
        "finite_result_magnitude",
        adjusted_magnitude,
        subnormal_magnitude,
        has_normal,
    )
    bounded_result_magnitude = program.select(
        "bounded_result_magnitude",
        infinity,
        finite_result_magnitude,
        is_overflow,
    )
    signed_result = program.binary(
        "signed_result", "or.bits512", bounded_result_magnitude, sign
    )
    payload_is_zero = program.compare_zero("payload_is_zero", magnitude)
    zero_fixed_result = program.select(
        "zero_fixed_result", sign, signed_result, payload_is_zero
    )
    payload_fixed_result = program.select(
        "payload_fixed_result",
        payload_program.canonical_nan,
        zero_fixed_result,
        payload_program.is_nan,
    )
    scale_is_nan = program.operation(
        "scale_is_nan", "cmp.eq.i32", "d0", s0=scale_byte, s1=byte_mask
    )
    program.operation(
        None,
        "select.i32x16",
        "d",
        s1=payload_program.canonical_nan,
        s2=payload_fixed_result,
        sel=scalar_mask("scale_nan_mask", scale_is_nan),
    )

    return DescriptorRule(
        source_op=vector.vector_decode,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("payload", _exact_vector("f8E4M3", lane_count)),
            Guard.value_storage_operand_schema("schema", schema),
            Guard.operand_segment_count("auxiliary", 1),
            Guard.value_type("auxiliary", _exact_vector("i32", 1), element=0),
            Guard.value_type("result", _exact_vector("bf16", lane_count)),
        ),
        emit=(*payload_program.emits, *program.emits),
        report_key=(f"native_mxfp8_e4m3fn_e8m0x{lane_count}_to_bfloat16x{lane_count}"),
    )


def _fp8_to_bf16_vector_rule(
    fp8_format: Float8PacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    program = _fp8_to_bf16_emits(fp8_format, result_name=None)
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", rule_shape.vector_type(fp8_format.element)),
            Guard.value_type("result", rule_shape.vector_type("bf16")),
        ),
        emit=program.emits,
        report_key=(
            f"native_{fp8_format.report_name}x{rule_shape.report_lane_range}_to_"
            f"bfloat16x{rule_shape.report_lane_range}"
        ),
    )


def _fp8_to_f32_vector_rule(
    fp8_format: Float8PacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    program = _fp8_to_bf16_emits(fp8_format, result_name="decoded_bf16")
    convert_emits = _bf16_to_f32_emits(
        rule_shape, program.result, temporary_prefix="fp8_widen_"
    )
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=_descriptor(
            f"amd.xdna.aie2p.convert.bf16x{rule_shape.native_lane_count}.to."
            f"f32x{rule_shape.native_lane_count}"
        ),
        guards=(
            Guard.value_type("input", rule_shape.vector_type(fp8_format.element)),
            Guard.value_type("result", rule_shape.vector_type("f32")),
        ),
        emit=(*program.emits, *convert_emits),
        report_key=(
            f"native_{fp8_format.report_name}x{rule_shape.report_lane_range}_to_"
            f"binary32x{rule_shape.report_lane_range}"
        ),
    )


def _fp8_narrow_state_emits(
    source_format: FloatPacketSourceFormat,
) -> tuple[ContractEmit, ...]:
    """Builds the shared packet-conversion state for one narrowing rule."""

    state_values = [("saturation", 1)]
    if source_format.bit_width == 16:
        state_values.append(("ups-mode", 0))
    state_values.extend(
        (
            ("rounding", BF16_CONVERSION_ROUNDING),
            ("srs-mode", 0),
            ("pack-size", 1),
        )
    )
    return tuple(
        EmitDescriptorOp(
            descriptor=_descriptor(f"amd.xdna.aie2p.state.{name}.immediate"),
            immediates={"i": value},
            form=DescriptorEmitForm.OP,
        )
        for name, value in state_values
    )


def _float_source_i32_chunks(
    program: _PacketProgram,
    source_format: FloatPacketSourceFormat,
    rule_shape: FloatPacketRuleShape,
) -> tuple[ValueRef, ...]:
    """Returns native X carriers containing sixteen source bit patterns each."""

    source = ValueRef.operand("input")
    if source_format.bit_width == 16:
        instruction = (
            _I16_TO_I32_W if rule_shape.native_lane_count == 16 else _I16_TO_I32_X
        )
        native_source = source
        if instruction.slice_input:
            native_source = program.temporary("source_w")
            program.emits.append(
                EmitRegisterSlice(
                    source=source,
                    result=native_source,
                    unit_count=1,
                )
            )
        wide = program.operation(
            "source_i32_accumulator",
            f"widen.{instruction.physical_shape}.unsigned.configured",
            "dst",
            src=native_source,
            su=program.shift(0),
        )
        chunks = []
        for index in range(instruction.accumulator_unit_count):
            accumulator = wide
            if instruction.accumulator_unit_count > 1:
                accumulator = program.temporary(f"source_i32_accumulator_{index}")
                program.emits.append(
                    EmitRegisterSlice(
                        source=wide,
                        result=accumulator,
                        unit_offset=index,
                        unit_count=1,
                    )
                )
            chunks.append(
                program.operation(
                    f"source_i32_chunk_{index}",
                    "move.accumulator512.to.vector512",
                    "dst",
                    src=accumulator,
                )
            )
        return tuple(chunks)

    if rule_shape.native_lane_count == 16:
        return (source,)

    chunks = []
    source_is_accumulator = (
        rule_shape.minimum_lane_count == rule_shape.maximum_lane_count == 32
    )
    for index in range(2):
        chunk = program.temporary(f"source_i32_chunk_{index}")
        if source_is_accumulator:
            accumulator = program.temporary(f"source_i32_accumulator_{index}")
            program.emits.append(
                EmitRegisterSlice(
                    source=source,
                    result=accumulator,
                    unit_offset=index,
                    unit_count=1,
                )
            )
            program.operation(
                f"source_i32_chunk_{index}",
                "move.accumulator512.to.vector512",
                "dst",
                src=accumulator,
            )
        else:
            program.emits.append(
                EmitRegisterSlice(
                    source=source,
                    result=chunk,
                    unit_offset=2 * index,
                    unit_count=2,
                )
            )
        chunks.append(chunk)
    return tuple(chunks)


def _i32_packet_accumulator(
    program: _PacketProgram,
    name: str,
    source: ValueRef,
) -> tuple[ValueRef, ValueRef]:
    """Moves one i32 X carrier to an accumulator and retains padding."""

    filler = program.temporary(f"{name}_filler_w")
    program.emits.append(
        EmitRegisterSlice(
            source=source,
            result=filler,
            unit_offset=1,
            unit_count=1,
        )
    )
    accumulator = program.operation(
        f"{name}_accumulator",
        "move.vector512.to.accumulator512",
        "dst",
        src=source,
    )
    return accumulator, filler


def _round_i32_packet_to_i16(
    program: _PacketProgram,
    name: str,
    accumulator: ValueRef,
    filler: ValueRef,
    shift: int,
    *,
    signed: bool = False,
) -> ValueRef:
    """Rounds sixteen i32 lanes into the low W of an X carrier."""

    result_w = program.operation(
        f"{name}_w",
        f"narrow.2x.b-to-w.{'signed' if signed else 'unsigned'}.configured",
        "dst",
        src=accumulator,
        su=program.shift(shift),
    )
    result = program.temporary(name)
    program.emits.append(
        EmitRegisterConcat(
            sources=(result_w, filler),
            result=result,
            result_type=_exact_vector("i16", 32),
        )
    )
    return result


@dataclass(frozen=True, slots=True)
class _DirectFloat8Narrowing:
    """Finite conversion for source and destination encodings with equal bias."""

    # Number of low source bits removed by rounding.
    shift: int


@dataclass(frozen=True, slots=True)
class _RebiasedFloat8Narrowing:
    """Finite conversion that reconstructs destination normal and subnormal bits."""

    # Mask selecting the explicit source significand.
    fraction_mask: ValueRef
    # Source significand bit implicit in normal encodings.
    hidden_bit: ValueRef
    # Encoded exponent delta between the source and destination formats.
    normal_bias: ValueRef
    # Source encodings where each destination subnormal candidate becomes active.
    exponent_thresholds: tuple[tuple[int, ValueRef], ...]


type _Float8NarrowingStrategy = _DirectFloat8Narrowing | _RebiasedFloat8Narrowing


def _float_chunk_to_fp8_i16(
    program: _PacketProgram,
    source: ValueRef,
    source_format: FloatPacketSourceFormat,
    fp8_format: Float8PacketFormat,
    chunk_index: int,
    *,
    nonsign_mask: ValueRef,
    sign_mask: ValueRef,
    infinity: ValueRef,
    clamp: ValueRef,
    one: ValueRef,
    zero: ValueRef,
    strategy: _Float8NarrowingStrategy,
) -> ValueRef:
    """Narrows one sixteen-lane i32 source carrier to packed i16 codes."""

    prefix = f"chunk_{chunk_index}"
    absolute = program.binary(f"{prefix}_absolute", "and.bits512", source, nonsign_mask)
    sign_bits = program.binary(f"{prefix}_sign_bits", "and.bits512", source, sign_mask)
    sign_accumulator, sign_filler = _i32_packet_accumulator(
        program, f"{prefix}_sign", sign_bits
    )
    result_sign = _round_i32_packet_to_i16(
        program,
        f"{prefix}_result_sign",
        sign_accumulator,
        sign_filler,
        source_format.bit_width - 8,
    )

    if isinstance(strategy, _DirectFloat8Narrowing):
        absolute_accumulator, filler = _i32_packet_accumulator(
            program, f"{prefix}_absolute", absolute
        )
        finite = _round_i32_packet_to_i16(
            program,
            f"{prefix}_finite",
            absolute_accumulator,
            filler,
            strategy.shift,
        )
    else:
        adjusted = program.binary(
            f"{prefix}_normal_adjusted",
            "sub.i32x16",
            absolute,
            strategy.normal_bias,
        )
        normal_accumulator, normal_filler = _i32_packet_accumulator(
            program, f"{prefix}_normal", adjusted
        )
        normal = _round_i32_packet_to_i16(
            program,
            f"{prefix}_normal",
            normal_accumulator,
            normal_filler,
            source_format.mantissa_bits - fp8_format.mantissa_bits,
            signed=True,
        )
        normal = program.binary(
            f"{prefix}_normal_nonnegative",
            "max.signed.i16x32",
            normal,
            zero,
        )

        fraction = program.binary(
            f"{prefix}_fraction", "and.bits512", absolute, strategy.fraction_mask
        )
        significand = program.binary(
            f"{prefix}_significand", "or.bits512", fraction, strategy.hidden_bit
        )
        significand_accumulator, significand_filler = _i32_packet_accumulator(
            program, f"{prefix}_significand", significand
        )
        subnormal = zero
        for unbiased_exponent, threshold in strategy.exponent_thresholds:
            shift = (
                source_format.mantissa_bits
                + 1
                - fp8_format.exponent_bias
                - fp8_format.mantissa_bits
                - unbiased_exponent
            )
            candidate = _round_i32_packet_to_i16(
                program,
                f"{prefix}_subnormal_{unbiased_exponent}",
                significand_accumulator,
                significand_filler,
                shift,
            )
            in_range = program.compare_unsigned_greater_equal(
                f"{prefix}_at_exponent_{unbiased_exponent}",
                absolute,
                threshold,
            )
            subnormal = program.select(
                f"{prefix}_subnormal_through_{unbiased_exponent}",
                candidate,
                subnormal,
                in_range,
                element_bits=16,
            )
        finite = program.binary(
            f"{prefix}_finite",
            "max.unsigned.i16x32",
            normal,
            subnormal,
        )

    finite = program.binary(f"{prefix}_clamped", "min.unsigned.i16x32", finite, clamp)
    special_delta = program.binary(
        f"{prefix}_special_delta", "sub.i32x16", absolute, infinity
    )
    special_accumulator, special_filler = _i32_packet_accumulator(
        program, f"{prefix}_special", special_delta
    )
    special = _round_i32_packet_to_i16(
        program,
        f"{prefix}_special",
        special_accumulator,
        special_filler,
        0,
        signed=True,
    )
    special = program.binary(
        f"{prefix}_special_nonnegative",
        "max.signed.i16x32",
        special,
        zero,
    )
    special = program.binary(
        f"{prefix}_special_flag", "min.unsigned.i16x32", special, one
    )
    magnitude = finite
    for increment in range(fp8_format.nan_payload - fp8_format.finite_clamp):
        magnitude = program.binary(
            f"{prefix}_magnitude_{increment + 1}",
            "add.i16x32",
            magnitude,
            special,
        )
    return program.binary(f"{prefix}_result", "or.bits512", magnitude, result_sign)


def _float_to_fp8_emits(
    source_format: FloatPacketSourceFormat,
    fp8_format: Float8PacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> tuple[ContractEmit, ...]:
    """Narrows up to thirty-two floating lanes into exact FP8 packets."""

    program = _PacketProgram(32, "fp8_narrow_")
    program.emits.extend(_fp8_narrow_state_emits(source_format))
    chunks = _float_source_i32_chunks(program, source_format, rule_shape)

    nonsign_mask = program.splat("nonsign_mask", source_format.nonsign_mask)
    sign_mask_value = source_format.sign_bit
    if sign_mask_value == 1 << 31:
        sign_mask_value = -(1 << 31)
    sign_mask = program.splat("sign_mask", sign_mask_value)
    infinity = program.splat("infinity", source_format.infinity_bits)
    clamp = program.splat("clamp", fp8_format.finite_clamp, element_bits=16)
    one = program.splat("one", 1, element_bits=16)
    zero = program.binary("zero", "sub.i16x32", one, one)

    if source_format.element == "f16" and fp8_format.element == "f8E5M2":
        strategy: _Float8NarrowingStrategy = _DirectFloat8Narrowing(
            shift=source_format.mantissa_bits - fp8_format.mantissa_bits
        )
    else:
        strategy = _RebiasedFloat8Narrowing(
            fraction_mask=program.splat("fraction_mask", source_format.fraction_mask),
            hidden_bit=program.splat("hidden_bit", source_format.hidden_bit),
            normal_bias=program.splat(
                "normal_bias",
                (source_format.exponent_bias - fp8_format.exponent_bias)
                << source_format.mantissa_bits,
            ),
            exponent_thresholds=tuple(
                (
                    unbiased_exponent,
                    program.splat(
                        f"exponent_{unbiased_exponent}_threshold",
                        source_format.exponent_bits_for(unbiased_exponent),
                    ),
                )
                for unbiased_exponent in range(
                    fp8_format.minimum_rounding_exponent,
                    fp8_format.minimum_normal_exponent,
                )
            ),
        )

    narrowed_chunks = tuple(
        _float_chunk_to_fp8_i16(
            program,
            chunk,
            source_format,
            fp8_format,
            index,
            nonsign_mask=nonsign_mask,
            sign_mask=sign_mask,
            infinity=infinity,
            clamp=clamp,
            one=one,
            zero=zero,
            strategy=strategy,
        )
        for index, chunk in enumerate(chunks)
    )

    code_words = []
    for index, narrowed in enumerate(narrowed_chunks):
        code_word = program.temporary(f"code_word_{index}")
        program.emits.append(
            EmitRegisterSlice(
                source=narrowed,
                result=code_word,
                unit_count=1,
            )
        )
        code_words.append(code_word)
    if len(code_words) == 1:
        unused_word = program.temporary("unused_code_word")
        program.emits.append(
            EmitRegisterSlice(
                source=narrowed_chunks[0],
                result=unused_word,
                unit_offset=1,
                unit_count=1,
            )
        )
        code_words.append(unused_word)

    packed_source = program.temporary("packed_source")
    program.emits.append(
        EmitRegisterConcat(
            sources=tuple(code_words),
            result=packed_source,
            result_type=_exact_vector("i16", 32),
        )
    )
    program.emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.saturation.immediate"),
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        )
    )
    packed_word = program.operation(
        "packed_w",
        "pack.w.trunc.configured",
        "dst",
        src=packed_source,
    )
    unused_word = program.temporary("packed_unused_w")
    program.emits.append(
        EmitRegisterSlice(
            source=packed_source,
            result=unused_word,
            unit_offset=1,
            unit_count=1,
        )
    )
    program.emits.append(
        EmitRegisterConcat(
            sources=(packed_word, unused_word),
            result=ValueRef.result("result"),
        )
    )
    return tuple(program.emits)


def _float_to_fp8_vector_rule(
    source_format: FloatPacketSourceFormat,
    fp8_format: Float8PacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    emits = _float_to_fp8_emits(source_format, fp8_format, rule_shape)
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=_descriptor("amd.xdna.aie2p.pack.w.trunc.configured"),
        guards=(
            Guard.value_type("input", rule_shape.vector_type(source_format.element)),
            Guard.value_type("result", rule_shape.vector_type(fp8_format.element)),
        ),
        emit=emits,
        report_key=(
            f"native_{source_format.report_name}x{rule_shape.report_lane_range}_to_"
            f"{fp8_format.report_name}x{rule_shape.report_lane_range}"
        ),
    )


def _integer_truncation_rule(
    rule_shape: IntegerTruncationRuleShape,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    source = ValueRef.operand("input")
    emits: list[ContractEmit] = []
    if rule_shape.source_carrier_count == 2:
        low_source = ValueRef.temporary("source_low")
        high_source = ValueRef.temporary("source_high")
        emits.extend(
            (
                EmitRegisterSlice(
                    source=source,
                    result=low_source,
                    unit_count=2,
                ),
                EmitRegisterSlice(
                    source=source,
                    result=high_source,
                    unit_offset=2,
                    unit_count=2,
                ),
            )
        )
    else:
        low_source = source
        high_source = source

    current = source
    for index, control_value in enumerate(rule_shape.instruction.shuffle_controls):
        control = ValueRef.temporary(f"shuffle_control_{index}")
        is_final = index + 1 == len(rule_shape.instruction.shuffle_controls)
        result = (
            ValueRef.result("result")
            if is_final
            else ValueRef.temporary(f"truncated_{index}")
        )
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=constant,
                    results={"dst": control},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": control_value},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=shuffle,
                    operands={
                        "s1": low_source if index == 0 else current,
                        "s2": high_source if index == 0 else current,
                        "mod": control,
                    },
                    results={"dst": result},
                    result_types=(
                        None if is_final else {"dst": DescriptorResultType()}
                    ),
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
        current = result

    return DescriptorRule(
        source_op=vector.vector_trunci,
        descriptor=shuffle,
        guards=(
            Guard.value_type("input", rule_shape.input_type),
            Guard.value_type("result", rule_shape.result_type),
        ),
        emit=tuple(emits),
        report_key=rule_shape.report_key,
    )


def _integer_pack_emits(
    pack_instruction: IntegerPackInstruction,
    pack: Descriptor,
    source: ValueRef,
    *,
    saturation: int = 0,
) -> tuple[ContractEmit, ...]:
    packed_result = (
        ValueRef.temporary("packed_w")
        if pack_instruction.pad_result
        else ValueRef.result("result")
    )
    result_emits: tuple[ContractEmit, ...] = ()
    if pack_instruction.pad_result:
        result_emits = (
            EmitRegisterSlice(
                source=source,
                result=ValueRef.temporary("unused_w"),
                unit_offset=1,
                unit_count=1,
            ),
            EmitRegisterConcat(
                sources=(packed_result, ValueRef.temporary("unused_w")),
                result=ValueRef.result("result"),
            ),
        )
    return (
        *integer_pack_state_emits(pack_instruction.pack_size, saturation=saturation),
        EmitDescriptorOp(
            descriptor=pack,
            operands={"src": source},
            results={"dst": packed_result},
            result_types=(
                {"dst": DescriptorResultType()} if pack_instruction.pad_result else None
            ),
            form=DescriptorEmitForm.OP,
        ),
        *result_emits,
    )


def _integer_pack_rule(rule_shape: IntegerPackRuleShape) -> DescriptorRule:
    pack_instruction = rule_shape.instruction
    pack = _descriptor(
        f"amd.xdna.aie2p.pack.{pack_instruction.physical_width}.trunc.configured"
    )
    return DescriptorRule(
        source_op=pack_instruction.source_op,
        descriptor=pack,
        guards=(
            Guard.value_type(
                pack_instruction.source_field,
                rule_shape.input_type,
            ),
            Guard.value_type("result", rule_shape.result_type),
            *(
                (
                    Guard.attr_kind("width", "i64"),
                    Guard.i64_range(
                        "width",
                        pack_instruction.bit_width,
                        pack_instruction.bit_width,
                    ),
                )
                if pack_instruction.bit_width is not None
                else ()
            ),
        ),
        emit=_integer_pack_emits(
            pack_instruction,
            pack,
            ValueRef.operand(pack_instruction.source_field),
        ),
        report_key=rule_shape.report_key,
    )


def _saturating_i4_pack_rule(
    pack_instruction: IntegerPackInstruction,
    outer_op: Op,
    inner_op: Op,
    value_fields: tuple[str, str],
) -> DescriptorRule:
    # The shared matcher requires each consumed clamp result to be adjacent and
    # single-use. The original input and bound constants may have other users.
    bounds = {vector.vector_maxsi: -8, vector.vector_minsi: 7}
    bound_fields = tuple("rhs" if field == "lhs" else "lhs" for field in value_fields)
    pack = _descriptor(
        f"amd.xdna.aie2p.pack.{pack_instruction.physical_width}.signed.configured"
    )
    source = ValueRef.operand(value_fields[1], source_node="inner")
    order = "min_max" if outer_op is vector.vector_minsi else "max_min"
    return DescriptorRule(
        source_op=vector.vector_bitpack,
        descriptor=pack,
        priority=1,
        guards=(
            Guard.value_type(
                "source", _exact_vector("i8", pack_instruction.native_lane_count)
            ),
            Guard.value_type(
                "result", _exact_vector("i8", pack_instruction.result_lanes)
            ),
            Guard.i64_range("width", 4, 4),
        ),
        source_nodes=(
            SourceNode.adjacent_definition(
                "outer",
                source_op=outer_op,
                parent_operand=ValueRef.operand("source"),
                node_result=ValueRef.result("result"),
                guards=(
                    Guard.value_exact_i64(bound_fields[0]),
                    Guard.value_i64_range(
                        bound_fields[0], bounds[outer_op], bounds[outer_op]
                    ),
                ),
            ),
            SourceNode.adjacent_definition(
                "inner",
                source_op=inner_op,
                parent="outer",
                parent_operand=ValueRef.operand(value_fields[0]),
                node_result=ValueRef.result("result"),
                guards=(
                    Guard.value_exact_i64(bound_fields[1]),
                    Guard.value_i64_range(
                        bound_fields[1], bounds[inner_op], bounds[inner_op]
                    ),
                ),
            ),
        ),
        emit=_integer_pack_emits(pack_instruction, pack, source, saturation=1),
        report_key=(
            f"native_saturating_signed_i8x{pack_instruction.native_lane_count}_to_i4_"
            f"{order}_{value_fields[0]}_{value_fields[1]}"
        ),
    )


AIE2P_PACKET_CONVERSION_RULES = (
    *(
        _mxfp8_e4m3fn_e8m0_to_bf16_rule(lane_count, schema)
        for lane_count, schema in _MXFP8_E4M3FN_E8M0_RULE_SHAPES
    ),
    *(
        _saturating_i4_pack_rule(instruction, outer_op, inner_op, value_fields)
        for instruction in INTEGER_PACK_INSTRUCTIONS
        if instruction.bit_width == 4
        for outer_op, inner_op in (
            (vector.vector_minsi, vector.vector_maxsi),
            (vector.vector_maxsi, vector.vector_minsi),
        )
        for value_fields in product(("lhs", "rhs"), repeat=2)
    ),
    *(
        _integer_shift_rule(source_op, rule_shape)
        for source_op in (vector.vector_shli, vector.vector_shrui, vector.vector_shrsi)
        for rule_shape in INTEGER_SHIFT_RULE_SHAPES
    ),
    *(
        _integer_bitunpack_rule(source_op, source_kind, source_lane_count)
        for source_op, source_kind in (
            (vector.vector_bitunpacku, "u"),
            (vector.vector_bitunpacks, "s"),
        )
        for source_lane_count in I4_UNPACK_SOURCE_LANE_COUNTS
    ),
    *(
        _integer_widen_rule(source_op, signedness, rule_shape)
        for source_op, signedness in (
            (vector.vector_extui, "unsigned"),
            (vector.vector_extsi, "signed"),
        )
        for rule_shape in INTEGER_WIDEN_RULE_SHAPES
    ),
    *(
        _integer_truncation_rule(rule_shape)
        for rule_shape in INTEGER_TRUNCATION_RULE_SHAPES
    ),
    *(_integer_pack_rule(rule_shape) for rule_shape in INTEGER_PACK_RULE_SHAPES),
    *(
        _float_to_fp8_vector_rule(source_format, fp8_format, rule_shape)
        for source_format in FLOAT_PACKET_SOURCE_FORMATS
        for fp8_format in FLOAT8_PACKET_FORMATS
        for rule_shape in (
            _FLOAT8_NARROW_16BIT_RULE_SHAPES
            if source_format.bit_width == 16
            else _FLOAT8_NARROW_F32_RULE_SHAPES
        )
    ),
    *(
        rule
        for fp8_format in FLOAT8_PACKET_FORMATS
        for rule_shape in FLOAT_PACKET_RULE_SHAPES
        for rule in (
            _fp8_to_bf16_vector_rule(fp8_format, rule_shape),
            _fp8_to_f32_vector_rule(fp8_format, rule_shape),
        )
    ),
    *(_f32_to_bf16_vector_rule(rule_shape) for rule_shape in FLOAT_PACKET_RULE_SHAPES),
    *(_bf16_to_f32_vector_rule(rule_shape) for rule_shape in FLOAT_PACKET_RULE_SHAPES),
)
