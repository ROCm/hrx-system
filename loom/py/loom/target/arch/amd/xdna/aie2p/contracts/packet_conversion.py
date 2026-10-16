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
from loom.target.arch.amd.xdna.aie2p.contracts.f32_accumulator import (
    F32AccumulatorProgram,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import (
    IntegerSignedness,
    PacketProgram,
    shift_integer_packet_fixed,
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
    TypePattern,
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
class ComposedIntegerWidenRuleShape:
    """Logical lane interval realized by a composed packet program."""

    # Source integer element type.
    input_element: str
    # Result integer element type.
    result_element: str
    # First logical lane count realized by this rule.
    minimum_lane_count: int
    # Last logical lane count realized by this rule.
    maximum_lane_count: int
    # Number of 512-bit result packets containing logical lanes.
    result_packet_count: int

    def __post_init__(self) -> None:
        result_lanes_per_packet = 512 // int(self.result_element[1:])
        if not (
            1 <= self.result_packet_count
            and 1 <= self.minimum_lane_count <= self.maximum_lane_count
            and self.minimum_lane_count
            > (self.result_packet_count - 1) * result_lanes_per_packet
            and self.maximum_lane_count
            <= self.result_packet_count * result_lanes_per_packet
        ):
            raise ValueError(
                "composed integer widening interval crosses a result packet boundary"
            )

    @property
    def input_type(self) -> Vector:
        """Source-visible input type interval."""

        return Vector(
            self.input_element,
            minimum_lanes=self.minimum_lane_count,
            maximum_lanes=self.maximum_lane_count,
        )

    @property
    def result_type(self) -> Vector:
        """Source-visible result type interval."""

        return Vector(
            self.result_element,
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
            f"native_{signedness}_{self.input_element}x{lane_range}_to_"
            f"{self.result_element}x{lane_range}"
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

    # Widened BF16 packet, or None when a consumer uses the components directly.
    result: ValueRef | None
    # Unsigned finite BF16 payload magnitude.
    finite_magnitude: ValueRef
    # BF16 sign bits for the source payload.
    sign: ValueRef
    # Canonical BF16 NaN packet.
    canonical_nan: ValueRef
    # E4M3FN NaN predicate, or None for formats with infinity encodings.
    is_nan: ValueRef | None
    # Ordered descriptor program defining the reusable values.
    emits: tuple[ContractEmit, ...]


@dataclass(frozen=True, slots=True)
class _MxBf16PayloadProgram:
    """BF16 components produced from one finite MX payload group."""

    # Unsigned finite BF16 payload magnitude.
    finite_magnitude: ValueRef
    # BF16 sign bits for the source payload.
    sign: ValueRef
    # Canonical BF16 NaN packet used for an E8M0 NaN scale.
    canonical_nan: ValueRef
    # Ordered descriptor program defining the reusable values.
    emits: tuple[ContractEmit, ...]


@dataclass(frozen=True, slots=True)
class FloatPacketFormat:
    """One IEEE-like floating-point encoding carried in native i32 packets."""

    # IR vector element type.
    element: str
    # Stable type spelling used by compile reports.
    report_name: str
    # Number of explicit significand bits.
    mantissa_bits: int
    # Encoded exponent bias.
    exponent_bias: int
    # Number of encoded exponent bits.
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
        """Mask retaining the encoded exponent and fraction."""

        return self.sign_bit - 1

    @property
    def fraction_mask(self) -> int:
        """Mask retaining the explicit significand bits."""

        return (1 << self.mantissa_bits) - 1

    @property
    def hidden_bit(self) -> int:
        """Implicit leading bit of a normal significand."""

        return 1 << self.mantissa_bits

    @property
    def infinity_bits(self) -> int:
        """Unsigned infinity encoding."""

        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def maximum_exponent(self) -> int:
        """Largest finite unbiased exponent."""

        return (1 << self.exponent_bits) - 2 - self.exponent_bias

    @property
    def minimum_normal_exponent(self) -> int:
        """Smallest normal unbiased exponent."""

        return 1 - self.exponent_bias

    @property
    def minimum_rounding_exponent(self) -> int:
        """Smallest exponent that can round to a nonzero value."""

        return self.minimum_normal_exponent - self.mantissa_bits - 1

    def exponent_bits_for(self, unbiased_exponent: int) -> int:
        """Returns the normal encoding for an unbiased exponent."""

        return (unbiased_exponent + self.exponent_bias) << self.mantissa_bits


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

# VUPS cannot produce sixteen-bit accumulator lanes. Byte-to-halfword
# extension instead interleaves every source byte with a zero or sign-fill byte
# in one or two ordinary X result packets.
BYTE_TO_HALFWORD_WIDEN_RULE_SHAPES = (
    ComposedIntegerWidenRuleShape("i8", "i16", 1, 32, 1),
    ComposedIntegerWidenRuleShape("i8", "i16", 33, 64, 2),
)

# Byte-to-i64 extension composes the native i8-to-i32 and i32-to-i64 VUPS
# forms. Wider authored vectors are split by the shared cast packetizer.
BYTE_TO_I64_WIDEN_RULE_SHAPES = (
    ComposedIntegerWidenRuleShape("i8", "i64", 1, 8, 1),
    ComposedIntegerWidenRuleShape("i8", "i64", 9, 16, 2),
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

# Exact packet programs do not participate in fused-memory matching or need a
# distinct report identity for the exact sixteen-lane shape. Share its physical
# carrier with the partial-width interval instead of duplicating a large emit
# program. The 32-lane shape remains distinct because its binary32 value uses
# accumulator storage while widths 17-31 use ordinary vector storage.
_EXACT_FLOAT_PACKET_RULE_SHAPES = (
    FloatPacketRuleShape(16, 1, 16),
    FloatPacketRuleShape(32, 17, 31),
    FloatPacketRuleShape(32, 32, 32),
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

_F16_PACKET_FORMAT = FloatPacketFormat("f16", "binary16", 10, 15, 5)
_BF16_PACKET_FORMAT = FloatPacketFormat("bf16", "bfloat16", 7, 127, 8)
_F32_PACKET_FORMAT = FloatPacketFormat("f32", "binary32", 23, 127, 8)
FLOAT_PACKET_FORMATS = (
    _F16_PACKET_FORMAT,
    _BF16_PACKET_FORMAT,
    _F32_PACKET_FORMAT,
)

# Exact IEEE-like 16-bit conversions use the generic integer packet programs
# below. Native rows remain ahead of these fallbacks so policy-compatible or
# proven inputs keep the shortest available sequence.
_EXACT_FLOAT16_PACKET_FORMATS = (_F16_PACKET_FORMAT,)

_FP8_PAYLOAD_AND_SIGN_MASK = 0x807F
_FP8_SUBNORMAL_THRESHOLD = 0x0080
_CANONICAL_BF16_NAN = 0x7FC0


def _mxfp8_e4m3fn_e8m0_schema(lane_count: int) -> EncodingOperandSummaryDef:
    """Builds one exact dense MXFP8 schema with groups of at most 32 lanes."""

    scale_group_element_count = min(lane_count, 32)
    return EncodingOperandSummaryDef(
        element_format=encoding.enum_fact(encoding.NumericFormat, "f8e4m3fn"),
        scale_format=encoding.enum_fact(encoding.NumericFormat, "e8m0"),
        payload_packing=encoding.enum_fact(encoding.PayloadPacking, "dense_lanes"),
        scale_topology=encoding.enum_fact(encoding.ScaleTopology, "block_1d"),
        affine_policy=encoding.enum_fact(encoding.AffinePolicy, "scale_only"),
        payload_element_count=lane_count,
        scale_group_element_count=scale_group_element_count,
        scale_group_shape=(scale_group_element_count,),
        scale_operand_count=1,
    )


MXFP8_E4M3FN_E8M0_X8_SCHEMA = _mxfp8_e4m3fn_e8m0_schema(8)
MXFP8_E4M3FN_E8M0_X32_SCHEMA = _mxfp8_e4m3fn_e8m0_schema(32)
MXFP8_E4M3FN_E8M0_X64_SCHEMA = _mxfp8_e4m3fn_e8m0_schema(64)

_MXFP8_E4M3FN_E8M0_RULE_SHAPES = (
    (8, MXFP8_E4M3FN_E8M0_X8_SCHEMA),
    (32, MXFP8_E4M3FN_E8M0_X32_SCHEMA),
    (64, MXFP8_E4M3FN_E8M0_X64_SCHEMA),
)


def _mxfp4_e2m1_e8m0_schema(lane_count: int) -> EncodingOperandSummaryDef:
    """Builds one exact packed MXFP4 group schema."""

    return EncodingOperandSummaryDef(
        element_format=encoding.enum_fact(encoding.NumericFormat, "f4e2m1"),
        scale_format=encoding.enum_fact(encoding.NumericFormat, "e8m0"),
        payload_packing=encoding.enum_fact(
            encoding.PayloadPacking, "little_endian_nibbles"
        ),
        scale_topology=encoding.enum_fact(encoding.ScaleTopology, "block_1d"),
        affine_policy=encoding.enum_fact(encoding.AffinePolicy, "scale_only"),
        zero_scale_fallback=True,
        payload_register_count=lane_count // 8,
        payload_element_count=lane_count,
        scale_group_element_count=32,
        scale_group_shape=(32,),
        scale_operand_count=1,
    )


MXFP4_E2M1_E8M0_X32_SCHEMA = _mxfp4_e2m1_e8m0_schema(32)
MXFP4_E2M1_E8M0_X64_SCHEMA = _mxfp4_e2m1_e8m0_schema(64)

_MXFP4_E2M1_E8M0_RULE_SHAPES = (
    (32, MXFP4_E2M1_E8M0_X32_SCHEMA),
    (64, MXFP4_E2M1_E8M0_X64_SCHEMA),
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


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _exact_vector(element: str, element_count: int) -> Vector:
    return Vector(element, lanes=element_count)


def integer_widen_state_emits(
    ups_mode: int,
    *,
    temporary_prefix: str = "",
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Builds the explicit configured state consumed by one VUPS form."""

    shift = ValueRef.temporary(f"{temporary_prefix}shift")
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
    *,
    bind_result_type: bool = False,
    temporary_prefix: str = "",
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Bridges a native accumulator result to its source-visible carrier."""

    if result_accumulator_unit_count is None:
        result_accumulator_unit_count = instruction.accumulator_unit_count
    if (
        instruction.direct_accumulator_result
        and result_accumulator_unit_count == instruction.accumulator_unit_count
    ):
        return result, ()

    native_result = ValueRef.temporary(f"{temporary_prefix}wide_result")
    output_emits: list[ContractEmit] = []
    vector_units: list[ValueRef] = []
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )
    for unit in range(result_accumulator_unit_count):
        accumulator_unit = native_result
        if instruction.accumulator_unit_count > 1:
            accumulator_unit = ValueRef.temporary(
                f"{temporary_prefix}accumulator_unit_{unit}"
            )
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
            else ValueRef.temporary(f"{temporary_prefix}vector_unit_{unit}")
        )
        output_emits.append(
            EmitDescriptorOp(
                descriptor=move_from_accumulator,
                operands={"src": accumulator_unit},
                results={"dst": vector_unit},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result_accumulator_unit_count > 1 or bind_result_type
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


def integer_widen_rule(
    source_op: Op,
    signedness: str,
    rule_shape: IntegerWidenRuleShape,
    *,
    input_type: TypePattern | None = None,
    result_type: TypePattern | None = None,
    source: ValueRef | None = None,
    prefix_emits: tuple[ContractEmit, ...] = (),
    extra_guards: tuple[Guard, ...] = (),
    report_key: str | None = None,
) -> DescriptorRule:
    instruction = rule_shape.instruction
    source = ValueRef.operand("input") if source is None else source
    input_emits = prefix_emits
    if instruction.slice_input:
        sliced_source = ValueRef.temporary("source_w")
        input_emits = (
            *input_emits,
            EmitRegisterSlice(
                source=source,
                result=sliced_source,
                unit_count=1,
            ),
        )
        source = sliced_source
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
            Guard.value_type(
                "input", rule_shape.input_type if input_type is None else input_type
            ),
            Guard.value_type(
                "result", rule_shape.result_type if result_type is None else result_type
            ),
            *extra_guards,
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
        report_key=(
            rule_shape.report_key(signedness) if report_key is None else report_key
        ),
    )


def _byte_to_halfword_widen_rule(
    source_op: Op,
    signedness: IntegerSignedness,
    rule_shape: ComposedIntegerWidenRuleShape,
) -> DescriptorRule:
    """Extends byte lanes by interleaving their low and fill bytes."""

    program = PacketProgram(8, "byte_to_halfword_")
    source = ValueRef.operand("input")
    zero = program.binary("zero", "sub.i8x64", source, source)
    fill = zero
    if signedness == "signed":
        negative = program.compare_signed_less_than("negative", source, zero)
        all_ones = program.splat("all_ones", -1)
        fill = program.select("sign_fill", all_ones, zero, negative)

    result_packets: list[ValueRef] = []
    for packet in range(rule_shape.result_packet_count):
        control = program.constant(
            f"packet_{packet}_control",
            I8_INTERLEAVE_CONTROL + packet,
            descriptor_key="amd.xdna.aie2p.constant.i32.mova",
        )
        result_packets.append(
            program.operation(
                (None if rule_shape.result_packet_count == 1 else f"packet_{packet}"),
                "shuffle.x.configured",
                "dst",
                s1=source,
                s2=fill,
                mod=control,
            )
        )
    if rule_shape.result_packet_count > 1:
        program.emits.append(
            EmitRegisterConcat(
                sources=result_packets,
                result=ValueRef.result("result"),
            )
        )

    descriptor = next(
        emit.descriptor
        for emit in reversed(program.emits)
        if isinstance(emit, EmitDescriptorOp)
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", rule_shape.input_type),
            Guard.value_type("result", rule_shape.result_type),
        ),
        emit=tuple(program.emits),
        report_key=rule_shape.report_key(signedness),
    )


def _byte_to_i64_widen_rule(
    source_op: Op,
    signedness: IntegerSignedness,
    rule_shape: ComposedIntegerWidenRuleShape,
) -> DescriptorRule:
    """Extends byte lanes through exact i32 and i64 VUPS stages."""

    source_w = ValueRef.temporary("byte_to_i64_source_w")
    intermediate = ValueRef.temporary("byte_to_i64_i32")
    shift, initial_state_emits = integer_widen_state_emits(
        _I8_TO_I32_W.ups_mode,
        temporary_prefix="byte_to_i64_",
    )
    intermediate_native, intermediate_output_emits = integer_widen_result_emits(
        _I8_TO_I32_W,
        intermediate,
        result_accumulator_unit_count=1,
        bind_result_type=True,
        temporary_prefix="byte_to_i64_i32_",
    )
    first_widen = _descriptor(
        f"amd.xdna.aie2p.widen.{_I8_TO_I32_W.physical_shape}.{signedness}.configured"
    )

    result_native, result_output_emits = integer_widen_result_emits(
        _I32_TO_I64_X,
        ValueRef.result("result"),
        result_accumulator_unit_count=rule_shape.result_packet_count,
        temporary_prefix="byte_to_i64_i64_",
    )
    final_widen = _descriptor(
        f"amd.xdna.aie2p.widen.{_I32_TO_I64_X.physical_shape}.{signedness}.configured"
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=final_widen,
        guards=(
            Guard.value_type("input", rule_shape.input_type),
            Guard.value_type("result", rule_shape.result_type),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
                result=source_w,
                unit_count=1,
            ),
            *initial_state_emits,
            EmitDescriptorOp(
                descriptor=first_widen,
                operands={"src": source_w, "su": shift},
                results={"dst": intermediate_native},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            *intermediate_output_emits,
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.state.ups-mode.immediate"),
                immediates={"i": _I32_TO_I64_X.ups_mode},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=final_widen,
                operands={"src": intermediate, "su": shift},
                results={"dst": result_native},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            *result_output_emits,
        ),
        report_key=rule_shape.report_key(signedness),
    )


def _widen_integer_packet_to_i32(
    input_element: str,
    signedness: IntegerSignedness,
    source: ValueRef,
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Widens one partial X-carried integer packet to sixteen i32 lanes."""

    instruction = {
        "i8": _I8_TO_I32_W,
        "i16": _I16_TO_I32_W,
    }[input_element]
    input_emits: list[ContractEmit] = []
    if instruction.slice_input:
        sliced_source = ValueRef.temporary("integer_to_f32_source_w")
        input_emits.append(
            EmitRegisterSlice(
                source=source,
                result=sliced_source,
                unit_count=1,
            )
        )
        source = sliced_source

    shift, state_emits = integer_widen_state_emits(instruction.ups_mode)
    result = ValueRef.temporary("integer_to_f32_widened")
    native_result, output_emits = integer_widen_result_emits(
        instruction,
        result,
        result_accumulator_unit_count=1,
        bind_result_type=True,
    )
    widen = _descriptor(
        f"amd.xdna.aie2p.widen.{instruction.physical_shape}.{signedness}.configured"
    )
    return result, (
        *input_emits,
        *state_emits,
        EmitDescriptorOp(
            descriptor=widen,
            operands={"src": source, "su": shift},
            results={"dst": native_result},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        *output_emits,
    )


def _integer_to_f32_packet_rule(
    source_op: Op,
    input_element: str,
    signedness: IntegerSignedness,
) -> DescriptorRule:
    """Converts one integer packet to correctly rounded binary32 lanes."""

    input_value = ValueRef.operand("input")
    input_emits: tuple[ContractEmit, ...] = ()
    if input_element != "i32":
        input_value, input_emits = _widen_integer_packet_to_i32(
            input_element, signedness, input_value
        )

    packet = PacketProgram(32, temporary_prefix="integer_to_f32_")
    accumulator = F32AccumulatorProgram(temporary_prefix="integer_to_f32_")
    if input_element != "i32":
        # The interior mantissa leaves the complete signed or unsigned i8/i16
        # domain in one unit-spaced F32 binade. Integer addition to the bit
        # pattern and floating subtraction therefore recover every lane
        # exactly.
        bias = packet.splat("bias", 0x4B010000)
        biased = packet.binary("biased", "add.i32x16", input_value, bias)
        biased_accumulator = accumulator.vector_to_accumulator("biased", biased)
        bias_accumulator = accumulator.vector_to_accumulator("bias", bias)
        result_accumulator = accumulator.binary(
            "result",
            biased_accumulator,
            bias_accumulator,
            "amd.xdna.aie2p.sub.f32x16.configured",
        )
        primary_descriptor = _descriptor("amd.xdna.aie2p.sub.f32x16.configured")
    else:
        # The high and low components are exact integers in F32. Their single
        # final addition performs exactly the rounding required by the source
        # i32-to-F32 conversion, including signed cancellation and midpoint
        # ties across the full input domain.
        high = shift_integer_packet_fixed(
            packet,
            "high",
            input_value,
            -16,
            signedness=signedness,
        )
        low_mask = packet.splat("low_mask", 0xFFFF)
        low = packet.binary("low", "and.bits512", input_value, low_mask)

        high_bias = packet.splat("high_bias", 0x53010000)
        high_biased = packet.binary("high_biased", "add.i32x16", high, high_bias)
        low_bias = packet.splat("low_bias", 0x4B010000)
        low_biased = packet.binary("low_biased", "add.i32x16", low, low_bias)

        high_biased_accumulator = accumulator.vector_to_accumulator(
            "high_biased", high_biased
        )
        high_bias_accumulator = accumulator.vector_to_accumulator(
            "high_bias", high_bias
        )
        high_float_accumulator = accumulator.binary(
            "high_float",
            high_biased_accumulator,
            high_bias_accumulator,
            "amd.xdna.aie2p.sub.f32x16.configured",
        )
        low_biased_accumulator = accumulator.vector_to_accumulator(
            "low_biased", low_biased
        )
        low_bias_accumulator = accumulator.vector_to_accumulator("low_bias", low_bias)
        low_float_accumulator = accumulator.binary(
            "low_float",
            low_biased_accumulator,
            low_bias_accumulator,
            "amd.xdna.aie2p.sub.f32x16.configured",
        )
        result_accumulator = accumulator.binary(
            "result",
            high_float_accumulator,
            low_float_accumulator,
            "amd.xdna.aie2p.add.f32x64.configured",
        )
        primary_descriptor = _descriptor("amd.xdna.aie2p.add.f32x64.configured")

    accumulator.accumulator_to_vector(
        "result", result_accumulator, ValueRef.result("result")
    )
    packet_type = Vector(
        input_element,
        minimum_static_elements=1,
        maximum_static_elements=16,
    )
    result_type = Vector(
        "f32",
        minimum_static_elements=1,
        maximum_static_elements=16,
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=primary_descriptor,
        guards=(
            Guard.value_type("input", packet_type),
            Guard.value_type("result", result_type),
        ),
        emit=(*input_emits, *packet.emits, *accumulator.emits),
        report_key=f"exact_{signedness}_{input_element}_to_binary32",
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
            Guard.value_not_subnormal_or_instance_flags_has_all(
                "input", "subnormal", "daz"
            ),
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
            Guard.value_not_subnormal_or_instance_flags_has_all(
                "input", "subnormal", "daz"
            ),
        ),
        emit=_bf16_to_f32_emits(rule_shape, ValueRef.operand("input")),
        report_key=(
            f"native_bfloat16x{rule_shape.report_lane_range}_to_"
            f"binary32x{rule_shape.report_lane_range}"
        ),
    )


@dataclass(frozen=True, slots=True)
class _Float16WideningState:
    """Shared constants for widening one IEEE-like 16-bit packet."""

    # Source format defining exponent and significand geometry.
    source_format: FloatPacketFormat
    # Mask selecting the source sign bit.
    sign_mask: ValueRef
    # Mask selecting the unsigned source payload.
    nonsign_mask: ValueRef
    # Smallest normal unsigned source payload.
    minimum_normal: ValueRef
    # Unsigned source infinity payload.
    infinity: ValueRef
    # Encoded exponent delta from finite source values to binary32.
    normal_bias: ValueRef
    # Encoded exponent delta from source special values to binary32.
    special_bias: ValueRef
    # Zero binary32 payload.
    zero: ValueRef


@dataclass(frozen=True, slots=True)
class _Float16NarrowingState:
    """Shared constants for narrowing binary32 to one IEEE-like 16-bit packet."""

    # Destination format defining exponent and significand geometry.
    result_format: FloatPacketFormat
    # Mask selecting the binary32 sign bit.
    sign_mask: ValueRef
    # Mask selecting the unsigned binary32 payload.
    nonsign_mask: ValueRef
    # Mask selecting the explicit binary32 significand bits.
    fraction_mask: ValueRef
    # Implicit leading bit of a normal binary32 significand.
    hidden_bit: ValueRef
    # Encoded exponent delta from binary32 to normal destination values.
    normal_bias: ValueRef
    # Smallest binary32 payload that maps to a normal destination value.
    minimum_normal: ValueRef
    # First binary32 exponent whose values all overflow the destination.
    overflow_exponent: ValueRef
    # Unsigned binary32 infinity payload.
    source_infinity: ValueRef
    # Unsigned destination infinity payload.
    result_infinity: ValueRef
    # Destination quiet-NaN payload bit.
    quiet_nan_bit: ValueRef
    # Zero destination payload.
    zero: ValueRef
    # One binary32 integer lane.
    one: ValueRef


def _float16_chunk_to_f32(
    program: PacketProgram,
    source: ValueRef,
    chunk_index: int,
    *,
    result_name: str | None,
    state: _Float16WideningState,
) -> ValueRef:
    """Widens one sixteen-lane IEEE-like bit packet to binary32 bits."""

    source_format = state.source_format
    position_shift = _F32_PACKET_FORMAT.mantissa_bits - source_format.mantissa_bits
    prefix = f"chunk_{chunk_index}"
    sign = program.binary(f"{prefix}_sign", "and.bits512", source, state.sign_mask)
    sign = shift_integer_packet_fixed(
        program,
        f"{prefix}_positioned_sign",
        sign,
        _F32_PACKET_FORMAT.bit_width - source_format.bit_width,
    )
    payload = program.binary(
        f"{prefix}_payload", "and.bits512", source, state.nonsign_mask
    )

    positioned_payload = shift_integer_packet_fixed(
        program, f"{prefix}_positioned_payload", payload, position_shift
    )
    normal = program.binary(
        f"{prefix}_normal", "add.i32x16", positioned_payload, state.normal_bias
    )
    special = program.binary(
        f"{prefix}_special", "add.i32x16", positioned_payload, state.special_bias
    )
    is_special = program.compare_unsigned_greater_equal(
        f"{prefix}_is_special", payload, state.infinity
    )
    normal_or_special = program.select(
        f"{prefix}_normal_or_special", special, normal, is_special
    )

    subnormal = state.zero
    for normalization_shift in range(source_format.mantissa_bits, 0, -1):
        threshold = program.splat(
            f"{prefix}_subnormal_{normalization_shift}_threshold",
            1 << (source_format.mantissa_bits - normalization_shift),
        )
        active = program.compare_unsigned_greater_equal(
            f"{prefix}_subnormal_{normalization_shift}_active",
            payload,
            threshold,
        )
        positioned = shift_integer_packet_fixed(
            program,
            f"{prefix}_subnormal_{normalization_shift}_positioned",
            payload,
            position_shift + normalization_shift,
        )
        bias = program.splat(
            f"{prefix}_subnormal_{normalization_shift}_bias",
            (
                _F32_PACKET_FORMAT.exponent_bias
                - source_format.exponent_bias
                - normalization_shift
            )
            << _F32_PACKET_FORMAT.mantissa_bits,
        )
        candidate = program.binary(
            f"{prefix}_subnormal_{normalization_shift}",
            "add.i32x16",
            positioned,
            bias,
        )
        subnormal = program.select(
            f"{prefix}_subnormal_through_{normalization_shift}",
            candidate,
            subnormal,
            active,
        )

    is_normal = program.compare_unsigned_greater_equal(
        f"{prefix}_is_normal", payload, state.minimum_normal
    )
    magnitude = program.select(
        f"{prefix}_magnitude", normal_or_special, subnormal, is_normal
    )
    return program.operation(
        result_name,
        "or.bits512",
        "d",
        s1=magnitude,
        s2=sign,
    )


def _float16_to_f32_emits(
    source_format: FloatPacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> tuple[ContractEmit, ...]:
    """Widens one source-visible IEEE-like 16-bit packet to binary32."""

    if source_format.bit_width != 16:
        raise ValueError("software float widening requires a 16-bit source")
    program = PacketProgram(32, f"{source_format.element}_widen_")
    program.state("saturation", 1)
    program.state("ups-mode", 0)
    chunks = _float_source_i32_chunks(program, source_format, rule_shape)

    state = _Float16WideningState(
        source_format=source_format,
        sign_mask=program.splat("sign_mask", source_format.sign_bit),
        nonsign_mask=program.splat("nonsign_mask", source_format.nonsign_mask),
        minimum_normal=program.splat("minimum_normal", source_format.hidden_bit),
        infinity=program.splat("infinity", source_format.infinity_bits),
        normal_bias=program.splat(
            "normal_bias",
            (_F32_PACKET_FORMAT.exponent_bias - source_format.exponent_bias)
            << _F32_PACKET_FORMAT.mantissa_bits,
        ),
        special_bias=program.splat(
            "special_bias",
            (
                (1 << _F32_PACKET_FORMAT.exponent_bits)
                - (1 << source_format.exponent_bits)
            )
            << _F32_PACKET_FORMAT.mantissa_bits,
        ),
        zero=program.splat("zero", 0),
    )

    direct_result = len(chunks) == 1
    results = tuple(
        _float16_chunk_to_f32(
            program,
            chunk,
            index,
            result_name=None if direct_result else f"chunk_{index}_result",
            state=state,
        )
        for index, chunk in enumerate(chunks)
    )
    if direct_result:
        return tuple(program.emits)

    if rule_shape.minimum_lane_count == rule_shape.maximum_lane_count == 32:
        accumulator_results = tuple(
            program.operation(
                f"chunk_{index}_accumulator",
                "move.vector512.to.accumulator512",
                "dst",
                src=result,
            )
            for index, result in enumerate(results)
        )
        program.emits.append(
            EmitRegisterConcat(
                sources=accumulator_results,
                result=ValueRef.result("result"),
                result_type=_exact_vector("f32", 32),
            )
        )
    else:
        program.emits.append(
            EmitRegisterConcat(sources=results, result=ValueRef.result("result"))
        )
    return tuple(program.emits)


def _float16_to_f32_vector_rule(
    source_format: FloatPacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=_descriptor("amd.xdna.aie2p.narrow.2x.c-to-x.unsigned.configured"),
        guards=(
            Guard.value_type("input", rule_shape.vector_type(source_format.element)),
            Guard.value_type("result", rule_shape.vector_type("f32")),
        ),
        emit=_float16_to_f32_emits(source_format, rule_shape),
        report_key=f"exact_{source_format.report_name}_to_binary32_packet",
    )


def _f32_chunk_to_float16(
    program: PacketProgram,
    source: ValueRef,
    chunk_index: int,
    *,
    result_name: str | None,
    state: _Float16NarrowingState,
) -> ValueRef:
    """Narrows one sixteen-lane binary32 bit packet to a 16-bit format."""

    result_format = state.result_format
    mantissa_shift = _F32_PACKET_FORMAT.mantissa_bits - result_format.mantissa_bits
    prefix = f"chunk_{chunk_index}"
    sign_bits = program.binary(
        f"{prefix}_sign_bits", "and.bits512", source, state.sign_mask
    )
    sign_accumulator, sign_filler = _i32_packet_accumulator(
        program, f"{prefix}_sign", sign_bits
    )
    sign = _round_i32_packet_to_i16(
        program,
        f"{prefix}_sign",
        sign_accumulator,
        sign_filler,
        _F32_PACKET_FORMAT.bit_width - result_format.bit_width,
    )

    absolute = program.binary(
        f"{prefix}_absolute", "and.bits512", source, state.nonsign_mask
    )
    normal_adjusted = program.binary(
        f"{prefix}_normal_adjusted", "sub.i32x16", absolute, state.normal_bias
    )
    normal_accumulator, normal_filler = _i32_packet_accumulator(
        program, f"{prefix}_normal", normal_adjusted
    )
    normal = _round_i32_packet_to_i16(
        program,
        f"{prefix}_normal",
        normal_accumulator,
        normal_filler,
        mantissa_shift,
    )

    fraction = program.binary(
        f"{prefix}_fraction", "and.bits512", absolute, state.fraction_mask
    )
    significand = program.binary(
        f"{prefix}_significand", "or.bits512", fraction, state.hidden_bit
    )
    significand_accumulator, significand_filler = _i32_packet_accumulator(
        program, f"{prefix}_significand", significand
    )
    subnormal = state.zero
    for unbiased_exponent in range(
        result_format.minimum_rounding_exponent,
        result_format.minimum_normal_exponent,
    ):
        source_exponent = unbiased_exponent + _F32_PACKET_FORMAT.exponent_bias
        candidate = _round_i32_packet_to_i16(
            program,
            f"{prefix}_subnormal_{source_exponent}",
            significand_accumulator,
            significand_filler,
            (
                _F32_PACKET_FORMAT.mantissa_bits
                + 1
                - result_format.exponent_bias
                - result_format.mantissa_bits
                - unbiased_exponent
            ),
        )
        threshold = program.splat(
            f"{prefix}_exponent_{source_exponent}_threshold",
            _F32_PACKET_FORMAT.exponent_bits_for(unbiased_exponent),
        )
        active = program.compare_unsigned_greater_equal(
            f"{prefix}_at_exponent_{source_exponent}", absolute, threshold
        )
        subnormal = program.select(
            f"{prefix}_subnormal_through_{source_exponent}",
            candidate,
            subnormal,
            active,
            element_bits=result_format.bit_width,
        )

    has_normal_exponent = program.compare_unsigned_greater_equal(
        f"{prefix}_has_normal_exponent", absolute, state.minimum_normal
    )
    finite = program.select(
        f"{prefix}_finite",
        normal,
        subnormal,
        has_normal_exponent,
        element_bits=result_format.bit_width,
    )
    overflows = program.compare_unsigned_greater_equal(
        f"{prefix}_overflows", absolute, state.overflow_exponent
    )
    finite = program.select(
        f"{prefix}_finite_or_infinity",
        state.result_infinity,
        finite,
        overflows,
        element_bits=result_format.bit_width,
    )

    shifted_nan_payload = shift_integer_packet_fixed(
        program, f"{prefix}_nan_payload_shifted", fraction, -mantissa_shift
    )
    program.state("saturation", 1)
    program.state("rounding", BF16_CONVERSION_ROUNDING)
    program.state("srs-mode", 0)
    nan_accumulator, nan_filler = _i32_packet_accumulator(
        program, f"{prefix}_nan", shifted_nan_payload
    )
    nan_payload = _round_i32_packet_to_i16(
        program, f"{prefix}_nan_payload", nan_accumulator, nan_filler, 0
    )
    nan_payload = program.binary(
        f"{prefix}_quiet_nan_payload",
        "or.bits512",
        nan_payload,
        state.quiet_nan_bit,
    )
    nan = program.binary(
        f"{prefix}_nan", "or.bits512", nan_payload, state.result_infinity
    )
    is_nan = program.compare_unsigned_greater_equal(
        f"{prefix}_is_nan", fraction, state.one
    )
    special = program.select(
        f"{prefix}_special",
        nan,
        state.result_infinity,
        is_nan,
        element_bits=result_format.bit_width,
    )
    is_special = program.compare_unsigned_greater_equal(
        f"{prefix}_is_special", absolute, state.source_infinity
    )
    magnitude = program.select(
        f"{prefix}_magnitude",
        special,
        finite,
        is_special,
        element_bits=result_format.bit_width,
    )
    return program.operation(
        result_name,
        "or.bits512",
        "d",
        s1=magnitude,
        s2=sign,
    )


def _f32_to_float16_emits(
    result_format: FloatPacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> tuple[ContractEmit, ...]:
    """Narrows one source-visible binary32 packet to a 16-bit format."""

    if result_format.bit_width != 16:
        raise ValueError("software float narrowing requires a 16-bit result")
    program = PacketProgram(32, f"{result_format.element}_narrow_")
    chunks = _float_source_i32_chunks(program, _F32_PACKET_FORMAT, rule_shape)
    program.state("saturation", 1)
    program.state("rounding", BF16_CONVERSION_ROUNDING)
    program.state("srs-mode", 0)

    state = _Float16NarrowingState(
        result_format=result_format,
        sign_mask=program.splat("sign_mask", -_F32_PACKET_FORMAT.sign_bit),
        nonsign_mask=program.splat("nonsign_mask", _F32_PACKET_FORMAT.nonsign_mask),
        fraction_mask=program.splat("fraction_mask", _F32_PACKET_FORMAT.fraction_mask),
        hidden_bit=program.splat("hidden_bit", _F32_PACKET_FORMAT.hidden_bit),
        normal_bias=program.splat(
            "normal_bias",
            (_F32_PACKET_FORMAT.exponent_bias - result_format.exponent_bias)
            << _F32_PACKET_FORMAT.mantissa_bits,
        ),
        minimum_normal=program.splat(
            "minimum_normal",
            _F32_PACKET_FORMAT.exponent_bits_for(result_format.minimum_normal_exponent),
        ),
        overflow_exponent=program.splat(
            "overflow_exponent",
            _F32_PACKET_FORMAT.exponent_bits_for(result_format.maximum_exponent + 1),
        ),
        source_infinity=program.splat(
            "source_infinity", _F32_PACKET_FORMAT.infinity_bits
        ),
        result_infinity=program.splat(
            "result_infinity",
            result_format.infinity_bits,
            element_bits=result_format.bit_width,
        ),
        quiet_nan_bit=program.splat(
            "quiet_nan_bit",
            1 << (result_format.mantissa_bits - 1),
            element_bits=result_format.bit_width,
        ),
        zero=program.splat("zero", 0, element_bits=result_format.bit_width),
        one=program.splat("one", 1),
    )

    direct_result = len(chunks) == 1
    results = tuple(
        _f32_chunk_to_float16(
            program,
            chunk,
            index,
            result_name=None if direct_result else f"chunk_{index}_result",
            state=state,
        )
        for index, chunk in enumerate(chunks)
    )
    if direct_result:
        return tuple(program.emits)

    result_words = []
    for index, result in enumerate(results):
        result_word = program.temporary(f"chunk_{index}_result_w")
        program.emits.append(
            EmitRegisterSlice(source=result, result=result_word, unit_count=1)
        )
        result_words.append(result_word)
    program.emits.append(
        EmitRegisterConcat(
            sources=tuple(result_words), result=ValueRef.result("result")
        )
    )
    return tuple(program.emits)


def _f32_to_float16_vector_rule(
    result_format: FloatPacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=_descriptor("amd.xdna.aie2p.narrow.2x.b-to-w.unsigned.configured"),
        guards=(
            Guard.value_type("input", rule_shape.vector_type("f32")),
            Guard.value_type("result", rule_shape.vector_type(result_format.element)),
        ),
        emit=_f32_to_float16_emits(result_format, rule_shape),
        report_key=f"exact_binary32_to_{result_format.report_name}_packet",
    )


def _fp8_to_bf16_emits(
    fp8_format: Float8PacketFormat,
    *,
    result_name: str | None,
    source: ValueRef | None = None,
    materialize_result: bool = True,
    temporary_prefix: str = "fp8_",
) -> _Float8WidenProgram:
    """Widens up to thirty-two FP8 lanes into exact BF16 bit patterns."""

    program = PacketProgram(16, temporary_prefix)
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
    finite = (
        program.binary("finite", "or.bits512", finite_unsigned, sign)
        if materialize_result
        else None
    )
    canonical_nan = program.splat("canonical_nan", _CANONICAL_BF16_NAN)

    is_nan = None
    if fp8_format.has_infinity:
        if finite is None:
            raise ValueError("component-only FP8 widening requires a finite format")
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
        if finite is not None:
            result = program.select(result_name, canonical_nan, finite, is_nan)
        else:
            result = None

    return _Float8WidenProgram(
        result=result,
        finite_magnitude=finite_unsigned,
        sign=sign,
        canonical_nan=canonical_nan,
        is_nan=is_nan,
        emits=tuple(program.emits),
    )


def _e8m0_scale_bf16_group_emits(
    *,
    finite_magnitude: ValueRef,
    sign: ValueRef,
    canonical_nan: ValueRef,
    payload_is_nan: ValueRef | None,
    scale: ValueRef,
    scale_byte_ordinal: int,
    result_name: str | None,
    temporary_prefix: str,
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Applies one packed E8M0 scale byte to a BF16 packet."""

    if scale_byte_ordinal < 0 or scale_byte_ordinal > 3:
        raise ValueError("E8M0 scale byte ordinal must be in [0, 3]")
    program = PacketProgram(16, temporary_prefix)

    scale_word = program.operation(
        "scale_word",
        "extract.i32.immediate",
        "dst",
        immediates={"idx": 0},
        s1=scale,
    )
    if scale_byte_ordinal:
        scale_word = program.operation(
            "selected_scale_word",
            "lshl.i32",
            "d0",
            s0=scale_word,
            s1=program.constant("scale_byte_shift", -8 * scale_byte_ordinal),
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

    infinity = program.splat("infinity", 0x7F80)
    magnitude = finite_magnitude

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

    subnormal_accumulators: list[ValueRef] = []
    for unit_ordinal in range(2):
        magnitude_unit = program.temporary(f"magnitude_{unit_ordinal}")
        program.emits.append(
            EmitRegisterSlice(
                source=magnitude,
                result=magnitude_unit,
                unit_offset=unit_ordinal,
                unit_count=1,
            )
        )
        subnormal_units_i32 = program.operation(
            f"subnormal_units_{unit_ordinal}_i32",
            "convert.floor.bf16x16.to.i32x16",
            "dst",
            src=magnitude_unit,
            shft=program.shift(9),
        )
        subnormal_accumulators.append(
            program.operation(
                f"subnormal_units_{unit_ordinal}_accumulator",
                "move.vector512.to.accumulator512",
                "dst",
                src=subnormal_units_i32,
            )
        )
    shift_zero = program.shift(0)
    shift_one = program.shift(1)
    shift_two = program.shift(2)
    shift_three = program.shift(3)
    program.state("rounding", 12)
    program.state("srs-mode", 0)
    program.state("saturation", 0)

    def narrow_subnormal_units(name: str, shift: ValueRef) -> ValueRef:
        narrowed_units = tuple(
            program.operation(
                f"{name}_{unit_ordinal}",
                "narrow.2x.b-to-w.unsigned.configured",
                "dst",
                src=accumulator,
                su=shift,
            )
            for unit_ordinal, accumulator in enumerate(subnormal_accumulators)
        )
        result = program.temporary(name)
        program.emits.append(
            EmitRegisterConcat(
                sources=narrowed_units,
                result=result,
                result_type=_exact_vector("i16", 32),
            )
        )
        return result

    units = narrow_subnormal_units("units", shift_zero)
    scale_two = narrow_subnormal_units("scale_two", shift_one)
    scale_one = narrow_subnormal_units("scale_one", shift_two)
    scale_zero = narrow_subnormal_units("scale_zero", shift_three)

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
    payload_fixed_result = zero_fixed_result
    if payload_is_nan is not None:
        payload_fixed_result = program.select(
            "payload_fixed_result",
            canonical_nan,
            zero_fixed_result,
            payload_is_nan,
        )
    scale_is_nan = program.operation(
        "scale_is_nan", "cmp.eq.i32", "d0", s0=scale_byte, s1=byte_mask
    )
    result = program.operation(
        result_name,
        "select.i32x16",
        "d",
        s1=canonical_nan,
        s2=payload_fixed_result,
        sel=scalar_mask("scale_nan_mask", scale_is_nan),
    )

    return result, tuple(program.emits)


def _mxfp8_e4m3fn_e8m0_to_bf16_rule(
    lane_count: int,
    schema: EncodingOperandSummaryDef,
) -> DescriptorRule:
    """Decodes one or two MXFP8 groups without scalarizing payload lanes."""

    group_count = 2 if lane_count == 64 else 1
    emits: list[ContractEmit] = []
    group_results: list[ValueRef] = []
    payload_source = ValueRef.operand("payload")
    for group_ordinal in range(group_count):
        if group_ordinal:
            high_payload = ValueRef.temporary("mxfp8_high_payload_w")
            payload_source = ValueRef.temporary("mxfp8_high_payload_x")
            emits.extend(
                (
                    EmitRegisterSlice(
                        source=ValueRef.operand("payload"),
                        result=high_payload,
                        unit_offset=1,
                        unit_count=1,
                    ),
                    EmitRegisterConcat(
                        sources=(high_payload, high_payload),
                        result=payload_source,
                        result_type=_exact_vector("f8E4M3", 64),
                    ),
                )
            )
        payload_program = _fp8_to_bf16_emits(
            _F8E4M3_PACKET_FORMAT,
            result_name=None,
            source=payload_source,
            materialize_result=False,
            temporary_prefix=f"mxfp8_g{group_ordinal}_payload_",
        )
        if payload_program.is_nan is None:
            raise ValueError("MXFP8 E4M3FN widening must produce a NaN predicate")
        emits.extend(payload_program.emits)
        group_result_name = None if group_count == 1 else "result"
        group_result, group_emits = _e8m0_scale_bf16_group_emits(
            finite_magnitude=payload_program.finite_magnitude,
            sign=payload_program.sign,
            canonical_nan=payload_program.canonical_nan,
            payload_is_nan=payload_program.is_nan,
            scale=ValueRef.operand("auxiliary", element=0),
            scale_byte_ordinal=group_ordinal,
            result_name=group_result_name,
            temporary_prefix=f"mxfp8_g{group_ordinal}_scale_",
        )
        emits.extend(group_emits)
        group_results.append(group_result)

    if group_count == 2:
        emits.append(
            EmitRegisterConcat(
                sources=group_results,
                result=ValueRef.result("result"),
            )
        )

    return DescriptorRule(
        source_op=vector.vector_decode,
        descriptor=next(
            emit.descriptor
            for emit in reversed(emits)
            if isinstance(emit, EmitDescriptorOp)
        ),
        guards=(
            Guard.value_type("payload", _exact_vector("f8E4M3", lane_count)),
            Guard.value_storage_operand_schema("schema", schema),
            Guard.operand_segment_count("auxiliary", 1),
            Guard.value_type("auxiliary", _exact_vector("i32", 1), element=0),
            Guard.value_type("result", _exact_vector("bf16", lane_count)),
        ),
        emit=tuple(emits),
        report_key=(f"native_mxfp8_e4m3fn_e8m0x{lane_count}_to_bfloat16x{lane_count}"),
    )


def _mxfp4_e2m1_to_bf16_components(
    source: ValueRef,
    *,
    temporary_prefix: str,
) -> _MxBf16PayloadProgram:
    """Expands thirty-two unpacked E2M1 lanes into BF16 components."""

    program = PacketProgram(16, temporary_prefix)
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

    magnitude_mask = program.splat("magnitude_mask", 0x0007)
    magnitude = program.binary("magnitude", "and.bits512", low_bytes, magnitude_mask)
    sign_mask = program.splat("sign_mask", 0x0800)
    sign = program.binary("sign_x1", "and.bits512", high_bytes, sign_mask)
    for shift in range(1, 5):
        sign = program.binary(f"sign_x{1 << shift}", "add.i16x32", sign, sign)

    positioned_magnitude = magnitude
    for shift in range(1, 7):
        positioned_magnitude = program.binary(
            f"magnitude_x{1 << shift}",
            "add.i16x32",
            positioned_magnitude,
            positioned_magnitude,
        )
    normal_base = program.splat("normal_base", 0x3F00)
    normal = program.binary("normal", "add.i16x32", normal_base, positioned_magnitude)
    two = program.splat("two", 2)
    below_two = program.compare_unsigned_less_than("below_two", magnitude, two)
    is_zero = program.compare_zero("is_zero", magnitude)
    small = program.select("small", zero, normal_base, is_zero)
    finite_magnitude = program.select("finite_magnitude", small, normal, below_two)
    canonical_nan = program.splat("canonical_nan", _CANONICAL_BF16_NAN)
    return _MxBf16PayloadProgram(
        finite_magnitude=finite_magnitude,
        sign=sign,
        canonical_nan=canonical_nan,
        emits=tuple(program.emits),
    )


def _mxfp4_e2m1_e8m0_to_bf16_rule(
    lane_count: int,
    schema: EncodingOperandSummaryDef,
) -> DescriptorRule:
    """Decodes one or two packed MXFP4 groups without scalar lane operations."""

    group_count = lane_count // 32
    packed_payload = ValueRef.temporary("mxfp4_packed_payload_w")
    unpacked_payload = ValueRef.temporary("mxfp4_unpacked_payload_x")
    unpack = _descriptor("amd.xdna.aie2p.unpack.u4x64.to.u8x64.configured")
    emits: list[ContractEmit] = [
        EmitRegisterSlice(
            source=ValueRef.operand("payload"),
            result=packed_payload,
            unit_count=1,
        ),
        *integer_unpack_state_emits(),
        EmitDescriptorOp(
            descriptor=unpack,
            operands={"src": packed_payload},
            results={"dst": unpacked_payload},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
    ]
    group_results: list[ValueRef] = []
    payload_source = unpacked_payload
    for group_ordinal in range(group_count):
        if group_ordinal:
            high_payload = ValueRef.temporary("mxfp4_high_payload_w")
            payload_source = ValueRef.temporary("mxfp4_high_payload_x")
            emits.extend(
                (
                    EmitRegisterSlice(
                        source=unpacked_payload,
                        result=high_payload,
                        unit_offset=1,
                        unit_count=1,
                    ),
                    EmitRegisterConcat(
                        sources=(high_payload, high_payload),
                        result=payload_source,
                        result_type=_exact_vector("i8", 64),
                    ),
                )
            )
        payload_program = _mxfp4_e2m1_to_bf16_components(
            payload_source,
            temporary_prefix=f"mxfp4_g{group_ordinal}_payload_",
        )
        emits.extend(payload_program.emits)
        group_result_name = None if group_count == 1 else "result"
        group_result, group_emits = _e8m0_scale_bf16_group_emits(
            finite_magnitude=payload_program.finite_magnitude,
            sign=payload_program.sign,
            canonical_nan=payload_program.canonical_nan,
            payload_is_nan=None,
            scale=ValueRef.operand("auxiliary", element=0),
            scale_byte_ordinal=group_ordinal,
            result_name=group_result_name,
            temporary_prefix=f"mxfp4_g{group_ordinal}_scale_",
        )
        emits.extend(group_emits)
        group_results.append(group_result)

    if group_count == 2:
        emits.append(
            EmitRegisterConcat(
                sources=group_results,
                result=ValueRef.result("result"),
            )
        )

    return DescriptorRule(
        source_op=vector.vector_decode,
        descriptor=next(
            emit.descriptor
            for emit in reversed(emits)
            if isinstance(emit, EmitDescriptorOp)
        ),
        guards=(
            Guard.value_type("payload", _exact_vector("i32", lane_count // 8)),
            Guard.value_storage_operand_schema("schema", schema),
            Guard.operand_segment_count("auxiliary", 1),
            Guard.value_type("auxiliary", _exact_vector("i32", 1), element=0),
            Guard.value_type("result", _exact_vector("bf16", lane_count)),
        ),
        emit=tuple(emits),
        report_key=(f"native_mxfp4_e2m1_e8m0x{lane_count}_to_bfloat16x{lane_count}"),
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
    if program.result is None:
        raise ValueError("FP8-to-F32 widening requires a materialized BF16 packet")
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
    source_format: FloatPacketFormat,
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
    program: PacketProgram,
    source_format: FloatPacketFormat,
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
    program: PacketProgram,
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
    program: PacketProgram,
    name: str | None,
    accumulator: ValueRef,
    filler: ValueRef,
    shift: int,
    *,
    signed: bool = False,
) -> ValueRef:
    """Rounds sixteen i32 lanes into the low W of an X carrier."""

    result_w = program.operation(
        f"{name or 'result'}_w",
        f"narrow.2x.b-to-w.{'signed' if signed else 'unsigned'}.configured",
        "dst",
        src=accumulator,
        su=program.shift(shift),
    )
    result = ValueRef.result("result") if name is None else program.temporary(name)
    program.emits.append(
        EmitRegisterConcat(
            sources=(result_w, filler),
            result=result,
            result_type=_exact_vector("i16", 32),
        )
    )
    return result


@dataclass(frozen=True, slots=True)
class _ExactBF16NarrowingState:
    """Shared constants for exact binary32-to-bfloat16 packet conversion."""

    # One in every i32 lane, used to select the retained significand parity.
    one: ValueRef
    # Round-to-nearest-even bias excluding the retained significand parity.
    round_bias: ValueRef
    # Mask selecting the unsigned binary32 payload.
    nonsign_mask: ValueRef
    # Unsigned binary32 infinity payload.
    infinity: ValueRef
    # Quiet-NaN payload bit in the source binary32 representation.
    quiet_nan_bit: ValueRef


def _bind_exact_bf16_widening_results(
    program: PacketProgram,
    results: tuple[ValueRef, ...],
    rule_shape: FloatPacketRuleShape,
) -> None:
    """Binds one or two exact binary32 packet results to the source result."""

    if len(results) == 1:
        return
    if rule_shape.minimum_lane_count == rule_shape.maximum_lane_count == 32:
        accumulator_results = tuple(
            program.operation(
                f"chunk_{index}_accumulator",
                "move.vector512.to.accumulator512",
                "dst",
                src=result,
            )
            for index, result in enumerate(results)
        )
        program.emits.append(
            EmitRegisterConcat(
                sources=accumulator_results,
                result=ValueRef.result("result"),
                result_type=_exact_vector("f32", 32),
            )
        )
    else:
        program.emits.append(
            EmitRegisterConcat(sources=results, result=ValueRef.result("result"))
        )


def _exact_bf16_to_f32_emits(
    rule_shape: FloatPacketRuleShape,
) -> tuple[ContractEmit, ...]:
    """Widens bfloat16 bits exactly, including bfloat16 subnormals."""

    program = PacketProgram(32, "bf16_exact_widen_")
    program.state("saturation", 1)
    program.state("ups-mode", 0)
    chunks = _float_source_i32_chunks(program, _BF16_PACKET_FORMAT, rule_shape)
    direct_result = len(chunks) == 1
    results = tuple(
        shift_integer_packet_fixed(
            program,
            None if direct_result else f"chunk_{index}_result",
            chunk,
            16,
        )
        for index, chunk in enumerate(chunks)
    )
    _bind_exact_bf16_widening_results(program, results, rule_shape)
    return tuple(program.emits)


def _exact_bf16_to_f32_vector_rule(
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=_descriptor("amd.xdna.aie2p.narrow.2x.c-to-x.unsigned.configured"),
        guards=(
            Guard.value_type("input", rule_shape.vector_type("bf16")),
            Guard.value_type("result", rule_shape.vector_type("f32")),
        ),
        emit=_exact_bf16_to_f32_emits(rule_shape),
        report_key="exact_bfloat16_to_binary32_packet",
    )


def _exact_f32_chunk_to_bf16(
    program: PacketProgram,
    source: ValueRef,
    chunk_index: int,
    *,
    result_name: str | None,
    state: _ExactBF16NarrowingState,
) -> ValueRef:
    """Narrows sixteen binary32 bit patterns to exact bfloat16 encodings."""

    prefix = f"chunk_{chunk_index}"
    upper = shift_integer_packet_fixed(program, f"{prefix}_upper", source, -16)
    upper_lsb = program.binary(f"{prefix}_upper_lsb", "and.bits512", upper, state.one)
    bias = program.binary(f"{prefix}_bias", "add.i32x16", state.round_bias, upper_lsb)
    rounded = program.binary(f"{prefix}_rounded", "add.i32x16", source, bias)
    absolute = program.binary(
        f"{prefix}_absolute", "and.bits512", source, state.nonsign_mask
    )
    is_nan = program.compare_unsigned_less_than(
        f"{prefix}_is_nan", state.infinity, absolute
    )
    nan = program.binary(f"{prefix}_nan", "or.bits512", source, state.quiet_nan_bit)
    selected = program.select(f"{prefix}_selected", nan, rounded, is_nan)

    program.state("srs-mode", 0)
    accumulator, filler = _i32_packet_accumulator(
        program, f"{prefix}_selected", selected
    )
    return _round_i32_packet_to_i16(
        program,
        result_name,
        accumulator,
        filler,
        16,
    )


def _exact_f32_to_bf16_emits(
    rule_shape: FloatPacketRuleShape,
) -> tuple[ContractEmit, ...]:
    """Narrows binary32 packets with exact RNE and NaN preservation."""

    program = PacketProgram(32, "bf16_exact_narrow_")
    chunks = _float_source_i32_chunks(program, _F32_PACKET_FORMAT, rule_shape)
    program.state("saturation", 0)
    program.state("rounding", 0)
    program.state("srs-mode", 0)
    state = _ExactBF16NarrowingState(
        one=program.splat("one", 1),
        round_bias=program.splat("round_bias", 0x7FFF),
        nonsign_mask=program.splat("nonsign_mask", _F32_PACKET_FORMAT.nonsign_mask),
        infinity=program.splat("infinity", _F32_PACKET_FORMAT.infinity_bits),
        quiet_nan_bit=program.splat("quiet_nan_bit", 0x00400000),
    )

    direct_result = len(chunks) == 1
    results = tuple(
        _exact_f32_chunk_to_bf16(
            program,
            chunk,
            index,
            result_name=None if direct_result else f"chunk_{index}_result",
            state=state,
        )
        for index, chunk in enumerate(chunks)
    )
    if not direct_result:
        result_words = []
        for index, result in enumerate(results):
            result_word = program.temporary(f"chunk_{index}_packed_w")
            program.emits.append(
                EmitRegisterSlice(source=result, result=result_word, unit_count=1)
            )
            result_words.append(result_word)
        program.emits.append(
            EmitRegisterConcat(
                sources=tuple(result_words), result=ValueRef.result("result")
            )
        )
    return tuple(program.emits)


def _exact_f32_to_bf16_vector_rule(
    rule_shape: FloatPacketRuleShape,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=_descriptor("amd.xdna.aie2p.narrow.2x.b-to-w.unsigned.configured"),
        guards=(
            Guard.value_type("input", rule_shape.vector_type("f32")),
            Guard.value_type("result", rule_shape.vector_type("bf16")),
        ),
        emit=_exact_f32_to_bf16_emits(rule_shape),
        report_key="exact_binary32_to_bfloat16_packet",
    )


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
    program: PacketProgram,
    source: ValueRef,
    source_format: FloatPacketFormat,
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
    source_format: FloatPacketFormat,
    fp8_format: Float8PacketFormat,
    rule_shape: FloatPacketRuleShape,
) -> tuple[ContractEmit, ...]:
    """Narrows up to thirty-two floating lanes into exact FP8 packets."""

    program = PacketProgram(32, "fp8_narrow_")
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
    source_format: FloatPacketFormat,
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


def integer_truncation_emits(
    rule_shape: IntegerTruncationRuleShape,
    source: ValueRef,
    result: ValueRef,
    *,
    result_type: DescriptorResultType | None = None,
) -> tuple[ContractEmit, ...]:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
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
        step_result = result if is_final else ValueRef.temporary(f"truncated_{index}")
        if is_final:
            step_result_types = (
                {"dst": result_type} if result_type is not None else None
            )
        else:
            step_result_types = {"dst": DescriptorResultType()}
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
                    results={"dst": step_result},
                    result_types=step_result_types,
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
        current = step_result

    return tuple(emits)


def integer_truncation_rule(
    rule_shape: IntegerTruncationRuleShape,
    *,
    source_op: Op = vector.vector_trunci,
    input_type: TypePattern | None = None,
    result_type: TypePattern | None = None,
    extra_guards: tuple[Guard, ...] = (),
    report_key: str | None = None,
) -> DescriptorRule:
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")

    return DescriptorRule(
        source_op=source_op,
        descriptor=shuffle,
        guards=(
            Guard.value_type(
                "input", rule_shape.input_type if input_type is None else input_type
            ),
            Guard.value_type(
                "result", rule_shape.result_type if result_type is None else result_type
            ),
            *extra_guards,
        ),
        emit=integer_truncation_emits(
            rule_shape,
            ValueRef.operand("input"),
            ValueRef.result("result"),
        ),
        report_key=rule_shape.report_key if report_key is None else report_key,
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
        _mxfp4_e2m1_e8m0_to_bf16_rule(lane_count, schema)
        for lane_count, schema in _MXFP4_E2M1_E8M0_RULE_SHAPES
    ),
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
        _integer_bitunpack_rule(source_op, source_kind, source_lane_count)
        for source_op, source_kind in (
            (vector.vector_bitunpacku, "u"),
            (vector.vector_bitunpacks, "s"),
        )
        for source_lane_count in I4_UNPACK_SOURCE_LANE_COUNTS
    ),
    *(
        _byte_to_halfword_widen_rule(source_op, signedness, rule_shape)
        for source_op, signedness in (
            (vector.vector_extui, "unsigned"),
            (vector.vector_extsi, "signed"),
        )
        for rule_shape in BYTE_TO_HALFWORD_WIDEN_RULE_SHAPES
    ),
    *(
        _byte_to_i64_widen_rule(source_op, signedness, rule_shape)
        for source_op, signedness in (
            (vector.vector_extui, "unsigned"),
            (vector.vector_extsi, "signed"),
        )
        for rule_shape in BYTE_TO_I64_WIDEN_RULE_SHAPES
    ),
    *(
        integer_widen_rule(source_op, signedness, rule_shape)
        for source_op, signedness in (
            (vector.vector_extui, "unsigned"),
            (vector.vector_extsi, "signed"),
        )
        for rule_shape in INTEGER_WIDEN_RULE_SHAPES
    ),
    *(
        _integer_to_f32_packet_rule(source_op, input_element, signedness)
        for source_op, signedness in (
            (vector.vector_sitofp, "signed"),
            (vector.vector_uitofp, "unsigned"),
        )
        for input_element in ("i8", "i16", "i32")
    ),
    *(
        integer_truncation_rule(rule_shape)
        for rule_shape in INTEGER_TRUNCATION_RULE_SHAPES
    ),
    *(_integer_pack_rule(rule_shape) for rule_shape in INTEGER_PACK_RULE_SHAPES),
    *(
        _float_to_fp8_vector_rule(source_format, fp8_format, rule_shape)
        for source_format in FLOAT_PACKET_FORMATS
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
    *(
        rule
        for rule_shape in _EXACT_FLOAT_PACKET_RULE_SHAPES
        for rule in (
            _exact_f32_to_bf16_vector_rule(rule_shape),
            _exact_bf16_to_f32_vector_rule(rule_shape),
        )
    ),
    *(
        rule
        for float16_format in _EXACT_FLOAT16_PACKET_FORMATS
        for rule_shape in _EXACT_FLOAT_PACKET_RULE_SHAPES
        for rule in (
            _float16_to_f32_vector_rule(float16_format, rule_shape),
            _f32_to_float16_vector_rule(float16_format, rule_shape),
        )
    ),
)
