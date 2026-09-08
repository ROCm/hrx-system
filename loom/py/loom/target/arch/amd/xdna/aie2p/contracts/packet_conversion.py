# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native AIE2P packet-conversion contracts."""

from __future__ import annotations

from dataclasses import dataclass

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
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
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

# Source-visible lane counts supported by native BF16/F32 packet conversion.
BF16_F32_PACKET_LANE_COUNTS = (16, 32)


@dataclass(frozen=True, slots=True)
class IntegerWidenCase:
    """One source-visible shape supported by native VUPS forms."""

    # Source vector element type.
    input_element: str
    # Result vector element type.
    result_element: str
    # Logical lane count preserved by the conversion.
    lane_count: int

    @property
    def memory_width_bits(self) -> int:
        """Number of source bits consumed by a fused widening load."""

        return self.lane_count * int(self.input_element[1:])

    @property
    def result_width_bits(self) -> int:
        """Number of result bits produced by the widening operation."""

        return self.lane_count * int(self.result_element[1:])

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
class IntegerPackCase:
    """One source-visible shape supported by native VPACK forms."""

    # Source vector element type.
    input_element: str
    # Number of source lanes packed by the conversion.
    input_lanes: int
    # Required vector.bitpack width, or None for vector.trunci.
    bit_width: int | None

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

        return 8 if self.bit_width is None else self.bit_width

    @property
    def result_lanes(self) -> int:
        """Number of physical i8 lanes carrying the packed result."""

        return self.input_lanes * self.output_element_bits // 8

    @property
    def memory_width_bits(self) -> int:
        """Number of result bits written by a fused packing store."""

        return self.result_lanes * 8

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
                f"native_trunc_{self.input_element}x{self.input_lanes}_to_"
                f"i8x{self.result_lanes}"
            )
        return (
            f"native_bitpack_{self.input_element}x{self.input_lanes}_to_"
            f"i{self.bit_width}x{self.input_lanes}"
        )

    @property
    def pad_result(self) -> bool:
        """Whether the result preserves an unused X-carrier half."""

        return self.memory_width_bits == 256


INTEGER_WIDEN_CASES = (
    IntegerWidenCase("i16", "i32", 16),
    IntegerWidenCase("i32", "i64", 8),
    IntegerWidenCase("i8", "i32", 32),
    IntegerWidenCase("i16", "i64", 16),
    IntegerWidenCase("i16", "i32", 32),
    IntegerWidenCase("i32", "i64", 16),
    IntegerWidenCase("i8", "i32", 64),
    IntegerWidenCase("i16", "i64", 32),
)

INTEGER_PACK_CASES = (
    IntegerPackCase("i16", 32, None),
    IntegerPackCase("i16", 64, None),
    IntegerPackCase("i8", 64, 4),
    IntegerPackCase("i8", 128, 4),
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _exact_vector(element: str, element_count: int) -> Vector:
    return Vector(element, lanes=element_count)


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
    widen_case: IntegerWidenCase,
    result: ValueRef,
) -> tuple[ValueRef, tuple[ContractEmit, ...]]:
    """Bridges a native accumulator result to its source-visible carrier."""

    if widen_case.direct_accumulator_result:
        return result, ()

    native_result = ValueRef.temporary("wide_result")
    output_emits: list[ContractEmit] = []
    vector_units: list[ValueRef] = []
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )
    for unit in range(widen_case.accumulator_unit_count):
        accumulator_unit = native_result
        if widen_case.accumulator_unit_count > 1:
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
            if widen_case.accumulator_unit_count == 1
            else ValueRef.temporary(f"vector_unit_{unit}")
        )
        output_emits.append(
            EmitDescriptorOp(
                descriptor=move_from_accumulator,
                operands={"src": accumulator_unit},
                results={"dst": vector_unit},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if widen_case.accumulator_unit_count > 1
                    else None
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        vector_units.append(vector_unit)
    if widen_case.accumulator_unit_count > 1:
        output_emits.append(
            EmitRegisterConcat(
                sources=vector_units,
                result=result,
            )
        )
    return native_result, tuple(output_emits)


def integer_pack_state_emits(pack_size: int) -> tuple[ContractEmit, ...]:
    """Builds the explicit configured state consumed by one VPACK form."""

    return (
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.saturation.immediate"),
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.pack-size.immediate"),
            immediates={"i": pack_size},
            form=DescriptorEmitForm.OP,
        ),
    )


def _integer_widen_rule(
    source_op: Op,
    signedness: str,
    widen_case: IntegerWidenCase,
) -> DescriptorRule:
    input_type = _exact_vector(widen_case.input_element, widen_case.lane_count)
    result_type = _exact_vector(widen_case.result_element, widen_case.lane_count)
    source = ValueRef.operand("input")
    input_emits: tuple[ContractEmit, ...] = ()
    if widen_case.slice_input:
        source = ValueRef.temporary("source_w")
        input_emits = (
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
                result=source,
                unit_count=1,
            ),
        )
    shift, state_emits = integer_widen_state_emits(widen_case.ups_mode)
    result = ValueRef.result("result")
    native_result, output_emits = integer_widen_result_emits(widen_case, result)
    widen = _descriptor(
        f"amd.xdna.aie2p.widen.{widen_case.physical_shape}.{signedness}.configured"
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=widen,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
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
                    if widen_case.direct_accumulator_result
                    else {"dst": DescriptorResultType()}
                ),
                form=DescriptorEmitForm.OP,
            ),
            *output_emits,
        ),
        report_key=(
            f"native_{signedness}_{widen_case.input_element}x"
            f"{widen_case.lane_count}_to_{widen_case.result_element}x"
            f"{widen_case.lane_count}"
        ),
    )


def _f32_to_bf16_vector_rule(lane_count: int) -> DescriptorRule:
    set_rounding = _descriptor("amd.xdna.aie2p.state.rounding.immediate")
    convert = _descriptor(
        f"amd.xdna.aie2p.convert.f32x{lane_count}.to.bf16x{lane_count}"
    )
    source_type = _exact_vector("f32", lane_count)
    result_type = _exact_vector("bf16", lane_count)
    native_source = ValueRef.operand("input")
    input_emits: tuple[ContractEmit, ...] = ()
    native_result = ValueRef.result("result")
    result_types = None
    output_emits: tuple[ContractEmit, ...] = ()
    if lane_count == 16:
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
        report_key=f"native_binary32x{lane_count}_to_bfloat16x{lane_count}",
    )


def _bf16_to_f32_vector_rule(lane_count: int) -> DescriptorRule:
    convert = _descriptor(
        f"amd.xdna.aie2p.convert.bf16x{lane_count}.to.f32x{lane_count}"
    )
    source_type = _exact_vector("bf16", lane_count)
    result_type = _exact_vector("f32", lane_count)
    native_source = ValueRef.operand("input")
    input_emits: tuple[ContractEmit, ...] = ()
    native_result = ValueRef.result("result")
    result_types = None
    output_emits: tuple[ContractEmit, ...] = ()
    if lane_count == 16:
        native_source = ValueRef.temporary("source_w")
        native_result = ValueRef.temporary("converted_accumulator")
        result_types = {"dst": DescriptorResultType()}
        input_emits = (
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
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
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=convert,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            *input_emits,
            EmitDescriptorOp(
                descriptor=convert,
                operands={"src": native_source},
                results={"dst": native_result},
                result_types=result_types,
                form=DescriptorEmitForm.OP,
            ),
            *output_emits,
        ),
        report_key=f"native_bfloat16x{lane_count}_to_binary32x{lane_count}",
    )


def _integer_pack_rule(pack_case: IntegerPackCase) -> DescriptorRule:
    pack = _descriptor(
        f"amd.xdna.aie2p.pack.{pack_case.physical_width}.trunc.configured"
    )
    packed_result = (
        ValueRef.temporary("packed_w")
        if pack_case.pad_result
        else ValueRef.result("result")
    )
    result_emits: tuple[ContractEmit, ...] = ()
    if pack_case.pad_result:
        result_emits = (
            EmitRegisterSlice(
                source=ValueRef.operand(pack_case.source_field),
                result=ValueRef.temporary("unused_w"),
                unit_offset=1,
                unit_count=1,
            ),
            EmitRegisterConcat(
                sources=(packed_result, ValueRef.temporary("unused_w")),
                result=ValueRef.result("result"),
            ),
        )
    return DescriptorRule(
        source_op=pack_case.source_op,
        descriptor=pack,
        guards=(
            Guard.value_type(
                pack_case.source_field,
                _exact_vector(pack_case.input_element, pack_case.input_lanes),
            ),
            Guard.value_type("result", _exact_vector("i8", pack_case.result_lanes)),
            *(
                (
                    Guard.attr_kind("width", "i64"),
                    Guard.i64_range(
                        "width",
                        pack_case.bit_width,
                        pack_case.bit_width,
                    ),
                )
                if pack_case.bit_width is not None
                else ()
            ),
        ),
        emit=(
            *integer_pack_state_emits(pack_case.pack_size),
            EmitDescriptorOp(
                descriptor=pack,
                operands={"src": ValueRef.operand(pack_case.source_field)},
                results={"dst": packed_result},
                result_types=(
                    {"dst": DescriptorResultType()} if pack_case.pad_result else None
                ),
                form=DescriptorEmitForm.OP,
            ),
            *result_emits,
        ),
        report_key=pack_case.report_key,
    )


AIE2P_PACKET_CONVERSION_RULES = (
    *(
        _integer_widen_rule(source_op, signedness, widen_case)
        for source_op, signedness in (
            (vector.vector_extui, "unsigned"),
            (vector.vector_extsi, "signed"),
        )
        for widen_case in INTEGER_WIDEN_CASES
    ),
    *(_integer_pack_rule(pack_case) for pack_case in INTEGER_PACK_CASES),
    *(
        _f32_to_bf16_vector_rule(lane_count)
        for lane_count in BF16_F32_PACKET_LANE_COUNTS
    ),
    *(
        _bf16_to_f32_vector_rule(lane_count)
        for lane_count in BF16_F32_PACKET_LANE_COUNTS
    ),
)
