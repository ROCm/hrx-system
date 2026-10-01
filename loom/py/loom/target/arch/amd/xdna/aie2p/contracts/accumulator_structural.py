# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P accumulator structural selection rules."""

from dataclasses import dataclass

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    ValueTypeProject,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


# Rank-one source types with accumulator-file representations. Each native
# packet occupies one 512-bit X register before or after the explicit move.
_F32X32_ACCUMULATOR = Vector("f32", lanes=32)


def _vector_lane_interval(
    element_type: str,
    minimum_lane_count: int,
    maximum_lane_count: int,
) -> TypePattern:
    """Returns a canonical exact or interval rank-one vector pattern."""

    if minimum_lane_count == maximum_lane_count:
        return Vector(element_type, lanes=minimum_lane_count)
    return Vector(
        element_type,
        minimum_lanes=minimum_lane_count,
        maximum_lanes=maximum_lane_count,
    )


@dataclass(frozen=True, slots=True)
class _AccumulatorVectorShape:
    """Logical interval sharing one accumulator carrier shape."""

    # Source vector element type.
    element_type: str
    # First logical lane count carried by the shape.
    minimum_lane_count: int
    # Last logical lane count carried by the shape.
    maximum_lane_count: int
    # Logical lanes carried by one 512-bit packet.
    packet_lane_count: int
    # Number of packets that can contain logical lanes.
    logical_packet_count: int
    # Number of MBMS units in the allocatable physical carrier.
    register_unit_count: int

    @property
    def source_type(self) -> TypePattern:
        """Source-visible vector interval using this carrier."""

        return _vector_lane_interval(
            self.element_type,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )


# AIE2P exposes allocatable two-unit and four-unit accumulator views. There is
# no three-unit physical view, so a three-packet logical value retains a padded
# four-unit carrier and leaves the final unit outside its value domain.
_ACCUMULATOR_VECTOR_SHAPES = (
    _AccumulatorVectorShape("f32", 32, 32, 16, 2, 2),
    _AccumulatorVectorShape("f32", 33, 48, 16, 3, 4),
    _AccumulatorVectorShape("f32", 49, 64, 16, 4, 4),
    _AccumulatorVectorShape("i32", 33, 48, 16, 3, 4),
    _AccumulatorVectorShape("i32", 49, 64, 16, 4, 4),
    _AccumulatorVectorShape("i64", 17, 24, 8, 3, 4),
    _AccumulatorVectorShape("i64", 25, 32, 8, 4, 4),
)


@dataclass(frozen=True, slots=True)
class _AccumulatorSliceResultShape:
    """Logical result interval sharing one carrier and packet count."""

    # Result vector element type.
    element_type: str
    # First logical lane count carried by the shape.
    minimum_lane_count: int
    # Last logical lane count carried by the shape.
    maximum_lane_count: int
    # Logical lanes carried by one 512-bit packet.
    packet_lane_count: int
    # Number of packets that can contain logical lanes.
    logical_packet_count: int
    # Number of units in the allocatable physical carrier.
    register_unit_count: int
    # Low register class carrying the result packets.
    register_class: str

    @property
    def result_type(self) -> TypePattern:
        """Source-visible vector interval using this carrier."""

        return _vector_lane_interval(
            self.element_type,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )

    @property
    def is_accumulator(self) -> bool:
        """Whether the result packets stay in the accumulator register file."""

        return self.register_class == "aie2p.mbms"


def _accumulator_slice_result_shapes() -> tuple[_AccumulatorSliceResultShape, ...]:
    """Returns every result carrier reachable from an accumulator source."""

    ordinary_lane_maximums = {
        "f32": 31,
        "i32": 32,
        "i64": 16,
    }
    result_shapes: list[_AccumulatorSliceResultShape] = []
    for element_type, packet_lane_count in (("f32", 16), ("i32", 16), ("i64", 8)):
        ordinary_lane_maximum = ordinary_lane_maximums[element_type]
        result_shapes.append(
            _AccumulatorSliceResultShape(
                element_type,
                1,
                packet_lane_count,
                packet_lane_count,
                1,
                2,
                "aie2p.vec256",
            )
        )
        if ordinary_lane_maximum > packet_lane_count:
            result_shapes.append(
                _AccumulatorSliceResultShape(
                    element_type,
                    packet_lane_count + 1,
                    ordinary_lane_maximum,
                    packet_lane_count,
                    2,
                    4,
                    "aie2p.vec256",
                )
            )
        result_shapes.extend(
            _AccumulatorSliceResultShape(
                shape.element_type,
                shape.minimum_lane_count,
                shape.maximum_lane_count,
                shape.packet_lane_count,
                shape.logical_packet_count,
                shape.register_unit_count,
                "aie2p.mbms",
            )
            for shape in _ACCUMULATOR_VECTOR_SHAPES
            if shape.element_type == element_type
        )
    return tuple(result_shapes)


_ACCUMULATOR_SLICE_RESULT_SHAPES = _accumulator_slice_result_shapes()


def _accumulator_slice_guards(
    source_type: TypePattern,
    result_type: TypePattern,
    offset_minimum: int,
    offset_maximum: int,
) -> tuple[Guard, ...]:
    """Selects one static rank-one interval from an accumulator source."""

    return (
        Guard.value_type("source", source_type),
        Guard.value_type("result", result_type),
        Guard.operand_segment_count("offsets", 0),
        Guard.i64_array_count("static_offsets", 1),
        Guard.i64_array_element_range(
            "static_offsets",
            element=0,
            minimum=offset_minimum,
            maximum=offset_maximum,
        ),
    )


def _accumulator_slice_unit(
    source: ValueRef,
    unit_index: int,
    name: str,
    emits: list[ContractEmit],
) -> ValueRef:
    """Projects one MBMS unit from an accumulator carrier."""

    unit = ValueRef.temporary(name)
    emits.append(
        EmitRegisterSlice(
            source=source,
            result=unit,
            unit_offset=unit_index,
            unit_count=1,
        )
    )
    return unit


def _accumulator_slice_x_packet(
    source: ValueRef,
    unit_index: int,
    name: str,
    emits: list[ContractEmit],
    *,
    result: ValueRef | None = None,
) -> ValueRef:
    """Moves one selected MBMS unit into a 512-bit X carrier."""

    accumulator_unit = _accumulator_slice_unit(
        source,
        unit_index,
        f"{name}_accumulator",
        emits,
    )
    vector_packet = result or ValueRef.temporary(f"{name}_vector")
    emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.move.accumulator512.to.vector512"),
            operands={"src": accumulator_unit},
            results={"dst": vector_packet},
            result_types=(
                None if result is not None else {"dst": DescriptorResultType()}
            ),
            form=DescriptorEmitForm.OP,
        )
    )
    return vector_packet


def _accumulator_slice_aligned_rule(
    source_shape: _AccumulatorVectorShape,
    result_shape: _AccumulatorSliceResultShape,
    source_packet_index: int,
) -> ValueAliasRule | DescriptorRule:
    """Retains aligned MBMS packets or moves them into X carriers."""

    offset = source_packet_index * source_shape.packet_lane_count
    guards = _accumulator_slice_guards(
        source_shape.source_type,
        result_shape.result_type,
        offset,
        offset,
    )
    source = ValueRef.operand("source")
    result = ValueRef.result("result")
    if (
        result_shape.is_accumulator
        and source_packet_index == 0
        and source_shape.register_unit_count == result_shape.register_unit_count
    ):
        return ValueAliasRule(
            source_op=vector.vector_slice,
            source=source,
            result=result,
            guards=guards,
        )

    if (
        result_shape.is_accumulator
        and result_shape.logical_packet_count == result_shape.register_unit_count
    ):
        return DescriptorRule(
            source_op=vector.vector_slice,
            guards=guards,
            emit=(
                EmitRegisterSlice(
                    source=source,
                    result=result,
                    unit_offset=source_packet_index,
                ),
            ),
        )

    emits: list[ContractEmit] = []
    packets: list[ValueRef] = []
    if result_shape.is_accumulator:
        packets.extend(
            _accumulator_slice_unit(
                source,
                source_packet_index + packet_index,
                f"result_accumulator_{packet_index}",
                emits,
            )
            for packet_index in range(result_shape.logical_packet_count)
        )
        packets.extend(
            (packets[-1],) * (result_shape.register_unit_count - len(packets))
        )
        emits.append(EmitRegisterConcat(sources=packets, result=result))
        return DescriptorRule(
            source_op=vector.vector_slice,
            guards=guards,
            emit=tuple(emits),
        )

    packets.extend(
        (
            _accumulator_slice_x_packet(
                source,
                source_packet_index + packet_index,
                f"result_packet_{packet_index}",
                emits,
                result=(result if result_shape.logical_packet_count == 1 else None),
            )
        )
        for packet_index in range(result_shape.logical_packet_count)
    )
    if len(packets) > 1:
        emits.append(EmitRegisterConcat(sources=packets, result=result))
    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=_descriptor("amd.xdna.aie2p.move.accumulator512.to.vector512"),
        guards=guards,
        emit=tuple(emits),
    )


def _accumulator_slice_shifted_rule(
    source_shape: _AccumulatorVectorShape,
    result_shape: _AccumulatorSliceResultShape,
    source_packet_index: int,
    offset_minimum: int,
    offset_maximum: int,
    *,
    include_adjacent_packet: bool,
) -> DescriptorRule:
    """Shifts selected MBMS packets through X and materializes the result."""

    source = ValueRef.operand("source")
    result = ValueRef.result("result")
    source_packet_count = result_shape.logical_packet_count + int(
        include_adjacent_packet
    )
    emits: list[ContractEmit] = []
    source_packets = tuple(
        _accumulator_slice_x_packet(
            source,
            source_packet_index + packet_index,
            f"source_packet_{packet_index}",
            emits,
        )
        for packet_index in range(source_packet_count)
    )
    byte_offset = ValueRef.temporary("byte_offset")
    emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.constant.i32.mova"),
            results={"dst": byte_offset},
            result_types={"dst": DescriptorResultType()},
            immediates={
                "i": AttrProject.i64_array_lane_byte_offset(
                    "static_offsets",
                    element=0,
                    bytes_per_lane=64 // source_shape.packet_lane_count,
                    base_byte_offset=-64 * source_packet_index,
                )
            },
            form=DescriptorEmitForm.CONST,
        )
    )

    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    shifted_packets: list[ValueRef] = []
    for packet_index in range(result_shape.logical_packet_count):
        is_direct_result = (
            not result_shape.is_accumulator and result_shape.logical_packet_count == 1
        )
        shifted_packet = (
            result
            if is_direct_result
            else ValueRef.temporary(f"shifted_packet_{packet_index}")
        )
        high_packet_index = min(packet_index + 1, len(source_packets) - 1)
        emits.append(
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "s1": source_packets[packet_index],
                    "s2": source_packets[high_packet_index],
                    "shift": byte_offset,
                },
                results={"d": shifted_packet},
                result_types=(
                    None if is_direct_result else {"d": DescriptorResultType()}
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        shifted_packets.append(shifted_packet)

    if result_shape.is_accumulator:
        accumulator_packets: list[ValueRef] = []
        move = _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
        for packet_index, shifted_packet in enumerate(shifted_packets):
            accumulator_packet = ValueRef.temporary(
                f"result_accumulator_{packet_index}"
            )
            emits.append(
                EmitDescriptorOp(
                    descriptor=move,
                    operands={"src": shifted_packet},
                    results={"dst": accumulator_packet},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                )
            )
            accumulator_packets.append(accumulator_packet)
        accumulator_packets.extend(
            (accumulator_packets[-1],)
            * (result_shape.register_unit_count - len(accumulator_packets))
        )
        emits.append(
            EmitRegisterConcat(
                sources=accumulator_packets,
                result=result,
            )
        )
    elif result_shape.logical_packet_count > 1:
        emits.append(EmitRegisterConcat(sources=shifted_packets, result=result))

    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=shift,
        guards=_accumulator_slice_guards(
            source_shape.source_type,
            result_shape.result_type,
            offset_minimum,
            offset_maximum,
        ),
        emit=tuple(emits),
    )


def _accumulator_vector_slice_rules() -> tuple[ValueAliasRule | DescriptorRule, ...]:
    """Builds complete static slice rules for every accumulator carrier."""

    rules: list[ValueAliasRule | DescriptorRule] = []
    for source_shape in _ACCUMULATOR_VECTOR_SHAPES:
        result_shapes = tuple(
            shape
            for shape in _ACCUMULATOR_SLICE_RESULT_SHAPES
            if shape.element_type == source_shape.element_type
        )
        for result_shape in result_shapes:
            maximum_start_packet = (
                source_shape.logical_packet_count - result_shape.logical_packet_count
            )
            if maximum_start_packet < 0:
                continue
            for source_packet_index in range(maximum_start_packet + 1):
                rules.append(
                    _accumulator_slice_aligned_rule(
                        source_shape,
                        result_shape,
                        source_packet_index,
                    )
                )
                packet_offset = source_packet_index * source_shape.packet_lane_count
                shifted_offset_minimum = packet_offset + 1
                shifted_offset_maximum = min(
                    packet_offset + source_shape.packet_lane_count - 1,
                    source_shape.maximum_lane_count - result_shape.minimum_lane_count,
                )
                if shifted_offset_minimum > shifted_offset_maximum:
                    continue
                # One carrier interval can contain both slices that stay within
                # its logical packets and slices that cross into the next one.
                # Materializing that adjacent packet shares one rule across the
                # interval while bounding the extra work to one register move.
                rules.append(
                    _accumulator_slice_shifted_rule(
                        source_shape,
                        result_shape,
                        source_packet_index,
                        shifted_offset_minimum,
                        shifted_offset_maximum,
                        include_adjacent_packet=(
                            source_packet_index + result_shape.logical_packet_count
                            < source_shape.logical_packet_count
                        ),
                    )
                )
    return tuple(rules)


_ACCUMULATOR_VECTOR_SLICE_RULES = _accumulator_vector_slice_rules()


@dataclass(frozen=True, slots=True)
class _AccumulatorConcatOperandShape:
    """One input interval with a fixed logical packet and carrier shape."""

    # Source vector element type.
    element_type: str
    # First logical lane count in the interval.
    minimum_lane_count: int
    # Last logical lane count in the interval.
    maximum_lane_count: int
    # Logical lanes carried by one 512-bit packet.
    packet_lane_count: int
    # Number of packets that can contain logical lanes.
    logical_packet_count: int
    # MBMS carrier units, or None when the input uses ordinary X registers.
    accumulator_unit_count: int | None

    @property
    def source_type(self) -> TypePattern:
        """Source-visible vector interval using this operand shape."""

        return _vector_lane_interval(
            self.element_type,
            self.minimum_lane_count,
            self.maximum_lane_count,
        )

    @property
    def is_packet_aligned(self) -> bool:
        """Whether this interval denotes one exact packet boundary."""

        return (
            self.minimum_lane_count == self.maximum_lane_count
            and self.maximum_lane_count
            == self.logical_packet_count * self.packet_lane_count
        )


_ACCUMULATOR_CONCAT_TYPE_SPECS = (
    ("f32", 4, 16),
    ("i32", 4, 16),
    ("i64", 8, 8),
)


def _accumulator_concat_unit_count(
    element_type: str,
    lane_count: int,
    packet_lane_count: int,
) -> int | None:
    """Returns the physical accumulator width for one logical endpoint."""

    if element_type == "f32" and lane_count == 2 * packet_lane_count:
        return 2
    if lane_count > 2 * packet_lane_count:
        return 4
    return None


def _accumulator_concat_left_shapes(
    element_type: str,
    packet_lane_count: int,
) -> tuple[_AccumulatorConcatOperandShape, ...]:
    """Partitions left operands where packet alignment changes emission."""

    shapes: list[_AccumulatorConcatOperandShape] = []
    for packet_count in range(1, 5):
        minimum_lane_count = (packet_count - 1) * packet_lane_count + 1
        aligned_lane_count = packet_count * packet_lane_count
        if minimum_lane_count < aligned_lane_count:
            shapes.append(
                _AccumulatorConcatOperandShape(
                    element_type,
                    minimum_lane_count,
                    aligned_lane_count - 1,
                    packet_lane_count,
                    packet_count,
                    _accumulator_concat_unit_count(
                        element_type,
                        minimum_lane_count,
                        packet_lane_count,
                    ),
                )
            )
        shapes.append(
            _AccumulatorConcatOperandShape(
                element_type,
                aligned_lane_count,
                aligned_lane_count,
                packet_lane_count,
                packet_count,
                _accumulator_concat_unit_count(
                    element_type,
                    aligned_lane_count,
                    packet_lane_count,
                ),
            )
        )
    return tuple(shapes)


def _accumulator_concat_right_shapes(
    element_type: str,
    packet_lane_count: int,
) -> tuple[_AccumulatorConcatOperandShape, ...]:
    """Partitions right operands only where packet count or carrier changes."""

    shapes: list[_AccumulatorConcatOperandShape] = []
    for shape in _accumulator_concat_left_shapes(element_type, packet_lane_count):
        if (
            shapes
            and shapes[-1].maximum_lane_count + 1 == shape.minimum_lane_count
            and shapes[-1].logical_packet_count == shape.logical_packet_count
            and shapes[-1].accumulator_unit_count == shape.accumulator_unit_count
        ):
            previous = shapes[-1]
            shapes[-1] = _AccumulatorConcatOperandShape(
                previous.element_type,
                previous.minimum_lane_count,
                shape.maximum_lane_count,
                previous.packet_lane_count,
                previous.logical_packet_count,
                previous.accumulator_unit_count,
            )
        else:
            shapes.append(shape)
    return tuple(shapes)


_ACCUMULATOR_BITCAST_TYPE_GROUPS = (
    (_F32X32_ACCUMULATOR,),
    (
        Vector("f32", minimum_lanes=33, maximum_lanes=64),
        Vector("i32", minimum_lanes=33, maximum_lanes=64),
        Vector("i64", minimum_lanes=17, maximum_lanes=32),
    ),
)


def _vector_concat_merge_emits(
    left: ValueRef,
    right: ValueRef,
    result: ValueRef,
    *,
    left_byte_count: ValueTypeProject,
    remaining_byte_count: ValueTypeProject,
    temporary_prefix: str = "",
    result_type: DescriptorResultType | None = None,
) -> tuple[ContractEmit, ...]:
    """Joins the suffix-aligned left payload to the right carrier prefix."""

    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    left_bytes = ValueRef.temporary(f"{temporary_prefix}left_bytes")
    rotated_left = ValueRef.temporary(f"{temporary_prefix}rotated_left")
    remaining_bytes = ValueRef.temporary(f"{temporary_prefix}remaining_bytes")
    return (
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": left_bytes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": left_byte_count},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=shift,
            operands={"s1": left, "s2": left, "shift": left_bytes},
            results={"d": rotated_left},
            result_types={"d": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": remaining_bytes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": remaining_byte_count},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=shift,
            operands={
                "s1": rotated_left,
                "s2": right,
                "shift": remaining_bytes,
            },
            results={"d": result},
            result_types=({"d": result_type} if result_type is not None else None),
            form=DescriptorEmitForm.OP,
        ),
    )


def _accumulator_concat_x_packet(
    shape: _AccumulatorConcatOperandShape,
    input_index: int,
    packet_index: int,
    temporary_prefix: str,
    emits: list[ContractEmit],
) -> ValueRef:
    """Returns one operand packet in an X carrier."""

    source = ValueRef.operand("inputs", element=input_index)
    vector_packet = ValueRef.temporary(f"{temporary_prefix}_vector")
    if shape.accumulator_unit_count is not None:
        accumulator_unit = ValueRef.temporary(f"{temporary_prefix}_accumulator")
        emits.extend(
            (
                EmitRegisterSlice(
                    source=source,
                    result=accumulator_unit,
                    unit_offset=packet_index,
                    unit_count=1,
                ),
                EmitDescriptorOp(
                    descriptor=_descriptor(
                        "amd.xdna.aie2p.move.accumulator512.to.vector512"
                    ),
                    operands={"src": accumulator_unit},
                    results={"dst": vector_packet},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
        return vector_packet
    if shape.logical_packet_count == 1:
        return source
    emits.append(
        EmitRegisterSlice(
            source=source,
            result=vector_packet,
            unit_offset=2 * packet_index,
            unit_count=2,
        )
    )
    return vector_packet


def _accumulator_concat_accumulator_packet(
    shape: _AccumulatorConcatOperandShape,
    input_index: int,
    packet_index: int,
    temporary_prefix: str,
    emits: list[ContractEmit],
) -> ValueRef:
    """Returns one operand packet in an MBMS unit."""

    source = ValueRef.operand("inputs", element=input_index)
    accumulator_unit = ValueRef.temporary(f"{temporary_prefix}_accumulator")
    if shape.accumulator_unit_count is not None:
        emits.append(
            EmitRegisterSlice(
                source=source,
                result=accumulator_unit,
                unit_offset=packet_index,
                unit_count=1,
            )
        )
        return accumulator_unit
    vector_packet = _accumulator_concat_x_packet(
        shape,
        input_index,
        packet_index,
        temporary_prefix,
        emits,
    )
    emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512"),
            operands={"src": vector_packet},
            results={"dst": accumulator_unit},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        )
    )
    return accumulator_unit


def _accumulator_concat_move_x_packet(
    vector_packet: ValueRef,
    packet_index: int,
    emits: list[ContractEmit],
) -> ValueRef:
    """Moves one repacked X packet into its result MBMS unit."""

    accumulator_unit = ValueRef.temporary(f"result_accumulator_{packet_index}")
    emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512"),
            operands={"src": vector_packet},
            results={"dst": accumulator_unit},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        )
    )
    return accumulator_unit


def _accumulator_concat_rule(
    left_shape: _AccumulatorConcatOperandShape,
    right_shape: _AccumulatorConcatOperandShape,
    result_shape: _AccumulatorVectorShape,
    result_lane_minimum: int,
    result_lane_maximum: int,
    element_byte_count: int,
) -> DescriptorRule:
    """Concatenates one logical interval into its native accumulator carrier."""

    guards = (
        Guard.i64_range("axis", 0, 0),
        Guard.operand_segment_count("inputs", 2),
        Guard.value_type("inputs", left_shape.source_type, element=0),
        Guard.value_type("inputs", right_shape.source_type, element=1),
        Guard.value_type(
            "result",
            _vector_lane_interval(
                result_shape.element_type,
                result_lane_minimum,
                result_lane_maximum,
            ),
        ),
    )

    # F32x32 is the only complete two-unit accumulator operand. Preserve the
    # existing zero-instruction F32x32 + F32x32 -> F32x64 transition.
    if (
        left_shape.accumulator_unit_count == left_shape.logical_packet_count == 2
        and right_shape.accumulator_unit_count == right_shape.logical_packet_count == 2
        and result_shape.register_unit_count == result_shape.logical_packet_count == 4
    ):
        return DescriptorRule(
            source_op=vector.vector_concat,
            guards=guards,
            emit=(
                EmitRegisterConcat(
                    sources=(
                        ValueRef.operand("inputs", element=0),
                        ValueRef.operand("inputs", element=1),
                    ),
                    result=ValueRef.result("result"),
                ),
            ),
        )

    emits: list[ContractEmit] = []
    accumulator_units: list[ValueRef] = []
    if left_shape.is_packet_aligned:
        for input_index, shape, name in (
            (0, left_shape, "left"),
            (1, right_shape, "right"),
        ):
            accumulator_units.extend(
                _accumulator_concat_accumulator_packet(
                    shape,
                    input_index,
                    packet_index,
                    f"{name}_{packet_index}",
                    emits,
                )
                for packet_index in range(shape.logical_packet_count)
            )
    else:
        accumulator_units.extend(
            _accumulator_concat_accumulator_packet(
                left_shape,
                0,
                packet_index,
                f"left_{packet_index}",
                emits,
            )
            for packet_index in range(left_shape.logical_packet_count - 1)
        )

        left_tail = _accumulator_concat_x_packet(
            left_shape,
            0,
            left_shape.logical_packet_count - 1,
            "left_tail",
            emits,
        )
        right_packets = tuple(
            _accumulator_concat_x_packet(
                right_shape,
                1,
                packet_index,
                f"right_{packet_index}",
                emits,
            )
            for packet_index in range(right_shape.logical_packet_count)
        )
        merged_packet_count = (
            result_shape.logical_packet_count - left_shape.logical_packet_count + 1
        )
        if merged_packet_count not in (
            right_shape.logical_packet_count,
            right_shape.logical_packet_count + 1,
        ):
            raise ValueError("accumulator concat packet partition is inconsistent")

        first_packet_index = left_shape.logical_packet_count - 1
        merged_packets = [ValueRef.temporary(f"result_packet_{first_packet_index}")]
        temporary_prefix = "accumulator_boundary_"
        emits.extend(
            _vector_concat_merge_emits(
                left_tail,
                right_packets[0],
                merged_packets[0],
                left_byte_count=ValueTypeProject.static_dim_scaled(
                    ValueRef.operand("inputs", element=0),
                    scale=element_byte_count,
                    addend=-first_packet_index * 64,
                ),
                remaining_byte_count=(
                    ValueTypeProject.literal_minus_static_dim_scaled(
                        ValueRef.operand("inputs", element=0),
                        scale=element_byte_count,
                        literal=(first_packet_index + 1) * 64,
                    )
                ),
                temporary_prefix=temporary_prefix,
                result_type=DescriptorResultType(),
            )
        )
        remaining_bytes = ValueRef.temporary(f"{temporary_prefix}remaining_bytes")
        shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
        for merged_index in range(1, merged_packet_count):
            merged_packet = ValueRef.temporary(
                f"result_packet_{first_packet_index + merged_index}"
            )
            emits.append(
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={
                        "s1": right_packets[merged_index - 1],
                        "s2": (
                            right_packets[merged_index]
                            if merged_index < len(right_packets)
                            else right_packets[-1]
                        ),
                        "shift": remaining_bytes,
                    },
                    results={"d": merged_packet},
                    result_types={"d": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                )
            )
            merged_packets.append(merged_packet)
        accumulator_units.extend(
            _accumulator_concat_move_x_packet(
                packet,
                first_packet_index + packet_index,
                emits,
            )
            for packet_index, packet in enumerate(merged_packets)
        )

    if len(accumulator_units) != result_shape.logical_packet_count:
        raise ValueError("accumulator concat emitted the wrong logical packet count")
    accumulator_units.extend(
        (accumulator_units[-1],)
        * (result_shape.register_unit_count - len(accumulator_units))
    )
    emits.append(
        EmitRegisterConcat(
            sources=tuple(accumulator_units),
            result=ValueRef.result("result"),
        )
    )
    return DescriptorRule(
        source_op=vector.vector_concat,
        descriptor=(
            _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
            if left_shape.is_packet_aligned
            else _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
        ),
        guards=guards,
        emit=tuple(emits),
    )


def _accumulator_concat_rules() -> tuple[DescriptorRule, ...]:
    """Builds disjoint rules for every legal binary accumulator partition."""

    rules: list[DescriptorRule] = []
    for (
        element_type,
        element_byte_count,
        packet_lane_count,
    ) in _ACCUMULATOR_CONCAT_TYPE_SPECS:
        left_shapes = _accumulator_concat_left_shapes(
            element_type,
            packet_lane_count,
        )
        right_shapes = _accumulator_concat_right_shapes(
            element_type,
            packet_lane_count,
        )
        result_shapes = tuple(
            shape
            for shape in _ACCUMULATOR_VECTOR_SHAPES
            if shape.element_type == element_type
        )
        for left_shape in left_shapes:
            for right_shape in right_shapes:
                sum_minimum = (
                    left_shape.minimum_lane_count + right_shape.minimum_lane_count
                )
                sum_maximum = (
                    left_shape.maximum_lane_count + right_shape.maximum_lane_count
                )
                for result_shape in result_shapes:
                    result_lane_minimum = max(
                        sum_minimum,
                        result_shape.minimum_lane_count,
                    )
                    result_lane_maximum = min(
                        sum_maximum,
                        result_shape.maximum_lane_count,
                    )
                    if result_lane_minimum > result_lane_maximum:
                        continue
                    rules.append(
                        _accumulator_concat_rule(
                            left_shape,
                            right_shape,
                            result_shape,
                            result_lane_minimum,
                            result_lane_maximum,
                            element_byte_count,
                        )
                    )
    return tuple(rules)


_ACCUMULATOR_CONCAT_RULES = _accumulator_concat_rules()
