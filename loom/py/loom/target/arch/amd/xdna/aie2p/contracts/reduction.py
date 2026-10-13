# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P vector reduction selection rules."""

from typing import Literal

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    F32_ACCUMULATOR_ADD_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.contracts.floating import (
    emit_bf16_extremum,
    emit_f32x16_extremum,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import PairPacketProgram
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
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
    EmitRegisterCopy,
    EmitRegisterMove,
    EmitRegisterSlice,
    Guard,
    Scalar,
    ValueRef,
    ValueTypeProject,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F32 = Scalar("f32")
_BF16 = Scalar("bf16")
_F32X16_VECTOR = Vector("f32", lanes=16)
_F32X64_ACCUMULATOR = Vector("f32", lanes=64)

# AIE2P's 512-bit shuffle network halves the active i32 lanes with each
# control. These sequences are independently witnessed against the AIE API
# reduce_add implementation for the hardware-native vector widths.
_I32_REDUCTION_CONTROLS = (
    (4, (7, 5)),
    (8, (9, 7, 5)),
    (16, (11, 9, 7, 5)),
)

# The pinned AIE API accumulator reduction and its emitted object use these
# odd-half filters to halve the live F32 lanes at each stage.
_F32X16_REDUCTION_CONTROLS = (11, 9, 7, 5)


def _reduction_shapes(
    maximum_lane_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """Partitions one power-of-two carrier into packed reduction trees."""

    if maximum_lane_count <= 0 or maximum_lane_count & (maximum_lane_count - 1):
        raise ValueError("reduction carrier lane count must be a power of two")
    shapes = []
    tree_lane_count = 1
    while tree_lane_count <= maximum_lane_count:
        padded_minimum = tree_lane_count // 2 + 1
        padded_maximum = tree_lane_count - 1
        if padded_minimum <= padded_maximum:
            shapes.append((padded_minimum, padded_maximum, tree_lane_count))
        shapes.append((tree_lane_count, tree_lane_count, tree_lane_count))
        tree_lane_count *= 2
    return tuple(shapes)


# AIE2P reduces native vectors by successively combining each active half with
# the other half. Power-of-two widths use a direct tree; intervening widths
# fill the next tree width with the authored initial value.
_BF16_REDUCTION_SHAPES = _reduction_shapes(32)

_NATIVE_INTEGER_REDUCTION_FAMILIES = (
    ("i8", 8),
    ("i16", 16),
    ("i32", 32),
)
_I64_REDUCTION_SHAPES = _reduction_shapes(8)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _constant_emit(result: ValueRef, value: int) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=_descriptor("amd.xdna.aie2p.constant.i32.mova"),
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _op_emit(
    descriptor: Descriptor,
    *,
    operands: dict[str, ValueRef],
    results: dict[str, ValueRef],
    descriptor_result_type: bool = False,
    immediates: dict[str, int] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands=operands,
        results=results,
        result_types=(
            {name: DescriptorResultType() for name in results}
            if descriptor_result_type
            else None
        ),
        immediates=immediates or {},
        form=DescriptorEmitForm.OP,
    )


def _reduce_add_i32_rule(
    lane_count: int, controls: tuple[int, ...], *, zero_init: bool
) -> DescriptorRule:
    input_type = Vector("i32", lanes=lane_count)
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    vector_add = _descriptor("amd.xdna.aie2p.add.i32x16")
    extract = _descriptor("amd.xdna.aie2p.extract.i32.immediate")
    scalar_add = _descriptor("amd.xdna.aie2p.add.i32")

    emits: list[ContractEmit] = []
    current = ValueRef.operand("input")
    for stage, control in enumerate(controls):
        control_value = ValueRef.temporary(f"control_{stage}")
        shuffled = ValueRef.temporary(f"shuffled_{stage}")
        reduced = ValueRef.temporary(f"reduced_{stage}")
        emits.extend(
            (
                _constant_emit(control_value, control),
                _op_emit(
                    shuffle,
                    operands={"s1": current, "s2": current, "mod": control_value},
                    results={"dst": shuffled},
                    descriptor_result_type=True,
                ),
                _op_emit(
                    vector_add,
                    operands={"s1": current, "s2": shuffled},
                    results={"d": reduced},
                    descriptor_result_type=True,
                ),
            )
        )
        current = reduced

    extracted = (
        ValueRef.result("result") if zero_init else ValueRef.temporary("reduced_scalar")
    )
    emits.append(
        _op_emit(
            extract,
            operands={"s1": current},
            results={"dst": extracted},
            descriptor_result_type=not zero_init,
            immediates={"idx": 0},
        )
    )
    if not zero_init:
        emits.append(
            _op_emit(
                scalar_add,
                operands={"s0": ValueRef.operand("init"), "s1": extracted},
                results={"d0": ValueRef.result("result")},
            )
        )

    guards = [
        Guard.enum_attr_equals("kind", "addi"),
        Guard.value_type("input", input_type),
        Guard.value_type("init", _I32),
        Guard.value_type("result", _I32),
    ]
    if zero_init:
        guards.append(Guard.value_i64_range("init", 0, 0))
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=extract if zero_init else scalar_add,
        guards=tuple(guards),
        emit=tuple(emits),
    )


def _reduce_add_f32x16_rule() -> DescriptorRule:
    clear = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    move_to_accumulator = _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.to.accumulator512.configured")
    add = _descriptor("amd.xdna.aie2p.add.f32x64.configured")
    extract = _descriptor("amd.xdna.aie2p.extract.i32.immediate")

    emits: list[ContractEmit] = []
    for accumulator in ("current_padding", "shuffle_padding"):
        cleared_accumulator = ValueRef.temporary(f"{accumulator}_clear")
        emits.append(
            _op_emit(
                clear,
                operands={},
                results={"dst": cleared_accumulator},
                descriptor_result_type=True,
            )
        )
        emits.append(
            EmitRegisterMove(
                source=cleared_accumulator,
                result=ValueRef.temporary(accumulator),
                result_type=cleared_accumulator,
            )
        )
        emits.extend(
            EmitRegisterSlice(
                source=ValueRef.temporary(accumulator),
                result=ValueRef.temporary(f"{accumulator}_unit_{unit}"),
                unit_offset=unit,
                unit_count=1,
            )
            for unit in range(1, 4)
        )

    emits.extend(
        (
            _op_emit(
                move_to_accumulator,
                operands={"src": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("current_unit")},
                descriptor_result_type=True,
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("current_unit"),
                    *(
                        ValueRef.temporary(f"current_padding_unit_{unit}")
                        for unit in range(1, 4)
                    ),
                ),
                result=ValueRef.temporary("current_accumulator"),
                result_type=_F32X64_ACCUMULATOR,
            ),
            _constant_emit(
                ValueRef.temporary("add_control"), F32_ACCUMULATOR_ADD_CONTROL
            ),
        )
    )

    current_vector = ValueRef.operand("input")
    current_accumulator = ValueRef.temporary("current_accumulator")
    for stage, control in enumerate(_F32X16_REDUCTION_CONTROLS):
        shuffle_control = ValueRef.temporary(f"shuffle_control_{stage}")
        shuffled_unit = ValueRef.temporary(f"shuffled_unit_{stage}")
        shuffled_storage_unit = ValueRef.temporary(f"shuffled_storage_unit_{stage}")
        shuffled_accumulator = ValueRef.temporary(f"shuffled_accumulator_{stage}")
        reduced_accumulator = ValueRef.temporary(f"reduced_accumulator_{stage}")
        reduced_unit = ValueRef.temporary(f"reduced_unit_{stage}")
        reduced_vector = ValueRef.temporary(f"reduced_vector_{stage}")
        emits.extend(
            (
                _constant_emit(shuffle_control, control),
                _op_emit(
                    shuffle,
                    operands={
                        "s1": current_vector,
                        "s2": current_vector,
                        "mod": shuffle_control,
                    },
                    results={"dst": shuffled_unit},
                    descriptor_result_type=True,
                ),
                EmitRegisterCopy(
                    source=shuffled_unit,
                    result=shuffled_storage_unit,
                    result_type=ValueRef.temporary("shuffle_padding_unit_1"),
                ),
                EmitRegisterConcat(
                    sources=(
                        shuffled_storage_unit,
                        *(
                            ValueRef.temporary(f"shuffle_padding_unit_{unit}")
                            for unit in range(1, 4)
                        ),
                    ),
                    result=shuffled_accumulator,
                    result_type=_F32X64_ACCUMULATOR,
                ),
                _op_emit(
                    add,
                    operands={
                        "acc1": current_accumulator,
                        "acc2": shuffled_accumulator,
                        "acc": ValueRef.temporary("add_control"),
                    },
                    results={"dst": reduced_accumulator},
                    descriptor_result_type=True,
                ),
                EmitRegisterSlice(
                    source=reduced_accumulator,
                    result=reduced_unit,
                    unit_count=1,
                ),
                _op_emit(
                    move_from_accumulator,
                    operands={"src": reduced_unit},
                    results={"dst": reduced_vector},
                    descriptor_result_type=True,
                ),
            )
        )
        current_accumulator = reduced_accumulator
        current_vector = reduced_vector

    emits.append(
        _op_emit(
            extract,
            operands={"s1": current_vector},
            results={"dst": ValueRef.result("result")},
            descriptor_result_type=True,
            immediates={"idx": 0},
        )
    )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=add,
        guards=(
            Guard.enum_attr_equals("kind", "addf"),
            Guard.value_type("input", _F32X16_VECTOR),
            Guard.value_type("init", _F32),
            Guard.value_type("result", _F32),
            Guard.value_float_equals("init", 0.0),
            Guard.instance_flags_has_all("fastmath", "reassoc"),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "ninf"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        ),
        emit=tuple(emits),
        report_key="f32x16_accumulator_tree",
    )


def _reduce_extremum_f32x16_rule(
    kind: Literal["minnumf", "minimumf", "maxnumf", "maximumf"],
    operation: Literal["minimum", "maximum"],
) -> DescriptorRule:
    splat = _descriptor("amd.xdna.aie2p.splat.i32x16")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    extract = _descriptor("amd.xdna.aie2p.extract.i32.immediate")

    zero = ValueRef.temporary("extremum_zero")
    zero_vector = ValueRef.temporary("extremum_zero_vector")
    init_vector = ValueRef.temporary("extremum_init_vector")
    current = ValueRef.temporary("extremum_initial")
    emits: list[ContractEmit] = [
        _constant_emit(zero, 0),
        _op_emit(
            splat,
            operands={"src": zero},
            results={"dst": zero_vector},
            descriptor_result_type=True,
        ),
        _op_emit(
            splat,
            operands={"src": ValueRef.operand("init")},
            results={"dst": init_vector},
            descriptor_result_type=True,
        ),
        *emit_f32x16_extremum(
            ValueRef.operand("input"),
            init_vector,
            current,
            operation,
            temporary_prefix="extremum_initial_",
            zero_vector=zero_vector,
        ),
    ]
    for stage, control in enumerate(_F32X16_REDUCTION_CONTROLS):
        control_value = ValueRef.temporary(f"extremum_control_{stage}")
        shuffled = ValueRef.temporary(f"extremum_shuffled_{stage}")
        reduced = ValueRef.temporary(f"extremum_reduced_{stage}")
        emits.extend(
            (
                _constant_emit(control_value, control),
                _op_emit(
                    shuffle,
                    operands={"s1": current, "s2": current, "mod": control_value},
                    results={"dst": shuffled},
                    descriptor_result_type=True,
                ),
                *emit_f32x16_extremum(
                    current,
                    shuffled,
                    reduced,
                    operation,
                    temporary_prefix=f"extremum_stage_{stage}_",
                    zero_vector=zero_vector,
                ),
            )
        )
        current = reduced

    emits.append(
        _op_emit(
            extract,
            operands={"s1": current},
            results={"dst": ValueRef.result("result")},
            descriptor_result_type=True,
            immediates={"idx": 0},
        )
    )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=extract,
        guards=(
            Guard.enum_attr_equals("kind", kind),
            Guard.value_type("input", _F32X16_VECTOR),
            Guard.value_type("init", _F32),
            Guard.value_type("result", _F32),
            Guard.instance_flags_has_all("fastmath", "reassoc"),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        ),
        emit=tuple(emits),
        report_key=f"f32x16_packed_{operation}_tree",
    )


def _reduction_input_type(
    element_type: str, minimum_lane_count: int, maximum_lane_count: int
) -> Vector:
    if minimum_lane_count == maximum_lane_count:
        return Vector(element_type, lanes=minimum_lane_count)
    return Vector(
        element_type,
        minimum_static_elements=minimum_lane_count,
        maximum_static_elements=maximum_lane_count,
    )


def _reduction_shift_byte_counts(
    tree_lane_count: int, element_byte_count: int
) -> tuple[int, ...]:
    half_lane_count = tree_lane_count // 2
    byte_counts = []
    while half_lane_count:
        byte_counts.append(half_lane_count * element_byte_count)
        half_lane_count //= 2
    return tuple(byte_counts)


def _pad_reduction_input_emits(
    input_value: ValueRef,
    init_vector: ValueRef,
    result: ValueRef,
    *,
    element_width: int,
    minimum_lane_count: int,
) -> tuple[ContractEmit, ...]:
    predicate_constant = _descriptor("amd.xdna.aie2p.constant.i32.predicate.low32")
    if minimum_lane_count > 32:
        low_immediate = -1
        high_immediate = ValueTypeProject.static_dim_low_bits_mask(
            input_value, addend=-32
        )
        predicate_complete = _descriptor(
            "amd.xdna.aie2p.predicate.complete.constant.high32"
        )
    else:
        low_immediate = ValueTypeProject.static_dim_low_bits_mask(input_value)
        high_immediate = 0
        predicate_complete = _descriptor(
            "amd.xdna.aie2p.predicate.complete.zero.high32"
        )
    select_suffix = (
        f"i{element_width}x{512 // element_width}"
        if element_width == 8
        else f"i{element_width}x{512 // element_width}.mask64"
    )
    select = _descriptor(f"amd.xdna.aie2p.select.{select_suffix}")
    active_low = ValueRef.temporary("extremum_active_low")
    active = ValueRef.temporary("extremum_active")
    return (
        EmitDescriptorOp(
            descriptor=predicate_constant,
            results={"dst": active_low},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": low_immediate},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=predicate_complete,
            operands={"storage": active_low},
            results={"dst": active},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": high_immediate},
            form=DescriptorEmitForm.OP,
        ),
        _op_emit(
            select,
            operands={"s1": init_vector, "s2": input_value, "sel": active},
            results={"d": result},
            descriptor_result_type=True,
        ),
    )


def _reduce_extremum_bf16_rule(
    kinds: tuple[str, str],
    operation: Literal["minimum", "maximum"],
    minimum_lane_count: int,
    maximum_lane_count: int,
    tree_lane_count: int,
) -> DescriptorRule:
    input_type = _reduction_input_type("bf16", minimum_lane_count, maximum_lane_count)
    broadcast = _descriptor("amd.xdna.aie2p.splat.i16x32")
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    extract = _descriptor("amd.xdna.aie2p.extract.i16.immediate")

    init_vector = ValueRef.temporary("extremum_init_vector")
    current = ValueRef.temporary("extremum_initial")
    emits: list[ContractEmit] = [
        _op_emit(
            broadcast,
            operands={"src": ValueRef.operand("init")},
            results={"dst": init_vector},
            descriptor_result_type=True,
        )
    ]
    if minimum_lane_count == maximum_lane_count == tree_lane_count:
        emits.extend(
            emit_bf16_extremum(
                ValueRef.operand("input"),
                init_vector,
                current,
                operation,
                temporary_prefix="extremum_initial_",
            )
        )
        report_mode = "packed"
    else:
        emits.extend(
            _pad_reduction_input_emits(
                ValueRef.operand("input"),
                init_vector,
                current,
                element_width=16,
                minimum_lane_count=minimum_lane_count,
            )
        )
        report_mode = "padded"

    for stage, byte_count in enumerate(
        _reduction_shift_byte_counts(tree_lane_count, 2)
    ):
        control = ValueRef.temporary(f"extremum_control_{stage}")
        shifted = ValueRef.temporary(f"extremum_shifted_{stage}")
        reduced = ValueRef.temporary(f"extremum_reduced_{stage}")
        emits.extend(
            (
                _constant_emit(control, byte_count),
                _op_emit(
                    shift,
                    operands={"s1": current, "s2": current, "shift": control},
                    results={"d": shifted},
                    descriptor_result_type=True,
                ),
                *emit_bf16_extremum(
                    current,
                    shifted,
                    reduced,
                    operation,
                    temporary_prefix=f"extremum_stage_{stage}_",
                ),
            )
        )
        current = reduced

    emits.append(
        _op_emit(
            extract,
            operands={"s1": current},
            results={"dst": ValueRef.result("result")},
            descriptor_result_type=True,
            immediates={"idx": 0},
        )
    )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=extract,
        guards=(
            Guard.enum_attr_in("kind", kinds),
            Guard.value_type("input", input_type),
            Guard.value_type("init", _BF16),
            Guard.value_type("result", _BF16),
            Guard.instance_flags_has_all("fastmath", "reassoc"),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        ),
        emit=tuple(emits),
        report_key=f"bf16_{report_mode}_{operation}_tree",
    )


def _reduce_extremum_integer_rule(
    kind: Literal["minsi", "maxsi", "minui", "maxui"],
    operation: Literal["min", "max"],
    signedness: Literal["signed", "unsigned"],
    element_type: Literal["i8", "i16", "i32"],
    element_width: int,
    minimum_lane_count: int,
    maximum_lane_count: int,
    tree_lane_count: int,
) -> DescriptorRule:
    input_type = _reduction_input_type(
        element_type, minimum_lane_count, maximum_lane_count
    )
    scalar_type = Scalar(element_type)
    lane_count = 512 // element_width
    broadcast = _descriptor(f"amd.xdna.aie2p.splat.i{element_width}x{lane_count}")
    extremum = _descriptor(
        f"amd.xdna.aie2p.{operation}.{signedness}.i{element_width}x{lane_count}"
    )
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    extract = _descriptor(f"amd.xdna.aie2p.extract.i{element_width}.immediate")

    init_vector = ValueRef.temporary("extremum_init_vector")
    current = ValueRef.temporary("extremum_initial")
    emits: list[ContractEmit] = [
        _op_emit(
            broadcast,
            operands={"src": ValueRef.operand("init")},
            results={"dst": init_vector},
            descriptor_result_type=True,
        )
    ]
    if minimum_lane_count == maximum_lane_count == tree_lane_count:
        emits.append(
            _op_emit(
                extremum,
                operands={"s1": ValueRef.operand("input"), "s2": init_vector},
                results={"d": current},
                descriptor_result_type=True,
            )
        )
        report_mode = "packed"
    else:
        emits.extend(
            _pad_reduction_input_emits(
                ValueRef.operand("input"),
                init_vector,
                current,
                element_width=element_width,
                minimum_lane_count=minimum_lane_count,
            )
        )
        report_mode = "padded"

    for stage, byte_count in enumerate(
        _reduction_shift_byte_counts(tree_lane_count, element_width // 8)
    ):
        control = ValueRef.temporary(f"extremum_control_{stage}")
        shifted = ValueRef.temporary(f"extremum_shifted_{stage}")
        reduced = ValueRef.temporary(f"extremum_reduced_{stage}")
        emits.extend(
            (
                _constant_emit(control, byte_count),
                _op_emit(
                    shift,
                    operands={"s1": current, "s2": current, "shift": control},
                    results={"d": shifted},
                    descriptor_result_type=True,
                ),
                _op_emit(
                    extremum,
                    operands={"s1": current, "s2": shifted},
                    results={"d": reduced},
                    descriptor_result_type=True,
                ),
            )
        )
        current = reduced

    emits.append(
        _op_emit(
            extract,
            operands={"s1": current},
            results={"dst": ValueRef.result("result")},
            descriptor_result_type=True,
            immediates={"idx": 0},
        )
    )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=extract,
        guards=(
            Guard.enum_attr_equals("kind", kind),
            Guard.value_type("input", input_type),
            Guard.value_type("init", scalar_type),
            Guard.value_type("result", scalar_type),
        ),
        emit=tuple(emits),
        report_key=f"{element_type}_{report_mode}_{kind}_tree",
    )


def _emit_i64_extremum(
    lhs: tuple[ValueRef, ValueRef],
    rhs: tuple[ValueRef, ValueRef],
    *,
    operation: Literal["min", "max"],
    signedness: Literal["signed", "unsigned"],
    temporary_prefix: str,
) -> tuple[tuple[ContractEmit, ...], tuple[ValueRef, ValueRef]]:
    """Selects packed i64 extrema from deinterleaved i32 word streams.

    The high-word extremum is the result high word. When the high words tie,
    the unsigned low-word extremum is the result low word. Two strict high-word
    comparisons steer either source low word around that tie result, avoiding
    a separate high-word equality sequence.
    """

    high_extremum = _descriptor(f"amd.xdna.aie2p.{operation}.{signedness}.i32x16")
    low_extremum = _descriptor(f"amd.xdna.aie2p.{operation}.unsigned.i32x16")
    high_compare = _descriptor(f"amd.xdna.aie2p.cmp.lt.{signedness}.i32x16.el.low32")
    select = _descriptor("amd.xdna.aie2p.select.i32x16.mask64")

    result_high = ValueRef.temporary(f"{temporary_prefix}high")
    tied_low = ValueRef.temporary(f"{temporary_prefix}tied_low")
    lhs_high_better = ValueRef.temporary(f"{temporary_prefix}lhs_high_better")
    rhs_high_better = ValueRef.temporary(f"{temporary_prefix}rhs_high_better")
    low_unless_rhs_high = ValueRef.temporary(f"{temporary_prefix}low_unless_rhs_high")
    result_low = ValueRef.temporary(f"{temporary_prefix}low")

    ordered_lhs, ordered_rhs = (lhs, rhs) if operation == "min" else (rhs, lhs)
    return (
        (
            _op_emit(
                high_extremum,
                operands={"s1": lhs[1], "s2": rhs[1]},
                results={"d": result_high},
                descriptor_result_type=True,
            ),
            _op_emit(
                low_extremum,
                operands={"s1": lhs[0], "s2": rhs[0]},
                results={"d": tied_low},
                descriptor_result_type=True,
            ),
            _op_emit(
                high_compare,
                operands={"s1": ordered_lhs[1], "s2": ordered_rhs[1]},
                results={"cmp": lhs_high_better},
                descriptor_result_type=True,
            ),
            _op_emit(
                high_compare,
                operands={"s1": ordered_rhs[1], "s2": ordered_lhs[1]},
                results={"cmp": rhs_high_better},
                descriptor_result_type=True,
            ),
            _op_emit(
                select,
                operands={
                    "s1": tied_low,
                    "s2": lhs[0],
                    "sel": lhs_high_better,
                },
                results={"d": low_unless_rhs_high},
                descriptor_result_type=True,
            ),
            _op_emit(
                select,
                operands={
                    "s1": low_unless_rhs_high,
                    "s2": rhs[0],
                    "sel": rhs_high_better,
                },
                results={"d": result_low},
                descriptor_result_type=True,
            ),
        ),
        (result_low, result_high),
    )


def _reduce_extremum_i64_rule(
    kind: Literal["minsi", "maxsi", "minui", "maxui"],
    operation: Literal["min", "max"],
    signedness: Literal["signed", "unsigned"],
    minimum_lane_count: int,
    maximum_lane_count: int,
    tree_lane_count: int,
) -> DescriptorRule:
    input_type = _reduction_input_type("i64", minimum_lane_count, maximum_lane_count)
    # Deinterleave once and keep the low and high words in independent packed
    # streams through the entire tree. Repeatedly splitting and rejoining i64
    # vectors would make every tree stage pay four additional shuffles.
    program = PairPacketProgram("extremum_")
    input_words = program.split_i64_words(ValueRef.operand("input"), "input")
    emits: list[ContractEmit] = list(program.emits)

    init_words = (
        ValueRef.temporary("extremum_init_low"),
        ValueRef.temporary("extremum_init_high"),
    )
    init_vectors = (
        ValueRef.temporary("extremum_init_low_vector"),
        ValueRef.temporary("extremum_init_high_vector"),
    )
    splat = _descriptor("amd.xdna.aie2p.splat.i32x16")
    for word_index, (init_word, init_vector) in enumerate(
        zip(init_words, init_vectors, strict=True)
    ):
        emits.extend(
            (
                EmitRegisterSlice(
                    source=ValueRef.operand("init"),
                    result=init_word,
                    unit_offset=word_index,
                    unit_count=1,
                ),
                _op_emit(
                    splat,
                    operands={"src": init_word},
                    results={"dst": init_vector},
                    descriptor_result_type=True,
                ),
            )
        )

    exact_tree = minimum_lane_count == maximum_lane_count == tree_lane_count
    if exact_tree:
        initial_emits, current_words = _emit_i64_extremum(
            input_words,
            init_vectors,
            operation=operation,
            signedness=signedness,
            temporary_prefix="extremum_initial_",
        )
        emits.extend(initial_emits)
        report_mode = "packed"
    else:
        active = ValueRef.temporary("extremum_active")
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.constant.i32.predicate.low32"),
                results={"dst": active},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "i": ValueTypeProject.static_dim_low_bits_mask(
                        ValueRef.operand("input")
                    )
                },
                form=DescriptorEmitForm.CONST,
            )
        )
        select = _descriptor("amd.xdna.aie2p.select.i32x16.mask64")
        current_words = (
            ValueRef.temporary("extremum_initial_low"),
            ValueRef.temporary("extremum_initial_high"),
        )
        for input_word, init_vector, current_word in zip(
            input_words, init_vectors, current_words, strict=True
        ):
            emits.append(
                _op_emit(
                    select,
                    operands={"s1": init_vector, "s2": input_word, "sel": active},
                    results={"d": current_word},
                    descriptor_result_type=True,
                )
            )
        report_mode = "padded"

    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    for stage, byte_count in enumerate(
        _reduction_shift_byte_counts(tree_lane_count, 4)
    ):
        control = ValueRef.temporary(f"extremum_control_{stage}")
        shifted_words = (
            ValueRef.temporary(f"extremum_shifted_{stage}_low"),
            ValueRef.temporary(f"extremum_shifted_{stage}_high"),
        )
        emits.append(_constant_emit(control, byte_count))
        for current_word, shifted_word in zip(
            current_words, shifted_words, strict=True
        ):
            emits.append(
                _op_emit(
                    shift,
                    operands={
                        "s1": current_word,
                        "s2": current_word,
                        "shift": control,
                    },
                    results={"d": shifted_word},
                    descriptor_result_type=True,
                )
            )
        stage_emits, current_words = _emit_i64_extremum(
            current_words,
            shifted_words,
            operation=operation,
            signedness=signedness,
            temporary_prefix=f"extremum_stage_{stage}_",
        )
        emits.extend(stage_emits)

    extract = _descriptor("amd.xdna.aie2p.extract.i32.immediate")
    result_words = (
        ValueRef.temporary("extremum_result_low"),
        ValueRef.temporary("extremum_result_high"),
    )
    for current_word, result_word in zip(current_words, result_words, strict=True):
        emits.append(
            _op_emit(
                extract,
                operands={"s1": current_word},
                results={"dst": result_word},
                descriptor_result_type=True,
                immediates={"idx": 0},
            )
        )
    emits.append(
        EmitRegisterConcat(
            sources=result_words,
            result=ValueRef.result("result"),
        )
    )
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=extract,
        guards=(
            Guard.enum_attr_equals("kind", kind),
            Guard.value_type("input", input_type),
            Guard.value_type("init", _I64),
            Guard.value_type("result", _I64),
        ),
        emit=tuple(emits),
        report_key=f"i64_{report_mode}_{kind}_tree",
    )


def _reduce_i1_rule(
    lane_count: int, kind: str, *, identity_init: bool
) -> DescriptorRule:
    # The carrier is 64 bits; bits outside the logical vector are undefined.
    program = ScalarProgram()
    result_name = None if identity_init else "reduced"
    word_count = (lane_count + 31) // 32
    words = []
    for word_index, word in enumerate(("low32", "high32")[:word_count]):
        width = min(32, lane_count - 32 * word_index)
        mask = program.constant(f"mask_{word}", -1 if width == 32 else (1 << width) - 1)
        active = program.binary(
            f"active_{word}", f"predicate.mask.{word}", ValueRef.operand("input"), mask
        )
        words.append(
            program.binary(
                result_name if word_count == 1 else f"all_{word}",
                "cmp.eq.i32",
                active,
                mask,
            )
            if kind == "andi"
            else active
        )
    combined = (
        program.binary(
            result_name if kind == "andi" else "combined",
            "and.i32" if kind == "andi" else "or.i32",
            *words,
        )
        if len(words) == 2
        else words[0]
    )
    if kind == "ori":
        program.unary(result_name, "cmp.nez.i32", combined)
    if not identity_init:
        program.binary(
            None,
            "and.i32" if kind == "andi" else "or.i32",
            ValueRef.temporary("reduced"),
            ValueRef.operand("init"),
        )
    guards = [
        Guard.enum_attr_equals("kind", kind),
        Guard.value_type(
            "input",
            Vector(
                "i1",
                minimum_static_elements=lane_count,
                maximum_static_elements=lane_count,
            ),
        ),
        Guard.value_type("init", Scalar("i1")),
        Guard.value_type("result", Scalar("i1")),
    ]
    if identity_init:
        identity = int(kind == "andi")
        guards.append(Guard.value_i64_range("init", identity, identity))
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(guards),
        emit=tuple(program.emits),
    )


AIE2P_REDUCTION_RULES = (
    _reduce_add_f32x16_rule(),
    *(
        _reduce_extremum_f32x16_rule(kind, operation)
        for kind, operation in (
            ("minnumf", "minimum"),
            ("minimumf", "minimum"),
            ("maxnumf", "maximum"),
            ("maximumf", "maximum"),
        )
    ),
    *(
        _reduce_extremum_bf16_rule(
            kinds,
            operation,
            minimum_lane_count,
            maximum_lane_count,
            tree_lane_count,
        )
        for kinds, operation in (
            (("minnumf", "minimumf"), "minimum"),
            (("maxnumf", "maximumf"), "maximum"),
        )
        for minimum_lane_count, maximum_lane_count, tree_lane_count in (
            _BF16_REDUCTION_SHAPES
        )
    ),
    *(
        _reduce_extremum_integer_rule(
            kind,
            operation,
            signedness,
            element_type,
            element_width,
            minimum_lane_count,
            maximum_lane_count,
            tree_lane_count,
        )
        for kind, operation, signedness in (
            ("minsi", "min", "signed"),
            ("maxsi", "max", "signed"),
            ("minui", "min", "unsigned"),
            ("maxui", "max", "unsigned"),
        )
        for element_type, element_width in _NATIVE_INTEGER_REDUCTION_FAMILIES
        for minimum_lane_count, maximum_lane_count, tree_lane_count in (
            _reduction_shapes(512 // element_width)
        )
    ),
    *(
        _reduce_extremum_i64_rule(
            kind,
            operation,
            signedness,
            minimum_lane_count,
            maximum_lane_count,
            tree_lane_count,
        )
        for kind, operation, signedness in (
            ("minsi", "min", "signed"),
            ("maxsi", "max", "signed"),
            ("minui", "min", "unsigned"),
            ("maxui", "max", "unsigned"),
        )
        for minimum_lane_count, maximum_lane_count, tree_lane_count in (
            _I64_REDUCTION_SHAPES
        )
    ),
    *(
        _reduce_add_i32_rule(lane_count, controls, zero_init=zero_init)
        for lane_count, controls in _I32_REDUCTION_CONTROLS
        for zero_init in (True, False)
    ),
    *(
        _reduce_i1_rule(lane_count, kind, identity_init=identity_init)
        for kind in ("ori", "andi")
        for lane_count in range(1, 65)
        for identity_init in (True, False)
    ),
)
