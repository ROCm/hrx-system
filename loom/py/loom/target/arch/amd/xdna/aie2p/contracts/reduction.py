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

# AIE2P reduces native BF16 vectors by successively comparing each active
# half with the other half. Power-of-two widths use the direct vendor tree;
# intervening widths fill the next tree width with the authored initial value.
_BF16_REDUCTION_SHAPES = (
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 4),
    (4, 4, 4),
    (5, 7, 8),
    (8, 8, 8),
    (9, 15, 16),
    (16, 16, 16),
    (17, 31, 32),
    (32, 32, 32),
)


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


def _bf16_reduction_input_type(
    minimum_lane_count: int, maximum_lane_count: int
) -> Vector:
    if minimum_lane_count == maximum_lane_count:
        return Vector("bf16", lanes=minimum_lane_count)
    return Vector(
        "bf16",
        minimum_static_elements=minimum_lane_count,
        maximum_static_elements=maximum_lane_count,
    )


def _bf16_reduction_shift_byte_counts(tree_lane_count: int) -> tuple[int, ...]:
    half_lane_count = tree_lane_count // 2
    byte_counts = []
    while half_lane_count:
        byte_counts.append(half_lane_count * 2)
        half_lane_count //= 2
    return tuple(byte_counts)


def _pad_bf16_reduction_input_emits(
    input_value: ValueRef,
    init_vector: ValueRef,
    result: ValueRef,
) -> tuple[ContractEmit, ...]:
    predicate_constant = _descriptor("amd.xdna.aie2p.constant.i32.predicate.low32")
    predicate_complete = _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
    select = _descriptor("amd.xdna.aie2p.select.i16x32.mask64")
    active_low = ValueRef.temporary("extremum_active_low")
    active = ValueRef.temporary("extremum_active")
    return (
        EmitDescriptorOp(
            descriptor=predicate_constant,
            results={"dst": active_low},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": ValueTypeProject.static_dim_low_bits_mask(input_value)},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=predicate_complete,
            operands={"storage": active_low},
            results={"dst": active},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": 0},
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
    input_type = _bf16_reduction_input_type(minimum_lane_count, maximum_lane_count)
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
            _pad_bf16_reduction_input_emits(
                ValueRef.operand("input"), init_vector, current
            )
        )
        report_mode = "padded"

    for stage, byte_count in enumerate(
        _bf16_reduction_shift_byte_counts(tree_lane_count)
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
