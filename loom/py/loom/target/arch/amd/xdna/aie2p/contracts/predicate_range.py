# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P scalar construction of contiguous predicate ranges."""

from dataclasses import dataclass

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    ValueTypeProject,
    Vector,
    descriptor_by_key,
)


@dataclass(frozen=True, slots=True)
class _CoordinateType:
    """One scalar carrier whose non-wrapping range fits native i32 ALU ops."""

    pattern: Scalar
    maximum: int


_COORDINATE_TYPES = (
    _CoordinateType(Scalar("i8"), 2**7 - 1),
    _CoordinateType(Scalar("i16"), 2**15 - 1),
    _CoordinateType(Scalar("i32"), 2**31 - 1),
    _CoordinateType(Scalar("index"), 2**31 - 1),
)


def _descriptor(key: str):
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _result_lane_count(*, addend: int = 0) -> ValueTypeProject:
    return ValueTypeProject.static_dim_scaled(
        ValueRef.result("result"), scale=1, addend=addend
    )


def _emit_clamped_lane_count(program: ScalarProgram) -> ValueRef:
    """Computes clamp(upper - lower, 0, result lane count)."""

    delta = program.binary(
        "range_delta",
        "sub.i32",
        ValueRef.operand("upper_bound"),
        ValueRef.operand("lower_bound"),
    )
    zero = program.constant("range_zero", 0)
    positive = program.binary("range_is_positive", "cmp.slt.i32", zero, delta)
    nonnegative = program.select("range_nonnegative", delta, zero, positive)
    lane_count = program.constant("range_lane_count", _result_lane_count())
    within_result = program.binary(
        "range_within_result", "cmp.slt.i32", nonnegative, lane_count
    )
    return program.select("range_count", nonnegative, lane_count, within_result)


def _emit_partial_word_mask(
    program: ScalarProgram,
    count: ValueRef,
    *,
    prefix: str,
) -> ValueRef:
    """Returns (1 << count) - 1 for a count proven in [0, 31]."""

    one = program.constant(f"{prefix}_one", 1)
    shifted = program.binary(f"{prefix}_shifted", "lshl.i32", one, count)
    return program.add_immediate(f"{prefix}_mask", shifted, -1)


def _emit_full_word_mask(
    program: ScalarProgram,
    count: ValueRef,
    *,
    prefix: str,
) -> ValueRef:
    """Returns a low-bit mask for a count proven in [0, 32]."""

    word_width = program.constant(f"{prefix}_width", 32)
    below_width = program.binary(
        f"{prefix}_is_partial", "cmp.slt.i32", count, word_width
    )
    maximum_shift = program.constant(f"{prefix}_maximum_shift", 31)
    safe_count = program.select(
        f"{prefix}_safe_count", count, maximum_shift, below_width
    )
    partial = _emit_partial_word_mask(program, safe_count, prefix=f"{prefix}_partial")
    full = program.constant(f"{prefix}_full", -1)
    return program.select(f"{prefix}_mask", partial, full, below_width)


def _append_low_word(
    program: ScalarProgram,
    mask: ValueRef,
) -> ValueRef:
    low = program.temporary("range_low")
    program.emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.predicate.copy.scalar.to.low32"),
            operands={"s0": mask, "s1": mask},
            results={"d0": low},
            result_types={"d0": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        )
    )
    return low


def _append_zero_high_word(program: ScalarProgram, low: ValueRef) -> None:
    program.emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.predicate.complete.zero.high32"),
            operands={"storage": low},
            results={"dst": ValueRef.result("result")},
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        )
    )


def _append_high_word(
    program: ScalarProgram,
    low: ValueRef,
    mask: ValueRef,
) -> None:
    program.emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.predicate.copy.scalar.to.high32"),
            operands={"s0": mask, "s1": mask, "storage": low},
            results={"d0": ValueRef.result("result")},
            form=DescriptorEmitForm.OP,
        )
    )


def _prefix_range_emits(
    *,
    maximum_lanes: int,
) -> tuple[EmitDescriptorOp, ...]:
    program = ScalarProgram()
    count = _emit_clamped_lane_count(program)

    if maximum_lanes <= 31:
        low_mask = _emit_partial_word_mask(program, count, prefix="range_low")
        _append_zero_high_word(program, _append_low_word(program, low_mask))
        return tuple(program.emits)

    low_mask = _emit_full_word_mask(program, count, prefix="range_low")
    low = _append_low_word(program, low_mask)
    if maximum_lanes == 32:
        _append_zero_high_word(program, low)
        return tuple(program.emits)

    thirty_two = program.constant("range_high_base", 32)
    high_delta = program.binary("range_high_delta", "sub.i32", count, thirty_two)
    has_high = program.binary("range_has_high", "cmp.slt.i32", thirty_two, count)
    zero = program.constant("range_high_zero", 0)
    high_count = program.select("range_high_count", high_delta, zero, has_high)
    high_mask = (
        _emit_full_word_mask(program, high_count, prefix="range_high")
        if maximum_lanes == 64
        else _emit_partial_word_mask(program, high_count, prefix="range_high")
    )
    _append_high_word(program, low, high_mask)
    return tuple(program.emits)


def _prefix_range_rule(
    coordinate: _CoordinateType,
    *,
    minimum_lanes: int,
    maximum_lanes: int,
) -> DescriptorRule:
    result_type = Vector(
        "i1",
        minimum_static_elements=minimum_lanes,
        maximum_static_elements=maximum_lanes,
    )
    emits = _prefix_range_emits(
        maximum_lanes=maximum_lanes,
    )
    return DescriptorRule(
        source_op=vector.vector_mask_range,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("lower_bound", coordinate.pattern),
            Guard.value_type("upper_bound", coordinate.pattern),
            Guard.value_type("step", coordinate.pattern),
            Guard.value_type("result", result_type),
            Guard.value_i64_range(
                "lower_bound", 0, coordinate.maximum - (maximum_lanes - 1)
            ),
            Guard.value_i64_range("upper_bound", 0, coordinate.maximum),
            Guard.value_i64_range("step", 1, 1),
        ),
        emit=emits,
        report_key="predicate_range.scalar_prefix",
    )


# One predicate carrier owns at most 64 lanes. The partitions distinguish the
# shift-by-32 boundary and keep the common 1..31 lane path minimal. Wider masks,
# i64 coordinates, non-unit steps, negative coordinates, and wrapping fixed-
# width ranges retain the exact reference lowering because their semantics are
# not a single scalar prefix.
AIE2P_PREDICATE_RANGE_RULES = tuple(
    _prefix_range_rule(
        coordinate,
        minimum_lanes=minimum_lanes,
        maximum_lanes=maximum_lanes,
    )
    for coordinate in _COORDINATE_TYPES
    for minimum_lanes, maximum_lanes in (
        (1, 31),
        (32, 32),
        (33, 63),
        (64, 64),
    )
)
