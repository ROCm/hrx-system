# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BF16 widening across vector and accumulator carriers."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import PacketProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    DescriptorRule,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterSlice,
    Guard,
    ValueRef,
    Vector,
    descriptor_by_key,
)


def _widen_rule(
    minimum_lanes: int,
    maximum_lanes: int,
    *,
    short_accumulator: bool = False,
    preserve_subnormals: bool = True,
) -> DescriptorRule:
    program = PacketProgram(16, "bf16_exact_widen_")
    source = ValueRef.operand("input")
    result = ValueRef.result("result")
    short_packet = maximum_lanes <= 16
    packet_count = (maximum_lanes + 31) // 32
    accumulator_units = 1 if short_packet else 2
    accumulator_result = short_accumulator or minimum_lanes >= 32
    descriptor_key = (
        "widen.2x.w-to-b.unsigned.configured"
        if short_packet
        else "widen.2x.x-to-c.unsigned.configured"
    )
    if preserve_subnormals:
        # Unsaturated unsigned upshift places all BF16 payload bits in the F32
        # high halfword. It preserves signed zero, subnormals, infinities, and NaNs
        # without passing through floating arithmetic or narrowing a result.
        program.state("saturation", 0)
        program.state("ups-mode", 0)
        operands = {"su": program.shift(16)}
    else:
        lanes = 16 if short_packet else 32
        descriptor_key = f"convert.bf16x{lanes}.to.f32x{lanes}"
        operands = {}
    results = []
    for packet in range(packet_count):
        operand = source
        if short_packet or packet_count > 1:
            operand = program.temporary(f"source_{packet}")
            program.emits.append(
                EmitRegisterSlice(
                    source=source,
                    result=operand,
                    unit_offset=packet * 2,
                    unit_count=1 if short_packet else 2,
                )
            )
        accumulator = program.operation(
            f"accumulator_{packet}", descriptor_key, "dst", src=operand, **operands
        )
        if accumulator_result:
            results.append(accumulator)
        else:
            for unit in range(accumulator_units):
                chunk = accumulator
                if accumulator_units > 1:
                    chunk = program.temporary(f"chunk_{unit}")
                    program.emits.append(
                        EmitRegisterSlice(
                            source=accumulator,
                            result=chunk,
                            unit_offset=unit,
                            unit_count=1,
                        )
                    )
                results.append(
                    program.operation(
                        None if accumulator_units == 1 else f"result_{unit}",
                        "move.accumulator512.to.vector512",
                        "dst",
                        src=chunk,
                    )
                )
    if len(results) > 1:
        program.emits.append(EmitRegisterConcat(sources=tuple(results), result=result))
    elif results[0] != result:
        program.emits.append(EmitRegisterCopy(source=results[0], result=result))
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=descriptor_by_key(
            AIE2P_CORE_DESCRIPTOR_SET, f"amd.xdna.aie2p.{descriptor_key}"
        ),
        guards=(
            Guard.value_type(
                "input",
                Vector(
                    "bf16", minimum_lanes=minimum_lanes, maximum_lanes=maximum_lanes
                ),
            ),
            Guard.value_type(
                "result",
                Vector("f32", minimum_lanes=minimum_lanes, maximum_lanes=maximum_lanes),
            ),
            *(
                (
                    Guard.value_not_subnormal_or_instance_flags_has_all(
                        "input", "subnormal", "daz"
                    ),
                )
                if not preserve_subnormals
                else ()
            ),
            *(
                (
                    Guard.low_value_register_class(
                        "result", "aie2p.ebmll" if short_accumulator else "aie2p.vec256"
                    ),
                )
                if short_packet
                else ()
            ),
        ),
        emit=tuple(program.emits),
        report_key=(
            "exact_bfloat16_to_binary32_packet"
            if preserve_subnormals
            else f"native_bfloat16x{minimum_lanes}-{maximum_lanes}_to_"
            f"binary32x{minimum_lanes}-{maximum_lanes}"
        ),
    )


# The exact 32-lane and wider F32 shapes use accumulator storage; shorter
# shapes use vector storage, with a second 16-lane accumulator specialization.
AIE2P_BFLOAT_WIDENING_RULES = (
    _widen_rule(33, 64, preserve_subnormals=False),
    _widen_rule(1, 16),
    _widen_rule(1, 16, short_accumulator=True),
    _widen_rule(17, 31),
    _widen_rule(32, 32),
    _widen_rule(33, 64),
)
