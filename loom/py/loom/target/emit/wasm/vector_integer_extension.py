# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packed Wasm SIMD128 integer lane-extension contract rules."""

from collections.abc import Callable

from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_INTEGER_WIDTHS = (8, 16, 32, 64)


def _extension_descriptor_key(
    source_bit_count: int,
    result_bit_count: int,
    signedness: str,
) -> str:
    return (
        f"wasm.i{result_bit_count}x{128 // result_bit_count}.extend_low_"
        f"i{source_bit_count}x{128 // source_bit_count}_{signedness}"
    )


def _extension_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    source_bit_count: int,
    result_bit_count: int,
    signedness: str,
) -> DescriptorRule:
    source_type = Vector(
        f"i{source_bit_count}",
        minimum_lanes=1,
        maximum_lanes=128 // result_bit_count,
    )
    result_type = Vector(
        f"i{result_bit_count}",
        minimum_lanes=1,
        maximum_lanes=128 // result_bit_count,
    )
    current = ValueRef.operand("input")
    emits: list[EmitDescriptorOp] = []
    width = source_bit_count
    while width < result_bit_count:
        next_width = width * 2
        descriptor = descriptor_lookup(
            _extension_descriptor_key(width, next_width, signedness)
        )
        result = (
            ValueRef.result("result")
            if next_width == result_bit_count
            else ValueRef.temporary(f"i{next_width}")
        )
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": current},
                results={"dst": result},
                result_types=(
                    None
                    if next_width == result_bit_count
                    else {"dst": DescriptorResultType()}
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        current = result
        width = next_width

    return DescriptorRule(
        source_op=(vector.vector_extsi if signedness == "s" else vector.vector_extui),
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", source_type),
            type_guard("result", result_type),
            Guard.value_static_element_count_eq("input", "result"),
        ),
        emit=tuple(emits),
        report_key=(
            "wasm.integer_extension.native"
            if len(emits) == 1
            else "wasm.integer_extension.composed"
        ),
    )


def integer_extension_rules(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> tuple[DescriptorRule, ...]:
    """Returns every packed integer widening pair supported by SIMD128."""

    return tuple(
        _extension_rule(
            descriptor_lookup,
            type_guard,
            source_bit_count,
            result_bit_count,
            signedness,
        )
        for source_index, source_bit_count in enumerate(_INTEGER_WIDTHS[:-1])
        for result_bit_count in _INTEGER_WIDTHS[source_index + 1 :]
        for signedness in ("s", "u")
    )
