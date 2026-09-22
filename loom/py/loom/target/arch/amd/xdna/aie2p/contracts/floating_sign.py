# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact floating-point sign operations using integer data paths."""

from __future__ import annotations

from typing import Literal

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    Vector,
    descriptor_by_key,
)

type _SignOperation = Literal["abs", "neg", "copysign"]


def _scalar_sign_rule(
    source_op: Op, element_type: str, width: int, operation: _SignOperation
) -> DescriptorRule:
    program = ScalarProgram()
    absolute_mask = (1 << (width - 1)) - 1
    sign_mask = (1 << (width - 1)) if width < 32 else -(1 << 31)
    if operation == "abs":
        mask = program.constant("absolute_mask", absolute_mask)
        program.binary(None, "and.i32", ValueRef.operand("input"), mask)
    elif operation == "neg":
        mask = program.constant("sign_mask", sign_mask)
        program.binary(None, "xor.i32", ValueRef.operand("input"), mask)
    else:
        magnitude_mask = program.constant("absolute_mask", absolute_mask)
        sign_bits = program.constant("sign_mask", sign_mask)
        magnitude = program.binary(
            "magnitude", "and.i32", ValueRef.operand("lhs"), magnitude_mask
        )
        sign = program.binary("sign", "and.i32", ValueRef.operand("rhs"), sign_bits)
        program.binary(None, "or.i32", magnitude, sign)
    fields = (
        ("lhs", "rhs", "result") if operation == "copysign" else ("input", "result")
    )
    format_name = "binary32" if element_type == "f32" else element_type
    return DescriptorRule(
        source_op=source_op,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(Guard.value_type(field, Scalar(element_type)) for field in fields),
        emit=tuple(program.emits),
        report_key=f"exact_{format_name}_{operation}",
    )


def _vector_sign_rule(
    source_op: Op,
    element_type: str,
    width: int,
    operation: _SignOperation,
) -> DescriptorRule:
    source_type = Vector(
        element_type, minimum_static_elements=1, maximum_static_elements=512 // width
    )
    input_fields = ("lhs", "rhs") if operation == "copysign" else ("input",)
    input_values = {field: ValueRef.operand(field) for field in input_fields}
    emits: list[ContractEmit] = []
    input_value = input_values[input_fields[0]]
    vector_type = input_value
    output_value = ValueRef.result("result")

    def mask(name: str, bits: int) -> ValueRef:
        scalar = ValueRef.temporary(f"{name}_scalar")
        result = ValueRef.temporary(name)
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=descriptor_by_key(
                        AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.constant.i32"
                    ),
                    results={"dst": scalar},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": bits},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=descriptor_by_key(
                        AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.splat.i32x16"
                    ),
                    operands={"src": scalar},
                    results={"dst": result},
                    result_types={"dst": vector_type},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
        return result

    def binary(name: str | None, opcode: str, lhs: ValueRef, rhs: ValueRef) -> ValueRef:
        result = output_value if name is None else ValueRef.temporary(name)
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_by_key(
                    AIE2P_CORE_DESCRIPTOR_SET, f"amd.xdna.aie2p.{opcode}"
                ),
                operands={"s1": lhs, "s2": rhs},
                results={"d": result},
                result_types=(
                    {} if result == ValueRef.result("result") else {"d": vector_type}
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        return result

    # Repeat each element's sign position within the broadcast 32-bit word.
    sign_bits = sum(1 << (position - 1) for position in range(width, 33, width))
    absolute_bits = 0xFFFFFFFF ^ sign_bits
    signed_sign_bits = sign_bits - (1 << 32)
    if operation == "abs":
        absolute_mask = mask("absolute_mask", absolute_bits)
        binary(None, "and.bits512", input_value, absolute_mask)
    elif operation == "neg":
        sign_mask = mask("sign_mask", signed_sign_bits)
        union = binary("union", "or.bits512", input_value, sign_mask)
        intersection = binary("intersection", "and.bits512", input_value, sign_mask)
        # (a | b) - (a & b) equals a ^ b without cross-element borrowing.
        binary(None, "sub.i32x16", union, intersection)
    else:
        absolute_mask = mask("absolute_mask", absolute_bits)
        sign_mask = mask("sign_mask", signed_sign_bits)
        magnitude = binary("magnitude", "and.bits512", input_value, absolute_mask)
        sign = binary("sign", "and.bits512", input_values["rhs"], sign_mask)
        binary(None, "or.bits512", magnitude, sign)
    descriptor = emits[-1].descriptor
    fields = (
        ("lhs", "rhs", "result") if operation == "copysign" else ("input", "result")
    )
    format_name = "binary32" if element_type == "f32" else element_type
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=tuple(Guard.value_type(field, source_type) for field in fields),
        emit=tuple(emits),
        report_key=f"native_vector_{format_name}_{operation}",
    )


_SIGN_OPERATIONS: tuple[tuple[Op, Op, _SignOperation], ...] = (
    (scalar_arithmetic.scalar_absf, vector.vector_absf, "abs"),
    (scalar_arithmetic.scalar_negf, vector.vector_negf, "neg"),
    (scalar_arithmetic.scalar_copysignf, vector.vector_copysignf, "copysign"),
)

AIE2P_FLOATING_SIGN_RULES = tuple(
    build_rule(source_op, element_type, width, operation)
    for element_type, width in (("f32", 32), ("bf16", 16))
    for scalar_op, vector_op, operation in _SIGN_OPERATIONS
    for build_rule, source_op in (
        (_scalar_sign_rule, scalar_op),
        (_vector_sign_rule, vector_op),
    )
)
