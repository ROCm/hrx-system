# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native byte dot products with wrapping source accumulation."""

from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.descriptor_rule import (
    IntegerRecipe,
    descriptor_feature_guards,
    logical_core_descriptor,
)
from loom.target.contracts import DescriptorRule, Guard, ValueRef, Vector


def integer_dot_rules() -> tuple[DescriptorRule, ...]:
    rules = []
    for kind, operation, left, right in (
        ("s8s8", "s_dot", "lhs", "rhs"),
        ("u8u8", "u_dot", "lhs", "rhs"),
        ("s8u8", "su_dot", "lhs", "rhs"),
        ("u8s8", "su_dot", "rhs", "lhs"),
    ):
        recipe = IntegerRecipe(32, 1)
        lhs, rhs = ValueRef.operand(left), ValueRef.operand(right)
        if operation == "u_dot":
            lhs = recipe.emit("bitcast.v4i8.v4u8", {"input": lhs})
        if operation != "s_dot":
            rhs = recipe.emit("bitcast.v4i8.v4u8", {"input": rhs})
        descriptor = logical_core_descriptor(f"spirv.op_{operation}.v4i8")
        dot = recipe.emit(f"{operation}.v4i8", {"lhs": lhs, "rhs": rhs})
        if operation == "u_dot":
            dot = recipe.emit("bitcast.u32.i32", {"input": dot})
        # AccSat instructions saturate the accumulator addition. The source
        # operation wraps, so retain the ordinary dot-plus-IAdd pattern.
        recipe.emit(
            "iadd.i32",
            {"lhs": dot, "rhs": ValueRef.operand("acc")},
            result=ValueRef.result("result"),
        )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_dot4i,
                descriptor=descriptor,
                guards=(
                    Guard.enum_attr_equals("kind", kind),
                    Guard.value_type("lhs", Vector("i8", lanes=4)),
                    Guard.value_type("rhs", Vector("i8", lanes=4)),
                    Guard.value_type("acc", Vector("i32", lanes=1)),
                    Guard.value_type("result", Vector("i32", lanes=1)),
                    *descriptor_feature_guards(
                        *(emit.descriptor for emit in recipe.emits)
                    ),
                ),
                emit=tuple(recipe.emits),
            )
        )
    return tuple(rules)
