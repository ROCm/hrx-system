# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shader-legal ordered and unordered floating-point comparison recipes."""

from loom.dsl import Op
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.scalar_alu import FLOAT_COMPARE_TYPES
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    Guard,
    Scalar,
    ValueRef,
    Vector,
)


def float_classification_rules(
    source_op: Op, lane_count: int | None = None
) -> tuple[DescriptorRule, ...]:
    # OpOrdered and OpUnordered require Kernel and are illegal in Vulkan.
    # Self-comparisons classify each operand using the same floating-point
    # environment as the ordinary comparisons; Boolean operations combine them.
    rules = []
    boolean_suffix = "bool" if lane_count in (None, 1) else f"v{lane_count}bool"
    result_pattern = (
        Scalar("i1") if lane_count is None else Vector("i1", lanes=lane_count)
    )
    for scalar in FLOAT_COMPARE_TYPES:
        suffix = (
            scalar.suffix
            if lane_count in (None, 1)
            else f"v{lane_count}{scalar.suffix}"
        )
        pattern = (
            Scalar(scalar.source_type)
            if lane_count is None
            else Vector(scalar.source_type, lanes=lane_count)
        )
        for predicate, comparison, combination in (
            ("ord", "f_ord_equal", "logical_and"),
            ("uno", "f_unord_not_equal", "logical_or"),
        ):
            compare = logical_core_descriptor(f"spirv.op_{comparison}.{suffix}")
            combine = logical_core_descriptor(
                f"spirv.op_{combination}.{boolean_suffix}"
            )
            emits = (
                *(
                    emit_descriptor_op(
                        descriptor=compare,
                        operands={
                            "lhs": ValueRef.operand(name),
                            "rhs": ValueRef.operand(name),
                        },
                        results={"dst": ValueRef.temporary(name + "_classified")},
                        result_types={"dst": DescriptorResultType()},
                    )
                    for name in ("lhs", "rhs")
                ),
                emit_descriptor_op(
                    descriptor=combine,
                    operands={
                        "lhs": ValueRef.temporary("lhs_classified"),
                        "rhs": ValueRef.temporary("rhs_classified"),
                    },
                    results={"dst": ValueRef.result("result")},
                ),
            )
            rules.append(
                DescriptorRule(
                    source_op=source_op,
                    descriptor=combine,
                    guards=(
                        Guard.value_type("lhs", pattern),
                        Guard.value_type("rhs", pattern),
                        Guard.value_type("result", result_pattern),
                        Guard.enum_attr_equals("predicate", predicate),
                        *descriptor_feature_guards(compare, combine),
                    ),
                    emit=emits,
                )
            )
    return tuple(rules)
