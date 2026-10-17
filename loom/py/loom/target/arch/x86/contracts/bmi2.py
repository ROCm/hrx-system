# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional BMI2 scalar variable-shift contract fragment."""

from __future__ import annotations

from collections.abc import Mapping

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dsl import Op
from loom.target.arch.x86.descriptors import X86_SCALAR_DESCRIPTOR_SET
from loom.target.arch.x86.scalar_families import X86_SCALAR_VARIABLE_SHIFT_FAMILIES
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    descriptor_by_key,
)

_I32 = Scalar("i32")
_I64 = Scalar("i64")
_INDEX = Scalar("index")

_SCALAR_OPERATIONS: Mapping[str, Op] = {
    "shli": scalar_bitwise.scalar_shli,
    "shrsi": scalar_bitwise.scalar_shrsi,
    "shrui": scalar_bitwise.scalar_shrui,
}
_INDEX_OPERATIONS: Mapping[str, Op] = {
    "shli": index.index_shli,
    "shrsi": index.index_shrsi,
    "shrui": index.index_shrui,
}


def _variable_shift_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_operation: str,
    bit_count: int,
) -> DescriptorRule:
    descriptor = descriptor_by_key(
        X86_SCALAR_DESCRIPTOR_SET,
        f"x86.bmi2.{descriptor_operation}.gpr{bit_count}",
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *(
                Guard.value_type(field, type_pattern)
                for field in ("lhs", "rhs", "result")
            ),
            Guard.value_not_exact_i64("rhs"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _cases() -> tuple[DescriptorRule, ...]:
    return tuple(
        _variable_shift_rule(
            operations[family.source_operation],
            type_pattern,
            family.descriptor_operation,
            bit_count,
        )
        for type_pattern, bit_count, operations in (
            (_I32, 32, _SCALAR_OPERATIONS),
            (_I64, 64, _SCALAR_OPERATIONS),
            (_INDEX, 64, _INDEX_OPERATIONS),
        )
        for family in X86_SCALAR_VARIABLE_SHIFT_FAMILIES
    )


X86_BMI2_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
}

X86_BMI2_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.bmi2",
    descriptor_set=X86_SCALAR_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/bmi2.h",
    cases=_cases(),
    # The scalar fragment owns legality for these source operations. BMI2 is
    # an optional realization selected only when its descriptor facts apply.
    target_contract_query=False,
)
