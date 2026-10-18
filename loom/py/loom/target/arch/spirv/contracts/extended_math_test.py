# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.extended_math import (
    SPIRV_EXTENDED_MATH_CONTRACT_CASES,
)
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.extended_math import EXTENDED_MATH_OPERATIONS
from loom.target.contracts import GuardKind, TypePattern, ValueRef

_OPERATIONS_BY_SOURCE_KEY = {
    operation.source_op_key: operation for operation in EXTENDED_MATH_OPERATIONS
}


def _expected_rule_matrix() -> set[tuple[str, TypePattern, str]]:
    expected: set[tuple[str, TypePattern, str]] = set()
    for operation in EXTENDED_MATH_OPERATIONS:
        for source_type in operation.source_types:
            expected.add(
                (
                    f"scalar.{operation.source_op_key}",
                    TypePattern.scalar(source_type),
                    "spirv.op_ext_inst.glsl_std_450."
                    f"{operation.source_op_key}.{source_type}",
                )
            )
            for lane_count in (1, 2, 3, 4):
                suffix = (
                    source_type if lane_count == 1 else f"v{lane_count}{source_type}"
                )
                expected.add(
                    (
                        f"vector.{operation.source_op_key}",
                        TypePattern.vector(source_type, lanes=lane_count),
                        "spirv.op_ext_inst.glsl_std_450."
                        f"{operation.source_op_key}.{suffix}",
                    )
                )
    return expected


def test_extended_math_contract_covers_scalar_and_native_vector_shapes() -> None:
    actual: set[tuple[str, TypePattern, str]] = set()
    for rule in SPIRV_EXTENDED_MATH_CONTRACT_CASES:
        type_guards = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE
        }
        operation_key = rule.source_op.name.split(".", 1)[1]
        operation = _OPERATIONS_BY_SOURCE_KEY[operation_key]
        operand_names = operation.operand_names
        assert type_guards.keys() == {*operand_names, "result"}
        assert all(
            type_guards[operand_name] == type_guards["result"]
            for operand_name in operand_names
        )
        actual.add((rule.source_op.name, type_guards["result"], rule.descriptor.key))

        flag_guards = [
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
        ]
        flag_field = "math" if operation_key in ("sinf", "cosf") else "fastmath"
        assert all(guard.field == flag_field for guard in flag_guards)
        assert tuple(guard.enum_keyword for guard in flag_guards) == (
            operation.required_fastmath_flags
        )

        assert len(rule.emit) == 1
        emit = rule.emit[0]
        assert emit.descriptor is rule.descriptor
        assert emit.operands == {
            operand_name: ValueRef.operand(operand_name)
            for operand_name in operand_names
        }
        assert emit.results == {"dst": ValueRef.result("result")}

    assert actual == _expected_rule_matrix()


def test_extended_math_contract_is_bound_into_the_shipping_fragment() -> None:
    fragment_case_ids = {
        id(contract_case)
        for contract_case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
    }
    assert all(
        id(contract_case) in fragment_case_ids
        for contract_case in SPIRV_EXTENDED_MATH_CONTRACT_CASES
    )
