# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.scalar import comparison as scalar_comparison
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.scalar_alu import (
    FLOAT_COMPARE_PREDICATES,
    FLOAT_COMPARE_TYPES,
)
from loom.target.contracts import DescriptorRule, GuardKind, TypePattern, ValueRef

_EXPECTED_FLOAT_PREDICATES = (
    ("oeq", "f_ord_equal", "OpFOrdEqual", "LOOM_SPIRV_OP_F_ORD_EQUAL"),
    (
        "ogt",
        "f_ord_greater_than",
        "OpFOrdGreaterThan",
        "LOOM_SPIRV_OP_F_ORD_GREATER_THAN",
    ),
    (
        "oge",
        "f_ord_greater_than_equal",
        "OpFOrdGreaterThanEqual",
        "LOOM_SPIRV_OP_F_ORD_GREATER_THAN_EQUAL",
    ),
    (
        "olt",
        "f_ord_less_than",
        "OpFOrdLessThan",
        "LOOM_SPIRV_OP_F_ORD_LESS_THAN",
    ),
    (
        "ole",
        "f_ord_less_than_equal",
        "OpFOrdLessThanEqual",
        "LOOM_SPIRV_OP_F_ORD_LESS_THAN_EQUAL",
    ),
    (
        "one",
        "f_ord_not_equal",
        "OpFOrdNotEqual",
        "LOOM_SPIRV_OP_F_ORD_NOT_EQUAL",
    ),
    ("ord", "ordered", "OpOrdered", "LOOM_SPIRV_OP_ORDERED"),
    ("ueq", "f_unord_equal", "OpFUnordEqual", "LOOM_SPIRV_OP_F_UNORD_EQUAL"),
    (
        "ugt",
        "f_unord_greater_than",
        "OpFUnordGreaterThan",
        "LOOM_SPIRV_OP_F_UNORD_GREATER_THAN",
    ),
    (
        "uge",
        "f_unord_greater_than_equal",
        "OpFUnordGreaterThanEqual",
        "LOOM_SPIRV_OP_F_UNORD_GREATER_THAN_EQUAL",
    ),
    (
        "ult",
        "f_unord_less_than",
        "OpFUnordLessThan",
        "LOOM_SPIRV_OP_F_UNORD_LESS_THAN",
    ),
    (
        "ule",
        "f_unord_less_than_equal",
        "OpFUnordLessThanEqual",
        "LOOM_SPIRV_OP_F_UNORD_LESS_THAN_EQUAL",
    ),
    (
        "une",
        "f_unord_not_equal",
        "OpFUnordNotEqual",
        "LOOM_SPIRV_OP_F_UNORD_NOT_EQUAL",
    ),
    ("uno", "unordered", "OpUnordered", "LOOM_SPIRV_OP_UNORDERED"),
)


def test_float_predicates_match_the_complete_spirv_core_family() -> None:
    assert (
        tuple(
            (
                predicate.source_predicate,
                predicate.descriptor_suffix,
                predicate.mnemonic,
                predicate.opcode,
            )
            for predicate in FLOAT_COMPARE_PREDICATES
        )
        == _EXPECTED_FLOAT_PREDICATES
    )


def test_float_comparisons_require_only_operand_type_features() -> None:
    expected_type_features = {
        "f16": ("float16",),
        "f32": (),
        "f64": ("float64",),
    }
    assert {
        scalar.source_type: scalar.feature_atoms for scalar in FLOAT_COMPARE_TYPES
    } == expected_type_features


def test_scalar_float_comparison_contract_covers_every_type_and_predicate() -> None:
    rules = tuple(
        case
        for case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is scalar_comparison.scalar_cmpf
    )
    actual = {}
    for rule in rules:
        type_guards = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE
        }
        assert type_guards.keys() == {"lhs", "rhs", "result"}
        assert type_guards["lhs"] == type_guards["rhs"]
        assert type_guards["result"] == TypePattern.scalar("i1")
        predicate_guard = next(
            guard for guard in rule.guards if guard.kind == GuardKind.ENUM_ATTR_EQUALS
        )
        key = (type_guards["lhs"].element, predicate_guard.enum_keyword)
        assert key not in actual
        actual[key] = rule

        availability_guards = tuple(
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
        )
        assert bool(availability_guards) == bool(rule.descriptor.feature_mask_words)
        assert len(rule.emit) == 1
        assert rule.emit[0].descriptor is rule.descriptor
        assert rule.emit[0].operands == {
            "lhs": ValueRef.operand("lhs"),
            "rhs": ValueRef.operand("rhs"),
        }
        assert rule.emit[0].results == {"dst": ValueRef.result("result")}

    expected = {
        (scalar.source_type, predicate.source_predicate): (
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}"
        )
        for scalar in FLOAT_COMPARE_TYPES
        for predicate in FLOAT_COMPARE_PREDICATES
    }
    assert actual.keys() == expected.keys()
    for key, descriptor_key in expected.items():
        assert actual[key].descriptor.key == descriptor_key
