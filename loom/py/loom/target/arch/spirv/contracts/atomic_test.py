# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.dialect.view import defs as view
from loom.target.arch.spirv.atomic import (
    ATOMIC_FLOAT_OPERATIONS,
    ATOMIC_FLOAT_SCALARS,
    ATOMIC_INTEGER_OPERATIONS,
    ATOMIC_INTEGER_SCALARS,
    ATOMIC_SCOPES,
    ATOMIC_STORAGE_CLASSES,
    atomic_descriptor_key,
    cmpxchg_failure_orderings,
    float_atomic_descriptor_key,
)
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import DescriptorRule, GuardKind


def _atomic_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        contract_case
        for contract_case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(contract_case, DescriptorRule)
        and contract_case.descriptor.key.startswith("spirv.atomic.")
    )


def _rule_signature(rule: DescriptorRule) -> tuple[str, ...]:
    enum_guards = {
        guard.field: guard.enum_keyword
        for guard in rule.guards
        if guard.kind == GuardKind.ENUM_ATTR_EQUALS
    }
    return (
        rule.source_op.name,
        rule.descriptor.key,
        enum_guards.get("kind", ""),
        enum_guards.get("ordering", ""),
        enum_guards.get("success_ordering", ""),
        enum_guards.get("failure_ordering", ""),
        enum_guards.get("scope", ""),
    )


def _expected_rule_signatures() -> Counter[tuple[str, ...]]:
    expected = Counter[tuple[str, ...]]()
    for scalar in ATOMIC_INTEGER_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for ordering in scope.orderings:
                    for operation in ATOMIC_INTEGER_OPERATIONS:
                        forms = (
                            ("rmw", "reduce") if operation.supports_reduce else ("rmw",)
                        )
                        for form in forms:
                            source_op = (
                                view.view_atomic_rmw
                                if form == "rmw"
                                else view.view_atomic_reduce
                            )
                            key = atomic_descriptor_key(
                                form,
                                scalar,
                                storage_class,
                                scope,
                                operation=operation,
                            )
                            expected[
                                (
                                    source_op.name,
                                    key,
                                    operation.source_kind,
                                    ordering.source_keyword,
                                    "",
                                    "",
                                    scope.source_keyword,
                                )
                            ] += 1
                for success_ordering in scope.orderings:
                    key = atomic_descriptor_key(
                        "cmpxchg",
                        scalar,
                        storage_class,
                        scope,
                        success_ordering=success_ordering,
                    )
                    for failure_ordering in cmpxchg_failure_orderings(success_ordering):
                        expected[
                            (
                                view.view_atomic_cmpxchg.name,
                                key,
                                "",
                                "",
                                success_ordering.source_keyword,
                                failure_ordering.source_keyword,
                                scope.source_keyword,
                            )
                        ] += 1

    for scalar in ATOMIC_FLOAT_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for ordering in scope.orderings:
                    for operation in ATOMIC_FLOAT_OPERATIONS:
                        strategies = []
                        if operation.native_opcode is not None:
                            strategies.append("native")
                        if scalar.integer_scalar_enum is not None:
                            strategies.append(
                                "bitcast" if operation.source_kind == "xchgf" else "cas"
                            )
                        forms = (
                            ("rmw", "reduce") if operation.supports_reduce else ("rmw",)
                        )
                        for strategy in strategies:
                            for form in forms:
                                source_op = (
                                    view.view_atomic_rmw
                                    if form == "rmw"
                                    else view.view_atomic_reduce
                                )
                                key = float_atomic_descriptor_key(
                                    form,
                                    strategy,
                                    scalar,
                                    storage_class,
                                    scope,
                                    operation=operation,
                                )
                                expected[
                                    (
                                        source_op.name,
                                        key,
                                        operation.source_kind,
                                        ordering.source_keyword,
                                        "",
                                        "",
                                        scope.source_keyword,
                                    )
                                ] += 1
                if scalar.integer_scalar_enum is None:
                    continue
                for success_ordering in scope.orderings:
                    key = float_atomic_descriptor_key(
                        "cmpxchg",
                        "bitcast",
                        scalar,
                        storage_class,
                        scope,
                        success_ordering=success_ordering,
                    )
                    for failure_ordering in cmpxchg_failure_orderings(success_ordering):
                        expected[
                            (
                                view.view_atomic_cmpxchg.name,
                                key,
                                "",
                                "",
                                success_ordering.source_keyword,
                                failure_ordering.source_keyword,
                                scope.source_keyword,
                            )
                        ] += 1
    return expected


def test_atomic_contract_rule_matrix_is_complete() -> None:
    actual = Counter(_rule_signature(rule) for rule in _atomic_rules())
    expected = _expected_rule_signatures()

    assert len(actual) == 1946
    assert actual == expected


def test_native_float_rules_precede_integer_fallbacks() -> None:
    rules = _atomic_rules()
    rule_indices = {_rule_signature(rule): index for index, rule in enumerate(rules)}
    for scalar in ATOMIC_FLOAT_SCALARS:
        if scalar.integer_scalar_enum is None:
            continue
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for ordering in scope.orderings:
                    for operation in ATOMIC_FLOAT_OPERATIONS:
                        if operation.native_opcode is None:
                            continue
                        fallback_strategy = (
                            "bitcast" if operation.source_kind == "xchgf" else "cas"
                        )
                        forms = (
                            ("rmw", "reduce") if operation.supports_reduce else ("rmw",)
                        )
                        for form in forms:
                            source_op = (
                                view.view_atomic_rmw
                                if form == "rmw"
                                else view.view_atomic_reduce
                            )
                            common = (
                                source_op.name,
                                operation.source_kind,
                                ordering.source_keyword,
                                "",
                                "",
                                scope.source_keyword,
                            )
                            native_key = float_atomic_descriptor_key(
                                form,
                                "native",
                                scalar,
                                storage_class,
                                scope,
                                operation=operation,
                            )
                            fallback_key = float_atomic_descriptor_key(
                                form,
                                fallback_strategy,
                                scalar,
                                storage_class,
                                scope,
                                operation=operation,
                            )
                            native_signature = (
                                common[0],
                                native_key,
                                *common[1:],
                            )
                            fallback_signature = (
                                common[0],
                                fallback_key,
                                *common[1:],
                            )
                            assert (
                                rule_indices[native_signature]
                                < rule_indices[fallback_signature]
                            )


def test_workgroup_rules_require_the_allocation_carrier_class() -> None:
    for rule in _atomic_rules():
        is_workgroup = ".workgroup." in rule.descriptor.key
        carrier_guards = tuple(
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.LOW_VALUE_REGISTER_CLASS
        )
        if not is_workgroup:
            assert carrier_guards == ()
            continue
        assert len(carrier_guards) == 1
        assert carrier_guards[0].field == "view"
        assert carrier_guards[0].register_class is not None
        assert carrier_guards[0].register_class.startswith("spirv.ptr.workgroup.array.")
