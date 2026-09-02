# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V logical atomic source-to-low contract rules."""

from loom.dialect.view import defs as view
from loom.target.arch.spirv.atomic import (
    ATOMIC_FLOAT_OPERATIONS,
    ATOMIC_FLOAT_SCALARS,
    ATOMIC_INTEGER_OPERATIONS,
    ATOMIC_INTEGER_SCALARS,
    ATOMIC_SCOPES,
    ATOMIC_STORAGE_CLASSES,
    AtomicFloatOperation,
    AtomicFloatScalar,
    AtomicIntegerOperation,
    AtomicIntegerScalar,
    AtomicOrdering,
    AtomicScope,
    AtomicStorageClass,
    atomic_descriptor_key,
    cmpxchg_failure_orderings,
    float_atomic_descriptor_key,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.contracts.memory import (
    source_memory_address_feature_guards,
    storage_buffer_address_materializer,
    storage_buffer_source_memory,
    workgroup_address_materializer,
    workgroup_carrier_guard,
    workgroup_source_memory,
)
from loom.target.arch.spirv.scalar_memory import (
    StorageBufferScalar,
    storage_buffer_scalar_by_suffix,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorRule,
    Guard,
    Scalar,
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
    SourceMemoryOperation,
    ValueRef,
    View,
)


def _atomic_carrier_guards(
    storage_class: AtomicStorageClass,
    scalar_suffix: str,
) -> tuple[Guard, ...]:
    return (
        (workgroup_carrier_guard(scalar_suffix),)
        if storage_class.suffix == "workgroup"
        else ()
    )


def _atomic_memory_contract(
    operation: SourceMemoryOperation,
    source_scalar: StorageBufferScalar,
    address_scalar: StorageBufferScalar,
    storage_class: AtomicStorageClass,
) -> tuple[SourceMemoryConstraint, SourceMemoryAddressMaterializer]:
    if storage_class.suffix == "storage_buffer":
        return (
            storage_buffer_source_memory(operation, source_scalar),
            storage_buffer_address_materializer(address_scalar),
        )
    if storage_class.suffix == "workgroup":
        return (
            workgroup_source_memory(operation, source_scalar),
            workgroup_address_materializer(address_scalar),
        )
    raise ValueError(f"unknown atomic storage class '{storage_class.suffix}'")


def _integer_atomic_rule(
    form: str,
    scalar: AtomicIntegerScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
    ordering: AtomicOrdering,
    operation: AtomicIntegerOperation,
) -> DescriptorRule:
    descriptor = logical_core_descriptor(
        atomic_descriptor_key(
            form,
            scalar,
            storage_class,
            scope,
            operation=operation,
        )
    )
    storage_scalar = storage_buffer_scalar_by_suffix(scalar.suffix)
    if storage_scalar is None:
        raise ValueError(f"missing SPIR-V storage scalar '{scalar.suffix}'")
    source_memory_operation = (
        SourceMemoryOperation.ATOMIC_REDUCE
        if form == "reduce"
        else SourceMemoryOperation.ATOMIC_RMW
    )
    source_memory, address_materializer = _atomic_memory_contract(
        source_memory_operation,
        storage_scalar,
        storage_scalar,
        storage_class,
    )
    scalar_type = Scalar(scalar.source_type)
    results = {"old": ValueRef.result("result")} if form == "rmw" else {}
    return DescriptorRule(
        source_op=(
            view.view_atomic_reduce if form == "reduce" else view.view_atomic_rmw
        ),
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("kind", operation.source_kind),
            Guard.enum_attr_equals("ordering", ordering.source_keyword),
            Guard.enum_attr_equals("scope", scope.source_keyword),
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type("value", scalar_type),
            *((Guard.value_type("result", scalar_type),) if form == "rmw" else ()),
            *_atomic_carrier_guards(storage_class, storage_scalar.suffix),
            *source_memory_address_feature_guards(address_materializer),
            *descriptor_feature_guards(descriptor),
        ),
        emit=(
            emit_descriptor_op(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "value": ValueRef.operand("value"),
                },
                results=results,
                immediates={
                    "ordering": AttrProject.enum_ordinal("ordering"),
                },
                source_memory=source_memory,
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _integer_atomic_cmpxchg_rule(
    scalar: AtomicIntegerScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
    success_ordering: AtomicOrdering,
    failure_ordering: AtomicOrdering,
) -> DescriptorRule:
    descriptor = logical_core_descriptor(
        atomic_descriptor_key(
            "cmpxchg",
            scalar,
            storage_class,
            scope,
            success_ordering=success_ordering,
        )
    )
    storage_scalar = storage_buffer_scalar_by_suffix(scalar.suffix)
    if storage_scalar is None:
        raise ValueError(f"missing SPIR-V storage scalar '{scalar.suffix}'")
    source_memory, address_materializer = _atomic_memory_contract(
        SourceMemoryOperation.ATOMIC_CMPXCHG,
        storage_scalar,
        storage_scalar,
        storage_class,
    )
    scalar_type = Scalar(scalar.source_type)
    return DescriptorRule(
        source_op=view.view_atomic_cmpxchg,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("success_ordering", success_ordering.source_keyword),
            Guard.enum_attr_equals("failure_ordering", failure_ordering.source_keyword),
            Guard.enum_attr_equals("scope", scope.source_keyword),
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type("expected", scalar_type),
            Guard.value_type("replacement", scalar_type),
            Guard.value_type("old", scalar_type),
            *_atomic_carrier_guards(storage_class, storage_scalar.suffix),
            *source_memory_address_feature_guards(address_materializer),
            *descriptor_feature_guards(descriptor),
        ),
        emit=(
            emit_descriptor_op(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "replacement": ValueRef.operand("replacement"),
                    "expected": ValueRef.operand("expected"),
                },
                results={"old": ValueRef.result("old")},
                immediates={
                    "failure_ordering": AttrProject.enum_ordinal("failure_ordering"),
                },
                source_memory=source_memory,
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _integer_atomic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scalar in ATOMIC_INTEGER_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for ordering in scope.orderings:
                    for operation in ATOMIC_INTEGER_OPERATIONS:
                        if operation.supports_reduce:
                            rules.append(
                                _integer_atomic_rule(
                                    "reduce",
                                    scalar,
                                    storage_class,
                                    scope,
                                    ordering,
                                    operation,
                                )
                            )
                        rules.append(
                            _integer_atomic_rule(
                                "rmw",
                                scalar,
                                storage_class,
                                scope,
                                ordering,
                                operation,
                            )
                        )
                for success_ordering in scope.orderings:
                    rules.extend(
                        _integer_atomic_cmpxchg_rule(
                            scalar,
                            storage_class,
                            scope,
                            success_ordering,
                            failure_ordering,
                        )
                        for failure_ordering in cmpxchg_failure_orderings(
                            success_ordering
                        )
                    )
    return tuple(rules)


def _float_atomic_rule(
    form: str,
    strategy: str,
    scalar: AtomicFloatScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
    ordering: AtomicOrdering,
    operation: AtomicFloatOperation,
) -> DescriptorRule:
    descriptor = logical_core_descriptor(
        float_atomic_descriptor_key(
            form,
            strategy,
            scalar,
            storage_class,
            scope,
            operation=operation,
        )
    )
    storage_scalar = storage_buffer_scalar_by_suffix(scalar.suffix)
    if storage_scalar is None:
        raise ValueError(f"missing SPIR-V storage scalar '{scalar.suffix}'")
    address_scalar = storage_scalar
    if strategy != "native":
        if scalar.integer_suffix is None:
            raise ValueError(f"{scalar.source_type} has no integer atomic carrier")
        address_scalar = storage_buffer_scalar_by_suffix(scalar.integer_suffix)
        if address_scalar is None:
            raise ValueError(f"missing SPIR-V storage scalar '{scalar.integer_suffix}'")
    source_memory_operation = (
        SourceMemoryOperation.ATOMIC_REDUCE
        if form == "reduce"
        else SourceMemoryOperation.ATOMIC_RMW
    )
    source_memory, address_materializer = _atomic_memory_contract(
        source_memory_operation,
        storage_scalar,
        address_scalar,
        storage_class,
    )
    scalar_type = Scalar(scalar.source_type)
    results = {"old": ValueRef.result("result")} if form == "rmw" else {}
    return DescriptorRule(
        source_op=(
            view.view_atomic_reduce if form == "reduce" else view.view_atomic_rmw
        ),
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("kind", operation.source_kind),
            Guard.enum_attr_equals("ordering", ordering.source_keyword),
            Guard.enum_attr_equals("scope", scope.source_keyword),
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type("value", scalar_type),
            *((Guard.value_type("result", scalar_type),) if form == "rmw" else ()),
            *_atomic_carrier_guards(storage_class, address_scalar.suffix),
            *source_memory_address_feature_guards(address_materializer),
            *descriptor_feature_guards(descriptor),
        ),
        emit=(
            emit_descriptor_op(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "value": ValueRef.operand("value"),
                },
                results=results,
                immediates={
                    "ordering": AttrProject.enum_ordinal("ordering"),
                },
                source_memory=source_memory,
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _float_atomic_cmpxchg_rule(
    scalar: AtomicFloatScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
    success_ordering: AtomicOrdering,
    failure_ordering: AtomicOrdering,
) -> DescriptorRule:
    descriptor = logical_core_descriptor(
        float_atomic_descriptor_key(
            "cmpxchg",
            "bitcast",
            scalar,
            storage_class,
            scope,
            success_ordering=success_ordering,
        )
    )
    storage_scalar = storage_buffer_scalar_by_suffix(scalar.suffix)
    if storage_scalar is None:
        raise ValueError(f"missing SPIR-V storage scalar '{scalar.suffix}'")
    if scalar.integer_suffix is None:
        raise ValueError(f"{scalar.source_type} has no integer atomic carrier")
    address_scalar = storage_buffer_scalar_by_suffix(scalar.integer_suffix)
    if address_scalar is None:
        raise ValueError(f"missing SPIR-V storage scalar '{scalar.integer_suffix}'")
    source_memory, address_materializer = _atomic_memory_contract(
        SourceMemoryOperation.ATOMIC_CMPXCHG,
        storage_scalar,
        address_scalar,
        storage_class,
    )
    scalar_type = Scalar(scalar.source_type)
    return DescriptorRule(
        source_op=view.view_atomic_cmpxchg,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("success_ordering", success_ordering.source_keyword),
            Guard.enum_attr_equals("failure_ordering", failure_ordering.source_keyword),
            Guard.enum_attr_equals("scope", scope.source_keyword),
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type("expected", scalar_type),
            Guard.value_type("replacement", scalar_type),
            Guard.value_type("old", scalar_type),
            *_atomic_carrier_guards(storage_class, address_scalar.suffix),
            *source_memory_address_feature_guards(address_materializer),
            *descriptor_feature_guards(descriptor),
        ),
        emit=(
            emit_descriptor_op(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "replacement": ValueRef.operand("replacement"),
                    "expected": ValueRef.operand("expected"),
                },
                results={"old": ValueRef.result("old")},
                immediates={
                    "failure_ordering": AttrProject.enum_ordinal("failure_ordering"),
                },
                source_memory=source_memory,
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _float_atomic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scalar in ATOMIC_FLOAT_SCALARS:
        for storage_class in ATOMIC_STORAGE_CLASSES:
            for scope in ATOMIC_SCOPES:
                for ordering in scope.orderings:
                    for operation in ATOMIC_FLOAT_OPERATIONS:
                        if operation.native_opcode is not None:
                            if operation.supports_reduce:
                                rules.append(
                                    _float_atomic_rule(
                                        "reduce",
                                        "native",
                                        scalar,
                                        storage_class,
                                        scope,
                                        ordering,
                                        operation,
                                    )
                                )
                            rules.append(
                                _float_atomic_rule(
                                    "rmw",
                                    "native",
                                    scalar,
                                    storage_class,
                                    scope,
                                    ordering,
                                    operation,
                                )
                            )
                        if scalar.integer_scalar_enum is None:
                            continue
                        if operation.source_kind == "xchgf":
                            rules.append(
                                _float_atomic_rule(
                                    "rmw",
                                    "bitcast",
                                    scalar,
                                    storage_class,
                                    scope,
                                    ordering,
                                    operation,
                                )
                            )
                            continue
                        if operation.supports_reduce:
                            rules.append(
                                _float_atomic_rule(
                                    "reduce",
                                    "cas",
                                    scalar,
                                    storage_class,
                                    scope,
                                    ordering,
                                    operation,
                                )
                            )
                        rules.append(
                            _float_atomic_rule(
                                "rmw",
                                "cas",
                                scalar,
                                storage_class,
                                scope,
                                ordering,
                                operation,
                            )
                        )
                if scalar.integer_scalar_enum is None:
                    continue
                for success_ordering in scope.orderings:
                    rules.extend(
                        _float_atomic_cmpxchg_rule(
                            scalar,
                            storage_class,
                            scope,
                            success_ordering,
                            failure_ordering,
                        )
                        for failure_ordering in cmpxchg_failure_orderings(
                            success_ordering
                        )
                    )
    return tuple(rules)


SPIRV_ATOMIC_CONTRACT_CASES = (
    *_integer_atomic_rules(),
    *_float_atomic_rules(),
)
