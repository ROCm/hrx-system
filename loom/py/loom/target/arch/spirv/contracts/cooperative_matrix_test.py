# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.cooperative_matrix import (
    COOPERATIVE_INITIALIZER_MATERIALIZER,
    COOPERATIVE_MATRIX_INITIALIZER_RULES,
)
from loom.target.arch.spirv.cooperative_matrix import COOPERATIVE_MATRIX_TYPES
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import DescriptorRule, GuardKind, ValueAliasRule, Vector


def test_constructor_types_cover_components_without_duplicate_accumulators() -> None:
    actual = {
        (value.scalar.suffix, value.rows, value.columns, value.role)
        for value in COOPERATIVE_MATRIX_TYPES
    }
    expected = {
        (scalar, rows, columns, role)
        for scalar, reduction in (
            ("f16", 16),
            ("bf16", 16),
            ("i8", 16),
            ("u8", 16),
            ("i8", 32),
            ("u8", 32),
        )
        for rows, columns, role in ((16, reduction, "lhs"), (reduction, 16, "rhs"))
    } | {(scalar, 16, 16, "accumulator") for scalar in ("f16", "f32", "i32", "u32")}
    assert actual == expected
    assert len(COOPERATIVE_MATRIX_TYPES) == len(expected)
    storage_bits = feature_bits_value(
        ("storage_buffer_8bit_access", "storage_buffer_16bit_access")
    )
    for value in COOPERATIVE_MATRIX_TYPES:
        assert not value.feature_bits & storage_bits
        if value.scalar.suffix == "f32":
            assert value.feature_bits == feature_bits_value(("cooperative_matrix_khr",))


def test_initializers_cover_constants_origins_schemas_and_both_accumulator_roles() -> (
    None
):
    signatures = set()
    native_signatures = set()
    for rule in COOPERATIVE_MATRIX_INITIALIZER_RULES:
        assert rule.source_op is vector.vector_fragment
        if isinstance(rule, ValueAliasRule):
            continue
        assert isinstance(rule, DescriptorRule)
        role_guard = next(guard for guard in rule.guards if guard.field == "role")
        roles = (
            role_guard.enum_keywords
            if role_guard.kind == GuardKind.ENUM_ATTR_IN
            else (role_guard.enum_keyword,)
        )
        payload = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE
        )
        assert payload == Vector(
            payload.elements, minimum_lanes=1, maximum_lanes="LOOM_DIM_MAX_STATIC_SIZE"
        )
        schema = any(
            guard.kind == GuardKind.VALUE_STORAGE_ELEMENT_FORMAT
            for guard in rule.guards
        )
        constant = any(
            guard.kind in (GuardKind.VALUE_EXACT_FLOAT, GuardKind.VALUE_EXACT_I64)
            for guard in rule.guards
        )
        uniformity = [
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_MATERIALIZABLE
        ]
        assert len(uniformity) == 1
        assert uniformity[0].materializer == COOPERATIVE_INITIALIZER_MATERIALIZER.name
        native_signature = (rule.descriptor.key, schema, constant)
        assert native_signature not in native_signatures
        native_signatures.add(native_signature)
        signatures.update(
            (rule.descriptor.key, role, schema, constant) for role in roles
        )
        constructor = rule.emit[-1]
        assert constructor.descriptor is rule.descriptor
        assert tuple(constructor.operands) == ("component",)
        if ".cooperative.u" in rule.descriptor.key:
            assert schema
            assert rule.emit[-2].descriptor.key.startswith("spirv.op_bitcast.")
    expected = {
        (value.constructor_key, role, schema, constant)
        for value in COOPERATIVE_MATRIX_TYPES
        for role in value.source_roles
        for schema in (
            (True,) if value.scalar.suffix.startswith("u") else (False, True)
        )
        for constant in (True, False)
    }
    assert signatures == expected
    assert len(signatures) == 68
