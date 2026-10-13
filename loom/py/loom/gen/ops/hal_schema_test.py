# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declaration admission and the HAL source/native structural correspondence."""

from dataclasses import replace

import pytest

from loom.dialect.hal.methods import (
    QUEUE_COPY,
    Field,
    Optional,
    Range,
    Record,
    Value,
)
from loom.gen.ops import hal_schema
from loom.ir import ScalarTypeKind, scalar_type_name


def test_queue_copy_layout_preserves_nested_optional_range() -> None:
    layouts, roots, depths, projection = hal_schema.native_model()
    ordered, ordinals = hal_schema.nodes()
    root = layouts[roots[0]]
    assert len(root) == 11
    assert projection[ordinals[QUEUE_COPY.arguments]][0] == 15
    assert root[1] == root[2]  # Independent waits/signals with the same element.
    assert root[8] == root[9]  # Independent before/after boundaries.
    assert root[8][0] == "IREE_VM_NATIVE_CALL_FIELD_RANGE"
    optional_element = layouts[root[8][1]]
    assert len(optional_element) == 1
    assert optional_element[0][0] == "IREE_VM_NATIVE_CALL_FIELD_RANGE"
    assert depths[roots[0]] == 5  # Boundary/list/buffer/recipe/operations.
    for shape in ordered:
        width, child = projection[ordinals[shape]]
        if isinstance(shape, (Range, Optional)):
            assert width == 2
            assert child < len(layouts)
        elif isinstance(shape, Value):
            assert width == 1
        else:
            assert width == sum(projection[ordinals[field.value]][0] for field in shape.fields)


@pytest.mark.parametrize("kind", [kind for kind in ScalarTypeKind if kind.name not in ("INDEX", "OFFSET")])
def test_scalar_family_uses_vm_carriers(kind: ScalarTypeKind) -> None:
    name = scalar_type_name(kind)
    arguments = Record("scalars", (Field("value", Value(name)),))
    layouts, roots, _, _ = hal_schema.native_model((replace(QUEUE_COPY, arguments=arguments),))
    carrier = {"i1": "I32", "f8E4M3": "F8E4M3FN"}.get(name, name.upper())
    assert layouts[roots[0]] == ((f"IREE_VM_SCALAR_TYPE_{carrier}", 0),)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (Record("bad", (Field("x", Value("hal.missing")),)), "unsupported HAL value"),
        (Record("bad", (Field("x", Value("index")),)), "unsupported HAL value"),
        (Record("bad", (Field("x", Value("i32")), Field("x", Value("i64")))), "duplicate field"),
        (Record("bad-name", ()), "invalid HAL identifier"),
        (Record("bad", (Field("X", Value("i32")),)), "invalid HAL identifier"),
        (Record("bad", (Field("x", Optional(Optional(Value("i32")))),)), "ambiguous source presence"),
        (Record("bad", (Field("x", Record("bad", ())),)), "conflicting HAL record"),
    ],
)
def test_rejects_ambiguous_or_unrepresentable_declarations(arguments: Record, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        hal_schema.native_model((replace(QUEUE_COPY, arguments=arguments),))


@pytest.mark.parametrize("methods", [(), (QUEUE_COPY, QUEUE_COPY), (replace(QUEUE_COPY, version=-1),), (replace(QUEUE_COPY, version=1 << 32),)])
def test_rejects_invalid_directory(methods) -> None:
    with pytest.raises(ValueError, match="HAL method"):
        hal_schema.nodes(methods)


def test_generated_runtime_schema_is_a_checked_in_artifact() -> None:
    header = hal_schema.native_header()
    assert "IREE_HAL_NATIVE_QUEUE_COPY_VERSION = 1" in header
    assert "IREE_HAL_NATIVE_QUEUE_COPY_BARRIERS_AFTER = 12" in header
    assert "IREE_HAL_NATIVE_REF_QUEUE = 3" in header
    assert "loom/" not in header
