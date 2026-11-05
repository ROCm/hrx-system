# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source coordinates retained by structural register-slice recipes."""

from dataclasses import replace

import pytest

from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorRule,
    EmitRegisterSlice,
    Guard,
    LowerAttrCopyKind,
    SourceMemoryConstraint,
    SourceMemoryOperation,
    ValueRef,
)
from loom.target.contracts.lower_rules import compile_lower_rule_set
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET


@pytest.mark.parametrize(
    ("projection", "kind"),
    [
        (
            AttrProject.i64_array_element("static_offsets", element=0),
            LowerAttrCopyKind.I64_ARRAY_ELEMENT,
        ),
        (
            AttrProject.i64_array_element_plus_literal(
                "static_offsets", element=0, literal=2
            ),
            LowerAttrCopyKind.I64_ARRAY_ELEMENT_PLUS_LITERAL,
        ),
        (
            AttrProject.i64_array_literal_minus_element(
                "static_offsets", element=0, literal=4
            ),
            LowerAttrCopyKind.I64_ARRAY_LITERAL_MINUS_ELEMENT,
        ),
        (
            AttrProject.i64_array_element_quotient(
                "static_offsets", element=0, divisor=2
            ),
            LowerAttrCopyKind.I64_ARRAY_ELEMENT_QUOTIENT,
        ),
        (
            AttrProject.i64_array_element_remainder(
                "static_offsets", element=0, divisor=2
            ),
            LowerAttrCopyKind.I64_ARRAY_ELEMENT_REMAINDER,
        ),
        (
            AttrProject.i64_array_lane_byte_offset(
                "static_offsets", element=0, bytes_per_lane=4, base_byte_offset=8
            ),
            LowerAttrCopyKind.I64_ARRAY_LANE_BYTE,
        ),
    ],
)
def test_slice_retains_coordinate_projection(
    projection: AttrProject, kind: LowerAttrCopyKind
) -> None:
    rule = DescriptorRule(
        source_op=vector.vector_slice,
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=ValueRef.result("result"),
                unit_offset=projection,
            ),
        ),
    )
    fragment = ContractFragment(
        name="test.projection",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=(rule,),
    )
    table = compile_lower_rule_set(fragment, dialect_ops={"vector": ALL_VECTOR_OPS})
    (emit,) = table.emits
    assert emit.attr_copy_count == 1
    assert emit.structural_offset == emit.attr_copy_start
    attr = table.attr_copies[emit.attr_copy_start]
    assert attr.kind is kind
    assert attr.source_attr_index == 0
    assert attr.source_element_index == 0
    assert attr.literal_i64 == projection.literal_i64


@pytest.mark.parametrize(
    "projection",
    [
        AttrProject.direct("offset"),
        AttrProject.i64_literal_minus_attr("offset", literal=32),
        AttrProject.i64_literal_minus_attrs(
            "offset", other_source_attr="width", literal=32
        ),
        AttrProject.i64_attr_minus_literal("offset", literal=8),
    ],
)
def test_scalar_coordinate_projection(projection: AttrProject) -> None:
    projection.validate_register_offset(scalar_bitwise.scalar_bitfield_extracts)


def test_direct_coordinate_requires_integer_attribute() -> None:
    with pytest.raises(ValueError, match="requires an i64 attribute"):
        AttrProject.direct("static_offsets").validate_register_offset(
            vector.vector_slice
        )


@pytest.mark.parametrize(
    ("multiple", "diagnostic"),
    [
        (0, "positive i64 multiple"),
        (-1, "count must be non-negative"),
        (1 << 63, "positive i64 multiple"),
    ],
)
def test_coordinate_alignment_rejects_invalid_multiple(
    multiple: int, diagnostic: str
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        Guard.i64_array_element_multiple("static_offsets", 0, multiple).validate(
            vector.vector_slice
        )


@pytest.mark.parametrize(
    ("element", "diagnostic"),
    [(-1, "element must be non-negative"), (65536, "u16 element")],
)
def test_coordinate_alignment_rejects_invalid_element(
    element: int, diagnostic: str
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        Guard.i64_array_element_multiple("static_offsets", element, 2).validate(
            vector.vector_slice
        )


def test_memory_count_can_be_owned_by_source_shape_guards() -> None:
    memory = SourceMemoryConstraint(
        operation=SourceMemoryOperation.LOAD,
        memory_spaces=("generic",),
        element_byte_count=4,
        vector_lane_count=None,
        vector_lane_byte_stride=4,
        static_byte_offset=0,
    )
    assert memory.vector_lane_count is None
    for count in (0, -1, 1 << 32):
        with pytest.raises(ValueError, match="positive u32"):
            replace(memory, vector_lane_count=count)
