# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

import pytest

from loom.target.arch.spirv.ordinary_vector import (
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_INSTRUCTIONS,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
    OrdinaryVectorType,
)


def test_component_matrix_covers_the_spirv_scalar_type_vocabulary() -> None:
    assert tuple(
        source_type
        for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
        for source_type in component_type.source_types
    ) == (
        "i1",
        "i8",
        "i16",
        "i32",
        "index",
        "i64",
        "offset",
        "f16",
        "bf16",
        "f32",
        "f64",
    )

    by_suffix = {
        component_type.suffix: component_type
        for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
    }
    assert by_suffix["bool"].vector_value_class.endswith("BOOL_VECTOR")
    assert by_suffix["bool"].scalar_value_class.endswith("BOOL")
    assert by_suffix["i32"].source_types == ("i32", "index")
    assert by_suffix["offset64"].scalar_value_class.endswith("OFFSET64")
    assert by_suffix["offset64"].scalar_enum.endswith("U64")
    assert by_suffix["bf16"].feature_atoms == ("bfloat16_type_khr",)


def test_native_type_matrix_is_the_component_lane_cross_product() -> None:
    assert NATIVE_ORDINARY_VECTOR_LANE_COUNTS == (2, 3, 4)
    assert len(ORDINARY_VECTOR_TYPES) == (
        len(ORDINARY_VECTOR_COMPONENT_TYPES) * len(NATIVE_ORDINARY_VECTOR_LANE_COUNTS)
    )
    assert len({vector_type.suffix for vector_type in ORDINARY_VECTOR_TYPES}) == len(
        ORDINARY_VECTOR_TYPES
    )
    assert {vector_type.lane_count for vector_type in ORDINARY_VECTOR_TYPES} == set(
        NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    )


def test_structural_instruction_matrix_is_closed_per_native_type() -> None:
    instructions_by_type = Counter()
    for instruction in ORDINARY_VECTOR_INSTRUCTIONS:
        vector_type = (
            instruction.result_type
            if isinstance(instruction.result_type, OrdinaryVectorType)
            else instruction.operand_types[0]
        )
        assert isinstance(vector_type, OrdinaryVectorType)
        instructions_by_type[vector_type] += 1

    assert len(ORDINARY_VECTOR_INSTRUCTIONS) == 4 * len(ORDINARY_VECTOR_TYPES)
    assert set(instructions_by_type.values()) == {4}
    assert len({row.key for row in ORDINARY_VECTOR_INSTRUCTIONS}) == len(
        ORDINARY_VECTOR_INSTRUCTIONS
    )
    assert {row.packet_form for row in ORDINARY_VECTOR_INSTRUCTIONS} == {
        "LOOM_SPIRV_PACKET_FORM_COMPOSITE_CONSTRUCT",
        "LOOM_SPIRV_PACKET_FORM_COMPOSITE_EXTRACT",
        "LOOM_SPIRV_PACKET_FORM_COMPOSITE_INSERT",
        "LOOM_SPIRV_PACKET_FORM_SELECT",
    }


def test_component_invariants_reject_invalid_records() -> None:
    bool_component = next(
        component
        for component in ORDINARY_VECTOR_COMPONENT_TYPES
        if component.kind == OrdinaryVectorComponentKind.BOOLEAN
    )
    with pytest.raises(ValueError, match="requires a source type"):
        OrdinaryVectorComponentType(
            source_types=(),
            suffix="missing",
            kind=OrdinaryVectorComponentKind.SIGNED_INTEGER,
            scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
            bit_width=32,
        )
    with pytest.raises(ValueError, match="bit width must be positive"):
        OrdinaryVectorComponentType(
            source_types=("i32",),
            suffix="zero_width",
            kind=OrdinaryVectorComponentKind.SIGNED_INTEGER,
            scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
            bit_width=0,
        )
    with pytest.raises(ValueError, match="boolean vector components"):
        OrdinaryVectorComponentType(
            source_types=("i1",),
            suffix="invalid_bool",
            kind=OrdinaryVectorComponentKind.BOOLEAN,
            scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
            bit_width=1,
        )
    with pytest.raises(ValueError, match="numeric vector component"):
        OrdinaryVectorComponentType(
            source_types=("i32",),
            suffix="missing_scalar",
            kind=OrdinaryVectorComponentKind.SIGNED_INTEGER,
            scalar_enum="LOOM_SPIRV_SCALAR_TYPE_UNKNOWN",
            bit_width=32,
        )
    with pytest.raises(ValueError, match="source-ineligible"):
        OrdinaryVectorComponentType(
            source_types=("i32",),
            suffix="u32",
            kind=OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
            scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U32",
            bit_width=32,
        )
    unsigned_view = OrdinaryVectorComponentType(
        source_types=(),
        suffix="u32",
        kind=OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U32",
        bit_width=32,
    )
    assert unsigned_view.source_types == ()
    assert bool_component.bit_width == 1


def test_signed_component_ranges_cover_each_integer_width() -> None:
    integer_components = tuple(
        component
        for component in ORDINARY_VECTOR_COMPONENT_TYPES
        if component.kind == OrdinaryVectorComponentKind.SIGNED_INTEGER
    )
    assert {
        component.suffix: (component.signed_minimum, component.signed_maximum)
        for component in integer_components
    } == {
        "i8": (-128, 127),
        "i16": (-32768, 32767),
        "i32": (-(2**31), (2**31) - 1),
        "i64": (-(2**63), (2**63) - 1),
    }

    f32_component = next(
        component
        for component in ORDINARY_VECTOR_COMPONENT_TYPES
        if component.suffix == "f32"
    )
    with pytest.raises(ValueError, match="not a signed integer component"):
        _ = f32_component.signed_minimum
    with pytest.raises(ValueError, match="not a signed integer component"):
        _ = f32_component.signed_maximum


def test_lane_invariants_reject_non_native_counts() -> None:
    f32_component = next(
        component
        for component in ORDINARY_VECTOR_COMPONENT_TYPES
        if component.suffix == "f32"
    )
    for lane_count in (0, 1, 5, 8, 16):
        with pytest.raises(ValueError, match="require 2-4 lanes"):
            OrdinaryVectorType(f32_component, lane_count)
