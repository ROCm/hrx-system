# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.x86.packed_dot_data import (
    X86_PACKED_DOT_DESCRIPTORS,
    PackedDotDescriptor,
    packed_dot_native_layout,
)
from loom.target.native_contraction_layout import ContractionShape
from loom.target.native_coordinate_projection import coordinate_projection_plan


def _descriptor(key: str) -> PackedDotDescriptor:
    return next(
        descriptor for descriptor in X86_PACKED_DOT_DESCRIPTORS if descriptor.key == key
    )


def test_bf16_ymm_layout_maps_packed_pairs_to_independent_dots() -> None:
    layout = packed_dot_native_layout(_descriptor("x86.avx512_bf16.vdpbf16ps.ymm"))

    assert layout.shape == ContractionShape(block_count=8, m=1, n=1, k=2)
    assert layout.lhs.coordinate_map.source_dimensions[0].name == "value"
    assert layout.lhs.coordinate_map.source_dimensions[0].extent == 16
    assert layout.lhs.coordinate_map.destination_by_source == (
        0,
        8,
        1,
        9,
        2,
        10,
        3,
        11,
        4,
        12,
        5,
        13,
        6,
        14,
        7,
        15,
    )
    assert (
        layout.rhs.coordinate_map.destination_by_source
        == layout.lhs.coordinate_map.destination_by_source
    )
    assert layout.accumulator.coordinate_map.destination_by_source == tuple(range(8))
    assert layout.result.coordinate_map == layout.accumulator.coordinate_map


def test_i8_ymm_layout_maps_packed_quads_to_independent_dots() -> None:
    layout = packed_dot_native_layout(_descriptor("x86.avx_vnni_int8.vpdpbssd.ymm"))

    assert layout.shape == ContractionShape(block_count=8, m=1, n=1, k=4)
    assert layout.lhs.coordinate_map.source_dimensions[0].extent == 32
    assert layout.lhs.coordinate_map.destination_by_source == (
        0,
        8,
        16,
        24,
        1,
        9,
        17,
        25,
        2,
        10,
        18,
        26,
        3,
        11,
        19,
        27,
        4,
        12,
        20,
        28,
        5,
        13,
        21,
        29,
        6,
        14,
        22,
        30,
        7,
        15,
        23,
        31,
    )


def test_every_packed_dot_role_has_a_compact_coordinate_projection() -> None:
    plans = set()
    role_map_count = 0
    for descriptor in X86_PACKED_DOT_DESCRIPTORS:
        layout = packed_dot_native_layout(descriptor)
        for role_layout in (
            layout.lhs,
            layout.rhs,
            layout.accumulator,
            layout.result,
        ):
            plan = coordinate_projection_plan(role_layout.coordinate_map)
            assert plan is not None, f"{descriptor.key}/{role_layout.role}"
            plans.add(plan)
            role_map_count += 1

    assert role_map_count == len(X86_PACKED_DOT_DESCRIPTORS) * 4
    assert len(plans) == 3
