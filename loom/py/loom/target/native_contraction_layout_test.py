# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.native_contraction_layout import (
    ContractionShape,
    CoordinateDimension,
    contiguous_element_layout,
    exact_coordinate_map,
    grouped_dot_contraction_layout,
    ownership_relation,
    unique_ownership_coordinate_map,
)


def test_exact_map_preserves_replication() -> None:
    semantic_dimensions = (CoordinateDimension("row", 2),)
    source = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("participant", 4),),
        destination_dimensions=semantic_dimensions,
        evaluate=lambda coordinate: (coordinate[0] % 2,),
    )
    destination = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("value", 2),),
        destination_dimensions=semantic_dimensions,
        evaluate=lambda coordinate: coordinate,
    )

    relation = ownership_relation(source, destination)

    assert source.sources_by_destination == ((0, 2), (1, 3))
    assert source.evaluate_canonical_inverse((0,)) == (0,)
    assert source.evaluate_canonical_inverse((1,)) == (1,)
    assert relation.source_owners_by_destination == ((0, 2), (1, 3))
    assert unique_ownership_coordinate_map(relation) is None


def test_exact_map_accepts_non_power_of_two_dimensions() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("value", 6),),
        destination_dimensions=(
            CoordinateDimension("row", 2),
            CoordinateDimension("column", 3),
        ),
        evaluate=lambda coordinate: (coordinate[0] % 2, coordinate[0] // 2),
    )

    assert coordinate_map.destination_by_source == tuple(range(6))


def test_unique_ownership_map_preserves_destination_replication() -> None:
    semantic_dimensions = (CoordinateDimension("logical", 2),)
    source = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("value", 2),),
        destination_dimensions=semantic_dimensions,
        evaluate=lambda coordinate: coordinate,
    )
    destination = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("participant", 4),),
        destination_dimensions=semantic_dimensions,
        evaluate=lambda coordinate: (coordinate[0] % 2,),
    )

    owner_map = unique_ownership_coordinate_map(ownership_relation(source, destination))

    assert owner_map is not None
    assert owner_map.destination_by_source == (0, 1, 0, 1)
    assert owner_map.sources_by_destination == ((0, 2), (1, 3))


def test_grouped_dot_composes_carrier_atoms_with_contraction_roles() -> None:
    source = contiguous_element_layout(
        key="packed_i32_4x8",
        element_count=4,
        atom_bit_width=8,
    )
    accumulator = contiguous_element_layout(
        key="scalar_i32",
        element_count=1,
        atom_bit_width=32,
    )

    layout = grouped_dot_contraction_layout(
        group_size=4,
        lhs=source,
        rhs=source,
        accumulator=accumulator,
        result=accumulator,
    )

    assert layout.shape == ContractionShape(block_count=1, m=1, n=1, k=4)
    assert tuple(layout.lhs.coordinate_map.evaluate((atom,)) for atom in range(4)) == (
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
    )
