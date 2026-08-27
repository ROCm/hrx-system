# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.native_contraction_layout import (
    CoordinateDimension,
    exact_coordinate_map,
)
from loom.target.native_coordinate_projection import (
    CoordinateProjectionPlan,
    CoordinateProjectionTerm,
    coordinate_projection_plan,
)


def test_projection_compiles_binary_replication() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(
            CoordinateDimension("participant", 4),
            CoordinateDimension("value", 2),
        ),
        destination_dimensions=(CoordinateDimension("element", 4),),
        evaluate=lambda coordinate: ((coordinate[0] % 2) + 2 * coordinate[1],),
    )

    assert coordinate_projection_plan(coordinate_map) == CoordinateProjectionPlan(
        forward_terms=(
            CoordinateProjectionTerm("participant", "element", 1, 2, 1),
            CoordinateProjectionTerm("value", "element", 1, 0, 2),
        ),
        inverse_terms=(
            CoordinateProjectionTerm("element", "participant", 1, 2, 1),
            CoordinateProjectionTerm("element", "value", 2, 0, 1),
        ),
    )


def test_projection_compiles_non_binary_mixed_radix() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("value", 6),),
        destination_dimensions=(
            CoordinateDimension("row", 2),
            CoordinateDimension("column", 3),
        ),
        evaluate=lambda coordinate: (coordinate[0] % 2, coordinate[0] // 2),
    )

    assert coordinate_projection_plan(coordinate_map) == CoordinateProjectionPlan(
        forward_terms=(
            CoordinateProjectionTerm("value", "row", 1, 2, 1),
            CoordinateProjectionTerm("value", "column", 2, 0, 1),
        ),
        inverse_terms=(
            CoordinateProjectionTerm("row", "value", 1, 0, 1),
            CoordinateProjectionTerm("column", "value", 1, 0, 2),
        ),
    )


def test_projection_compiles_interleaved_source_digits() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(
            CoordinateDimension("participant", 2),
            CoordinateDimension("value", 16),
        ),
        destination_dimensions=(CoordinateDimension("row", 32),),
        evaluate=lambda coordinate: (
            coordinate[1] % 4 + 4 * coordinate[0] + 8 * ((coordinate[1] // 4) % 4),
        ),
    )

    assert coordinate_projection_plan(coordinate_map) == CoordinateProjectionPlan(
        forward_terms=(
            CoordinateProjectionTerm("participant", "row", 1, 0, 4),
            CoordinateProjectionTerm("value", "row", 1, 0, 1),
            CoordinateProjectionTerm("value", "row", 4, 0, 4),
        ),
        inverse_terms=(
            CoordinateProjectionTerm("row", "participant", 4, 2, 1),
            CoordinateProjectionTerm("row", "value", 1, 4, 1),
            CoordinateProjectionTerm("row", "value", 8, 0, 4),
        ),
    )


def test_projection_rejects_affine_offset() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("value", 8),),
        destination_dimensions=(CoordinateDimension("logical", 8),),
        evaluate=lambda coordinate: (coordinate[0] ^ 5,),
    )

    assert coordinate_projection_plan(coordinate_map) is None


def test_projection_rejects_arbitrary_lookup_permutation() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(CoordinateDimension("value", 4),),
        destination_dimensions=(CoordinateDimension("logical", 4),),
        evaluate=lambda coordinate: ((0, 1, 3, 2)[coordinate[0]],),
    )

    assert coordinate_projection_plan(coordinate_map) is None


def test_projection_rejects_cross_dimension_interaction() -> None:
    coordinate_map = exact_coordinate_map(
        source_dimensions=(
            CoordinateDimension("row", 2),
            CoordinateDimension("column", 2),
        ),
        destination_dimensions=(CoordinateDimension("logical", 2),),
        evaluate=lambda coordinate: (coordinate[0] ^ coordinate[1],),
    )

    assert coordinate_projection_plan(coordinate_map) is None


def test_projection_term_rejects_modulus_one() -> None:
    with pytest.raises(
        ValueError,
        match="coordinate projection modulus must be zero or at least two",
    ):
        CoordinateProjectionTerm(
            source_dimension="source",
            destination_dimension="destination",
            source_divisor=1,
            source_modulus=1,
            destination_multiplier=1,
        )
