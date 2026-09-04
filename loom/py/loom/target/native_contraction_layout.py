# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact finite layouts for target-native values and contractions.

This module is generation-time infrastructure. Targets describe finite
physical positions and their semantic coordinates. Generators may compile
those facts into smaller target-specific rows; shipping C does not reconstruct
or search the maps.

Compact executable projections and target movement recipes are derived from
the same exact maps. Opaque, scalable, metadata-dependent, and separately
addressable target tiles remain valid without an exact finite layout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

ROLE_LHS = "lhs"
ROLE_RHS = "rhs"
ROLE_ACCUMULATOR = "accumulator"
ROLE_RESULT = "result"

_ROLES = (ROLE_LHS, ROLE_RHS, ROLE_ACCUMULATOR, ROLE_RESULT)
_ROLE_LOGICAL_DIMENSION_NAMES = {
    ROLE_LHS: ("block", "m", "k"),
    ROLE_RHS: ("block", "k", "n"),
    ROLE_ACCUMULATOR: ("block", "m", "n"),
    ROLE_RESULT: ("block", "m", "n"),
}
_OPERATION_LOCAL_DIMENSION_NAMES = ("block", "row", "column")


@dataclass(frozen=True, slots=True)
class CoordinateDimension:
    """One named finite coordinate dimension."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coordinate dimension name must not be empty")
        if self.extent <= 0:
            raise ValueError(
                f"coordinate dimension '{self.name}' has non-positive extent"
            )

    @property
    def binary_bit_count(self) -> int | None:
        """Returns the number of bits in a binary extent, otherwise None."""

        if self.extent.bit_count() != 1:
            return None
        return (self.extent - 1).bit_length()


def _validate_unique_dimensions(
    dimensions: tuple[CoordinateDimension, ...], *, space: str
) -> None:
    names = tuple(dimension.name for dimension in dimensions)
    if len(names) != len(set(names)):
        raise ValueError(f"{space} coordinate dimensions must have unique names")


def _point_count(dimensions: tuple[CoordinateDimension, ...]) -> int:
    return prod(dimension.extent for dimension in dimensions)


def _pack_coordinate(
    dimensions: tuple[CoordinateDimension, ...], coordinate: tuple[int, ...]
) -> int:
    if len(coordinate) != len(dimensions):
        raise ValueError(
            f"coordinate has {len(coordinate)} values for {len(dimensions)} dimensions"
        )
    ordinal = 0
    stride = 1
    for dimension, value in zip(dimensions, coordinate, strict=True):
        if value < 0 or value >= dimension.extent:
            raise ValueError(
                f"coordinate {dimension.name}={value} is outside extent "
                f"{dimension.extent}"
            )
        ordinal += value * stride
        stride *= dimension.extent
    return ordinal


def _unpack_coordinate(
    dimensions: tuple[CoordinateDimension, ...], ordinal: int
) -> tuple[int, ...]:
    point_count = _point_count(dimensions)
    if ordinal < 0 or ordinal >= point_count:
        raise ValueError(f"point ordinal {ordinal} is outside extent {point_count}")
    return _unpack_coordinate_unchecked(dimensions, ordinal)


def _unpack_coordinate_unchecked(
    dimensions: tuple[CoordinateDimension, ...], ordinal: int
) -> tuple[int, ...]:
    coordinate: list[int] = []
    for dimension in dimensions:
        coordinate.append(ordinal % dimension.extent)
        ordinal //= dimension.extent
    return tuple(coordinate)


@dataclass(frozen=True, slots=True, init=False)
class ExactCoordinateMap:
    """Exact total and surjective map between named coordinate spaces."""

    source_dimensions: tuple[CoordinateDimension, ...]
    destination_dimensions: tuple[CoordinateDimension, ...]
    destination_by_source: tuple[int, ...]
    sources_by_destination: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        *,
        source_dimensions: tuple[CoordinateDimension, ...],
        destination_dimensions: tuple[CoordinateDimension, ...],
        destination_by_source: tuple[int, ...],
    ) -> None:
        _validate_unique_dimensions(source_dimensions, space="source")
        _validate_unique_dimensions(destination_dimensions, space="destination")
        source_point_count = _point_count(source_dimensions)
        destination_point_count = _point_count(destination_dimensions)
        if len(destination_by_source) != source_point_count:
            raise ValueError("coordinate map does not cover every source point")
        sources_by_destination: list[list[int]] = [
            [] for _ in range(destination_point_count)
        ]
        for source_ordinal, destination_ordinal in enumerate(destination_by_source):
            if (
                destination_ordinal < 0
                or destination_ordinal >= destination_point_count
            ):
                raise ValueError("coordinate map contains an invalid destination point")
            sources_by_destination[destination_ordinal].append(source_ordinal)
        if any(not sources for sources in sources_by_destination):
            raise ValueError("coordinate map does not cover its destination domain")
        object.__setattr__(self, "source_dimensions", source_dimensions)
        object.__setattr__(self, "destination_dimensions", destination_dimensions)
        object.__setattr__(self, "destination_by_source", destination_by_source)
        object.__setattr__(
            self,
            "sources_by_destination",
            tuple(tuple(sources) for sources in sources_by_destination),
        )

    @property
    def source_point_count(self) -> int:
        return _point_count(self.source_dimensions)

    @property
    def destination_point_count(self) -> int:
        return _point_count(self.destination_dimensions)

    @property
    def is_bijective(self) -> bool:
        return all(len(sources) == 1 for sources in self.sources_by_destination)

    def evaluate(self, source_coordinate: tuple[int, ...]) -> tuple[int, ...]:
        source_ordinal = _pack_coordinate(self.source_dimensions, source_coordinate)
        return _unpack_coordinate(
            self.destination_dimensions,
            self.destination_by_source[source_ordinal],
        )

    def evaluate_canonical_inverse(
        self, destination_coordinate: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Returns the first source that maps to one destination coordinate."""

        destination_ordinal = _pack_coordinate(
            self.destination_dimensions, destination_coordinate
        )
        return _unpack_coordinate(
            self.source_dimensions,
            self.sources_by_destination[destination_ordinal][0],
        )

    def rename_destination_dimensions(
        self, names: tuple[str, ...]
    ) -> ExactCoordinateMap:
        if len(names) != len(self.destination_dimensions):
            raise ValueError("destination rename rank does not match")
        return ExactCoordinateMap(
            source_dimensions=self.source_dimensions,
            destination_dimensions=tuple(
                CoordinateDimension(name, dimension.extent)
                for name, dimension in zip(
                    names, self.destination_dimensions, strict=True
                )
            ),
            destination_by_source=self.destination_by_source,
        )


def exact_coordinate_map(
    *,
    source_dimensions: tuple[CoordinateDimension, ...],
    destination_dimensions: tuple[CoordinateDimension, ...],
    evaluate: Callable[[tuple[int, ...]], tuple[int, ...]],
) -> ExactCoordinateMap:
    """Builds and exhaustively validates an exact finite coordinate map."""

    destination_by_source: list[int] = []
    for source_ordinal in range(_point_count(source_dimensions)):
        destination_ordinal = _pack_coordinate(
            destination_dimensions,
            evaluate(_unpack_coordinate_unchecked(source_dimensions, source_ordinal)),
        )
        destination_by_source.append(destination_ordinal)
    return ExactCoordinateMap(
        source_dimensions=source_dimensions,
        destination_dimensions=destination_dimensions,
        destination_by_source=tuple(destination_by_source),
    )


def compose_coordinate_maps(
    first: ExactCoordinateMap, second: ExactCoordinateMap
) -> ExactCoordinateMap:
    """Composes two coordinate maps through their shared coordinate space."""

    if first.destination_dimensions != second.source_dimensions:
        raise ValueError("coordinate maps do not share their composition space")
    return exact_coordinate_map(
        source_dimensions=first.source_dimensions,
        destination_dimensions=second.destination_dimensions,
        evaluate=lambda coordinate: _unpack_coordinate(
            second.destination_dimensions,
            second.destination_by_source[
                first.destination_by_source[
                    _pack_coordinate(first.source_dimensions, coordinate)
                ]
            ],
        ),
    )


@dataclass(frozen=True, slots=True)
class ExactElementLayout:
    """Exact placement of fixed-width semantic elements in physical atoms.

    Target carrier types, feature requirements, and selection preferences are
    deliberately outside this record. They belong to target descriptors and
    selected lowering plans. Opaque or scalable target values do not fabricate
    an exact element layout.
    """

    key: str
    atom_bit_width: int
    element_layout: ExactCoordinateMap

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("exact element layout identity must not be empty")
        if self.atom_bit_width <= 0:
            raise ValueError("exact element atom width must be positive")
        if self.element_layout.destination_dimensions != (
            CoordinateDimension("element", self.element_layout.destination_point_count),
        ):
            raise ValueError(
                "exact element layout must map physical atoms to one element axis"
            )
        if not self.element_layout.is_bijective:
            raise ValueError("exact element placement must be bijective")

    @property
    def element_count(self) -> int:
        return self.element_layout.destination_point_count

    @property
    def total_bit_count(self) -> int:
        return self.element_layout.source_point_count * self.atom_bit_width


def contiguous_element_layout(
    *,
    key: str,
    element_count: int,
    atom_bit_width: int,
    physical_dimension_name: str = "atom",
) -> ExactElementLayout:
    """Builds a contiguous fixed-width semantic element layout."""

    if element_count <= 0:
        raise ValueError(f"{key}: element count must be positive")
    element_layout = exact_coordinate_map(
        source_dimensions=(
            CoordinateDimension(physical_dimension_name, element_count),
        ),
        destination_dimensions=(CoordinateDimension("element", element_count),),
        evaluate=lambda coordinate: coordinate,
    )
    return ExactElementLayout(
        key=key,
        atom_bit_width=atom_bit_width,
        element_layout=element_layout,
    )


@dataclass(frozen=True, slots=True)
class ContractionShape:
    """Logical shape computed by one target-native contraction."""

    block_count: int
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        if any(extent <= 0 for extent in (self.block_count, self.m, self.n, self.k)):
            raise ValueError("contraction shape extents must be positive")

    def role_dimensions(self, role: str) -> tuple[CoordinateDimension, ...]:
        extents = {
            "block": self.block_count,
            "m": self.m,
            "n": self.n,
            "k": self.k,
        }
        try:
            names = _ROLE_LOGICAL_DIMENSION_NAMES[role]
        except KeyError as error:
            raise ValueError(f"unknown contraction role '{role}'") from error
        return tuple(CoordinateDimension(name, extents[name]) for name in names)


@dataclass(frozen=True, slots=True)
class ExactContractionRoleLayout:
    """Exact finite ownership proof for one native contraction role."""

    role: str
    coordinate_map: ExactCoordinateMap


def exact_contraction_role_layout(
    shape: ContractionShape,
    role: str,
    coordinate_map: ExactCoordinateMap,
) -> ExactContractionRoleLayout:
    expected_dimensions = shape.role_dimensions(role)
    if coordinate_map.destination_dimensions != expected_dimensions:
        raise ValueError(
            f"{role} coordinate map has destination dimensions "
            f"{coordinate_map.destination_dimensions}, expected "
            f"{expected_dimensions}"
        )
    return ExactContractionRoleLayout(role=role, coordinate_map=coordinate_map)


@dataclass(frozen=True, slots=True)
class ExactContractionLayout:
    """Exact finite role layouts for one target-native contraction."""

    shape: ContractionShape
    lhs: ExactContractionRoleLayout
    rhs: ExactContractionRoleLayout
    accumulator: ExactContractionRoleLayout
    result: ExactContractionRoleLayout

    def __post_init__(self) -> None:
        for expected_role, role_layout in zip(
            _ROLES,
            (self.lhs, self.rhs, self.accumulator, self.result),
            strict=True,
        ):
            if role_layout.role != expected_role:
                raise ValueError(
                    f"{role_layout.role} layout occupies the {expected_role} slot"
                )
            exact_contraction_role_layout(
                self.shape, expected_role, role_layout.coordinate_map
            )

    def role_layout(self, role: str) -> ExactContractionRoleLayout:
        try:
            return {
                ROLE_LHS: self.lhs,
                ROLE_RHS: self.rhs,
                ROLE_ACCUMULATOR: self.accumulator,
                ROLE_RESULT: self.result,
            }[role]
        except KeyError as error:
            raise ValueError(f"unknown contraction role '{role}'") from error


def transpose_contraction_layout(
    layout: ExactContractionLayout,
) -> ExactContractionLayout:
    """Exchanges M/N and LHS/RHS without changing physical ownership.

    Every physical position keeps its payload value. The two matrix axes are
    exchanged in semantic space, and the operand roles follow the contraction
    identity ``A * B = transpose(transpose(B) * transpose(A))``.
    """

    shape = ContractionShape(
        block_count=layout.shape.block_count,
        m=layout.shape.n,
        n=layout.shape.m,
        k=layout.shape.k,
    )

    def transpose_role(
        source: ExactContractionRoleLayout, destination_role: str
    ) -> ExactContractionRoleLayout:
        coordinate_map = source.coordinate_map

        def evaluate(physical_coordinate: tuple[int, ...]) -> tuple[int, ...]:
            block, first, second = coordinate_map.evaluate(physical_coordinate)
            return (block, second, first)

        return exact_contraction_role_layout(
            shape,
            destination_role,
            exact_coordinate_map(
                source_dimensions=coordinate_map.source_dimensions,
                destination_dimensions=shape.role_dimensions(destination_role),
                evaluate=evaluate,
            ),
        )

    return ExactContractionLayout(
        shape=shape,
        lhs=transpose_role(layout.rhs, ROLE_LHS),
        rhs=transpose_role(layout.lhs, ROLE_RHS),
        accumulator=transpose_role(layout.accumulator, ROLE_ACCUMULATOR),
        result=transpose_role(layout.result, ROLE_RESULT),
    )


def _grouped_dot_role_layout(
    shape: ContractionShape,
    role: str,
    element_layout: ExactElementLayout,
) -> ExactContractionRoleLayout:
    element_count = element_layout.element_count
    expected_element_count = (
        shape.block_count * shape.k
        if role in (ROLE_LHS, ROLE_RHS)
        else shape.block_count
    )
    if element_count != expected_element_count:
        raise ValueError(
            f"{element_layout.key}: {role} has {element_count} semantic elements, "
            f"expected {expected_element_count}"
        )

    def evaluate(coordinate: tuple[int, ...]) -> tuple[int, ...]:
        element = coordinate[0]
        if role == ROLE_LHS:
            return (element // shape.k, 0, element % shape.k)
        if role == ROLE_RHS:
            return (element // shape.k, element % shape.k, 0)
        return (element, 0, 0)

    semantic_role_map = exact_coordinate_map(
        source_dimensions=element_layout.element_layout.destination_dimensions,
        destination_dimensions=shape.role_dimensions(role),
        evaluate=evaluate,
    )
    return exact_contraction_role_layout(
        shape,
        role,
        compose_coordinate_maps(
            element_layout.element_layout,
            semantic_role_map,
        ),
    )


def grouped_dot_contraction_layout(
    *,
    group_size: int,
    lhs: ExactElementLayout,
    rhs: ExactElementLayout,
    accumulator: ExactElementLayout,
    result: ExactElementLayout,
) -> ExactContractionLayout:
    """Builds exact target ownership for independent grouped dot products."""

    if group_size <= 0:
        raise ValueError("grouped dot size must be positive")
    block_count = accumulator.element_count
    if result.element_count != block_count:
        raise ValueError("grouped dot accumulator and result counts differ")
    shape = ContractionShape(block_count=block_count, m=1, n=1, k=group_size)
    return ExactContractionLayout(
        shape=shape,
        lhs=_grouped_dot_role_layout(shape, ROLE_LHS, lhs),
        rhs=_grouped_dot_role_layout(shape, ROLE_RHS, rhs),
        accumulator=_grouped_dot_role_layout(
            shape,
            ROLE_ACCUMULATOR,
            accumulator,
        ),
        result=_grouped_dot_role_layout(shape, ROLE_RESULT, result),
    )


def operation_local_coordinate_map(
    role_layout: ExactContractionRoleLayout,
) -> ExactCoordinateMap:
    """Projects canonical role axes into fragment-repack shape positions."""

    return role_layout.coordinate_map.rename_destination_dimensions(
        _OPERATION_LOCAL_DIMENSION_NAMES
    )


@dataclass(frozen=True, slots=True)
class OwnershipRelation:
    """All exact source owners for every destination physical point."""

    source: ExactCoordinateMap
    destination: ExactCoordinateMap
    source_owners_by_destination: tuple[tuple[int, ...], ...]

    @property
    def edge_count(self) -> int:
        return sum(map(len, self.source_owners_by_destination))


def ownership_relation(
    source: ExactCoordinateMap, destination: ExactCoordinateMap
) -> OwnershipRelation:
    """Composes exact maps into a destination-to-source ownership relation."""

    if source.destination_dimensions != destination.destination_dimensions:
        raise ValueError(
            "source and destination maps have different semantic dimensions"
        )
    return OwnershipRelation(
        source=source,
        destination=destination,
        source_owners_by_destination=tuple(
            source.sources_by_destination[semantic_ordinal]
            for semantic_ordinal in destination.destination_by_source
        ),
    )


def unique_ownership_coordinate_map(
    relation: OwnershipRelation,
) -> ExactCoordinateMap | None:
    """Maps each destination position to its sole source physical owner.

    Relations with replicated source ownership remain relations: selecting one
    of several equivalent owners is a target movement decision, not a property
    of the coordinate map.
    """

    if any(len(owners) != 1 for owners in relation.source_owners_by_destination):
        return None
    destination_by_source = tuple(
        owners[0] for owners in relation.source_owners_by_destination
    )
    return ExactCoordinateMap(
        source_dimensions=relation.destination.source_dimensions,
        destination_dimensions=relation.source.source_dimensions,
        destination_by_source=destination_by_source,
    )
