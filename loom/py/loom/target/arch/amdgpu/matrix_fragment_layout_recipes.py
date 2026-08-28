# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU matrix fragment transition and publication recipes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from loom.target.arch.amdgpu.matrix_fragment_layout import (
    MATRIX_FRAGMENT_AXIS_NAMES,
    AmdgpuMatrixFragmentLayout,
    MatrixFragmentRoleLayout,
)
from loom.target.arch.amdgpu.matrix_fragment_layout_adaptation import (
    matrix_fragment_native_role_layout,
)
from loom.target.native_contraction_layout import (
    CoordinateDimension,
    operation_local_coordinate_map,
    ownership_relation,
    unique_ownership_coordinate_map,
)
from loom.target.native_coordinate_projection import (
    CoordinateProjectionPlan,
    CoordinateProjectionTerm,
    coordinate_projection_plan,
)


@dataclass(frozen=True, slots=True)
class MatrixFragmentResultToRhsPackedB16Projection:
    """Generated AMDGPU projection for one packed-B16 repack recipe."""

    exchange_participant_xor_mask: int
    reverse_participant_mask: int


@dataclass(frozen=True, slots=True)
class MatrixFragmentPackedB16PublicationProjection:
    """Exact result-owner projection for one packed-B16 memory packet."""

    publishing_participant_and_mask: int
    publishing_participant_equal_value: int
    paired_participant_xor_mask: int


@dataclass(frozen=True, slots=True)
class MatrixFragmentLaneBitProjection:
    """Contiguous lane bits projected by an AND followed by a right shift."""

    and_mask: int
    right_shift: int


@dataclass(frozen=True, slots=True)
class MatrixFragmentResultToLhsBf16Projection:
    """Generated AMDGPU movement facts for one result-to-LHS repack."""

    source_lane_group_byte_shift: int
    result_lane_div_byte_shift: int
    source_register_selector: MatrixFragmentLaneBitProjection
    source_lane_group: MatrixFragmentLaneBitProjection
    transpose_bit_count: int


def _contiguous_lane_bit_projection(
    start_bit: int, bit_count: int, input_bit_count: int
) -> MatrixFragmentLaneBitProjection:
    if bit_count == 0:
        return MatrixFragmentLaneBitProjection(and_mask=0, right_shift=0)
    if start_bit != 0 and start_bit + bit_count == input_bit_count:
        return MatrixFragmentLaneBitProjection(and_mask=0xFFFF, right_shift=start_bit)
    return MatrixFragmentLaneBitProjection(
        and_mask=((1 << bit_count) - 1) << start_bit,
        right_shift=start_bit,
    )


def _coordinate_projection_terms(
    plan: CoordinateProjectionPlan,
    source_dimension: str,
    destination_dimension: str,
) -> tuple[CoordinateProjectionTerm, ...]:
    return tuple(
        term
        for term in plan.forward_terms
        if term.source_dimension == source_dimension
        and term.destination_dimension == destination_dimension
    )


def _coordinate_projection_bit_transfer(
    term: CoordinateProjectionTerm,
    source_extent: int,
) -> tuple[int, int, int] | None:
    """Returns source start, destination start, and width for one bit digit."""

    if (
        term.source_divisor.bit_count() != 1
        or term.destination_multiplier.bit_count() != 1
        or source_extent % term.source_divisor != 0
    ):
        return None
    digit_extent = (
        term.source_modulus
        if term.source_modulus
        else source_extent // term.source_divisor
    )
    if digit_extent <= 1 or digit_extent.bit_count() != 1:
        return None
    return (
        term.source_divisor.bit_length() - 1,
        term.destination_multiplier.bit_length() - 1,
        digit_extent.bit_length() - 1,
    )


def matrix_fragment_result_to_lhs_bf16_projection(
    layout: AmdgpuMatrixFragmentLayout,
) -> MatrixFragmentResultToLhsBf16Projection | None:
    """Compiles exact result-to-LHS ownership into gather movement facts."""

    source = layout.result
    destination = layout.lhs
    if (
        layout.tile_shape[0] != 1
        or source.element_bit_count != 32
        or source.coordinate_element_stride != 1
        or source.payload_element_count != source.register_count
        or destination.element_bit_count != 16
        or destination.coordinate_element_stride != 1
        or destination.payload_element_count != destination.register_count * 2
        or source.register_count == 0
        or destination.register_count == 0
    ):
        return None

    source_layout = matrix_fragment_native_role_layout(layout, source)
    destination_layout = matrix_fragment_native_role_layout(layout, destination)
    if source_layout is None or destination_layout is None:
        return None
    source_coordinate_map = operation_local_coordinate_map(source_layout)
    destination_coordinate_map = operation_local_coordinate_map(destination_layout)
    if (
        source_coordinate_map.destination_dimensions
        != destination_coordinate_map.destination_dimensions
    ):
        return None
    relation = ownership_relation(source_coordinate_map, destination_coordinate_map)
    owner_map = unique_ownership_coordinate_map(relation)
    if owner_map is None:
        return None
    projection = coordinate_projection_plan(owner_map)
    if projection is None:
        return None

    participant_to_participant = _coordinate_projection_terms(
        projection, "participant", "participant"
    )
    participant_to_value = _coordinate_projection_terms(
        projection, "participant", "value"
    )
    value_to_participant = _coordinate_projection_terms(
        projection, "value", "participant"
    )
    value_to_value = _coordinate_projection_terms(projection, "value", "value")
    destination_participant_extent = owner_map.source_dimensions[0].extent
    destination_value_extent = owner_map.source_dimensions[1].extent
    destination_participant_bit_count = owner_map.source_dimensions[0].binary_bit_count
    destination_value_bit_count = owner_map.source_dimensions[1].binary_bit_count
    source_value_bit_count = owner_map.destination_dimensions[1].binary_bit_count
    row_bit_count = relation.source.destination_dimensions[1].binary_bit_count
    if (
        tuple(dimension.name for dimension in owner_map.source_dimensions)
        != ("participant", "value")
        or tuple(dimension.name for dimension in owner_map.destination_dimensions)
        != ("participant", "value")
        or destination_participant_bit_count is None
        or destination_value_bit_count is None
        or source_value_bit_count is None
        or row_bit_count is None
        or destination_value_bit_count == 0
        or destination_participant_bit_count < row_bit_count
        or value_to_value
        or len(value_to_participant) != 1
        or _coordinate_projection_bit_transfer(
            value_to_participant[0], destination_value_extent
        )
        != (0, 0, destination_value_bit_count)
    ):
        return None

    if source_value_bit_count == 0:
        if participant_to_value:
            return None
        source_register_selector = _contiguous_lane_bit_projection(0, 0, row_bit_count)
    else:
        if len(participant_to_value) != 1:
            return None
        source_register_transfer = _coordinate_projection_bit_transfer(
            participant_to_value[0], destination_participant_extent
        )
        if source_register_transfer is None:
            return None
        (
            source_register_start_bit,
            source_register_destination_start_bit,
            source_register_bit_count,
        ) = source_register_transfer
        if (
            source_register_destination_start_bit != 0
            or source_register_bit_count != source_value_bit_count
            or source_register_start_bit + source_register_bit_count > row_bit_count
        ):
            return None
        source_register_selector = _contiguous_lane_bit_projection(
            source_register_start_bit,
            source_register_bit_count,
            row_bit_count,
        )

    lane_mod_transfers: list[tuple[int, int, int]] = []
    lane_div_transfers: list[tuple[int, int, int]] = []
    for term in participant_to_participant:
        transfer = _coordinate_projection_bit_transfer(
            term, destination_participant_extent
        )
        if transfer is None:
            return None
        source_start_bit, _, bit_count = transfer
        if source_start_bit + bit_count <= row_bit_count:
            lane_mod_transfers.append(transfer)
        elif source_start_bit >= row_bit_count:
            lane_div_transfers.append(transfer)
        else:
            return None
    if len(lane_mod_transfers) > 1 or len(lane_div_transfers) > 1:
        return None

    source_lane_group_byte_shift = 0
    source_lane_group = _contiguous_lane_bit_projection(0, 0, row_bit_count)
    if lane_mod_transfers:
        (
            source_lane_group_start_bit,
            source_lane_group_destination_start_bit,
            source_lane_group_bit_count,
        ) = lane_mod_transfers[0]
        if source_lane_group_destination_start_bit < destination_value_bit_count:
            return None
        source_lane_group_byte_shift = source_lane_group_destination_start_bit + 2
        source_lane_group = _contiguous_lane_bit_projection(
            source_lane_group_start_bit,
            source_lane_group_bit_count,
            row_bit_count,
        )

    result_lane_div_byte_shift = 0
    if lane_div_transfers:
        (
            result_lane_div_start_bit,
            result_lane_div_destination_start_bit,
            result_lane_div_bit_count,
        ) = lane_div_transfers[0]
        if (
            result_lane_div_start_bit != row_bit_count
            or result_lane_div_bit_count
            != destination_participant_bit_count - row_bit_count
            or result_lane_div_destination_start_bit < destination_value_bit_count
        ):
            return None
        result_lane_div_byte_shift = result_lane_div_destination_start_bit + 2

    return MatrixFragmentResultToLhsBf16Projection(
        source_lane_group_byte_shift=source_lane_group_byte_shift,
        result_lane_div_byte_shift=result_lane_div_byte_shift,
        source_register_selector=source_register_selector,
        source_lane_group=source_lane_group,
        transpose_bit_count=min(
            source_value_bit_count, destination_value_bit_count - 1
        ),
    )


def matrix_fragment_result_to_rhs_packed_b16_projection(
    layout: AmdgpuMatrixFragmentLayout,
) -> MatrixFragmentResultToRhsPackedB16Projection | None:
    """Compiles exact ownership into the target's XOR-and-permute projection."""

    source = layout.result
    destination = layout.rhs
    if (
        layout.wave_size != 32
        or source.element_bit_count != 16
        or destination.element_bit_count != 16
        or source.coordinate_element_stride != 2
        or destination.coordinate_element_stride != 1
        or source.register_count == 0
        or source.register_count != destination.register_count
        or source.payload_element_count != source.register_count * 2
        or destination.payload_element_count != destination.register_count * 2
    ):
        return None

    source_layout = matrix_fragment_native_role_layout(layout, source)
    destination_layout = matrix_fragment_native_role_layout(layout, destination)
    if source_layout is None or destination_layout is None:
        return None
    source_coordinate_map = operation_local_coordinate_map(source_layout)
    destination_coordinate_map = operation_local_coordinate_map(destination_layout)
    if (
        source_coordinate_map.destination_dimensions
        != destination_coordinate_map.destination_dimensions
    ):
        return None
    relation = ownership_relation(source_coordinate_map, destination_coordinate_map)
    owner_map = unique_ownership_coordinate_map(relation)
    if owner_map is None:
        return None
    projection = coordinate_projection_plan(owner_map)
    if projection is None:
        return None

    participant_to_participant = _coordinate_projection_terms(
        projection, "participant", "participant"
    )
    participant_to_value = _coordinate_projection_terms(
        projection, "participant", "value"
    )
    value_to_participant = _coordinate_projection_terms(
        projection, "value", "participant"
    )
    value_to_value = _coordinate_projection_terms(projection, "value", "value")
    destination_participant_extent = owner_map.source_dimensions[0].extent
    destination_value_extent = owner_map.source_dimensions[1].extent
    destination_participant_bit_count = owner_map.source_dimensions[0].binary_bit_count
    destination_value_bit_count = owner_map.source_dimensions[1].binary_bit_count
    participant_bit_count = owner_map.destination_dimensions[0].binary_bit_count
    source_value_bit_count = owner_map.destination_dimensions[1].binary_bit_count
    if (
        tuple(dimension.name for dimension in owner_map.source_dimensions)
        != ("participant", "value")
        or tuple(dimension.name for dimension in owner_map.destination_dimensions)
        != ("participant", "value")
        or participant_bit_count is None
        or source_value_bit_count is None
        or destination_participant_bit_count is None
        or destination_value_bit_count is None
        or participant_bit_count == 0
        or destination_participant_bit_count != participant_bit_count
        or destination_value_bit_count != source_value_bit_count + 1
        or len(participant_to_participant) != 1
        or _coordinate_projection_bit_transfer(
            participant_to_participant[0], destination_participant_extent
        )
        != (0, 0, participant_bit_count - 1)
        or participant_to_value
        or len(value_to_participant) != 1
        or _coordinate_projection_bit_transfer(
            value_to_participant[0], destination_value_extent
        )
        != (0, participant_bit_count - 1, 1)
        or len(value_to_value) != 1
        or _coordinate_projection_bit_transfer(
            value_to_value[0], destination_value_extent
        )
        != (1, 0, source_value_bit_count)
    ):
        return None

    exchange_mask = 1 << (participant_bit_count - 1)
    reverse_participant_mask = sum(
        1 << participant
        for participant in range(layout.wave_size)
        if participant & exchange_mask
    )
    return MatrixFragmentResultToRhsPackedB16Projection(
        exchange_participant_xor_mask=exchange_mask,
        reverse_participant_mask=reverse_participant_mask,
    )


@cache
def matrix_fragment_packed_b16_publication_projection(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
    packed_axis: str,
) -> MatrixFragmentPackedB16PublicationProjection | None:
    """Compiles exact adjacent-axis owners into a packed publication recipe.

    The supported executable projection lets one participant bit predicate
    publish pairs. The publisher owns the low coordinate and the paired
    participant owns the adjacent high coordinate at the same local payload
    position. Exhaustive exact-map evaluation proves that the pairs cover the
    role's complete logical domain exactly once.
    """

    if packed_axis not in ("row", "column"):
        raise ValueError(f"unsupported packed-B16 publication axis '{packed_axis}'")
    packed_axis_index = MATRIX_FRAGMENT_AXIS_NAMES.index(packed_axis)
    if (
        layout.wave_size <= 0
        or (layout.wave_size & (layout.wave_size - 1)) != 0
        or layout.tile_shape[0] != 1
        or layout.tile_shape[packed_axis_index] % 2 != 0
        or role.element_bit_count != 32
        or role.coordinate_element_stride != 1
        or role.payload_element_count != role.coordinate_element_count
        or role.payload_element_count != role.register_count
        or role.reduction_group is not None
    ):
        return None

    native_role_layout = matrix_fragment_native_role_layout(layout, role)
    if native_role_layout is None:
        return None
    coordinate_map = operation_local_coordinate_map(native_role_layout)
    if (
        coordinate_map.source_dimensions
        != (
            CoordinateDimension("participant", layout.wave_size),
            CoordinateDimension("value", role.coordinate_element_count),
        )
        or coordinate_map.destination_dimensions
        != (
            CoordinateDimension("block", 1),
            CoordinateDimension("row", layout.tile_shape[1]),
            CoordinateDimension("column", layout.tile_shape[2]),
        )
        or not coordinate_map.is_bijective
    ):
        return None

    paired_participant_xor_mask: int | None = None
    publishing_positions: dict[int, set[int]] = {}
    paired_positions: dict[int, set[int]] = {}
    for row in range(layout.tile_shape[1]):
        for column in range(layout.tile_shape[2]):
            packed_coordinate = row if packed_axis == "row" else column
            if packed_coordinate % 2 != 0:
                continue
            paired_row = row + 1 if packed_axis == "row" else row
            paired_column = column + 1 if packed_axis == "column" else column
            publishing_participant, publishing_position = (
                coordinate_map.evaluate_canonical_inverse((0, row, column))
            )
            paired_participant, paired_position = (
                coordinate_map.evaluate_canonical_inverse(
                    (0, paired_row, paired_column)
                )
            )
            participant_xor_mask = publishing_participant ^ paired_participant
            if (
                participant_xor_mask == 0
                or paired_position != publishing_position
                or (
                    paired_participant_xor_mask is not None
                    and participant_xor_mask != paired_participant_xor_mask
                )
            ):
                return None
            paired_participant_xor_mask = participant_xor_mask
            publishing_positions.setdefault(publishing_participant, set()).add(
                publishing_position
            )
            paired_positions.setdefault(paired_participant, set()).add(paired_position)

    if paired_participant_xor_mask is None:
        return None
    expected_positions = set(range(role.coordinate_element_count))
    publishing_participants = frozenset(publishing_positions)
    paired_participants = frozenset(paired_positions)
    if (
        any(
            positions != expected_positions
            for positions in publishing_positions.values()
        )
        or any(
            positions != expected_positions for positions in paired_positions.values()
        )
        or publishing_participants & paired_participants
        or publishing_participants | paired_participants
        != frozenset(range(layout.wave_size))
        or any(
            participant ^ paired_participant_xor_mask not in paired_participants
            for participant in publishing_participants
        )
    ):
        return None

    first_publisher = min(publishing_participants)
    participant_bit_mask = layout.wave_size - 1
    publishing_participant_and_mask = participant_bit_mask
    for participant in publishing_participants:
        publishing_participant_and_mask &= ~(participant ^ first_publisher)
    publishing_participant_equal_value = (
        first_publisher & publishing_participant_and_mask
    )
    selected = frozenset(
        participant
        for participant in range(layout.wave_size)
        if (participant & publishing_participant_and_mask)
        == publishing_participant_equal_value
    )
    if publishing_participant_and_mask == 0 or selected != publishing_participants:
        return None
    return MatrixFragmentPackedB16PublicationProjection(
        publishing_participant_and_mask=publishing_participant_and_mask,
        publishing_participant_equal_value=publishing_participant_equal_value,
        paired_participant_xor_mask=paired_participant_xor_mask,
    )
