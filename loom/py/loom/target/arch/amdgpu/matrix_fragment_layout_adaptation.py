# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU adaptation to target-neutral native fragment layout facts."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from math import prod

from loom.target.arch.amdgpu.matrix_fragment_layout import (
    MATRIX_FRAGMENT_AXIS_NAMES,
    AmdgpuMatrixFragmentLayout,
    MatrixFragmentAxisLayout,
    MatrixFragmentRoleLayout,
    layout_roles,
)
from loom.target.native_contraction_layout import (
    ROLE_ACCUMULATOR,
    ROLE_LHS,
    ROLE_RESULT,
    ROLE_RHS,
    ContractionShape,
    CoordinateDimension,
    ExactContractionRoleLayout,
    ExactCoordinateMap,
    exact_contraction_role_layout,
    exact_coordinate_map,
)
from loom.target.native_coordinate_projection import (
    CoordinateProjectionPlan,
    coordinate_projection_plan,
)
from loom.target.native_layout_facts import (
    NativeContractionFacts,
    NativeContractionRoleFacts,
    NativeLayoutEvidence,
    NativeTransitionFacts,
    exact_native_contraction_role_facts,
    exact_native_transition_facts,
)

_ROLE_LOGICAL_AXIS_INDICES = {
    ROLE_LHS: (0, 1, 3),
    ROLE_RHS: (0, 3, 2),
    ROLE_ACCUMULATOR: (0, 1, 2),
    ROLE_RESULT: (0, 1, 2),
}
_ROLE_STORAGE_DIMENSION_NAMES = {
    ROLE_LHS: ("block", "row", "reduction"),
    ROLE_RHS: ("block", "reduction", "column"),
    ROLE_ACCUMULATOR: ("block", "row", "column"),
    ROLE_RESULT: ("block", "row", "column"),
}


@cache
def _role_storage_coordinate_evaluator(
    axes: tuple[MatrixFragmentAxisLayout | None, ...],
) -> Callable[[int, int], tuple[int | None, ...]]:
    """Precomputes one validated role's coordinate strides."""

    inner_element_count = prod(axis.element_count for axis in axes if axis is not None)
    outer_element_count = prod(axis.outer_count for axis in axes if axis is not None)
    axis_factors: list[tuple[MatrixFragmentAxisLayout, int, int] | None] = []
    inner_stride = inner_element_count
    outer_stride = outer_element_count
    for axis in axes:
        if axis is None:
            axis_factors.append(None)
            continue
        inner_stride //= axis.element_count
        outer_stride //= axis.outer_count
        axis_factors.append((axis, inner_stride, outer_stride))

    def evaluate(lane: int, coordinate_element_index: int) -> tuple[int | None, ...]:
        inner_linear_index = coordinate_element_index % inner_element_count
        outer_linear_index = coordinate_element_index // inner_element_count
        coordinates: list[int | None] = []
        for factor in axis_factors:
            if factor is None:
                coordinates.append(None)
                continue
            axis, axis_inner_stride, axis_outer_stride = factor
            element_coordinate = (
                inner_linear_index // axis_inner_stride
            ) % axis.element_count
            thread_coordinate = (lane // axis.thread_stride) % axis.thread_count
            outer_coordinate = (
                outer_linear_index // axis_outer_stride
            ) % axis.outer_count
            coordinates.append(
                axis.element_count
                * (thread_coordinate + axis.thread_count * outer_coordinate)
                + element_coordinate
            )
        return tuple(coordinates)

    return evaluate


@cache
def _matrix_fragment_packed_element_axis(
    wave_size: int,
    role: MatrixFragmentRoleLayout,
) -> str | None:
    if role.element_bit_count <= 0 or 32 % role.element_bit_count != 0:
        return None
    elements_per_register = 32 // role.element_bit_count
    if (
        elements_per_register <= 1
        or role.coordinate_element_offset != 0
        or role.coordinate_element_stride != 1
        or role.coordinate_element_count == 0
        or role.coordinate_element_count % elements_per_register != 0
    ):
        return None

    evaluate_coordinate = _role_storage_coordinate_evaluator(role.axes)
    packed_axis_index: int | None = None
    for participant in range(wave_size):
        for base_value in range(
            0, role.coordinate_element_count, elements_per_register
        ):
            base_coordinate = evaluate_coordinate(participant, base_value)
            for element in range(1, elements_per_register):
                coordinate = evaluate_coordinate(participant, base_value + element)
                changed_axes = tuple(
                    axis_index
                    for axis_index, (base, value) in enumerate(
                        zip(base_coordinate, coordinate, strict=True)
                    )
                    if base != value
                )
                if len(changed_axes) != 1:
                    return None
                changed_axis_index = changed_axes[0]
                base = base_coordinate[changed_axis_index]
                value = coordinate[changed_axis_index]
                if base is None or value != base + element:
                    return None
                if packed_axis_index is None:
                    packed_axis_index = changed_axis_index
                elif packed_axis_index != changed_axis_index:
                    return None

    return (
        None
        if packed_axis_index is None
        else MATRIX_FRAGMENT_AXIS_NAMES[packed_axis_index]
    )


def matrix_fragment_packed_element_axis(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
) -> str | None:
    """Returns the semantic axis densely packed within each register."""

    return _matrix_fragment_packed_element_axis(layout.wave_size, role)


@cache
def _matrix_fragment_role_storage_coordinate_map(
    wave_size: int,
    logical_axis_indices: tuple[int, ...],
    destination_dimension_names: tuple[str, ...],
    axes: tuple[MatrixFragmentAxisLayout | None, ...],
) -> ExactCoordinateMap:
    evaluate_coordinate = _role_storage_coordinate_evaluator(axes)
    coordinate_element_count = prod(
        axis.outer_count * axis.element_count for axis in axes if axis is not None
    )

    def evaluate(physical_coordinate: tuple[int, ...]) -> tuple[int, ...]:
        participant, value = physical_coordinate
        coordinate = evaluate_coordinate(participant, value)
        return tuple(
            0 if coordinate[index] is None else coordinate[index]
            for index in logical_axis_indices
        )

    destination_dimensions = []
    for dimension_name, axis_index in zip(
        destination_dimension_names,
        logical_axis_indices,
        strict=True,
    ):
        axis = axes[axis_index]
        extent = (
            1
            if axis is None
            else axis.outer_count * axis.thread_count * axis.element_count
        )
        destination_dimensions.append(CoordinateDimension(dimension_name, extent))
    return exact_coordinate_map(
        source_dimensions=(
            CoordinateDimension("participant", wave_size),
            CoordinateDimension("value", coordinate_element_count),
        ),
        destination_dimensions=tuple(destination_dimensions),
        evaluate=evaluate,
    )


def matrix_fragment_role_storage_coordinate_map(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
) -> ExactCoordinateMap:
    """Returns the exact participant/value-to-storage map for one role."""

    return _matrix_fragment_role_storage_coordinate_map(
        layout.wave_size,
        _ROLE_LOGICAL_AXIS_INDICES[role.role],
        _ROLE_STORAGE_DIMENSION_NAMES[role.role],
        role.axes,
    )


def matrix_fragment_native_role_layout(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
) -> ExactContractionRoleLayout | None:
    """Returns one exact role layout, or None when metadata owns coordinates."""

    if role.reduction_group is not None:
        return None
    block_count, m, n, k = layout.tile_shape
    shape = ContractionShape(block_count=block_count, m=m, n=n, k=k)

    coordinate_map = matrix_fragment_role_storage_coordinate_map(
        layout, role
    ).rename_destination_dimensions(
        tuple(dimension.name for dimension in shape.role_dimensions(role.role))
    )
    return exact_contraction_role_layout(shape, role.role, coordinate_map)


def matrix_fragment_role_storage_projection_plan(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
) -> CoordinateProjectionPlan:
    """Compiles one AMDGPU role into direct storage-coordinate terms."""

    plan = coordinate_projection_plan(
        matrix_fragment_role_storage_coordinate_map(layout, role)
    )
    if plan is None:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' "
            "does not have a direct storage-coordinate projection"
        )
    return plan


def _matrix_fragment_role_logical_coordinate_count(
    layout: AmdgpuMatrixFragmentLayout, role: MatrixFragmentRoleLayout
) -> int:
    block_count, row_count, column_count, reduction_count = layout.tile_shape
    if role.role == ROLE_LHS:
        return block_count * row_count * reduction_count
    if role.role == ROLE_RHS:
        return block_count * reduction_count * column_count
    return block_count * row_count * column_count


@cache
def matrix_fragment_native_contraction_facts(
    layout: AmdgpuMatrixFragmentLayout,
) -> NativeContractionFacts:
    """Summarizes one AMDGPU fragment layout for shipping consumers."""

    role_facts: list[NativeContractionRoleFacts] = []
    for role in layout_roles(layout):
        if role.reduction_group is None:
            role_facts.append(
                exact_native_contraction_role_facts(
                    role.role,
                    matrix_fragment_role_storage_coordinate_map(layout, role),
                    element_bit_count=role.element_bit_count,
                    register_count=role.register_count,
                    payload_element_count=role.payload_element_count,
                )
            )
            continue
        role_facts.append(
            NativeContractionRoleFacts(
                role=role.role,
                evidence=NativeLayoutEvidence.METADATA_DEPENDENT,
                element_bit_count=role.element_bit_count,
                register_count=role.register_count,
                payload_element_count=role.payload_element_count,
                physical_position_count=(
                    layout.wave_size * role.coordinate_element_count
                ),
                logical_coordinate_count=(
                    _matrix_fragment_role_logical_coordinate_count(layout, role)
                ),
                owner_multiplicity_minimum=None,
                owner_multiplicity_maximum=None,
            )
        )

    block_count, row_count, column_count, reduction_count = layout.tile_shape
    return NativeContractionFacts(
        shape=ContractionShape(
            block_count=block_count,
            m=row_count,
            n=column_count,
            k=reduction_count,
        ),
        participant_count=layout.wave_size,
        lhs=role_facts[0],
        rhs=role_facts[1],
        accumulator=role_facts[2],
        result=role_facts[3],
    )


@cache
def matrix_fragment_native_transition_facts(
    layout: AmdgpuMatrixFragmentLayout,
    source_role: str,
    destination_role: str,
) -> NativeTransitionFacts | None:
    """Compiles an exact AMDGPU role transition into source-owner facts."""

    source = matrix_fragment_native_role_layout(
        layout, next(role for role in layout_roles(layout) if role.role == source_role)
    )
    destination = matrix_fragment_native_role_layout(
        layout,
        next(role for role in layout_roles(layout) if role.role == destination_role),
    )
    if source is None or destination is None:
        return None
    return exact_native_transition_facts(
        matrix_fragment_native_contraction_facts(layout), source, destination
    )
