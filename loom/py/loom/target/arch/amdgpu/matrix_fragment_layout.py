# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU matrix fragment layout schema and validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

from loom.target.native_contraction_layout import (
    ROLE_ACCUMULATOR,
    ROLE_LHS,
    ROLE_RESULT,
    ROLE_RHS,
)

MATRIX_FRAGMENT_AXIS_NAMES = ("block", "row", "column", "reduction")
_MAX_FRAGMENT_REGISTER_COUNT = 32
_ROLE_AXIS_NAMES = {
    ROLE_LHS: frozenset(("row", "reduction")),
    ROLE_RHS: frozenset(("column", "reduction")),
    ROLE_ACCUMULATOR: frozenset(("row", "column")),
    ROLE_RESULT: frozenset(("row", "column")),
}


@dataclass(frozen=True, slots=True)
class MatrixFragmentAxisLayout:
    """Factorization of one semantic axis across payload elements and lanes."""

    outer_count: int
    thread_count: int
    thread_stride: int
    element_count: int


@dataclass(frozen=True, slots=True)
class MatrixFragmentReductionGroup:
    """Physical storage density for a compressed reduction axis."""

    storage_element_count: int
    logical_element_count: int


@dataclass(frozen=True, slots=True)
class MatrixFragmentRoleLayout:
    """One matrix role's payload storage and semantic coordinate layout."""

    role: str
    payload_element_count: int
    element_bit_count: int
    coordinate_element_stride: int
    reduction_group: MatrixFragmentReductionGroup | None
    axes: tuple[MatrixFragmentAxisLayout | None, ...]

    @property
    def register_count(self) -> int:
        payload_bit_count = self.payload_element_count * self.element_bit_count
        if payload_bit_count % 32 != 0:
            raise ValueError(
                f"matrix fragment role '{self.role}' payload occupies "
                f"{payload_bit_count} bits, not whole 32-bit registers"
            )
        return payload_bit_count // 32

    @property
    def coordinate_element_count(self) -> int:
        return prod(
            axis.outer_count * axis.element_count
            for axis in self.axes
            if axis is not None
        )


@dataclass(frozen=True, slots=True)
class AmdgpuMatrixFragmentLayout:
    """Target-owned matrix fragment layout table row."""

    key: str
    wave_size: int
    tile_shape: tuple[int, int, int, int]
    lhs: MatrixFragmentRoleLayout
    rhs: MatrixFragmentRoleLayout
    accumulator: MatrixFragmentRoleLayout
    result: MatrixFragmentRoleLayout
    family: str | None = None
    canonical_key: str | None = None
    instruction_operand_order: tuple[str, str] = ("lhs", "rhs")

    @property
    def c_kind(self) -> str:
        return f"LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_{self.key.upper()}"

    @property
    def name(self) -> str:
        return self.key.replace("_", ".")


def layout_roles(
    layout: AmdgpuMatrixFragmentLayout,
) -> tuple[MatrixFragmentRoleLayout, ...]:
    return (layout.lhs, layout.rhs, layout.accumulator, layout.result)


def _validate_role(
    layout: AmdgpuMatrixFragmentLayout, role: MatrixFragmentRoleLayout
) -> None:
    if len(role.axes) != len(MATRIX_FRAGMENT_AXIS_NAMES):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            f"{len(role.axes)} semantic axes"
        )
    expected_axis_names = _ROLE_AXIS_NAMES[role.role]
    if layout.tile_shape[0] > 1:
        expected_axis_names = expected_axis_names | {"block"}
    actual_axis_names = {
        axis_name
        for axis_name, axis in zip(MATRIX_FRAGMENT_AXIS_NAMES, role.axes, strict=True)
        if axis is not None
    }
    if actual_axis_names != expected_axis_names:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            f"semantic axes {sorted(actual_axis_names)}, expected "
            f"{sorted(expected_axis_names)}"
        )
    if (
        role.payload_element_count <= 0
        or role.payload_element_count > 0xFFFF
        or role.element_bit_count <= 0
        or role.element_bit_count > 0xFFFF
    ):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            "an invalid payload"
        )
    if role.register_count > _MAX_FRAGMENT_REGISTER_COUNT:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' uses "
            f"{role.register_count} payload registers, exceeding the "
            f"{_MAX_FRAGMENT_REGISTER_COUNT}-register architectural limit"
        )
    if role.coordinate_element_stride <= 0 or role.coordinate_element_stride > 0xFFFF:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            "an invalid coordinate element mapping"
        )
    payload_elements_per_register = (
        32 // role.element_bit_count
        if role.element_bit_count <= 32 and 32 % role.element_bit_count == 0
        else 0
    )
    if (
        payload_elements_per_register != 0
        and payload_elements_per_register % role.coordinate_element_stride != 0
    ):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            "a coordinate element mapping that cannot be addressed from "
            "32-bit payload registers"
        )
    reduction_group = role.reduction_group
    if reduction_group is not None:
        reduction_axis = MATRIX_FRAGMENT_AXIS_NAMES.index("reduction")
        if role.role not in ("lhs", "rhs") or role.axes[reduction_axis] is None:
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                "compresses a missing reduction axis"
            )
        if (
            reduction_group.storage_element_count <= 0
            or reduction_group.logical_element_count <= 0
            or reduction_group.storage_element_count
            >= reduction_group.logical_element_count
            or layout.tile_shape[reduction_axis] % reduction_group.logical_element_count
            != 0
        ):
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                "has an invalid reduction storage group"
            )
    last_payload_element = (
        role.coordinate_element_count - 1
    ) * role.coordinate_element_stride
    if last_payload_element >= role.payload_element_count:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' maps "
            f"coordinate element {role.coordinate_element_count - 1} to "
            f"payload element {last_payload_element} outside "
            f"{role.payload_element_count} elements"
        )
    thread_element_count = 1
    for axis_index, (axis_name, axis) in enumerate(
        zip(MATRIX_FRAGMENT_AXIS_NAMES, role.axes, strict=True)
    ):
        if axis is None:
            continue
        if (
            min(
                axis.outer_count,
                axis.thread_count,
                axis.thread_stride,
                axis.element_count,
            )
            <= 0
            or max(
                axis.outer_count,
                axis.thread_count,
                axis.thread_stride,
                axis.element_count,
            )
            > 0xFFFF
        ):
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"has an empty {axis_name} axis"
            )
        if axis.thread_count > 1 and (
            axis.thread_count.bit_count() != 1 or axis.thread_stride.bit_count() != 1
        ):
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"has a non-power-of-two {axis_name} lane factor"
            )
        expected_extent = layout.tile_shape[axis_index]
        if axis_name == "reduction" and reduction_group is not None:
            expected_extent = (
                expected_extent
                // reduction_group.logical_element_count
                * reduction_group.storage_element_count
            )
        actual_extent = axis.outer_count * axis.thread_count * axis.element_count
        if actual_extent != expected_extent:
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"{axis_name} axis covers {actual_extent}, expected "
                f"{expected_extent}"
            )
        thread_element_count *= axis.thread_count
    if layout.wave_size % thread_element_count != 0:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' "
            f"thread grid {thread_element_count} does not divide wave "
            f"size {layout.wave_size}"
        )
    thread_axes = tuple(axis for axis in role.axes if axis is not None)
    canonical_lanes: set[int] = set()
    for thread_coordinates in product(
        *(range(axis.thread_count) for axis in thread_axes)
    ):
        lane = sum(
            coordinate * axis.thread_stride
            for coordinate, axis in zip(thread_coordinates, thread_axes, strict=True)
        )
        if lane >= layout.wave_size:
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"canonical lane {lane} is outside wave {layout.wave_size}"
            )
        recovered_coordinates = tuple(
            (lane // axis.thread_stride) % axis.thread_count for axis in thread_axes
        )
        if recovered_coordinates != thread_coordinates:
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"thread strides do not invert coordinate "
                f"{thread_coordinates}"
            )
        if lane in canonical_lanes:
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"thread coordinates alias canonical lane {lane}"
            )
        canonical_lanes.add(lane)


def validate_matrix_fragment_layout(layout: AmdgpuMatrixFragmentLayout) -> None:
    """Raises ValueError when a generated layout row violates its contract."""

    if len(layout.tile_shape) != len(MATRIX_FRAGMENT_AXIS_NAMES):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' has "
            f"{len(layout.tile_shape)} tile dimensions"
        )
    if (
        min(layout.wave_size, *layout.tile_shape) <= 0
        or max(layout.wave_size, *layout.tile_shape) > 0xFFFF
    ):
        raise ValueError(f"matrix fragment layout '{layout.key}' has invalid shape")
    expected_roles = ("lhs", "rhs", "accumulator", "result")
    for expected_role, role in zip(expected_roles, layout_roles(layout), strict=True):
        if role.role != expected_role:
            raise ValueError(
                f"matrix fragment layout '{layout.key}' role '{role.role}' "
                f"occupies the {expected_role} slot"
            )
        _validate_role(layout, role)
    is_sparse = any(role.reduction_group is not None for role in layout_roles(layout))
    if is_sparse and layout.family not in ("smfmac", "swmmac"):
        raise ValueError(
            f"sparse matrix fragment layout '{layout.key}' has invalid "
            f"family '{layout.family}'"
        )
    if not is_sparse and layout.family is not None:
        raise ValueError(
            f"dense matrix fragment layout '{layout.key}' names family "
            f"'{layout.family}'"
        )
    if layout.instruction_operand_order not in (("lhs", "rhs"), ("rhs", "lhs")):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' has invalid instruction "
            f"operand order {layout.instruction_operand_order}"
        )
