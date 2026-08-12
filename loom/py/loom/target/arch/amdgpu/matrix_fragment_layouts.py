# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU matrix fragment semantic layout rows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

from loom.target.arch.amdgpu.matrix_formats import (
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)

_AXIS_NAMES = ("block", "row", "column", "reduction")
_MAX_FRAGMENT_REGISTER_COUNT = 32
_ROLE_AXIS_NAMES = {
    "lhs": frozenset(("row", "reduction")),
    "rhs": frozenset(("column", "reduction")),
    "accumulator": frozenset(("row", "column")),
    "result": frozenset(("row", "column")),
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


_STRUCTURED_2_TO_4_REDUCTION_GROUP = MatrixFragmentReductionGroup(2, 4)


@dataclass(frozen=True, slots=True)
class MatrixFragmentRoleLayout:
    """One matrix role's payload storage and semantic coordinate layout."""

    role: str
    payload_element_count: int
    element_bit_count: int
    coordinate_element_offset: int
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

    @property
    def c_kind(self) -> str:
        return f"LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_{self.key.upper()}"

    @property
    def name(self) -> str:
        return self.key.replace("_", ".")


def _axis(
    *, outer: int = 1, thread: int = 1, stride: int = 1, element: int = 1
) -> MatrixFragmentAxisLayout:
    return MatrixFragmentAxisLayout(outer, thread, stride, element)


def _axes(
    *,
    block: MatrixFragmentAxisLayout | None = None,
    row: MatrixFragmentAxisLayout | None = None,
    column: MatrixFragmentAxisLayout | None = None,
    reduction: MatrixFragmentAxisLayout | None = None,
) -> tuple[MatrixFragmentAxisLayout | None, ...]:
    return (block, row, column, reduction)


def _role(
    role: str,
    payload_element_count: int,
    element_bit_count: int,
    axes: tuple[MatrixFragmentAxisLayout | None, ...],
    *,
    coordinate_element_offset: int = 0,
    coordinate_element_stride: int = 1,
    reduction_group: MatrixFragmentReductionGroup | None = None,
) -> MatrixFragmentRoleLayout:
    return MatrixFragmentRoleLayout(
        role=role,
        payload_element_count=payload_element_count,
        element_bit_count=element_bit_count,
        coordinate_element_offset=coordinate_element_offset,
        coordinate_element_stride=coordinate_element_stride,
        reduction_group=reduction_group,
        axes=axes,
    )


def _single_tile_layout(
    key: str,
    *,
    wave_size: int,
    row_count: int,
    column_count: int,
    reduction_count: int,
    lhs_payload_element_count: int,
    rhs_payload_element_count: int,
    lhs_element_bit_count: int,
    rhs_element_bit_count: int,
    result_payload_element_count: int,
    result_element_bit_count: int = 32,
    lhs_reduction_group: MatrixFragmentReductionGroup | None = None,
    lhs_lane_replication: int = 1,
    rhs_lane_replication: int = 1,
    family: str | None = None,
) -> AmdgpuMatrixFragmentLayout:
    if wave_size % lhs_lane_replication != 0 or wave_size % rhs_lane_replication != 0:
        raise ValueError(
            f"matrix fragment layout '{key}' has a lane replication that "
            f"does not divide wave size {wave_size}"
        )
    lhs_wave_size = wave_size // lhs_lane_replication
    rhs_wave_size = wave_size // rhs_lane_replication
    if (
        lhs_wave_size % row_count != 0
        or rhs_wave_size % column_count != 0
        or wave_size % column_count != 0
    ):
        raise ValueError(
            f"matrix fragment layout '{key}' cannot factor wave size "
            f"{wave_size} over its tile"
        )
    lhs_row_thread_count = lhs_wave_size // row_count
    rhs_column_thread_count = rhs_wave_size // column_count
    result_column_thread_count = wave_size // column_count
    lhs = _role(
        "lhs",
        lhs_payload_element_count,
        lhs_element_bit_count,
        _axes(
            row=_axis(thread=row_count),
            reduction=_axis(
                thread=lhs_row_thread_count,
                stride=row_count,
                element=lhs_payload_element_count,
            ),
        ),
        reduction_group=lhs_reduction_group,
    )
    rhs = _role(
        "rhs",
        rhs_payload_element_count,
        rhs_element_bit_count,
        _axes(
            column=_axis(thread=column_count),
            reduction=_axis(
                thread=rhs_column_thread_count,
                stride=column_count,
                element=rhs_payload_element_count,
            ),
        ),
    )
    result_axes = _axes(
        row=_axis(
            thread=result_column_thread_count,
            stride=column_count,
            element=result_payload_element_count,
        ),
        column=_axis(thread=column_count),
    )
    return AmdgpuMatrixFragmentLayout(
        key=key,
        wave_size=wave_size,
        tile_shape=(1, row_count, column_count, reduction_count),
        lhs=lhs,
        rhs=rhs,
        accumulator=_role(
            "accumulator",
            result_payload_element_count,
            result_element_bit_count,
            result_axes,
        ),
        result=_role(
            "result",
            result_payload_element_count,
            result_element_bit_count,
            result_axes,
        ),
        family=family,
    )


_BLOCKED_MFMA_RESULT_AXES = {
    (16, 4): _axes(
        block=_axis(thread=16, stride=4),
        row=_axis(element=4),
        column=_axis(thread=4),
    ),
    (4, 16): _axes(
        block=_axis(element=4),
        row=_axis(thread=4, stride=16, element=4),
        column=_axis(thread=16),
    ),
    (2, 32): _axes(
        block=_axis(outer=2),
        row=_axis(outer=4, thread=2, stride=32, element=4),
        column=_axis(thread=32),
    ),
}


def _blocked_mfma_result_axes(
    block_count: int, row_count: int
) -> tuple[MatrixFragmentAxisLayout | None, ...]:
    axes = _BLOCKED_MFMA_RESULT_AXES.get((block_count, row_count))
    if axes is not None:
        return axes
    raise ValueError(f"unsupported blocked MFMA result shape {block_count}x{row_count}")


def _blocked_mfma_layout(
    key: str,
    *,
    block_count: int,
    row_count: int,
    reduction_count: int,
    source_payload_element_count: int,
    source_element_bit_count: int,
    result_payload_element_count: int,
    result_element_bit_count: int,
) -> AmdgpuMatrixFragmentLayout:
    lhs = _role(
        "lhs",
        source_payload_element_count,
        source_element_bit_count,
        _axes(
            block=_axis(thread=block_count, stride=row_count),
            row=_axis(thread=row_count),
            reduction=_axis(element=reduction_count),
        ),
    )
    rhs = _role(
        "rhs",
        source_payload_element_count,
        source_element_bit_count,
        _axes(
            block=_axis(thread=block_count, stride=row_count),
            column=_axis(thread=row_count),
            reduction=_axis(element=reduction_count),
        ),
    )
    result_axes = _blocked_mfma_result_axes(block_count, row_count)
    return AmdgpuMatrixFragmentLayout(
        key=key,
        wave_size=64,
        tile_shape=(block_count, row_count, row_count, reduction_count),
        lhs=lhs,
        rhs=rhs,
        accumulator=_role(
            "accumulator",
            result_payload_element_count,
            result_element_bit_count,
            result_axes,
        ),
        result=_role(
            "result",
            result_payload_element_count,
            result_element_bit_count,
            result_axes,
        ),
    )


def _blocked_f64_mfma_layout(key: str) -> AmdgpuMatrixFragmentLayout:
    lhs = _role(
        "lhs",
        1,
        64,
        _axes(
            block=_axis(thread=4, stride=4),
            row=_axis(thread=4),
            reduction=_axis(thread=4, stride=16),
        ),
    )
    rhs = _role(
        "rhs",
        1,
        64,
        _axes(
            block=_axis(thread=4, stride=4),
            column=_axis(thread=4),
            reduction=_axis(thread=4, stride=16),
        ),
    )
    result_axes = _axes(
        block=_axis(thread=4, stride=4),
        row=_axis(thread=4, stride=16),
        column=_axis(thread=4),
    )
    return AmdgpuMatrixFragmentLayout(
        key=key,
        wave_size=64,
        tile_shape=(4, 4, 4, 4),
        lhs=lhs,
        rhs=rhs,
        accumulator=_role("accumulator", 1, 64, result_axes),
        result=_role("result", 1, 64, result_axes),
    )


def _rdna3_layout(
    key: str,
    *,
    wave_size: int,
    source_payload_element_count: int,
    source_element_bit_count: int,
    result_element_bit_count: int,
    result_payload_element_count: int,
    result_coordinate_stride: int,
) -> AmdgpuMatrixFragmentLayout:
    source_axes_lhs = _axes(
        row=_axis(thread=16),
        reduction=_axis(element=source_payload_element_count),
    )
    source_axes_rhs = _axes(
        column=_axis(thread=16),
        reduction=_axis(element=source_payload_element_count),
    )
    result_coordinate_count = result_payload_element_count // result_coordinate_stride
    result_axes = _axes(
        row=_axis(
            outer=result_coordinate_count,
            thread=wave_size // 16,
            stride=16,
        ),
        column=_axis(thread=16),
    )
    return AmdgpuMatrixFragmentLayout(
        key=key,
        wave_size=wave_size,
        tile_shape=(1, 16, 16, 16),
        lhs=_role(
            "lhs",
            source_payload_element_count,
            source_element_bit_count,
            source_axes_lhs,
        ),
        rhs=_role(
            "rhs",
            source_payload_element_count,
            source_element_bit_count,
            source_axes_rhs,
        ),
        accumulator=_role(
            "accumulator",
            result_payload_element_count,
            result_element_bit_count,
            result_axes,
            coordinate_element_stride=result_coordinate_stride,
        ),
        result=_role(
            "result",
            result_payload_element_count,
            result_element_bit_count,
            result_axes,
            coordinate_element_stride=result_coordinate_stride,
        ),
    )


def _validate_role(
    layout: AmdgpuMatrixFragmentLayout, role: MatrixFragmentRoleLayout
) -> None:
    if len(role.axes) != len(_AXIS_NAMES):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            f"{len(role.axes)} semantic axes"
        )
    expected_axis_names = _ROLE_AXIS_NAMES[role.role]
    if layout.tile_shape[0] > 1:
        expected_axis_names = expected_axis_names | {"block"}
    actual_axis_names = {
        axis_name
        for axis_name, axis in zip(_AXIS_NAMES, role.axes, strict=True)
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
    if (
        role.coordinate_element_offset < 0
        or role.coordinate_element_offset > 0xFFFF
        or role.coordinate_element_stride <= 0
        or role.coordinate_element_stride > 0xFFFF
    ):
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' has "
            "an invalid coordinate element mapping"
        )
    payload_elements_per_register = (
        32 // role.element_bit_count
        if role.element_bit_count <= 32 and 32 % role.element_bit_count == 0
        else 0
    )
    if role.coordinate_element_offset != 0 or (
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
        reduction_axis = _AXIS_NAMES.index("reduction")
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
        role.coordinate_element_offset
        + (role.coordinate_element_count - 1) * role.coordinate_element_stride
    )
    if last_payload_element >= role.payload_element_count:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' maps "
            f"coordinate element {role.coordinate_element_count - 1} to "
            f"payload element {last_payload_element} outside "
            f"{role.payload_element_count} elements"
        )
    thread_element_count = 1
    for axis_index, (axis_name, axis) in enumerate(
        zip(_AXIS_NAMES, role.axes, strict=True)
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

    if len(layout.tile_shape) != len(_AXIS_NAMES):
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


def layout_roles(
    layout: AmdgpuMatrixFragmentLayout,
) -> tuple[MatrixFragmentRoleLayout, ...]:
    return (layout.lhs, layout.rhs, layout.accumulator, layout.result)


def role_coordinate(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
    lane: int,
    payload_element_index: int,
) -> tuple[int | None, ...] | None:
    """Returns the logical coordinate stored at one physical role location."""

    if (
        role.reduction_group is not None
        or lane < 0
        or lane >= layout.wave_size
        or payload_element_index < role.coordinate_element_offset
        or payload_element_index >= role.payload_element_count
    ):
        return None
    relative_payload_element = payload_element_index - role.coordinate_element_offset
    if relative_payload_element % role.coordinate_element_stride != 0:
        return None
    coordinate_element_index = (
        relative_payload_element // role.coordinate_element_stride
    )
    if coordinate_element_index >= role.coordinate_element_count:
        return None

    inner_element_count = prod(
        axis.element_count for axis in role.axes if axis is not None
    )
    outer_element_count = prod(
        axis.outer_count for axis in role.axes if axis is not None
    )
    inner_linear_index = coordinate_element_index % inner_element_count
    outer_linear_index = coordinate_element_index // inner_element_count
    inner_stride = inner_element_count
    outer_stride = outer_element_count
    coordinates: list[int | None] = []
    for axis_index, axis in enumerate(role.axes):
        if axis is None:
            coordinates.append(None)
            continue
        inner_stride //= axis.element_count
        outer_stride //= axis.outer_count
        element_coordinate = (inner_linear_index // inner_stride) % axis.element_count
        thread_coordinate = (lane // axis.thread_stride) % axis.thread_count
        outer_coordinate = (outer_linear_index // outer_stride) % axis.outer_count
        coordinate = (
            axis.element_count
            * (thread_coordinate + axis.thread_count * outer_coordinate)
            + element_coordinate
        )
        if coordinate >= layout.tile_shape[axis_index]:
            return None
        coordinates.append(coordinate)
    return tuple(coordinates)


def role_has_contiguous_lane_xor1_columns(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
) -> bool:
    """Returns whether lane xor 1 advances only the role's column axis."""

    row_axis = _AXIS_NAMES.index("row")
    column_axis = _AXIS_NAMES.index("column")
    reduction_axis = _AXIS_NAMES.index("reduction")
    column_layout = role.axes[column_axis]
    if (
        layout.wave_size % 2 != 0
        or role.axes[row_axis] is None
        or column_layout is None
        or role.axes[reduction_axis] is not None
        or role.coordinate_element_offset != 0
        or role.coordinate_element_count != role.payload_element_count
        or (role.payload_element_count > 1 and role.coordinate_element_stride != 1)
        or column_layout.element_count != 1
        or column_layout.thread_count % 2 != 0
        or column_layout.thread_stride != 1
    ):
        return False

    # The validated axis factorization makes the lane-pair relationship
    # algebraic. A stride-one column thread factor advances by one between an
    # even lane and its xor-one pair. Every other populated axis must have an
    # even thread stride so that the same pair remains within its lane factor.
    return all(
        axis_index == column_axis
        or axis is None
        or axis.thread_count == 1
        or axis.thread_stride % 2 == 0
        for axis_index, axis in enumerate(role.axes)
    )


AMDGPU_MATRIX_FRAGMENT_LAYOUTS: tuple[AmdgpuMatrixFragmentLayout, ...] = (
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_f16",
        wave_size=32,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=32,
        result_payload_element_count=8,
        result_coordinate_stride=1,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_bf16",
        wave_size=32,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=32,
        result_payload_element_count=8,
        result_coordinate_stride=1,
    ),
    _single_tile_layout(
        "cdna_mfma_f32_16x16x16_f16",
        wave_size=64,
        row_count=16,
        column_count=16,
        reduction_count=16,
        lhs_payload_element_count=4,
        rhs_payload_element_count=4,
        lhs_element_bit_count=16,
        rhs_element_bit_count=16,
        result_payload_element_count=4,
    ),
    _single_tile_layout(
        "cdna_mfma_f32_16x16x16_bf16",
        wave_size=64,
        row_count=16,
        column_count=16,
        reduction_count=16,
        lhs_payload_element_count=4,
        rhs_payload_element_count=4,
        lhs_element_bit_count=16,
        rhs_element_bit_count=16,
        result_payload_element_count=4,
    ),
    _single_tile_layout(
        "cdna_mfma_f32_16x16x4_f32",
        wave_size=64,
        row_count=16,
        column_count=16,
        reduction_count=4,
        lhs_payload_element_count=1,
        rhs_payload_element_count=1,
        lhs_element_bit_count=32,
        rhs_element_bit_count=32,
        result_payload_element_count=4,
    ),
    _single_tile_layout(
        "cdna_mfma_f64_16x16x4_f64",
        wave_size=64,
        row_count=16,
        column_count=16,
        reduction_count=4,
        lhs_payload_element_count=1,
        rhs_payload_element_count=1,
        lhs_element_bit_count=64,
        rhs_element_bit_count=64,
        result_payload_element_count=4,
        result_element_bit_count=64,
    ),
    *(
        _blocked_mfma_layout(
            key,
            block_count=block_count,
            row_count=row_count,
            reduction_count=reduction_count,
            source_payload_element_count=source_payload_element_count,
            source_element_bit_count=source_element_bit_count,
            result_payload_element_count=result_payload_element_count,
            result_element_bit_count=result_element_bit_count,
        )
        for (
            key,
            block_count,
            row_count,
            reduction_count,
            source_payload_element_count,
            source_element_bit_count,
            result_payload_element_count,
            result_element_bit_count,
        ) in (
            ("cdna_mfma_f32_4x4x1_f32_16b", 16, 4, 1, 1, 32, 4, 32),
            ("cdna_mfma_f32_4x4x2_bf16_16b", 16, 4, 2, 2, 16, 4, 32),
            ("cdna_mfma_f32_4x4x4_packed16_16b", 16, 4, 4, 4, 16, 4, 32),
            ("cdna_mfma_i32_4x4x4_i8_16b", 16, 4, 4, 4, 8, 4, 32),
            ("cdna_mfma_f32_16x16x1_f32_4b", 4, 16, 1, 1, 32, 16, 32),
            ("cdna_mfma_f32_16x16x2_bf16_4b", 4, 16, 2, 2, 16, 16, 32),
            ("cdna_mfma_f32_16x16x4_packed16_4b", 4, 16, 4, 4, 16, 16, 32),
            ("cdna_mfma_i32_16x16x4_i8_4b", 4, 16, 4, 4, 8, 16, 32),
            ("cdna_mfma_f32_32x32x1_f32_2b", 2, 32, 1, 1, 32, 32, 32),
            ("cdna_mfma_f32_32x32x2_bf16_2b", 2, 32, 2, 2, 16, 32, 32),
            ("cdna_mfma_f32_32x32x4_packed16_2b", 2, 32, 4, 4, 16, 32, 32),
            ("cdna_mfma_i32_32x32x4_i8_2b", 2, 32, 4, 4, 8, 32, 32),
        )
    ),
    _blocked_f64_mfma_layout("cdna_mfma_f64_4x4x4_f64_4b"),
    _rdna3_layout(
        "rdna3_wmmar3_f16_16x16x16_f16",
        wave_size=32,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=16,
        result_payload_element_count=16,
        result_coordinate_stride=2,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_bf16_16x16x16_bf16",
        wave_size=32,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=16,
        result_payload_element_count=16,
        result_coordinate_stride=2,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_f16_w64",
        wave_size=64,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=32,
        result_payload_element_count=4,
        result_coordinate_stride=1,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_bf16_w64",
        wave_size=64,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=32,
        result_payload_element_count=4,
        result_coordinate_stride=1,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f16_16x16x16_f16_w64",
        wave_size=64,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=16,
        result_payload_element_count=8,
        result_coordinate_stride=2,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_bf16_16x16x16_bf16_w64",
        wave_size=64,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_element_bit_count=16,
        result_payload_element_count=8,
        result_coordinate_stride=2,
    ),
    *(
        _rdna3_layout(
            key,
            wave_size=wave_size,
            source_payload_element_count=16,
            source_element_bit_count=source_element_bit_count,
            result_element_bit_count=32,
            result_payload_element_count=result_payload_element_count,
            result_coordinate_stride=1,
        )
        for (
            key,
            wave_size,
            source_element_bit_count,
            result_payload_element_count,
        ) in (
            ("rdna3_wmmar3_i32_16x16x16_iu8", 32, 8, 8),
            ("rdna3_wmmar3_i32_16x16x16_iu8_w64", 64, 8, 4),
            ("rdna3_wmmar3_i32_16x16x16_iu4", 32, 4, 8),
            ("rdna3_wmmar3_i32_16x16x16_iu4_w64", 64, 4, 4),
        )
    ),
    _single_tile_layout(
        "rdna4_wmma_f16_16x16x16_f16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=16,
        lhs_payload_element_count=8,
        rhs_payload_element_count=8,
        lhs_element_bit_count=16,
        rhs_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    _single_tile_layout(
        "rdna4_wmma_bf16_16x16x16_bf16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=16,
        lhs_payload_element_count=8,
        rhs_payload_element_count=8,
        lhs_element_bit_count=16,
        rhs_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    _single_tile_layout(
        "rdna4_wmma_f16_16x16x32_f16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=32,
        lhs_payload_element_count=16,
        rhs_payload_element_count=16,
        lhs_element_bit_count=16,
        rhs_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    _single_tile_layout(
        "rdna4_wmma_bf16_16x16x32_bf16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=32,
        lhs_payload_element_count=16,
        rhs_payload_element_count=16,
        lhs_element_bit_count=16,
        rhs_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    *(
        _single_tile_layout(
            key,
            wave_size=32,
            row_count=16,
            column_count=16,
            reduction_count=reduction_count,
            lhs_payload_element_count=source_payload_element_count,
            rhs_payload_element_count=source_payload_element_count,
            lhs_element_bit_count=source_element_bit_count,
            rhs_element_bit_count=source_element_bit_count,
            result_payload_element_count=8,
        )
        for (
            key,
            reduction_count,
            source_payload_element_count,
            source_element_bit_count,
        ) in (
            ("rdna4_wmma_f32_16x16x16_f16", 16, 8, 16),
            ("rdna4_wmma_f32_16x16x16_bf16", 16, 8, 16),
            ("rdna4_wmma_f32_16x16x32_f16", 32, 16, 16),
            ("rdna4_wmma_f32_16x16x32_bf16", 32, 16, 16),
            ("rdna4_wmma_f32_16x16x4_f32", 4, 2, 32),
            ("rdna4_wmma_f32_16x16x16_packed8", 16, 8, 8),
            ("rdna4_wmma_f32_16x16x64_packed8", 64, 32, 8),
            ("rdna4_wmma_f32_16x16x128_packed8", 128, 64, 8),
            ("rdna4_wmma_i32_16x16x16_iu8", 16, 8, 8),
            ("rdna4_wmma_i32_16x16x16_iu4", 16, 8, 4),
            ("rdna4_wmma_i32_16x16x32_iu4", 32, 16, 4),
            ("rdna4_wmma_i32_16x16x64_iu8", 64, 32, 8),
        )
    ),
    *(
        _single_tile_layout(
            key,
            wave_size=64,
            row_count=16,
            column_count=16,
            reduction_count=reduction_count,
            lhs_payload_element_count=source_payload_element_count,
            rhs_payload_element_count=source_payload_element_count,
            lhs_element_bit_count=source_element_bit_count,
            rhs_element_bit_count=source_element_bit_count,
            result_payload_element_count=result_payload_element_count,
            result_element_bit_count=result_element_bit_count,
            lhs_lane_replication=lane_replication,
            rhs_lane_replication=lane_replication,
        )
        for (
            key,
            reduction_count,
            source_payload_element_count,
            source_element_bit_count,
            result_payload_element_count,
            result_element_bit_count,
            lane_replication,
        ) in (
            ("rdna4_wmma_f32_16x16x16_f16_w64", 16, 4, 16, 4, 32, 1),
            ("rdna4_wmma_f32_16x16x16_bf16_w64", 16, 4, 16, 4, 32, 1),
            ("rdna4_wmma_f16_16x16x16_f16_w64", 16, 4, 16, 4, 16, 1),
            ("rdna4_wmma_bf16_16x16x16_bf16_w64", 16, 4, 16, 4, 16, 1),
            ("rdna4_wmma_f32_16x16x16_packed8_w64", 16, 4, 8, 4, 32, 1),
            ("rdna4_wmma_i32_16x16x16_iu8_w64", 16, 4, 8, 4, 32, 1),
            ("rdna4_wmma_i32_16x16x16_iu4_w64", 16, 8, 4, 4, 32, 2),
            ("rdna4_wmma_i32_16x16x32_iu4_w64", 32, 8, 4, 4, 32, 1),
        )
    ),
    *(
        _single_tile_layout(
            (f"gfx125x_wmma_f32_16x16x128_{lhs_format.token}_{rhs_format.token}"),
            wave_size=32,
            row_count=16,
            column_count=16,
            reduction_count=128,
            lhs_payload_element_count=64,
            rhs_payload_element_count=64,
            lhs_element_bit_count=lhs_format.element_bit_count,
            rhs_element_bit_count=rhs_format.element_bit_count,
            result_payload_element_count=8,
        )
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
        for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
    ),
    *(
        _single_tile_layout(
            key,
            wave_size=64,
            row_count=row_count,
            column_count=column_count,
            reduction_count=reduction_count,
            lhs_payload_element_count=source_payload_element_count,
            rhs_payload_element_count=source_payload_element_count,
            lhs_element_bit_count=source_element_bit_count,
            rhs_element_bit_count=source_element_bit_count,
            result_payload_element_count=result_payload_element_count,
        )
        for (
            key,
            row_count,
            column_count,
            reduction_count,
            source_payload_element_count,
            source_element_bit_count,
            result_payload_element_count,
        ) in (
            ("cdna_mfma_f32_16x16x32_f16", 16, 16, 32, 8, 16, 4),
            ("cdna_mfma_f32_16x16x32_bf16", 16, 16, 32, 8, 16, 4),
            ("cdna_mfma_f32_16x16x32_packed8", 16, 16, 32, 8, 8, 4),
            ("cdna_mfma_f32_32x32x16_f16", 32, 32, 16, 8, 16, 16),
            ("cdna_mfma_f32_32x32x16_bf16", 32, 32, 16, 8, 16, 16),
            ("cdna_mfma_f32_32x32x16_packed8", 32, 32, 16, 8, 8, 16),
            ("cdna_mfma_f32_16x16x8_packed16", 16, 16, 8, 2, 16, 4),
            ("cdna_mfma_f32_16x16x8_xf32", 16, 16, 8, 2, 32, 4),
            ("cdna_mfma_f32_32x32x4_packed16", 32, 32, 4, 2, 16, 16),
            ("cdna_mfma_f32_32x32x8_packed16", 32, 32, 8, 4, 16, 16),
            ("cdna_mfma_f32_32x32x4_xf32", 32, 32, 4, 2, 32, 16),
            ("cdna_mfma_f32_32x32x2_f32", 32, 32, 2, 1, 32, 16),
        )
    ),
    *(
        _single_tile_layout(
            key,
            wave_size=wave_size,
            row_count=row_count,
            column_count=column_count,
            reduction_count=reduction_count,
            lhs_payload_element_count=lhs_payload_element_count,
            rhs_payload_element_count=rhs_payload_element_count,
            lhs_element_bit_count=source_element_bit_count,
            rhs_element_bit_count=source_element_bit_count,
            result_payload_element_count=result_payload_element_count,
            result_element_bit_count=result_element_bit_count,
            lhs_reduction_group=_STRUCTURED_2_TO_4_REDUCTION_GROUP,
            family=family,
        )
        for (
            family,
            key,
            wave_size,
            row_count,
            column_count,
            reduction_count,
            lhs_payload_element_count,
            rhs_payload_element_count,
            source_element_bit_count,
            result_payload_element_count,
            result_element_bit_count,
        ) in (
            (
                "smfmac",
                "cdna_smfmac_32bit_16x16x32_packed16",
                64,
                16,
                16,
                32,
                4,
                8,
                16,
                4,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_16x16x64_packed8",
                64,
                16,
                16,
                64,
                8,
                16,
                8,
                4,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_16x16x64_packed16",
                64,
                16,
                16,
                64,
                8,
                16,
                16,
                4,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_16x16x128_packed8",
                64,
                16,
                16,
                128,
                16,
                32,
                8,
                4,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_32x32x16_packed16",
                64,
                32,
                32,
                16,
                4,
                8,
                16,
                16,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_32x32x32_packed8",
                64,
                32,
                32,
                32,
                8,
                16,
                8,
                16,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_32x32x32_packed16",
                64,
                32,
                32,
                32,
                8,
                16,
                16,
                16,
                32,
            ),
            (
                "smfmac",
                "cdna_smfmac_32bit_32x32x64_packed8",
                64,
                32,
                32,
                64,
                16,
                32,
                8,
                16,
                32,
            ),
            (
                "swmmac",
                "rdna4_swmmac_32bit_16x16x32_packed16",
                32,
                16,
                16,
                32,
                8,
                16,
                16,
                8,
                32,
            ),
            (
                "swmmac",
                "rdna4_swmmac_16bit_16x16x32_packed16",
                32,
                16,
                16,
                32,
                8,
                16,
                16,
                8,
                16,
            ),
            (
                "swmmac",
                "rdna4_swmmac_32bit_16x16x32_packed8",
                32,
                16,
                16,
                32,
                8,
                16,
                8,
                8,
                32,
            ),
            (
                "swmmac",
                "rdna4_swmmac_32bit_16x16x32_packed4",
                32,
                16,
                16,
                32,
                8,
                16,
                4,
                8,
                32,
            ),
            (
                "swmmac",
                "rdna4_swmmac_32bit_16x16x64_packed4",
                32,
                16,
                16,
                64,
                16,
                32,
                4,
                8,
                32,
            ),
            (
                "swmmac",
                "gfx1250_swmmac_32bit_16x16x64_packed16",
                32,
                16,
                16,
                64,
                16,
                32,
                16,
                8,
                32,
            ),
            (
                "swmmac",
                "gfx1250_swmmac_16bit_16x16x64_packed16",
                32,
                16,
                16,
                64,
                16,
                32,
                16,
                8,
                16,
            ),
            (
                "swmmac",
                "gfx1250_swmmac_32bit_16x16x128_packed8",
                32,
                16,
                16,
                128,
                32,
                64,
                8,
                8,
                32,
            ),
            (
                "swmmac",
                "gfx1250_swmmac_16bit_16x16x128_packed8",
                32,
                16,
                16,
                128,
                32,
                64,
                8,
                8,
                16,
            ),
        )
    ),
    *(
        _single_tile_layout(
            key,
            wave_size=64,
            row_count=16,
            column_count=16,
            reduction_count=reduction_count,
            lhs_payload_element_count=lhs_payload_element_count,
            rhs_payload_element_count=rhs_payload_element_count,
            lhs_element_bit_count=source_element_bit_count,
            rhs_element_bit_count=source_element_bit_count,
            result_payload_element_count=result_payload_element_count,
            result_element_bit_count=result_element_bit_count,
            lhs_reduction_group=_STRUCTURED_2_TO_4_REDUCTION_GROUP,
            lhs_lane_replication=lhs_lane_replication,
            family="swmmac",
        )
        for (
            key,
            reduction_count,
            lhs_payload_element_count,
            rhs_payload_element_count,
            source_element_bit_count,
            result_payload_element_count,
            result_element_bit_count,
            lhs_lane_replication,
        ) in (
            ("rdna4_swmmac_32bit_16x16x32_packed16_w64", 32, 4, 8, 16, 4, 32, 1),
            ("rdna4_swmmac_16bit_16x16x32_packed16_w64", 32, 4, 8, 16, 4, 16, 1),
            ("rdna4_swmmac_32bit_16x16x32_packed8_w64", 32, 4, 8, 8, 4, 32, 1),
            ("rdna4_swmmac_32bit_16x16x32_packed4_w64", 32, 8, 8, 4, 4, 32, 2),
            ("rdna4_swmmac_32bit_16x16x64_packed4_w64", 64, 8, 16, 4, 4, 32, 1),
        )
    ),
)


AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY = {
    layout.key: layout for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS
}

if len(AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY) != len(AMDGPU_MATRIX_FRAGMENT_LAYOUTS):
    raise ValueError("duplicate AMDGPU matrix fragment layout key")

for _layout_row in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
    validate_matrix_fragment_layout(_layout_row)
