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

_AXIS_NAMES = ("block", "row", "column", "reduction")
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
class MatrixFragmentRoleLayout:
    """One matrix role's payload storage and semantic coordinate layout."""

    role: str
    payload_element_count: int
    element_bit_count: int
    coordinate_element_offset: int
    coordinate_element_stride: int
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
) -> MatrixFragmentRoleLayout:
    return MatrixFragmentRoleLayout(
        role=role,
        payload_element_count=payload_element_count,
        element_bit_count=element_bit_count,
        coordinate_element_offset=coordinate_element_offset,
        coordinate_element_stride=coordinate_element_stride,
        axes=axes,
    )


def _single_tile_layout(
    key: str,
    *,
    wave_size: int,
    row_count: int,
    column_count: int,
    reduction_count: int,
    source_payload_element_count: int,
    source_element_bit_count: int,
    result_payload_element_count: int,
    result_element_bit_count: int = 32,
) -> AmdgpuMatrixFragmentLayout:
    row_thread_count = wave_size // row_count
    column_thread_count = wave_size // column_count
    lhs = _role(
        "lhs",
        source_payload_element_count,
        source_element_bit_count,
        _axes(
            row=_axis(thread=row_count),
            reduction=_axis(
                thread=row_thread_count,
                stride=row_count,
                element=source_payload_element_count,
            ),
        ),
    )
    rhs = _role(
        "rhs",
        source_payload_element_count,
        source_element_bit_count,
        _axes(
            column=_axis(thread=column_count),
            reduction=_axis(
                thread=column_thread_count,
                stride=column_count,
                element=source_payload_element_count,
            ),
        ),
    )
    result_axes = _axes(
        row=_axis(
            thread=column_thread_count,
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
    )


def _rdna3_layout(
    key: str,
    *,
    wave_size: int,
    result_element_bit_count: int,
    result_payload_element_count: int,
    result_coordinate_stride: int,
) -> AmdgpuMatrixFragmentLayout:
    source_axes_lhs = _axes(row=_axis(thread=16), reduction=_axis(element=16))
    source_axes_rhs = _axes(column=_axis(thread=16), reduction=_axis(element=16))
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
        lhs=_role("lhs", 16, 16, source_axes_lhs),
        rhs=_role("rhs", 16, 16, source_axes_rhs),
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
        expected_extent = layout.tile_shape[axis_index]
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
    if role.register_count > 0xFFFF:
        raise ValueError(
            f"matrix fragment layout '{layout.key}' role '{role.role}' uses "
            f"{role.register_count} payload registers"
        )


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


def layout_roles(
    layout: AmdgpuMatrixFragmentLayout,
) -> tuple[MatrixFragmentRoleLayout, ...]:
    return (layout.lhs, layout.rhs, layout.accumulator, layout.result)


def _role_coordinate(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
    lane: int,
    payload_element_index: int,
) -> tuple[int | None, ...] | None:
    if (
        lane < 0
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
    if (
        layout.wave_size % 2 != 0
        or role.axes[row_axis] is None
        or role.axes[column_axis] is None
        or role.axes[reduction_axis] is not None
    ):
        return False
    for payload_element_index in range(role.payload_element_count):
        for lane in range(0, layout.wave_size, 2):
            coordinate = _role_coordinate(layout, role, lane, payload_element_index)
            paired_coordinate = _role_coordinate(
                layout, role, lane ^ 1, payload_element_index
            )
            if coordinate is None or paired_coordinate is None:
                return False
            for axis_index in range(len(_AXIS_NAMES)):
                if axis_index == column_axis:
                    continue
                if coordinate[axis_index] != paired_coordinate[axis_index]:
                    return False
            column = coordinate[column_axis]
            paired_column = paired_coordinate[column_axis]
            if column is None or paired_column is None or paired_column != column + 1:
                return False
    return True


AMDGPU_MATRIX_FRAGMENT_LAYOUTS: tuple[AmdgpuMatrixFragmentLayout, ...] = (
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_f16",
        wave_size=32,
        result_element_bit_count=32,
        result_payload_element_count=8,
        result_coordinate_stride=1,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_bf16",
        wave_size=32,
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
        source_payload_element_count=4,
        source_element_bit_count=16,
        result_payload_element_count=4,
    ),
    _single_tile_layout(
        "cdna_mfma_f32_16x16x16_bf16",
        wave_size=64,
        row_count=16,
        column_count=16,
        reduction_count=16,
        source_payload_element_count=4,
        source_element_bit_count=16,
        result_payload_element_count=4,
    ),
    _single_tile_layout(
        "cdna_mfma_f32_16x16x4_f32",
        wave_size=64,
        row_count=16,
        column_count=16,
        reduction_count=4,
        source_payload_element_count=1,
        source_element_bit_count=32,
        result_payload_element_count=4,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f16_16x16x16_f16",
        wave_size=32,
        result_element_bit_count=16,
        result_payload_element_count=16,
        result_coordinate_stride=2,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_bf16_16x16x16_bf16",
        wave_size=32,
        result_element_bit_count=16,
        result_payload_element_count=16,
        result_coordinate_stride=2,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_f16_w64",
        wave_size=64,
        result_element_bit_count=32,
        result_payload_element_count=4,
        result_coordinate_stride=1,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f32_16x16x16_bf16_w64",
        wave_size=64,
        result_element_bit_count=32,
        result_payload_element_count=4,
        result_coordinate_stride=1,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_f16_16x16x16_f16_w64",
        wave_size=64,
        result_element_bit_count=16,
        result_payload_element_count=8,
        result_coordinate_stride=2,
    ),
    _rdna3_layout(
        "rdna3_wmmar3_bf16_16x16x16_bf16_w64",
        wave_size=64,
        result_element_bit_count=16,
        result_payload_element_count=8,
        result_coordinate_stride=2,
    ),
    _single_tile_layout(
        "rdna4_wmma_f16_16x16x16_f16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=16,
        source_payload_element_count=8,
        source_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    _single_tile_layout(
        "rdna4_wmma_bf16_16x16x16_bf16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=16,
        source_payload_element_count=8,
        source_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    _single_tile_layout(
        "rdna4_wmma_f16_16x16x32_f16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=32,
        source_payload_element_count=16,
        source_element_bit_count=16,
        result_payload_element_count=8,
        result_element_bit_count=16,
    ),
    _single_tile_layout(
        "rdna4_wmma_bf16_16x16x32_bf16",
        wave_size=32,
        row_count=16,
        column_count=16,
        reduction_count=32,
        source_payload_element_count=16,
        source_element_bit_count=16,
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
            source_payload_element_count=source_payload_element_count,
            source_element_bit_count=source_element_bit_count,
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
        )
    ),
    *(
        _single_tile_layout(
            key,
            wave_size=64,
            row_count=row_count,
            column_count=column_count,
            reduction_count=reduction_count,
            source_payload_element_count=source_payload_element_count,
            source_element_bit_count=source_element_bit_count,
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
)


AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY = {
    layout.key: layout for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS
}

if len(AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY) != len(AMDGPU_MATRIX_FRAGMENT_LAYOUTS):
    raise ValueError("duplicate AMDGPU matrix fragment layout key")

for _layout_row in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
    validate_matrix_fragment_layout(_layout_row)
