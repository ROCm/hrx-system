# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU matrix fragment ISA layout catalog."""

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.amdgpu.matrix_formats import (
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)
from loom.target.arch.amdgpu.matrix_fragment_layout import (
    AmdgpuMatrixFragmentLayout,
    MatrixFragmentAxisLayout,
    MatrixFragmentReductionGroup,
    MatrixFragmentRoleLayout,
    validate_matrix_fragment_layout,
)

_STRUCTURED_2_TO_4_REDUCTION_GROUP = MatrixFragmentReductionGroup(2, 4)


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
    coordinate_element_stride: int = 1,
    reduction_group: MatrixFragmentReductionGroup | None = None,
) -> MatrixFragmentRoleLayout:
    return MatrixFragmentRoleLayout(
        role=role,
        payload_element_count=payload_element_count,
        element_bit_count=element_bit_count,
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


def _rdna3_transposed_result_layout(
    key: str,
    *,
    canonical_key: str,
    source_payload_element_count: int,
) -> AmdgpuMatrixFragmentLayout:
    layout = _rdna3_layout(
        key,
        wave_size=32,
        source_payload_element_count=source_payload_element_count,
        source_element_bit_count=16,
        result_element_bit_count=32,
        result_payload_element_count=8,
        result_coordinate_stride=1,
    )
    transposed_result_axes = _axes(
        row=_axis(thread=16),
        column=_axis(outer=8, thread=2, stride=16),
    )
    return replace(
        layout,
        accumulator=replace(layout.accumulator, axes=transposed_result_axes),
        result=replace(layout.result, axes=transposed_result_axes),
        canonical_key=canonical_key,
        instruction_operand_order=("rhs", "lhs"),
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
    _rdna3_transposed_result_layout(
        "rdna3_wmmar3_f32_16x16x16_f16_transposed_result",
        canonical_key="rdna3_wmmar3_f32_16x16x16_f16",
        source_payload_element_count=16,
    ),
)


AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY = {
    layout.key: layout for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS
}

if len(AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY) != len(AMDGPU_MATRIX_FRAGMENT_LAYOUTS):
    raise ValueError("duplicate AMDGPU matrix fragment layout key")

for _layout_row in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
    validate_matrix_fragment_layout(_layout_row)
    if _layout_row.canonical_key is None:
        continue
    _canonical_layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY.get(
        _layout_row.canonical_key
    )
    if _canonical_layout is None or _canonical_layout.canonical_key is not None:
        raise ValueError(
            f"AMDGPU matrix fragment layout '{_layout_row.key}' names invalid "
            f"canonical layout '{_layout_row.canonical_key}'"
        )
    if _layout_row.tile_shape != _canonical_layout.tile_shape or tuple(
        (role.payload_element_count, role.element_bit_count)
        for role in (
            _layout_row.lhs,
            _layout_row.rhs,
            _layout_row.accumulator,
            _layout_row.result,
        )
    ) != tuple(
        (role.payload_element_count, role.element_bit_count)
        for role in (
            _canonical_layout.lhs,
            _canonical_layout.rhs,
            _canonical_layout.accumulator,
            _canonical_layout.result,
        )
    ):
        raise ValueError(
            f"AMDGPU matrix fragment layout '{_layout_row.key}' is not payload "
            f"compatible with canonical layout '{_layout_row.canonical_key}'"
        )
    if _layout_row.instruction_operand_order == ("rhs", "lhs"):
        canonical_lhs = _canonical_layout.lhs
        canonical_rhs = _canonical_layout.rhs
        # Reinterpreting canonical RHS storage as hardware LHS must encode
        # transpose(rhs), and the inverse reinterpretation must encode
        # transpose(lhs). The shared physical axis factorization proves both.
        canonical_result_axes = _canonical_layout.result.axes
        transposed_result_axes = (
            canonical_result_axes[0],
            canonical_result_axes[2],
            canonical_result_axes[1],
            canonical_result_axes[3],
        )
        source_roles_are_transpose_symmetric = (
            canonical_lhs.payload_element_count == canonical_rhs.payload_element_count
            and canonical_lhs.element_bit_count == canonical_rhs.element_bit_count
            and canonical_lhs.coordinate_element_stride
            == canonical_rhs.coordinate_element_stride
            and canonical_lhs.reduction_group == canonical_rhs.reduction_group
            and canonical_lhs.axes[0] == canonical_rhs.axes[0]
            and canonical_lhs.axes[1] == canonical_rhs.axes[2]
            and canonical_lhs.axes[3] == canonical_rhs.axes[3]
        )
        if (
            _canonical_layout.tile_shape[1] != _canonical_layout.tile_shape[2]
            or _layout_row.lhs != _canonical_layout.lhs
            or _layout_row.rhs != _canonical_layout.rhs
            or _canonical_layout.instruction_operand_order != ("lhs", "rhs")
            or not source_roles_are_transpose_symmetric
            or replace(_layout_row.accumulator, axes=_canonical_layout.accumulator.axes)
            != _canonical_layout.accumulator
            or replace(_layout_row.result, axes=_canonical_layout.result.axes)
            != _canonical_layout.result
            or _layout_row.accumulator.axes != transposed_result_axes
            or _layout_row.result.axes != transposed_result_axes
        ):
            raise ValueError(
                f"AMDGPU matrix fragment layout '{_layout_row.key}' does not "
                "pair its reversed inputs with an exact result transpose"
            )
