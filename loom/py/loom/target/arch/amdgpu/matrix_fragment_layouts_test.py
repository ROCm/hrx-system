# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS,
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
    MatrixFragmentAxisLayout,
    MatrixFragmentReductionGroup,
    role_coordinate,
    role_has_contiguous_lane_xor1_columns,
    validate_matrix_fragment_layout,
)


def test_lane_xor1_column_property_accounts_for_payload_padding() -> None:
    f32_layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f32_16x16x16_f16"]
    f16_layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f16_16x16x16_f16"]

    assert role_has_contiguous_lane_xor1_columns(f32_layout, f32_layout.result)
    assert not role_has_contiguous_lane_xor1_columns(f16_layout, f16_layout.result)


def test_validation_rejects_missing_and_extraneous_role_axes() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f32_16x16x16_f16"]
    result_axes = layout.result.axes
    missing_row_layout = replace(
        layout,
        result=replace(
            layout.result,
            axes=(result_axes[0], None, result_axes[2], result_axes[3]),
        ),
    )
    extraneous_block_layout = replace(
        layout,
        result=replace(
            layout.result,
            axes=(
                MatrixFragmentAxisLayout(1, 1, 1, 1),
                result_axes[1],
                result_axes[2],
                result_axes[3],
            ),
        ),
    )

    for malformed_layout in (missing_row_layout, extraneous_block_layout):
        with pytest.raises(ValueError, match="semantic axes"):
            validate_matrix_fragment_layout(malformed_layout)


def test_compressed_reduction_layout_separates_storage_and_logical_k() -> None:
    dense = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f32_16x16x32_packed8"]
    rhs_reduction = dense.rhs.axes[3]
    assert rhs_reduction is not None
    sparse = replace(
        dense,
        key="test_sparse_16x16x64_packed8",
        tile_shape=(*dense.tile_shape[:3], 64),
        lhs=replace(
            dense.lhs,
            reduction_group=MatrixFragmentReductionGroup(2, 4),
        ),
        rhs=replace(
            dense.rhs,
            payload_element_count=16,
            axes=(*dense.rhs.axes[:3], replace(rhs_reduction, element_count=16)),
        ),
    )

    validate_matrix_fragment_layout(sparse)
    assert role_coordinate(sparse, sparse.lhs, 0, 0) is None
    assert role_coordinate(sparse, sparse.rhs, 0, 0) == (None, None, 0, 0)


def test_validation_rejects_noncompressing_reduction_group() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f32_16x16x32_packed8"]
    malformed = replace(
        layout,
        lhs=replace(
            layout.lhs,
            reduction_group=MatrixFragmentReductionGroup(4, 4),
        ),
    )

    with pytest.raises(ValueError, match="reduction storage group"):
        validate_matrix_fragment_layout(malformed)


def test_result_to_lhs_partial_transpose_preserves_coordinates() -> None:
    checked_layout_count = 0
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        source_role = layout.result
        destination_role = layout.lhs
        if (
            source_role.element_bit_count != 32
            or destination_role.element_bit_count != 16
            or layout.tile_shape[2] != layout.tile_shape[3]
            or not role_has_contiguous_lane_xor1_columns(layout, source_role)
        ):
            continue
        checked_layout_count += 1

        source_locations: dict[tuple[int | None, ...], list[tuple[int, int]]] = (
            defaultdict(list)
        )
        for lane in range(layout.wave_size):
            for element_index in range(source_role.payload_element_count):
                coordinate = role_coordinate(layout, source_role, lane, element_index)
                if coordinate is not None:
                    source_locations[coordinate].append((lane, element_index))

        source_register_count = source_role.register_count
        destination_register_count = destination_role.register_count
        transpose_bit_count = min(
            source_register_count.bit_length() - 1,
            destination_register_count.bit_length() - 1,
        )
        transpose_mask = (1 << transpose_bit_count) - 1
        candidate_count = source_register_count >> transpose_bit_count
        lane_group_count = layout.wave_size // 16

        state = [
            [[None for _ in range(16)] for _ in range(source_register_count)]
            for _ in range(lane_group_count)
        ]
        for lane_group in range(lane_group_count):
            for source_register in range(source_register_count):
                for local_lane in range(0, 16, 2):
                    lane = lane_group * 16 + local_lane
                    state[lane_group][source_register][local_lane] = (
                        role_coordinate(layout, source_role, lane, source_register),
                        role_coordinate(layout, source_role, lane + 1, source_register),
                    )

        for bit_index in range(transpose_bit_count):
            register_xor = 1 << bit_index
            lane_xor = register_xor << 1
            next_state = [
                [register_lanes.copy() for register_lanes in lane_group]
                for lane_group in state
            ]
            for lane_group in range(lane_group_count):
                for source_register in range(source_register_count):
                    for local_lane in range(0, 16, 2):
                        if bool(source_register & register_xor) == bool(
                            local_lane & lane_xor
                        ):
                            continue
                        next_state[lane_group][source_register][local_lane] = state[
                            lane_group
                        ][source_register ^ register_xor][local_lane ^ lane_xor]
            state = next_state

        for destination_lane in range(layout.wave_size):
            for destination_register in range(destination_register_count):
                destination_coordinates = [
                    role_coordinate(
                        layout,
                        destination_role,
                        destination_lane,
                        destination_register * 2 + element_index,
                    )
                    for element_index in range(2)
                ]
                assert all(
                    coordinate is not None for coordinate in destination_coordinates
                )
                source_coordinates = [
                    (coordinate[0], coordinate[1], coordinate[3], coordinate[2])
                    for coordinate in destination_coordinates
                    if coordinate is not None
                ]
                source_positions = [
                    source_locations[coordinate] for coordinate in source_coordinates
                ]
                assert all(len(positions) == 1 for positions in source_positions)
                (
                    (source_lane_0, source_register_0),
                    (
                        source_lane_1,
                        source_register_1,
                    ),
                ) = (source_positions[0][0], source_positions[1][0])
                assert source_register_0 == source_register_1
                assert source_lane_0 // 16 == source_lane_1 // 16
                assert source_lane_0 % 2 == 0
                assert source_lane_1 == source_lane_0 + 1

                source_pair = (source_lane_0 % 16) // 2
                transposed_register = (
                    (source_register_0 >> transpose_bit_count) << transpose_bit_count
                ) | (source_pair & transpose_mask)
                transposed_lane = (source_lane_0 // 16) * 16 + 2 * (
                    (source_register_0 & transpose_mask)
                    | ((source_pair >> transpose_bit_count) << transpose_bit_count)
                )
                candidate_base = destination_register & transpose_mask
                candidate_registers = {
                    candidate_base + (candidate << transpose_bit_count)
                    for candidate in range(candidate_count)
                }
                assert transposed_register in candidate_registers
                assert state[transposed_lane // 16][transposed_register][
                    transposed_lane % 16
                ] == tuple(source_coordinates)

    assert checked_layout_count > 0
