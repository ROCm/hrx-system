# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.matrix_fragment_layout import (
    AmdgpuMatrixFragmentLayout,
    MatrixFragmentAxisLayout,
    MatrixFragmentReductionGroup,
    MatrixFragmentRoleLayout,
    validate_matrix_fragment_layout,
)
from loom.target.arch.amdgpu.matrix_fragment_layout_adaptation import (
    matrix_fragment_native_contraction_facts,
    matrix_fragment_packed_element_axis,
    matrix_fragment_role_storage_coordinate_map,
)
from loom.target.arch.amdgpu.matrix_fragment_layout_recipes import (
    MatrixFragmentPackedB16PublicationProjection,
    matrix_fragment_packed_b16_publication_projection,
)
from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
)
from loom.target.native_layout_facts import NativeLayoutEvidence


def _axis(
    *, thread_count: int = 1, thread_stride: int = 1, element_count: int = 1
) -> MatrixFragmentAxisLayout:
    return MatrixFragmentAxisLayout(
        outer_count=1,
        thread_count=thread_count,
        thread_stride=thread_stride,
        element_count=element_count,
    )


def _role(
    role: str,
    payload_element_count: int,
    element_bit_count: int,
    axes: tuple[MatrixFragmentAxisLayout | None, ...],
) -> MatrixFragmentRoleLayout:
    return MatrixFragmentRoleLayout(
        role=role,
        payload_element_count=payload_element_count,
        element_bit_count=element_bit_count,
        coordinate_element_stride=1,
        reduction_group=None,
        axes=axes,
    )


def _test_layout() -> AmdgpuMatrixFragmentLayout:
    lhs = _role(
        "lhs",
        2,
        16,
        (
            None,
            _axis(thread_count=2),
            None,
            _axis(element_count=2),
        ),
    )
    rhs = _role(
        "rhs",
        2,
        16,
        (
            None,
            None,
            _axis(thread_count=2),
            _axis(element_count=2),
        ),
    )
    result_axes = (
        None,
        _axis(thread_count=2, thread_stride=2),
        _axis(thread_count=2),
        None,
    )
    return AmdgpuMatrixFragmentLayout(
        key="test_exact_2x2x2",
        wave_size=4,
        tile_shape=(1, 2, 2, 2),
        lhs=lhs,
        rhs=rhs,
        accumulator=_role("accumulator", 1, 32, result_axes),
        result=_role("result", 1, 32, result_axes),
    )


def _replace_role_axis(
    role: MatrixFragmentRoleLayout,
    axis_index: int,
    axis: MatrixFragmentAxisLayout | None,
) -> MatrixFragmentRoleLayout:
    axes = list(role.axes)
    axes[axis_index] = axis
    return replace(role, axes=tuple(axes))


def test_exact_layout_adapts_to_shared_algebra_and_publication() -> None:
    layout = _test_layout()

    validate_matrix_fragment_layout(layout)
    facts = matrix_fragment_native_contraction_facts(layout)
    assert all(
        role.evidence is NativeLayoutEvidence.EXACT
        for role in (facts.lhs, facts.rhs, facts.accumulator, facts.result)
    )
    assert matrix_fragment_packed_element_axis(layout, layout.lhs) == "reduction"
    assert matrix_fragment_packed_element_axis(layout, layout.rhs) == "reduction"
    assert matrix_fragment_packed_element_axis(layout, layout.result) is None
    assert matrix_fragment_packed_b16_publication_projection(
        layout, layout.result, "column"
    ) == MatrixFragmentPackedB16PublicationProjection(
        publishing_participant_and_mask=1,
        publishing_participant_equal_value=0,
        paired_participant_xor_mask=1,
    )
    assert matrix_fragment_packed_b16_publication_projection(
        layout, layout.result, "row"
    ) == MatrixFragmentPackedB16PublicationProjection(
        publishing_participant_and_mask=2,
        publishing_participant_equal_value=0,
        paired_participant_xor_mask=2,
    )


def test_validation_rejects_missing_semantic_axis() -> None:
    layout = _test_layout()
    layout = replace(
        layout,
        result=_replace_role_axis(layout.result, 1, None),
    )

    with pytest.raises(ValueError, match="semantic axes"):
        validate_matrix_fragment_layout(layout)


def test_validation_rejects_non_power_of_two_lane_factor() -> None:
    layout = _test_layout()
    layout = replace(
        layout,
        lhs=_replace_role_axis(
            layout.lhs,
            1,
            _axis(thread_count=3),
        ),
    )

    with pytest.raises(ValueError, match="non-power-of-two"):
        validate_matrix_fragment_layout(layout)


def test_validation_rejects_noncompressing_reduction_group() -> None:
    layout = _test_layout()
    layout = replace(
        layout,
        lhs=replace(
            layout.lhs,
            reduction_group=MatrixFragmentReductionGroup(
                storage_element_count=2,
                logical_element_count=2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="invalid reduction storage group"):
        validate_matrix_fragment_layout(layout)


def test_validation_rejects_oversized_register_payload() -> None:
    layout = _test_layout()
    layout = replace(
        layout,
        lhs=replace(
            layout.lhs,
            payload_element_count=64,
            element_bit_count=32,
        ),
    )

    with pytest.raises(ValueError, match="32-register architectural limit"):
        validate_matrix_fragment_layout(layout)


def test_validation_rejects_unaddressable_coordinate_stride() -> None:
    layout = _test_layout()
    layout = replace(
        layout,
        lhs=replace(
            layout.lhs,
            payload_element_count=4,
            coordinate_element_stride=3,
        ),
    )

    with pytest.raises(ValueError, match="cannot be addressed"):
        validate_matrix_fragment_layout(layout)


def test_validation_accepts_elements_that_straddle_registers() -> None:
    layout = _test_layout()
    layout = replace(
        layout,
        lhs=replace(
            layout.lhs,
            payload_element_count=4,
            element_bit_count=24,
        ),
    )

    validate_matrix_fragment_layout(layout)
    assert layout.lhs.register_count == 3


def test_cdna_dense_input_layouts_match_isa_lane_equations() -> None:
    """Exhaustively checks the general CDNA MFMA input-layout equations."""

    layouts = tuple(
        layout
        for key, layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY.items()
        if key.startswith(("cdna_mfma_", "cdna4_mfma_"))
    )
    assert layouts

    for layout in layouts:
        block_count, row_count, column_count, reduction_count = layout.tile_shape
        assert layout.wave_size == 64
        for role, lane_dimension in (
            (layout.lhs, row_count),
            (layout.rhs, column_count),
        ):
            lane_group_count = layout.wave_size // (lane_dimension * block_count)
            reduction_items_per_lane = reduction_count // lane_group_count
            assert role.coordinate_element_count == reduction_items_per_lane
            coordinate_map = matrix_fragment_role_storage_coordinate_map(layout, role)

            for lane in range(layout.wave_size):
                lane_group = lane // lane_dimension
                block = lane_group % block_count
                lane_reduction_group = lane_group // block_count
                for item in range(reduction_items_per_lane):
                    reduction = item + (reduction_items_per_lane * lane_reduction_group)
                    expected = (
                        (block, lane % row_count, reduction)
                        if role.role == "lhs"
                        else (block, reduction, lane % column_count)
                    )
                    assert coordinate_map.evaluate((lane, item)) == expected, (
                        layout.key,
                        role.role,
                        lane,
                        item,
                    )


def test_cdna_result_layouts_match_isa_lane_equations() -> None:
    """Exhaustively checks the general CDNA MFMA output-layout equations."""

    layouts = tuple(
        layout
        for key, layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY.items()
        if key.startswith(("cdna_mfma_", "cdna4_mfma_", "cdna_smfmac_"))
    )
    assert layouts

    for layout in layouts:
        block_count, row_count, column_count, _ = layout.tile_shape
        for role in (layout.accumulator, layout.result):
            group_height = 1 if role.element_bit_count == 64 else 4
            elements_per_output_plane = column_count * row_count // group_height
            blocks_per_item = (
                layout.wave_size + elements_per_output_plane - 1
            ) // elements_per_output_plane
            lane_row_count = layout.wave_size // blocks_per_item // column_count
            row_group_count = row_count // group_height // lane_row_count
            coordinate_map = matrix_fragment_role_storage_coordinate_map(layout, role)

            for lane in range(layout.wave_size):
                block_in_item = (
                    lane // (column_count * lane_row_count)
                ) % blocks_per_item
                row_in_lane = (lane // column_count) % lane_row_count
                column = lane % column_count
                for item in range(role.coordinate_element_count):
                    row_in_group = item % group_height
                    outer_item = item // group_height
                    row_group = outer_item % row_group_count
                    block_group = outer_item // row_group_count
                    block = block_in_item + blocks_per_item * block_group
                    row = row_in_group + group_height * (
                        row_in_lane + lane_row_count * row_group
                    )
                    assert block < block_count
                    assert coordinate_map.evaluate((lane, item)) == (
                        block,
                        row,
                        column,
                    ), (layout.key, role.role, lane, item)
