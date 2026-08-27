# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.matrix_formats import (
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)
from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS,
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
    MatrixFragmentAxisLayout,
    MatrixFragmentLaneBitProjection,
    MatrixFragmentPackedB16PublicationProjection,
    MatrixFragmentReductionGroup,
    MatrixFragmentResultToLhsBf16Projection,
    MatrixFragmentResultToRhsPackedB16Projection,
    layout_roles,
    matrix_fragment_native_contraction_facts,
    matrix_fragment_native_layout,
    matrix_fragment_native_role_layout,
    matrix_fragment_native_transition_facts,
    matrix_fragment_packed_b16_publication_projection,
    matrix_fragment_packed_element_axis,
    matrix_fragment_result_to_lhs_bf16_projection,
    matrix_fragment_result_to_rhs_packed_b16_projection,
    matrix_fragment_role_storage_projection_plan,
    role_coordinate,
    validate_matrix_fragment_layout,
)
from loom.target.native_contraction_layout import (
    operation_local_coordinate_map,
    ownership_relation,
    unique_ownership_coordinate_map,
)
from loom.target.native_coordinate_projection import (
    CoordinateProjectionTerm,
    coordinate_projection_plan,
)
from loom.target.native_layout_facts import NativeLayoutEvidence


def test_dense_layouts_have_exact_native_contraction_maps() -> None:
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        roles = layout_roles(layout)
        role_layouts = tuple(
            matrix_fragment_native_role_layout(layout, role) for role in roles
        )
        assert tuple(role_layout is None for role_layout in role_layouts) == tuple(
            role.reduction_group is not None for role in roles
        )
        assert (matrix_fragment_native_layout(layout) is None) == any(
            role_layout is None for role_layout in role_layouts
        )


def test_native_contraction_facts_retain_exact_and_metadata_placements() -> None:
    metadata_role_count = 0
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        facts = matrix_fragment_native_contraction_facts(layout)
        assert facts.participant_count == layout.wave_size
        for role, role_facts in zip(
            layout_roles(layout),
            (facts.lhs, facts.rhs, facts.accumulator, facts.result),
            strict=True,
        ):
            if role.reduction_group is None:
                assert role_facts.evidence is NativeLayoutEvidence.EXACT
                assert role_facts.owner_multiplicity_minimum is not None
            else:
                metadata_role_count += 1
                assert role_facts.evidence is NativeLayoutEvidence.METADATA_DEPENDENT
                assert role_facts.owner_multiplicity_minimum is None

    assert metadata_role_count > 0


def test_gfx11_packed_b16_transition_facts_retain_owner_algebra() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f16_16x16x16_f16"]

    transition = matrix_fragment_native_transition_facts(layout, "result", "rhs")

    assert transition is not None
    assert transition.destination_position_count == 512
    assert transition.participant_change_count == 256
    assert transition.local_position_change_count == 480
    assert transition.destination_positions_per_source_minimum == 2
    assert transition.destination_positions_per_source_maximum == 2
    assert tuple(
        (
            factor.destination_dimension,
            factor.source_owner_dimension,
            factor.destination_divisor,
            factor.destination_modulus,
            factor.source_owner_multiplier,
        )
        for factor in transition.source_owner_factors
    ) == (
        ("participant", "participant", 1, 16, 1),
        ("value", "participant", 1, 2, 16),
        ("value", "value", 2, 0, 1),
    )


def test_roles_compile_to_bounded_storage_coordinate_projections() -> None:
    maximum_forward_term_count = 0
    maximum_inverse_term_count = 0
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        for role in layout_roles(layout):
            plan = matrix_fragment_role_storage_projection_plan(layout, role)
            maximum_forward_term_count = max(
                maximum_forward_term_count, len(plan.forward_terms)
            )
            maximum_inverse_term_count = max(
                maximum_inverse_term_count, len(plan.inverse_terms)
            )

    assert maximum_forward_term_count == 5
    assert maximum_inverse_term_count == 5


def test_storage_projections_have_one_participant_digit_per_axis() -> None:
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        for role in layout_roles(layout):
            plan = matrix_fragment_role_storage_projection_plan(layout, role)
            participant_destinations = [
                term.destination_dimension
                for term in plan.forward_terms
                if term.source_dimension == "participant"
            ]
            assert len(participant_destinations) == len(set(participant_destinations))


def test_packed_element_axes_are_proven_during_generation() -> None:
    packed_axes = {
        (layout.key, role.role): packed_axis
        for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS
        for role in layout_roles(layout)
        if (packed_axis := matrix_fragment_packed_element_axis(layout, role))
        is not None
    }

    assert packed_axes
    assert set(packed_axes.values()) <= {"row", "column", "reduction"}
    assert packed_axes[("rdna3_wmmar3_f16_16x16x16_f16", "lhs")] == "reduction"
    assert packed_axes[("rdna3_wmmar3_f16_16x16x16_f16", "rhs")] == "reduction"
    assert ("rdna3_wmmar3_f16_16x16x16_f16", "result") not in packed_axes


def test_packed_b16_result_to_rhs_ownership_is_generated_from_role_maps() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f16_16x16x16_f16"]
    native_layout = matrix_fragment_native_layout(layout)
    assert native_layout is not None

    relation = ownership_relation(
        operation_local_coordinate_map(native_layout.result),
        operation_local_coordinate_map(native_layout.rhs),
    )

    assert relation.edge_count == relation.destination.source_point_count
    assert set(relation.source_owners_by_destination) == {
        (source_ordinal,)
        for source_ordinal in range(relation.source.source_point_count)
    }
    owner_map = unique_ownership_coordinate_map(relation)
    assert owner_map is not None
    projection = coordinate_projection_plan(owner_map)
    assert projection is not None
    assert projection.forward_terms == (
        CoordinateProjectionTerm("participant", "participant", 1, 16, 1),
        CoordinateProjectionTerm("value", "participant", 1, 2, 16),
        CoordinateProjectionTerm("value", "value", 2, 0, 1),
    )


def test_packed_b16_result_to_rhs_projection_matches_shipping_layouts() -> None:
    projections = {
        layout.key: projection
        for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS
        if (projection := matrix_fragment_result_to_rhs_packed_b16_projection(layout))
        is not None
    }

    assert projections == {
        "rdna3_wmmar3_bf16_16x16x16_bf16": (
            MatrixFragmentResultToRhsPackedB16Projection(
                exchange_participant_xor_mask=16,
                reverse_participant_mask=0xFFFF0000,
            )
        ),
        "rdna3_wmmar3_f16_16x16x16_f16": (
            MatrixFragmentResultToRhsPackedB16Projection(
                exchange_participant_xor_mask=16,
                reverse_participant_mask=0xFFFF0000,
            )
        ),
    }


def test_result_to_lhs_projection_matches_shipping_layouts() -> None:
    layouts_by_projection: dict[MatrixFragmentResultToLhsBf16Projection, set[str]] = (
        defaultdict(set)
    )
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        projection = matrix_fragment_result_to_lhs_bf16_projection(layout)
        if projection is not None:
            layouts_by_projection[projection].add(layout.key)

    assert layouts_by_projection == {
        MatrixFragmentResultToLhsBf16Projection(
            source_lane_group_byte_shift=6,
            result_lane_div_byte_shift=0,
            source_register_selector=MatrixFragmentLaneBitProjection(0xFFFF, 1),
            source_lane_group=MatrixFragmentLaneBitProjection(1, 0),
            transpose_bit_count=3,
        ): {
            "rdna3_wmmar3_f32_16x16x16_bf16",
            "rdna3_wmmar3_f32_16x16x16_f16",
        },
        MatrixFragmentResultToLhsBf16Projection(
            source_lane_group_byte_shift=6,
            result_lane_div_byte_shift=4,
            source_register_selector=MatrixFragmentLaneBitProjection(3, 0),
            source_lane_group=MatrixFragmentLaneBitProjection(0xFFFF, 2),
            transpose_bit_count=1,
        ): {
            "cdna_mfma_f32_16x16x16_bf16",
            "cdna_mfma_f32_16x16x16_f16",
            "rdna4_wmma_f32_16x16x16_bf16_w64",
            "rdna4_wmma_f32_16x16x16_f16_w64",
        },
        MatrixFragmentResultToLhsBf16Projection(
            source_lane_group_byte_shift=6,
            result_lane_div_byte_shift=0,
            source_register_selector=MatrixFragmentLaneBitProjection(0xFFFF, 2),
            source_lane_group=MatrixFragmentLaneBitProjection(3, 0),
            transpose_bit_count=2,
        ): {
            "rdna3_wmmar3_f32_16x16x16_bf16_w64",
            "rdna3_wmmar3_f32_16x16x16_f16_w64",
        },
        MatrixFragmentResultToLhsBf16Projection(
            source_lane_group_byte_shift=6,
            result_lane_div_byte_shift=5,
            source_register_selector=MatrixFragmentLaneBitProjection(7, 0),
            source_lane_group=MatrixFragmentLaneBitProjection(0xFFFF, 3),
            transpose_bit_count=2,
        ): {
            "rdna4_wmma_f32_16x16x16_bf16",
            "rdna4_wmma_f32_16x16x16_f16",
        },
    }


def test_packed_b16_publication_accounts_for_payload_padding() -> None:
    f32_layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f32_16x16x16_f16"]
    f16_layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f16_16x16x16_f16"]

    assert matrix_fragment_packed_b16_publication_projection(
        f32_layout, f32_layout.result
    ) == MatrixFragmentPackedB16PublicationProjection(
        publishing_participant_modulus=2,
        publishing_participant_remainder=0,
        paired_participant_xor_mask=1,
    )
    assert (
        matrix_fragment_packed_b16_publication_projection(f16_layout, f16_layout.result)
        is None
    )


@pytest.mark.parametrize(
    "layout_key",
    [
        "rdna3_wmmar3_f32_16x16x16_f16",
        "rdna3_wmmar3_f32_16x16x16_f16_w64",
        "cdna_mfma_f32_16x16x16_f16",
        "rdna4_wmma_f32_16x16x16_f16",
        "gfx125x_wmma_f32_16x16x128_f8_f8",
    ],
)
def test_packed_b16_publication_spans_matrix_families(layout_key: str) -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY[layout_key]
    expected = MatrixFragmentPackedB16PublicationProjection(
        publishing_participant_modulus=2,
        publishing_participant_remainder=0,
        paired_participant_xor_mask=1,
    )

    assert (
        matrix_fragment_packed_b16_publication_projection(layout, layout.accumulator)
        == expected
    )
    assert (
        matrix_fragment_packed_b16_publication_projection(layout, layout.result)
        == expected
    )


def test_packed_b16_publication_requires_adjacent_column_owners() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f32_16x16x16_f16"]
    row_paired_role = replace(
        layout.result,
        axes=(
            None,
            MatrixFragmentAxisLayout(1, 16, 1, 1),
            MatrixFragmentAxisLayout(8, 2, 16, 1),
            None,
        ),
    )
    row_paired_layout = replace(layout, result=row_paired_role)
    validate_matrix_fragment_layout(row_paired_layout)

    assert role_coordinate(row_paired_layout, row_paired_role, 0, 0) == (
        None,
        0,
        0,
        None,
    )
    assert role_coordinate(row_paired_layout, row_paired_role, 1, 0) == (
        None,
        1,
        0,
        None,
    )
    assert (
        matrix_fragment_packed_b16_publication_projection(
            row_paired_layout, row_paired_role
        )
        is None
    )


@pytest.mark.parametrize(
    "layout_key",
    [
        "rdna3_wmmar3_i32_16x16x16_iu8",
        "rdna3_wmmar3_i32_16x16x16_iu8_w64",
        "rdna3_wmmar3_i32_16x16x16_iu4",
        "rdna3_wmmar3_i32_16x16x16_iu4_w64",
    ],
)
def test_rdna3_integer_wmma_layout_matches_instruction_coordinates(
    layout_key: str,
) -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY[layout_key]

    for lane in range(layout.wave_size):
        for element in range(layout.lhs.payload_element_count):
            assert role_coordinate(layout, layout.lhs, lane, element) == (
                None,
                lane % 16,
                None,
                element,
            )
            assert role_coordinate(layout, layout.rhs, lane, element) == (
                None,
                None,
                lane % 16,
                element,
            )
        for role in (layout.accumulator, layout.result):
            for element in range(role.payload_element_count):
                assert role_coordinate(layout, role, lane, element) == (
                    None,
                    (layout.wave_size // 16) * element + lane // 16,
                    lane % 16,
                    None,
                )


def test_gfx125x_selector_layouts_cover_every_physical_operand_abi() -> None:
    for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
        for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
            layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY[
                f"gfx125x_wmma_f32_16x16x128_{lhs_format.token}_{rhs_format.token}"
            ]
            assert layout.tile_shape == (1, 16, 16, 128)
            assert (
                layout.lhs.payload_element_count,
                layout.lhs.register_count,
                layout.lhs.element_bit_count,
            ) == (
                64,
                lhs_format.register_count_for(64),
                lhs_format.element_bit_count,
            )
            assert (
                layout.rhs.payload_element_count,
                layout.rhs.register_count,
                layout.rhs.element_bit_count,
            ) == (
                64,
                rhs_format.register_count_for(64),
                rhs_format.element_bit_count,
            )
            assert (
                layout.result.payload_element_count,
                layout.result.register_count,
                layout.result.element_bit_count,
            ) == (8, 8, 32)

            for lane in (0, 15, 16, 31):
                for element in (0, 5, 63):
                    reduction = 64 * (lane // 16) + element
                    assert role_coordinate(layout, layout.lhs, lane, element) == (
                        None,
                        lane % 16,
                        None,
                        reduction,
                    )
                    assert role_coordinate(layout, layout.rhs, lane, element) == (
                        None,
                        None,
                        lane % 16,
                        reduction,
                    )


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
        family="smfmac",
    )

    validate_matrix_fragment_layout(sparse)
    assert role_coordinate(sparse, sparse.lhs, 0, 0) is None
    assert role_coordinate(sparse, sparse.rhs, 0, 0) == (None, None, 0, 0)


def test_rdna4m_wave64_iu4_layouts_model_ignored_upper_lanes() -> None:
    dense_iu4 = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna4_wmma_i32_16x16x16_iu4_w64"]
    dense_f16 = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna4_wmma_f32_16x16x16_f16_w64"]
    sparse_iu4 = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY[
        "rdna4_swmmac_32bit_16x16x32_packed4_w64"
    ]

    assert role_coordinate(dense_iu4, dense_iu4.lhs, 0, 0) == role_coordinate(
        dense_iu4, dense_iu4.lhs, 32, 0
    )
    assert role_coordinate(dense_iu4, dense_iu4.rhs, 0, 0) == role_coordinate(
        dense_iu4, dense_iu4.rhs, 32, 0
    )
    assert role_coordinate(dense_f16, dense_f16.lhs, 0, 0) != role_coordinate(
        dense_f16, dense_f16.lhs, 32, 0
    )
    assert sparse_iu4.lhs.axes == dense_iu4.lhs.axes
    assert role_coordinate(sparse_iu4, sparse_iu4.rhs, 0, 0) != role_coordinate(
        sparse_iu4, sparse_iu4.rhs, 32, 0
    )


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


def test_validation_rejects_fragment_payload_above_architectural_limit() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f32_16x16x16_f16"]
    malformed = replace(
        layout,
        result=replace(layout.result, payload_element_count=33),
    )

    with pytest.raises(ValueError, match="32-register architectural limit"):
        validate_matrix_fragment_layout(malformed)


def test_validation_rejects_non_power_of_two_lane_factor() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f32_16x16x16_f16"]
    result_axes = layout.result.axes
    row_axis = result_axes[1]
    assert row_axis is not None
    malformed = replace(
        layout,
        result=replace(
            layout.result,
            axes=(
                result_axes[0],
                replace(row_axis, thread_stride=3),
                result_axes[2],
                result_axes[3],
            ),
        ),
    )

    with pytest.raises(ValueError, match="non-power-of-two row lane factor"):
        validate_matrix_fragment_layout(malformed)


def test_validation_accepts_elements_that_straddle_payload_registers() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f32_16x16x16_f16"]
    bitstream = replace(
        layout,
        result=replace(layout.result, element_bit_count=24),
    )

    validate_matrix_fragment_layout(bitstream)
    assert bitstream.result.register_count == 3


def test_validation_rejects_unaddressable_coordinate_element_mapping() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f32_16x16x16_f16"]
    malformed = replace(
        layout,
        result=replace(layout.result, coordinate_element_offset=1),
    )

    with pytest.raises(ValueError, match="cannot be addressed"):
        validate_matrix_fragment_layout(malformed)


@pytest.mark.parametrize(
    ("layout_key", "coordinate_oracle"),
    [
        (
            "cdna_mfma_f32_4x4x1_f32_16b",
            lambda lane, element: (lane // 4, element, lane % 4, None),
        ),
        (
            "cdna_mfma_f32_16x16x1_f32_4b",
            lambda lane, element: (
                element // 4,
                4 * (lane // 16) + element % 4,
                lane % 16,
                None,
            ),
        ),
        (
            "cdna_mfma_f32_32x32x1_f32_2b",
            lambda lane, element: (
                element // 16,
                element % 4 + 4 * (lane // 32) + 8 * ((element // 4) % 4),
                lane % 32,
                None,
            ),
        ),
        (
            "cdna_mfma_f64_4x4x4_f64_4b",
            lambda lane, element: (
                (lane // 4) % 4,
                lane // 16,
                lane % 4,
                None,
            ),
        ),
    ],
)
def test_blocked_mfma_result_layout_matches_instruction_coordinates(
    layout_key: str,
    coordinate_oracle: Callable[[int, int], tuple[int | None, ...]],
) -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY[layout_key]

    for role in (layout.accumulator, layout.result):
        for lane in range(layout.wave_size):
            for element in range(role.payload_element_count):
                assert role_coordinate(layout, role, lane, element) == (
                    coordinate_oracle(lane, element)
                )


def test_blocked_mfma_source_layouts_match_instruction_coordinates() -> None:
    checked_layout_count = 0
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        block_count, row_count, column_count, reduction_count = layout.tile_shape
        if (
            not layout.key.startswith("cdna_mfma_")
            or block_count == 1
            or layout.lhs.element_bit_count == 64
        ):
            continue
        checked_layout_count += 1
        assert block_count * row_count == layout.wave_size
        assert row_count == column_count
        assert layout.lhs.payload_element_count == reduction_count
        assert layout.rhs.payload_element_count == reduction_count

        for lane in range(layout.wave_size):
            block = lane // row_count
            matrix_lane = lane % row_count
            for element in range(reduction_count):
                assert role_coordinate(layout, layout.lhs, lane, element) == (
                    block,
                    matrix_lane,
                    None,
                    element,
                )
                assert role_coordinate(layout, layout.rhs, lane, element) == (
                    block,
                    None,
                    matrix_lane,
                    element,
                )

    assert checked_layout_count > 0


def test_blocked_f64_mfma_source_layout_matches_instruction_coordinates() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f64_4x4x4_f64_4b"]

    for lane in range(layout.wave_size):
        block = (lane // 4) % 4
        reduction = lane // 16
        matrix_lane = lane % 4
        assert role_coordinate(layout, layout.lhs, lane, 0) == (
            block,
            matrix_lane,
            None,
            reduction,
        )
        assert role_coordinate(layout, layout.rhs, lane, 0) == (
            block,
            None,
            matrix_lane,
            reduction,
        )


def test_f64_mfma_16x16_layout_matches_instruction_coordinates() -> None:
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["cdna_mfma_f64_16x16x4_f64"]

    for lane in range(layout.wave_size):
        assert role_coordinate(layout, layout.lhs, lane, 0) == (
            None,
            lane % 16,
            None,
            lane // 16,
        )
        assert role_coordinate(layout, layout.rhs, lane, 0) == (
            None,
            None,
            lane % 16,
            lane // 16,
        )
        for role in (layout.accumulator, layout.result):
            for element in range(role.payload_element_count):
                assert role_coordinate(layout, role, lane, element) == (
                    None,
                    4 * (lane // 16) + element,
                    lane % 16,
                    None,
                )


def test_result_to_lhs_partial_transpose_preserves_coordinates() -> None:
    checked_layout_count = 0
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        source_role = layout.result
        destination_role = layout.lhs
        if (
            source_role.element_bit_count != 32
            or destination_role.element_bit_count != 16
            or destination_role.reduction_group is not None
            or layout.tile_shape[2] != layout.tile_shape[3]
            or matrix_fragment_packed_b16_publication_projection(layout, source_role)
            is None
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
