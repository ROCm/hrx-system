# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
    MatrixFragmentAxisLayout,
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
