// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/matrix_fragment_layout.h"

#include "iree/testing/gtest.h"

namespace {

TEST(MatrixFragmentLayoutTest, AppliesCoordinateTerms) {
  const loom_matrix_fragment_coordinate_projection_term_t terms[] = {
      {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
       LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW, 1, 2, 1},
      {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
       LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN, 2, 3, 1},
  };
  uint32_t source_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT] = {};
  source_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE] = 5;
  uint32_t result_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT];
  loom_matrix_fragment_apply_coordinate_projection(terms, IREE_ARRAYSIZE(terms),
                                                   source_terms, result_terms);
  EXPECT_EQ(result_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW], 1);
  EXPECT_EQ(result_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN], 2);
}

TEST(MatrixFragmentLayoutTest, MapsAxesToCoordinateDimensions) {
  for (int axis = 0; axis < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++axis) {
    const auto axis_value = (loom_matrix_fragment_axis_t)axis;
    EXPECT_EQ(loom_matrix_fragment_coordinate_dimension_axis(
                  loom_matrix_fragment_axis_coordinate_dimension(axis_value)),
              axis_value);
  }
  EXPECT_EQ(loom_matrix_fragment_coordinate_dimension_axis(
                LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT),
            LOOM_MATRIX_FRAGMENT_AXIS_COUNT);
}

TEST(MatrixFragmentLayoutTest, SelectsRoleLayouts) {
  loom_matrix_fragment_layout_t layout = {};
  layout.lhs.role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
  layout.rhs.role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
  layout.accumulator.role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
  layout.result.role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;

  EXPECT_EQ(
      loom_matrix_fragment_role_layout(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS),
      &layout.lhs);
  EXPECT_EQ(
      loom_matrix_fragment_role_layout(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS),
      &layout.rhs);
  EXPECT_EQ(loom_matrix_fragment_role_layout(
                &layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR),
            &layout.accumulator);
  EXPECT_EQ(loom_matrix_fragment_role_layout(&layout,
                                             LOOM_CONTRACT_OPERAND_ROLE_RESULT),
            &layout.result);
  EXPECT_EQ(loom_matrix_fragment_role_layout(
                &layout, LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN),
            nullptr);
  EXPECT_EQ(
      loom_matrix_fragment_role_layout(nullptr, LOOM_CONTRACT_OPERAND_ROLE_LHS),
      nullptr);
}

}  // namespace
