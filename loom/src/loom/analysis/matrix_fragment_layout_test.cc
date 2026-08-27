// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/matrix_fragment_layout.h"

#include "iree/testing/gtest.h"

namespace {

constexpr loom_matrix_fragment_coordinate_flags_t kRowReduction =
    LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
    LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
constexpr loom_matrix_fragment_coordinate_flags_t kColumnReduction =
    LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |
    LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
constexpr loom_matrix_fragment_coordinate_flags_t kRowColumn =
    LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
    LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN;
constexpr loom_matrix_fragment_coordinate_flags_t kBlockRowColumn =
    LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK |
    LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
    LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN;

static const loom_matrix_fragment_coordinate_projection_term_t kLhsTerms[] = {
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW, 1, 16, 1},
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION, 1, 0, 1},
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT, 1, 0, 1},
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE, 1, 0, 1},
};
static const loom_matrix_fragment_coordinate_projection_plan_t kLhsPlan = {
    /*.terms=*/kLhsTerms,
    /*.forward_term_count=*/2,
    /*.inverse_term_count=*/2,
};

static const loom_matrix_fragment_coordinate_projection_term_t kRhsTerms[] = {
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN, 1, 16, 1},
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION, 1, 0, 1},
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT, 1, 0, 1},
    {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION,
     LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE, 1, 0, 1},
};
static const loom_matrix_fragment_coordinate_projection_plan_t kRhsPlan = {
    /*.terms=*/kRhsTerms,
    /*.forward_term_count=*/2,
    /*.inverse_term_count=*/2,
};

static const loom_matrix_fragment_coordinate_projection_term_t kResultTerms[] =
    {
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW, 16, 0, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW, 1, 0, 2},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN, 1, 16, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT, 1, 2, 16},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT, 1, 0, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE, 2, 0, 1},
};
static const loom_matrix_fragment_coordinate_projection_plan_t kResultPlan = {
    /*.terms=*/kResultTerms,
    /*.forward_term_count=*/3,
    /*.inverse_term_count=*/3,
};

static const loom_matrix_fragment_coordinate_projection_term_t
    kBlockedResultTerms[] = {
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK, 4, 0, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW, 16, 4, 4},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW, 1, 4, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN, 1, 16, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT, 4, 0, 16},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT, 1, 0, 1},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE, 1, 0, 4},
        {LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW,
         LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE, 1, 4, 1},
};
static const loom_matrix_fragment_coordinate_projection_plan_t
    kBlockedResultPlan = {
        /*.terms=*/kBlockedResultTerms,
        /*.forward_term_count=*/4,
        /*.inverse_term_count=*/4,
};

loom_matrix_fragment_role_layout_t RoleLayout(
    loom_contract_operand_role_t role, uint16_t register_count,
    uint16_t element_bit_count, uint16_t payload_element_count,
    uint16_t coordinate_element_count, uint16_t coordinate_element_offset,
    uint16_t coordinate_element_stride,
    loom_matrix_fragment_coordinate_flags_t coordinate_flags,
    loom_matrix_fragment_coordinate_flags_t packed_element_coordinate_flag,
    const loom_matrix_fragment_coordinate_projection_plan_t* projection) {
  return (loom_matrix_fragment_role_layout_t){
      /*.role=*/role,
      /*.register_count=*/register_count,
      /*.element_bit_count=*/element_bit_count,
      /*.payload_element_count=*/payload_element_count,
      /*.coordinate_element_count=*/coordinate_element_count,
      /*.coordinate_element_offset=*/coordinate_element_offset,
      /*.coordinate_element_stride=*/coordinate_element_stride,
      /*.packed_b16_publication=*/{},
      /*.coordinate_flags=*/coordinate_flags,
      /*.packed_element_coordinate_flag=*/packed_element_coordinate_flag,
      /*.reduction_group=*/{},
      /*.coordinate_projection_plan=*/projection,
  };
}

loom_matrix_fragment_layout_t TestLayout() {
  return (loom_matrix_fragment_layout_t){
      /*.kind=*/1,
      /*.name=*/IREE_SV("test.projection"),
      /*.wave_size=*/32,
      /*.tile_shape=*/{1, 16, 16, 16},
      /*.lhs=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_LHS, 8, 16, 16, 16, 0, 1,
                 kRowReduction, LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION,
                 &kLhsPlan),
      /*.rhs=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RHS, 8, 16, 16, 16, 0, 1,
                 kColumnReduction, LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION,
                 &kRhsPlan),
      /*.accumulator=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 8, 32, 8, 8, 0, 1,
                 kRowColumn, 0, &kResultPlan),
      /*.result=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 8, 32, 8, 8, 0, 1,
                 kRowColumn, 0, &kResultPlan),
  };
}

loom_matrix_fragment_layout_t BlockedLayout() {
  loom_matrix_fragment_layout_t layout = TestLayout();
  layout.kind = 2;
  layout.name = IREE_SV("test.blocked.projection");
  layout.wave_size = 64;
  layout.tile_shape = {4, 16, 16, 16};
  layout.result = RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 16, 32, 16, 16,
                             0, 1, kBlockRowColumn, 0, &kBlockedResultPlan);
  return layout;
}

loom_matrix_fragment_coordinate_t RowReduction(uint16_t row,
                                               uint16_t reduction) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kRowReduction,
      /*.block=*/0,
      /*.row=*/row,
      /*.column=*/0,
      /*.reduction=*/reduction,
  };
}

loom_matrix_fragment_coordinate_t RowColumn(uint16_t row, uint16_t column) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kRowColumn,
      /*.block=*/0,
      /*.row=*/row,
      /*.column=*/column,
      /*.reduction=*/0,
  };
}

void ExpectCoordinate(const loom_matrix_fragment_layout_t* layout,
                      loom_contract_operand_role_t role, uint16_t participant,
                      uint16_t payload_element, uint16_t block, uint16_t row,
                      uint16_t column, uint16_t reduction) {
  loom_matrix_fragment_coordinate_t coordinate = {};
  ASSERT_TRUE(loom_matrix_fragment_coordinate(layout, role, participant,
                                              payload_element, &coordinate));
  EXPECT_EQ(coordinate.block, block);
  EXPECT_EQ(coordinate.row, row);
  EXPECT_EQ(coordinate.column, column);
  EXPECT_EQ(coordinate.reduction, reduction);
}

void ExpectPhysicalElement(const loom_matrix_fragment_layout_t* layout,
                           loom_contract_operand_role_t role,
                           loom_matrix_fragment_coordinate_t coordinate,
                           uint16_t occurrence, uint16_t participant,
                           uint16_t payload_element) {
  loom_matrix_fragment_physical_element_t element = {};
  ASSERT_TRUE(loom_matrix_fragment_physical_element(layout, role, coordinate,
                                                    occurrence, &element));
  EXPECT_EQ(element.lane, participant);
  EXPECT_EQ(element.payload_element_index, payload_element);
}

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

TEST(MatrixFragmentLayoutTest, FindsRoleLayouts) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  const loom_matrix_fragment_role_layout_t* lhs =
      loom_matrix_fragment_role_layout(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(lhs->coordinate_projection_plan, &kLhsPlan);
  EXPECT_EQ(loom_matrix_fragment_role_layout(
                &layout, LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN),
            nullptr);
  EXPECT_EQ(
      loom_matrix_fragment_role_layout(nullptr, LOOM_CONTRACT_OPERAND_ROLE_LHS),
      nullptr);
}

TEST(MatrixFragmentLayoutTest, RejectsMetadataDependentLogicalCoordinates) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  layout.lhs.reduction_group = {
      /*.storage_element_count=*/2,
      /*.logical_element_count=*/4,
  };
  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, &coordinate));
}

TEST(MatrixFragmentLayoutTest, MapsReplicatedInputCoordinates) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 15, 15, 0, 15, 0,
                   15);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 16, 15, 0, 0, 0,
                   15);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 31, 15, 0, 0, 15,
                   15);
}

TEST(MatrixFragmentLayoutTest, MapsInterleavedResultCoordinates) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 0, 0, 0, 0, 0,
                   0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 15, 3, 0, 6, 15,
                   0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 16, 0, 0, 1, 0,
                   0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 7, 0, 15, 15,
                   0);
}

TEST(MatrixFragmentLayoutTest, MapsIndependentBlockedCoordinates) {
  loom_matrix_fragment_layout_t layout = BlockedLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 0, 0, 0, 0, 0,
                   0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 0, 4, 1, 0, 0,
                   0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 15, 3, 15,
                   15, 0);
}

TEST(MatrixFragmentLayoutTest, RejectsPaddingPayloadElements) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  layout.result.payload_element_count = 16;
  layout.result.coordinate_element_stride = 2;
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 14, 0, 15,
                   15, 0);
  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 15, &coordinate));
}

TEST(MatrixFragmentLayoutTest, RejectsOutOfRangeElements) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 32, 0, &coordinate));
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 16, &coordinate));
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 0, 8, &coordinate));
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN, 0, 0, &coordinate));
}

TEST(MatrixFragmentLayoutTest, FindsReplicatedPhysicalElements) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  uint16_t count = 0;
  ASSERT_TRUE(loom_matrix_fragment_physical_element_count(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, RowReduction(0, 0), &count));
  EXPECT_EQ(count, 2);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                        RowReduction(0, 0), 0, 0, 0);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                        RowReduction(0, 0), 1, 16, 0);
}

TEST(MatrixFragmentLayoutTest, FindsUniquePhysicalElements) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  uint16_t count = 0;
  ASSERT_TRUE(loom_matrix_fragment_physical_element_count(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, RowColumn(15, 15), &count));
  EXPECT_EQ(count, 1);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                        RowColumn(15, 15), 0, 31, 7);
}

TEST(MatrixFragmentLayoutTest, RejectsMissingPhysicalElements) {
  loom_matrix_fragment_layout_t layout = TestLayout();
  uint16_t count = 0;
  EXPECT_FALSE(loom_matrix_fragment_physical_element_count(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, RowColumn(0, 0), &count));
  EXPECT_EQ(count, 0);

  loom_matrix_fragment_physical_element_t element = {};
  EXPECT_FALSE(loom_matrix_fragment_physical_element(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, RowReduction(0, 0), 2,
      &element));
  EXPECT_FALSE(loom_matrix_fragment_physical_element(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN, RowReduction(0, 0), 0,
      &element));
}

}  // namespace
