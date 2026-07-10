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
constexpr loom_matrix_fragment_coordinate_flags_t kBlockRowReduction =
    LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK |
    LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
    LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
constexpr loom_matrix_fragment_coordinate_flags_t kBlockColumnReduction =
    LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK |
    LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |
    LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
constexpr loom_matrix_fragment_coordinate_flags_t kBlockRowColumn =
    LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK |
    LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
    LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN;

loom_matrix_fragment_axis_layout_t AxisLayout(uint16_t outer_count,
                                              uint16_t thread_count,
                                              uint16_t thread_stride,
                                              uint16_t element_count) {
  return (loom_matrix_fragment_axis_layout_t){
      /*.outer_count=*/outer_count,
      /*.thread_count=*/thread_count,
      /*.thread_stride=*/thread_stride,
      /*.element_count=*/element_count,
  };
}

loom_matrix_fragment_role_layout_t SemanticRoleLayout(
    loom_contract_operand_role_t role, uint16_t register_count,
    uint16_t payload_element_count, uint16_t element_bit_count,
    uint16_t coordinate_element_offset, uint16_t coordinate_element_stride,
    loom_matrix_fragment_coordinate_flags_t coordinate_flags) {
  return (loom_matrix_fragment_role_layout_t){
      /*.role=*/role,
      /*.register_count=*/register_count,
      /*.element_bit_count=*/element_bit_count,
      /*.payload_element_count=*/payload_element_count,
      /*.coordinate_element_offset=*/coordinate_element_offset,
      /*.coordinate_element_stride=*/coordinate_element_stride,
      /*.flags=*/0,
      /*.coordinate_flags=*/coordinate_flags,
      /*.axes=*/{},
  };
}

loom_matrix_fragment_layout_t RdnaLayout() {
  loom_matrix_fragment_layout_t layout = {
      /*.kind=*/1,
      /*.name=*/IREE_SV("test.rdna"),
      /*.wave_size=*/32,
      /*.tile_shape=*/
      {
          /*.block_count=*/1,
          /*.result_row_count=*/16,
          /*.result_column_count=*/16,
          /*.reduction_count=*/16,
      },
      /*.lhs=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_LHS, 8, 16, 16, 0, 1,
                         kRowReduction),
      /*.rhs=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RHS, 8, 16, 16, 0, 1,
                         kColumnReduction),
      /*.accumulator=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 8, 8, 32, 0, 1,
                         kRowColumn),
      /*.result=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 8, 8, 32, 0, 1,
                         kRowColumn),
  };
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 16, 1, 1);
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 1, 1, 16);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] = AxisLayout(1, 16, 1, 1);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 1, 1, 16);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(8, 2, 16, 1);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(8, 2, 16, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

loom_matrix_fragment_layout_t RdnaWave64InterleavedLayout() {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  layout.kind = 5;
  layout.name = IREE_SV("test.rdna.wave64.interleaved");
  layout.wave_size = 64;
  layout.accumulator = SemanticRoleLayout(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 4, 4, 32, 0, 1, kRowColumn);
  layout.result = SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 4, 4,
                                     32, 0, 1, kRowColumn);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(4, 4, 16, 1);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(4, 4, 16, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

loom_matrix_fragment_layout_t RdnaLowSubwordLayout() {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  layout.kind = 2;
  layout.name = IREE_SV("test.rdna.low_subword");
  layout.accumulator = SemanticRoleLayout(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 8, 16, 16, 0, 2, kRowColumn);
  layout.result = SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 8, 16,
                                     16, 0, 2, kRowColumn);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(8, 2, 16, 1);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(8, 2, 16, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

loom_matrix_fragment_layout_t CdnaLayout() {
  loom_matrix_fragment_layout_t layout = {
      /*.kind=*/2,
      /*.name=*/IREE_SV("test.cdna"),
      /*.wave_size=*/64,
      /*.tile_shape=*/
      {
          /*.block_count=*/1,
          /*.result_row_count=*/16,
          /*.result_column_count=*/16,
          /*.reduction_count=*/16,
      },
      /*.lhs=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_LHS, 2, 4, 16, 0, 1,
                         kRowReduction),
      /*.rhs=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RHS, 2, 4, 16, 0, 1,
                         kColumnReduction),
      /*.accumulator=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 4, 4, 32, 0, 1,
                         kRowColumn),
      /*.result=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 4, 4, 32, 0, 1,
                         kRowColumn),
  };
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 16, 1, 1);
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 4, 16, 4);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] = AxisLayout(1, 16, 1, 1);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 4, 16, 4);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(1, 4, 16, 4);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 4, 16, 4);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

loom_matrix_fragment_layout_t Blocked16x16Layout() {
  loom_matrix_fragment_layout_t layout = {
      /*.kind=*/6,
      /*.name=*/IREE_SV("test.blocked.16x16"),
      /*.wave_size=*/64,
      /*.tile_shape=*/
      {
          /*.block_count=*/4,
          /*.result_row_count=*/16,
          /*.result_column_count=*/16,
          /*.reduction_count=*/4,
      },
      /*.lhs=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_LHS, 2, 4, 16, 0, 1,
                         kBlockRowReduction),
      /*.rhs=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RHS, 2, 4, 16, 0, 1,
                         kBlockColumnReduction),
      /*.accumulator=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 16, 32, 0,
                         1, kBlockRowColumn),
      /*.result=*/
      SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 16, 16, 32, 0, 1,
                         kBlockRowColumn),
  };
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_BLOCK] = AxisLayout(1, 4, 16, 1);
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 16, 1, 1);
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 1, 64, 4);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_BLOCK] = AxisLayout(1, 4, 16, 1);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] = AxisLayout(1, 16, 1, 1);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 1, 64, 4);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_BLOCK] =
      AxisLayout(1, 1, 64, 4);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(1, 4, 16, 4);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_BLOCK] = AxisLayout(1, 1, 64, 4);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 4, 16, 4);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

loom_matrix_fragment_layout_t LaneGroupLowSubwordLayout() {
  loom_matrix_fragment_layout_t layout = CdnaLayout();
  layout.kind = 3;
  layout.name = IREE_SV("test.lane_group.low_subword");
  layout.accumulator = SemanticRoleLayout(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 4, 8, 16, 0, 2, kRowColumn);
  layout.result = SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 4, 8,
                                     16, 0, 2, kRowColumn);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(1, 4, 16, 4);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 4, 16, 4);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

loom_matrix_fragment_layout_t LaneGroupPackedRowLayout() {
  loom_matrix_fragment_layout_t layout = CdnaLayout();
  layout.kind = 4;
  layout.name = IREE_SV("test.lane_group.packed_row");
  layout.wave_size = 32;
  layout.lhs = SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_LHS, 4, 8, 16, 0,
                                  1, kRowReduction);
  layout.rhs = SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RHS, 4, 8, 16, 0,
                                  1, kColumnReduction);
  layout.accumulator = SemanticRoleLayout(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 4, 8, 16, 0, 1, kRowColumn);
  layout.result = SemanticRoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT, 4, 8,
                                     16, 0, 1, kRowColumn);
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 16, 1, 1);
  layout.lhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 2, 16, 8);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] = AxisLayout(1, 16, 1, 1);
  layout.rhs.axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION] =
      AxisLayout(1, 2, 16, 8);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] =
      AxisLayout(1, 2, 16, 8);
  layout.accumulator.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW] = AxisLayout(1, 2, 16, 8);
  layout.result.axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN] =
      AxisLayout(1, 16, 1, 1);
  return layout;
}

void ExpectCoordinate(const loom_matrix_fragment_layout_t* layout,
                      loom_contract_operand_role_t role, uint16_t lane,
                      uint16_t register_index, uint16_t element_index,
                      loom_matrix_fragment_coordinate_flags_t coordinate_flags,
                      uint16_t row, uint16_t column, uint16_t reduction) {
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  ASSERT_NE(role_layout, nullptr);
  const uint16_t elements_per_register =
      (uint16_t)(32 / role_layout->element_bit_count);
  const uint16_t payload_element_index =
      (uint16_t)(register_index * elements_per_register + element_index);
  loom_matrix_fragment_coordinate_t coordinate = {};
  ASSERT_TRUE(loom_matrix_fragment_coordinate(
      layout, role, lane, payload_element_index, &coordinate));
  EXPECT_EQ(coordinate.coordinate_flags, coordinate_flags);
  EXPECT_EQ(coordinate.row, row);
  EXPECT_EQ(coordinate.column, column);
  EXPECT_EQ(coordinate.reduction, reduction);
}

void ExpectPayloadCoordinate(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane,
    uint16_t payload_element_index,
    loom_matrix_fragment_coordinate_flags_t coordinate_flags, uint16_t block,
    uint16_t row, uint16_t column, uint16_t reduction) {
  loom_matrix_fragment_coordinate_t coordinate = {};
  ASSERT_TRUE(loom_matrix_fragment_coordinate(
      layout, role, lane, payload_element_index, &coordinate));
  EXPECT_EQ(coordinate.coordinate_flags, coordinate_flags);
  EXPECT_EQ(coordinate.block, block);
  EXPECT_EQ(coordinate.row, row);
  EXPECT_EQ(coordinate.column, column);
  EXPECT_EQ(coordinate.reduction, reduction);
}

loom_matrix_fragment_coordinate_t RowReduction(uint16_t row,
                                               uint16_t reduction) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kRowReduction,
      /*.block=*/{},
      /*.row=*/row,
      /*.column=*/{},
      /*.reduction=*/reduction,
  };
}

loom_matrix_fragment_coordinate_t ColumnReduction(uint16_t column,
                                                  uint16_t reduction) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kColumnReduction,
      /*.block=*/{},
      /*.row=*/{},
      /*.column=*/column,
      /*.reduction=*/reduction,
  };
}

loom_matrix_fragment_coordinate_t RowColumn(uint16_t row, uint16_t column) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kRowColumn,
      /*.block=*/{},
      /*.row=*/row,
      /*.column=*/column,
  };
}

void ExpectPhysicalElement(const loom_matrix_fragment_layout_t* layout,
                           loom_contract_operand_role_t role,
                           loom_matrix_fragment_coordinate_t coordinate,
                           uint16_t occurrence_index, uint16_t lane,
                           uint16_t register_index, uint16_t element_index) {
  loom_matrix_fragment_physical_element_t element = {};
  ASSERT_TRUE(loom_matrix_fragment_physical_element(
      layout, role, coordinate, occurrence_index, &element));
  EXPECT_EQ(element.lane, lane);
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  ASSERT_NE(role_layout, nullptr);
  const uint16_t elements_per_register =
      (uint16_t)(32 / role_layout->element_bit_count);
  EXPECT_EQ(element.payload_element_index,
            register_index * elements_per_register + element_index);
}

void ExpectPhysicalElementCount(const loom_matrix_fragment_layout_t* layout,
                                loom_contract_operand_role_t role,
                                loom_matrix_fragment_coordinate_t coordinate,
                                uint16_t count) {
  uint16_t actual_count = 0;
  ASSERT_TRUE(loom_matrix_fragment_physical_element_count(
      layout, role, coordinate, &actual_count));
  EXPECT_EQ(actual_count, count);
}

TEST(MatrixFragmentLayoutTest, FindsRoleLayouts) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  const loom_matrix_fragment_role_layout_t* lhs =
      loom_matrix_fragment_role_layout(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(lhs->coordinate_flags, kRowReduction);
  EXPECT_EQ(lhs->axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW].thread_count, 16);
  EXPECT_EQ(lhs->axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION].element_count, 16);
  EXPECT_EQ(loom_matrix_fragment_role_layout(
                &layout, LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN),
            nullptr);
  EXPECT_EQ(
      loom_matrix_fragment_role_layout(nullptr, LOOM_CONTRACT_OPERAND_ROLE_LHS),
      nullptr);
}

TEST(MatrixFragmentLayoutTest, MapsLaneModPackedReductionCoordinates) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0,
                   kRowReduction, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 15, 0, 1,
                   kRowReduction, 15, 0, 1);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 16, 7, 1,
                   kRowReduction, 0, 0, 15);

  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 0, 0, 0,
                   kColumnReduction, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 15, 0, 1,
                   kColumnReduction, 0, 15, 1);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 31, 7, 1,
                   kColumnReduction, 0, 15, 15);
}

TEST(MatrixFragmentLayoutTest, MapsRegisterInterleavedRowColumnCoordinates) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 0,
                   kRowColumn, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15, 0, 0,
                   kRowColumn, 0, 15, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 0, 0,
                   kRowColumn, 1, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 7, 0,
                   kRowColumn, 15, 15, 0);
}

TEST(MatrixFragmentLayoutTest,
     MapsWave64RegisterInterleavedRowColumnCoordinates) {
  loom_matrix_fragment_layout_t layout = RdnaWave64InterleavedLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 0,
                   kRowColumn, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15, 1, 0,
                   kRowColumn, 4, 15, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 0, 0,
                   kRowColumn, 1, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 3, 0,
                   kRowColumn, 15, 15, 0);
}

TEST(MatrixFragmentLayoutTest, MapsRegisterInterleavedLowSubwordCoordinates) {
  loom_matrix_fragment_layout_t layout = RdnaLowSubwordLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 0,
                   kRowColumn, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15, 0, 0,
                   kRowColumn, 0, 15, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 0, 0,
                   kRowColumn, 1, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 7, 0,
                   kRowColumn, 15, 15, 0);

  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 15, &coordinate));
  ExpectPhysicalElementCount(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                             RowColumn(15, 15), 1);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                        RowColumn(15, 15), 0, 31, 7, 0);
}

TEST(MatrixFragmentLayoutTest, MapsLaneGroupPackedReductionCoordinates) {
  loom_matrix_fragment_layout_t layout = CdnaLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0,
                   kRowReduction, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 15, 1, 1,
                   kRowReduction, 15, 0, 3);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 16, 0, 0,
                   kRowReduction, 0, 0, 4);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 63, 1, 1,
                   kRowReduction, 15, 0, 15);

  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 0, 0, 0,
                   kColumnReduction, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 15, 1, 1,
                   kColumnReduction, 0, 15, 3);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 16, 0, 0,
                   kColumnReduction, 0, 0, 4);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 63, 1, 1,
                   kColumnReduction, 0, 15, 15);
}

TEST(MatrixFragmentLayoutTest, MapsLaneGroupRegisterRowColumnCoordinates) {
  loom_matrix_fragment_layout_t layout = CdnaLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 0,
                   kRowColumn, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15, 3, 0,
                   kRowColumn, 3, 15, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 0, 0,
                   kRowColumn, 4, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 3, 0,
                   kRowColumn, 15, 15, 0);
}

TEST(MatrixFragmentLayoutTest, MapsLaneGroupLowSubwordCoordinates) {
  loom_matrix_fragment_layout_t layout = LaneGroupLowSubwordLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 0,
                   kRowColumn, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15, 3, 0,
                   kRowColumn, 3, 15, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 0, 0,
                   kRowColumn, 4, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 3, 0,
                   kRowColumn, 15, 15, 0);

  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 7, &coordinate));
  ExpectPhysicalElementCount(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                             RowColumn(15, 15), 1);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                        RowColumn(15, 15), 0, 63, 3, 0);
}

TEST(MatrixFragmentLayoutTest, MapsLaneGroupPackedRowCoordinates) {
  loom_matrix_fragment_layout_t layout = LaneGroupPackedRowLayout();
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 0,
                   kRowColumn, 0, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0, 1,
                   kRowColumn, 1, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15, 3, 1,
                   kRowColumn, 7, 15, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16, 0, 0,
                   kRowColumn, 8, 0, 0);
  ExpectCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 3, 1,
                   kRowColumn, 15, 15, 0);

  ExpectPhysicalElementCount(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                             RowColumn(15, 15), 1);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                        RowColumn(15, 15), 0, 31, 3, 1);
}

TEST(MatrixFragmentLayoutTest, MapsIndependentBlockedCoordinates) {
  loom_matrix_fragment_layout_t layout = Blocked16x16Layout();

  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0,
                          kBlockRowReduction, 0, 0, 0, 0);
  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 63, 3,
                          kBlockRowReduction, 3, 15, 0, 3);
  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 31, 2,
                          kBlockColumnReduction, 1, 0, 15, 2);

  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0,
                          kBlockRowColumn, 0, 0, 0, 0);
  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 0, 4,
                          kBlockRowColumn, 1, 0, 0, 0);
  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 15,
                          kBlockRowColumn, 3, 15, 15, 0);
}

TEST(MatrixFragmentLayoutTest, RejectsPaddingPayloadElements) {
  loom_matrix_fragment_layout_t layout = Blocked16x16Layout();
  layout.result.payload_element_count = 32;
  layout.result.coordinate_element_offset = 0;
  layout.result.coordinate_element_stride = 2;

  ExpectPayloadCoordinate(&layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 30,
                          kBlockRowColumn, 3, 15, 15, 0);
  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 31, &coordinate));
}

TEST(MatrixFragmentLayoutTest, RejectsOutOfRangeElements) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
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

TEST(MatrixFragmentLayoutTest, FindsReplicatedRdnaPhysicalElements) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  ExpectPhysicalElementCount(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                             RowReduction(0, 0), 2);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                        RowReduction(0, 0), 0, 0, 0, 0);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                        RowReduction(0, 0), 1, 16, 0, 0);
  ExpectPhysicalElementCount(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
                             ColumnReduction(15, 15), 2);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
                        ColumnReduction(15, 15), 0, 15, 7, 1);
  ExpectPhysicalElement(&layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
                        ColumnReduction(15, 15), 1, 31, 7, 1);
}

TEST(MatrixFragmentLayoutTest, FindsUniqueResultPhysicalElements) {
  loom_matrix_fragment_layout_t rdna_layout = RdnaLayout();
  ExpectPhysicalElementCount(&rdna_layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                             RowColumn(15, 15), 1);
  ExpectPhysicalElement(&rdna_layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                        RowColumn(15, 15), 0, 31, 7, 0);

  loom_matrix_fragment_layout_t cdna_layout = CdnaLayout();
  ExpectPhysicalElementCount(&cdna_layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                             RowReduction(0, 4), 1);
  ExpectPhysicalElement(&cdna_layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
                        RowReduction(0, 4), 0, 16, 0, 0);
  ExpectPhysicalElementCount(
      &cdna_layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, RowColumn(4, 0), 1);
  ExpectPhysicalElement(&cdna_layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
                        RowColumn(4, 0), 0, 16, 0, 0);
}

TEST(MatrixFragmentLayoutTest, RejectsMissingPhysicalElements) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
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
