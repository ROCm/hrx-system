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

loom_matrix_fragment_role_layout_t RoleLayout(
    loom_contract_operand_role_t role, loom_matrix_fragment_map_kind_t map_kind,
    uint16_t register_count, uint16_t elements_per_register,
    loom_matrix_fragment_coordinate_flags_t coordinate_flags) {
  return (loom_matrix_fragment_role_layout_t){
      /*.role=*/role,
      /*.map_kind=*/map_kind,
      /*.register_count=*/register_count,
      /*.elements_per_register=*/elements_per_register,
      /*.element_bit_count=*/16,
      /*.coordinate_flags=*/coordinate_flags,
  };
}

loom_matrix_fragment_layout_t RdnaLayout() {
  return (loom_matrix_fragment_layout_t){
      /*.kind=*/1,
      /*.name=*/IREE_SV("test.rdna"),
      /*.wave_size=*/32,
      /*.tile_shape=*/
      {
          /*.result_row_count=*/16,
          /*.result_column_count=*/16,
          /*.reduction_count=*/16,
      },
      /*.lhs=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_LHS,
                 LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION, 8, 2,
                 kRowReduction),
      /*.rhs=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RHS,
                 LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION, 8,
                 2, kColumnReduction),
      /*.accumulator=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
                 LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN, 8, 1,
                 kRowColumn),
      /*.result=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                 LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN, 8, 1,
                 kRowColumn),
  };
}

loom_matrix_fragment_layout_t CdnaLayout() {
  return (loom_matrix_fragment_layout_t){
      /*.kind=*/2,
      /*.name=*/IREE_SV("test.cdna"),
      /*.wave_size=*/64,
      /*.tile_shape=*/
      {
          /*.result_row_count=*/16,
          /*.result_column_count=*/16,
          /*.reduction_count=*/16,
      },
      /*.lhs=*/
      RoleLayout(
          LOOM_CONTRACT_OPERAND_ROLE_LHS,
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION, 2,
          2, kRowReduction),
      /*.rhs=*/
      RoleLayout(
          LOOM_CONTRACT_OPERAND_ROLE_RHS,
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION,
          2, 2, kColumnReduction),
      /*.accumulator=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
                 LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1,
                 kRowColumn),
      /*.result=*/
      RoleLayout(LOOM_CONTRACT_OPERAND_ROLE_RESULT,
                 LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1,
                 kRowColumn),
  };
}

void ExpectCoordinate(const loom_matrix_fragment_layout_t* layout,
                      loom_contract_operand_role_t role, uint16_t lane,
                      uint16_t register_index, uint16_t element_index,
                      loom_matrix_fragment_coordinate_flags_t coordinate_flags,
                      uint16_t row, uint16_t column, uint16_t reduction) {
  loom_matrix_fragment_coordinate_t coordinate = {};
  ASSERT_TRUE(loom_matrix_fragment_coordinate(
      layout, role, lane, register_index, element_index, &coordinate));
  EXPECT_EQ(coordinate.coordinate_flags, coordinate_flags);
  EXPECT_EQ(coordinate.row, row);
  EXPECT_EQ(coordinate.column, column);
  EXPECT_EQ(coordinate.reduction, reduction);
}

loom_matrix_fragment_coordinate_t RowReduction(uint16_t row,
                                               uint16_t reduction) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kRowReduction,
      /*.row=*/row,
      /*.column=*/{},
      /*.reduction=*/reduction,
  };
}

loom_matrix_fragment_coordinate_t ColumnReduction(uint16_t column,
                                                  uint16_t reduction) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kColumnReduction,
      /*.row=*/{},
      /*.column=*/column,
      /*.reduction=*/reduction,
  };
}

loom_matrix_fragment_coordinate_t RowColumn(uint16_t row, uint16_t column) {
  return (loom_matrix_fragment_coordinate_t){
      /*.coordinate_flags=*/kRowColumn,
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
  EXPECT_EQ(element.register_index, register_index);
  EXPECT_EQ(element.element_index, element_index);
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
  EXPECT_EQ(lhs->map_kind,
            LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION);
  EXPECT_EQ(lhs->coordinate_flags, kRowReduction);
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

TEST(MatrixFragmentLayoutTest, RejectsOutOfRangeElements) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  loom_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 32, 0, 0, &coordinate));
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 8, 0, &coordinate));
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 2, &coordinate));
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN, 0, 0, 0, &coordinate));
}

TEST(MatrixFragmentLayoutTest, RejectsInvalidLayouts) {
  loom_matrix_fragment_layout_t layout = RdnaLayout();
  loom_matrix_fragment_coordinate_t coordinate = {};

  layout.tile_shape.reduction_count = 15;
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 16, 7, 1, &coordinate));

  layout = RdnaLayout();
  layout.tile_shape.result_row_count = 0;
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0, &coordinate));

  layout = RdnaLayout();
  layout.lhs.map_kind = LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN;
  EXPECT_FALSE(loom_matrix_fragment_coordinate(
      &layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0, &coordinate));
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
