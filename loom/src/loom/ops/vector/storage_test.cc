// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/storage.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(VectorStorageTest, StaticRank1LaneCountAcceptsMatchingVector) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                         loom_dim_pack_static(4), 0);

  EXPECT_EQ(loom_vector_static_rank1_lane_count(type, LOOM_SCALAR_TYPE_I32,
                                                /*maximum_lane_count=*/8),
            4u);
}

TEST(VectorStorageTest, StaticRank1LaneCountRejectsWrongShape) {
  loom_type_t dynamic_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(0), 0);
  loom_type_t rank2_type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(2), loom_dim_pack_static(2), 0);
  loom_type_t f32_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t over_limit_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(9), 0);

  EXPECT_EQ(loom_vector_static_rank1_lane_count(dynamic_type,
                                                LOOM_SCALAR_TYPE_I32, 8),
            0u);
  EXPECT_EQ(
      loom_vector_static_rank1_lane_count(rank2_type, LOOM_SCALAR_TYPE_I32, 8),
      0u);
  EXPECT_EQ(
      loom_vector_static_rank1_lane_count(f32_type, LOOM_SCALAR_TYPE_I32, 8),
      0u);
  EXPECT_EQ(loom_vector_static_rank1_lane_count(over_limit_type,
                                                LOOM_SCALAR_TYPE_I32, 8),
            0u);
}

TEST(VectorStorageTest, PackedIntegerStorageShapeRoundsToStorageUnits) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                                         loom_dim_pack_static(5), 0);

  loom_vector_packed_integer_storage_shape_t shape;
  ASSERT_TRUE(loom_vector_packed_integer_storage_shape(
      type, /*storage_unit_bit_count=*/32, /*maximum_storage_unit_count=*/2,
      &shape));
  EXPECT_TRUE(loom_type_equal(shape.type, type));
  EXPECT_EQ(shape.element_type, LOOM_SCALAR_TYPE_I8);
  EXPECT_EQ(shape.lane_count, 5u);
  EXPECT_EQ(shape.element_bit_count, 8u);
  EXPECT_EQ(shape.payload_bit_count, 40u);
  EXPECT_EQ(shape.storage_unit_bit_count, 32u);
  EXPECT_EQ(shape.storage_unit_count, 2u);
}

TEST(VectorStorageTest, PackedIntegerStorageShapeRejectsInvalidShapes) {
  loom_vector_packed_integer_storage_shape_t shape;
  loom_type_t f32_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t dynamic_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_dynamic(0), 0);
  loom_type_t over_limit_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(9), 0);

  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(f32_type, 32, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(dynamic_type, 32, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(over_limit_type, 32, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(over_limit_type, 0, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(over_limit_type, 32, 0, &shape));
}

TEST(VectorStorageTest, PackedIntegerPayloadFromLanesAcceptsPackedPayload) {
  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(4), 0);

  loom_vector_packed_integer_payload_from_lanes_match_t match;
  ASSERT_TRUE(loom_vector_packed_integer_payload_from_lanes_match(
      source_type, result_type, /*width=*/4,
      /*storage_unit_bit_count=*/32, /*maximum_result_storage_unit_count=*/1,
      &match));
  EXPECT_EQ(match.width, 4u);
  EXPECT_EQ(match.source_shape.lane_count, 8u);
  EXPECT_EQ(match.result_shape.payload_bit_count, 32u);
  EXPECT_EQ(match.result_shape.storage_unit_count, 1u);
}

TEST(VectorStorageTest,
     PackedIntegerPayloadFromLanesRejectsInvalidStorageRelations) {
  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(4), 0);
  loom_type_t oversized_result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(8), 0);

  EXPECT_FALSE(loom_vector_packed_integer_payload_from_lanes_match(
      source_type, result_type, /*width=*/3, /*storage_unit_bit_count=*/32,
      /*maximum_result_storage_unit_count=*/1, NULL));
  EXPECT_FALSE(loom_vector_packed_integer_payload_from_lanes_match(
      source_type, oversized_result_type, /*width=*/4,
      /*storage_unit_bit_count=*/32, /*maximum_result_storage_unit_count=*/1,
      NULL));
}

TEST(VectorStorageTest, PackedIntegerLanesFromPayloadAcceptsUnpackedLanes) {
  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(1), 0);
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);

  loom_vector_packed_integer_lanes_from_payload_match_t match;
  ASSERT_TRUE(loom_vector_packed_integer_lanes_from_payload_match(
      source_type, result_type, /*width=*/4, /*storage_unit_bit_count=*/32,
      /*maximum_source_storage_unit_count=*/1, /*maximum_lane_count=*/8,
      &match));
  EXPECT_EQ(match.width, 4u);
  EXPECT_EQ(match.lane_count, 8u);
  EXPECT_EQ(match.source_shape.payload_bit_count, 32u);
  EXPECT_EQ(match.result_shape.lane_count, 8u);
}

TEST(VectorStorageTest, PackedIntegerLanesFromPayloadAcceptsPackedResultLanes) {
  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(1), 0);
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(4), 0);

  loom_vector_packed_integer_lanes_from_payload_match_t match;
  ASSERT_TRUE(loom_vector_packed_integer_lanes_from_payload_match(
      source_type, result_type, /*width=*/8, /*storage_unit_bit_count=*/32,
      /*maximum_source_storage_unit_count=*/1, /*maximum_lane_count=*/4,
      &match));
  EXPECT_EQ(match.lane_count, 4u);
  EXPECT_EQ(match.result_shape.element_bit_count, 8u);
  EXPECT_EQ(match.result_shape.storage_unit_count, 1u);
}

TEST(VectorStorageTest,
     PackedIntegerLanesFromPayloadRejectsInvalidStorageRelations) {
  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(1), 0);
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
  loom_type_t short_result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(7), 0);

  EXPECT_FALSE(loom_vector_packed_integer_lanes_from_payload_match(
      source_type, result_type, /*width=*/3, /*storage_unit_bit_count=*/32,
      /*maximum_source_storage_unit_count=*/1, /*maximum_lane_count=*/8, NULL));
  EXPECT_FALSE(loom_vector_packed_integer_lanes_from_payload_match(
      source_type, result_type, /*width=*/4, /*storage_unit_bit_count=*/32,
      /*maximum_source_storage_unit_count=*/1, /*maximum_lane_count=*/7, NULL));
  EXPECT_FALSE(loom_vector_packed_integer_lanes_from_payload_match(
      source_type, short_result_type, /*width=*/4,
      /*storage_unit_bit_count=*/32, /*maximum_source_storage_unit_count=*/1,
      /*maximum_lane_count=*/8, NULL));
}

}  // namespace
}  // namespace loom
