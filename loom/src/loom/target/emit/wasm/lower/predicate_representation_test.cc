// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower/predicate_representation.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

loom_type_t Vector1D(loom_scalar_type_t element_type, int64_t lane_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(lane_count),
                             /*encoding_id=*/0);
}

loom_type_t Vector2D(loom_scalar_type_t element_type, int64_t outer_count,
                     int64_t inner_count) {
  return loom_type_shaped_2d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(outer_count),
                             loom_dim_pack_static(inner_count),
                             /*encoding_id=*/0);
}

TEST(PredicateRepresentationTest, CoversEveryOnePacketLaneWidth) {
  static constexpr loom_low_representation_id_t kRepresentations[] = {
      LOOM_WASM_PREDICATE_REPRESENTATION_I8X16,
      LOOM_WASM_PREDICATE_REPRESENTATION_I16X8,
      LOOM_WASM_PREDICATE_REPRESENTATION_I32X4,
      LOOM_WASM_PREDICATE_REPRESENTATION_I64X2,
  };
  for (uint32_t lane_count = 1; lane_count <= 16; ++lane_count) {
    const loom_type_t type = Vector1D(LOOM_SCALAR_TYPE_I1, (int64_t)lane_count);
    uint32_t actual_lane_count = 0;
    ASSERT_TRUE(loom_wasm_predicate_type(type, &actual_lane_count));
    EXPECT_EQ(actual_lane_count, lane_count);

    loom_low_representation_candidate_t
        candidates[LOOM_WASM_PREDICATE_REPRESENTATION_COUNT];
    const iree_host_size_t candidate_count =
        loom_wasm_predicate_representation_candidates(
            type, LOOM_LOW_REPRESENTATION_ID_NONE, candidates);
    iree_host_size_t expected_candidate_count = 0;
    loom_low_representation_id_t expected_default =
        LOOM_LOW_REPRESENTATION_ID_NONE;
    for (loom_low_representation_id_t representation : kRepresentations) {
      const bool expected = lane_count * representation <= 128;
      EXPECT_EQ(loom_wasm_predicate_representation_available(lane_count,
                                                             representation),
                expected);
      if (!expected) {
        EXPECT_EQ(
            loom_wasm_predicate_carrier(type, representation).packet_count, 0);
        continue;
      }
      ASSERT_LT(expected_candidate_count, candidate_count);
      EXPECT_EQ(candidates[expected_candidate_count].representation,
                representation);
      EXPECT_EQ(candidates[expected_candidate_count].cost.runtime, 0);
      EXPECT_EQ(candidates[expected_candidate_count].cost.code_size, 0);
      ++expected_candidate_count;
      expected_default = representation;

      const loom_wasm_vector_carrier_t carrier =
          loom_wasm_predicate_carrier(type, representation);
      EXPECT_EQ(carrier.element_bit_count, representation);
      EXPECT_EQ(carrier.payload_byte_count, lane_count * representation / 8u);
      EXPECT_EQ(carrier.packet_count, 1);
    }
    EXPECT_EQ(candidate_count, expected_candidate_count);
    EXPECT_EQ(loom_wasm_predicate_default_representation(type),
              expected_default);
    EXPECT_EQ(loom_wasm_predicate_carrier(type, LOOM_LOW_REPRESENTATION_ID_NONE)
                  .element_bit_count,
              expected_default);
    EXPECT_EQ(loom_wasm_vector_carrier_for_type(type).element_bit_count,
              expected_default);
  }
}

TEST(PredicateRepresentationTest, PricesOneShuffleFromNativeWidth) {
  const loom_type_t type = Vector1D(LOOM_SCALAR_TYPE_I1, 2);
  loom_low_representation_candidate_t
      candidates[LOOM_WASM_PREDICATE_REPRESENTATION_COUNT];
  const iree_host_size_t candidate_count =
      loom_wasm_predicate_representation_candidates(
          type, LOOM_WASM_PREDICATE_REPRESENTATION_I32X4, candidates);
  ASSERT_EQ(candidate_count, 4);
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (candidates[i].representation ==
        LOOM_WASM_PREDICATE_REPRESENTATION_I32X4) {
      EXPECT_EQ(candidates[i].cost.runtime, 0);
      EXPECT_EQ(candidates[i].cost.code_size, 0);
    } else {
      EXPECT_EQ(candidates[i].cost.runtime, 1);
      EXPECT_EQ(candidates[i].cost.code_size, 18);
    }
  }
}

TEST(PredicateRepresentationTest, UsesElementCountAcrossLogicalShapes) {
  const loom_type_t type = Vector2D(LOOM_SCALAR_TYPE_I1, 2, 4);
  uint32_t element_count = 0;
  ASSERT_TRUE(loom_wasm_predicate_type(type, &element_count));
  EXPECT_EQ(element_count, 8);
  EXPECT_EQ(loom_wasm_predicate_default_representation(type),
            LOOM_WASM_PREDICATE_REPRESENTATION_I16X8);
  const loom_wasm_vector_carrier_t carrier = loom_wasm_predicate_carrier(
      type, LOOM_WASM_PREDICATE_REPRESENTATION_I8X16);
  EXPECT_EQ(carrier.element_bit_count, 8);
  EXPECT_EQ(carrier.payload_byte_count, 8);
  EXPECT_EQ(carrier.packet_count, 1);
}

TEST(PredicateRepresentationTest, RejectsNonPredicateAndOverwideShapes) {
  EXPECT_FALSE(
      loom_wasm_predicate_type(Vector1D(LOOM_SCALAR_TYPE_I8, 16), nullptr));
  EXPECT_FALSE(
      loom_wasm_predicate_type(Vector1D(LOOM_SCALAR_TYPE_I1, 17), nullptr));
  EXPECT_EQ(loom_wasm_predicate_default_representation(
                Vector1D(LOOM_SCALAR_TYPE_I1, 17)),
            LOOM_LOW_REPRESENTATION_ID_NONE);
}

}  // namespace
