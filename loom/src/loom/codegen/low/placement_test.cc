// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/placement.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowPlacementTest, ClassifiesEdgeCauses) {
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT));
  EXPECT_TRUE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH));
  EXPECT_TRUE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY));
  EXPECT_TRUE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD));
  EXPECT_TRUE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION));
  EXPECT_FALSE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY));
}

TEST(LowPlacementTest, ComposedRelationHasNoDirectSourceOperand) {
  loom_low_placement_relation_t source_to_intermediate = {
      .result_ordinal = 1,
      .source_ordinal = 0,
      .result_unit_offset = 2,
      .source_unit_offset = 4,
      .unit_count = 3,
      .source_operand_index = 1};
  loom_low_placement_relation_t intermediate_to_result = {
      .result_ordinal = 2,
      .source_ordinal = 1,
      .result_unit_offset = 8,
      .source_unit_offset = 3,
      .unit_count = 3,
      .source_operand_index = 2};

  loom_low_placement_relation_t composed = {};
  ASSERT_TRUE(loom_low_placement_relation_compose(
      &source_to_intermediate, &intermediate_to_result, &composed));
  EXPECT_EQ(composed.source_ordinal, 0u);
  EXPECT_EQ(composed.result_ordinal, 2u);
  EXPECT_EQ(composed.source_unit_offset, 5u);
  EXPECT_EQ(composed.result_unit_offset, 8u);
  EXPECT_EQ(composed.unit_count, 2u);
  EXPECT_EQ(composed.source_operand_index,
            LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE);
}

TEST(LowPlacementTest, StorageCompositionDoesNotImplyBitIdentity) {
  loom_low_placement_relation_t write = {};
  write.source_ordinal = 0;
  write.result_ordinal = 1;
  write.unit_count = 2;
  write.cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
  write.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
                LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE;
  loom_low_placement_relation_t edge = {};
  edge.source_ordinal = 1;
  edge.result_ordinal = 2;
  edge.unit_count = 2;
  edge.cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH;
  edge.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
               LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE;

  loom_low_placement_relation_t composed = {};
  ASSERT_TRUE(loom_low_placement_relation_compose(&write, &edge, &composed));
  EXPECT_EQ(composed.source_ordinal, 0u);
  EXPECT_EQ(composed.result_ordinal, 2u);
  EXPECT_EQ(composed.flags, LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE);
}

}  // namespace
}  // namespace loom
