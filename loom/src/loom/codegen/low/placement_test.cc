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
  EXPECT_TRUE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_FOR));
  EXPECT_TRUE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD));
  EXPECT_FALSE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY));
}

}  // namespace
}  // namespace loom
