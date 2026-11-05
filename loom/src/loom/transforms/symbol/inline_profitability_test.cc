// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_profitability.h"

#include "iree/testing/gtest.h"

namespace {

using ::testing::Test;

constexpr loom_inline_profitability_options_t kDefaults = {
    .budget = LOOM_INLINE_PROFITABILITY_DEFAULT_BUDGET,
    .hot_budget = LOOM_INLINE_PROFITABILITY_DEFAULT_HOT_BUDGET,
};

TEST(InlineProfitabilityTest, SelectsWholeDefaultCohortAtBoundary) {
  auto selected = loom_inline_profitability_select(
      kDefaults, /*expanded_body_cost=*/16, /*ordinary_call_count=*/4,
      /*hot_call_count=*/4);
  EXPECT_TRUE(selected.ordinary);
  EXPECT_TRUE(selected.hot);

  auto retained = loom_inline_profitability_select(
      kDefaults, /*expanded_body_cost=*/17, /*ordinary_call_count=*/4,
      /*hot_call_count=*/4);
  EXPECT_FALSE(retained.ordinary);
  EXPECT_TRUE(retained.hot);
}

TEST(InlineProfitabilityTest, SelectsOnlyHotCohortAtHotBoundary) {
  auto selected = loom_inline_profitability_select(
      kDefaults, /*expanded_body_cost=*/128, /*ordinary_call_count=*/2,
      /*hot_call_count=*/4);
  EXPECT_FALSE(selected.ordinary);
  EXPECT_TRUE(selected.hot);

  auto retained = loom_inline_profitability_select(
      kDefaults, /*expanded_body_cost=*/129, /*ordinary_call_count=*/2,
      /*hot_call_count=*/4);
  EXPECT_FALSE(retained.ordinary);
  EXPECT_FALSE(retained.hot);
}

TEST(InlineProfitabilityTest, SingleCallStillRespectsMaterializationBudget) {
  auto selected = loom_inline_profitability_select(
      kDefaults, /*expanded_body_cost=*/128, /*ordinary_call_count=*/1,
      /*hot_call_count=*/0);
  EXPECT_TRUE(selected.ordinary);

  auto retained = loom_inline_profitability_select(
      kDefaults, /*expanded_body_cost=*/129, /*ordinary_call_count=*/1,
      /*hot_call_count=*/0);
  EXPECT_FALSE(retained.ordinary);
}

TEST(InlineProfitabilityTest, ExpansionArithmeticSaturates) {
  EXPECT_EQ(loom_inline_profitability_add_expansion(8, 1), 8u);
  EXPECT_EQ(loom_inline_profitability_add_expansion(UINT32_MAX - 2, 8),
            UINT32_MAX);

  auto retained = loom_inline_profitability_select(kDefaults, UINT32_MAX,
                                                   UINT32_MAX, UINT32_MAX);
  EXPECT_FALSE(retained.ordinary);
  EXPECT_FALSE(retained.hot);
}

}  // namespace
