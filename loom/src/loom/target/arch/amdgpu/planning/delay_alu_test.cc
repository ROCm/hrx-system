// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/delay_alu.h"

#include "iree/testing/gtest.h"

namespace {

loom_amdgpu_delay_alu_match_t Match(const loom_amdgpu_delay_alu_state_t& state,
                                    const loom_amdgpu_delay_alu_info_t& info) {
  loom_amdgpu_delay_alu_accumulator_t accumulator = {};
  loom_amdgpu_delay_alu_accumulate_info(&state, 0, &info, &accumulator);
  return loom_amdgpu_delay_alu_select(&accumulator);
}

TEST(DelayAluTest, CompletionFollowsExactProducerAcrossIssueAndReset) {
  for (auto type :
       {LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS}) {
    loom_amdgpu_delay_alu_state_t state = {};
    EXPECT_FALSE(loom_amdgpu_delay_alu_reset(&state));
    const auto first = loom_amdgpu_delay_alu_make_info(&state, 0, type, 8, 17);
    loom_amdgpu_delay_alu_advance(&state, type, 1);
    const auto first_wait = Match(state, first);
    ASSERT_NE(first_wait.cycle_count, 0);
    loom_amdgpu_delay_alu_complete(&state, first_wait.delay_alu_immediate);

    const auto second = loom_amdgpu_delay_alu_make_info(&state, 1, type, 8, 18);
    loom_amdgpu_delay_alu_advance(&state, type, 1);
    EXPECT_EQ(Match(state, first).cycle_count, 0);
    EXPECT_EQ(Match(state, second).delay_alu_immediate,
              first_wait.delay_alu_immediate);

    // A structural packet can advance beyond the entire encoded window.
    loom_amdgpu_delay_alu_advance(&state, type, 64);
    EXPECT_EQ(Match(state, first).cycle_count, 0);
    EXPECT_EQ(Match(state, second).cycle_count, 0);
    const auto third = loom_amdgpu_delay_alu_make_info(&state, 66, type, 8, 19);
    loom_amdgpu_delay_alu_advance(&state, type, 1);
    EXPECT_EQ(Match(state, third).delay_alu_immediate,
              first_wait.delay_alu_immediate);

    EXPECT_FALSE(loom_amdgpu_delay_alu_reset(&state));
    EXPECT_EQ(Match(state, third).cycle_count, 0);
    const auto next = loom_amdgpu_delay_alu_make_info(&state, 0, type, 8, 20);
    loom_amdgpu_delay_alu_advance(&state, type, 1);
    EXPECT_EQ(Match(state, next).delay_alu_immediate,
              first_wait.delay_alu_immediate);
  }
}

TEST(DelayAluTest, ScalarCycleSelectorsDoNotCompleteVectorProducers) {
  loom_amdgpu_delay_alu_state_t state = {};
  EXPECT_FALSE(loom_amdgpu_delay_alu_reset(&state));
  const auto valu = loom_amdgpu_delay_alu_make_info(
      &state, 0, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, 4, 1);
  const auto trans = loom_amdgpu_delay_alu_make_info(
      &state, 0, LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS, 8, 2);
  loom_amdgpu_delay_alu_advance(&state, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, 1);
  loom_amdgpu_delay_alu_advance(&state, LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS, 1);
  loom_amdgpu_delay_alu_complete(&state, 9 | (11 << 7));
  EXPECT_EQ(Match(state, valu).delay_alu_immediate, 1);
  EXPECT_EQ(Match(state, trans).delay_alu_immediate, 5);
}

TEST(DelayAluTest, EpochWrapRequestsClearingRetainedLocationFacts) {
  loom_amdgpu_delay_alu_state_t state = {.epoch = UINT32_MAX};
  loom_amdgpu_delay_alu_complete(&state, 1 | (5 << 7));
  EXPECT_TRUE(loom_amdgpu_delay_alu_reset(&state));
  EXPECT_NE(state.epoch, 0);
  const auto next = loom_amdgpu_delay_alu_make_info(
      &state, 0, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, 4, 1);
  loom_amdgpu_delay_alu_advance(&state, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, 1);
  EXPECT_EQ(Match(state, next).delay_alu_immediate, 1);
}

}  // namespace
