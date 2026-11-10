// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
struct State {
  std::array<int, 3> history = {};
  std::array<int, 2> hidden = {1, 7};
  std::vector<std::pair<uint32_t, uint32_t>> order;
  static iree_status_t Emit(void* self, uint32_t bank, uint32_t stage) {
    auto& state = *static_cast<State*>(self);
    state.order.emplace_back(bank, stage);
    auto& value = state.hidden[bank];
    if (stage % 3 == 0) {
      value += 3 * state.history[stage / 3];
      state.history[stage / 3] = value;
    } else if (stage % 3 == 1) {
      value *= 2;
    } else {
      value += 1;
    }
    return iree_ok_status();
  }
};

TEST(ExecutionPlanTest, InterleaveMatchesIndependentSerialStateAndQuality) {
  uint32_t stages[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  loom_serve_plan_description_t desc = {1, 3, 9, stages};
  loom_serve_execution_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_plan_create(
      &desc, true, iree_allocator_system(), &plan));
  // Source descriptor storage can be reused immediately after preparation.
  stages[0] = 999;
  State grouped;
  IREE_ASSERT_OK(
      loom_serve_execution_plan_encode(plan, 0, 0, State::Emit, &grouped));
  State serial;
  for (unsigned bank = 0; bank < 2; ++bank) {
    for (unsigned stage = 0; stage < 9; ++stage) {
      IREE_ASSERT_OK(State::Emit(&serial, bank, stage));
    }
  }
  EXPECT_EQ(grouped.hidden, serial.hidden);
  EXPECT_EQ(grouped.history, serial.history);
  ASSERT_EQ(grouped.order.size(), 18u);
  EXPECT_EQ(grouped.order[2], std::make_pair(1u, 0u));
  for (unsigned layer = 0; layer < 3; ++layer) {
    auto a = std::find(grouped.order.begin(), grouped.order.end(),
                       std::make_pair(0u, layer * 3));
    auto b = std::find(grouped.order.begin(), grouped.order.end(),
                       std::make_pair(1u, layer * 3));
    EXPECT_LT(a, b);
  }
  // Teacher-forced toy logits use the independent retained-state oracle.
  // Exact scheduling has a zero KL budget and unchanged target-token NLL.
  double kl = 0, nll_a = 0, nll_b = 0;
  for (unsigned bank = 0; bank < 2; ++bank) {
    double za = 0, zb = 0;
    for (unsigned token = 0; token < 5; ++token) {
      za += std::exp(double((serial.hidden[bank] + token) % 7));
      zb += std::exp(double((grouped.hidden[bank] + token) % 7));
    }
    for (unsigned token = 0; token < 5; ++token) {
      double pa = std::exp(double((serial.hidden[bank] + token) % 7)) / za;
      double pb = std::exp(double((grouped.hidden[bank] + token) % 7)) / zb;
      kl += pa * std::log(pa / pb);
      if (token == bank) {
        nll_a -= std::log(pa);
        nll_b -= std::log(pb);
      }
    }
  }
  EXPECT_DOUBLE_EQ(kl, 0);
  EXPECT_DOUBLE_EQ(std::exp((nll_b - nll_a) / 2), 1);
  loom_serve_execution_plan_release(plan);
}

TEST(ExecutionPlanTest, ShapesAndSequentialProviderUseSameContract) {
  const uint32_t stages[] = {0, UINT32_MAX, 1, 2, UINT32_MAX, 3};
  const loom_serve_plan_description_t desc = {2, 1, 4, stages};
  loom_serve_execution_plan_t* plan = nullptr;
  const auto* provider = loom_serve_sequential_plan_provider();
  IREE_ASSERT_OK(
      provider->prepare(provider->self, &desc, iree_allocator_system(), &plan));
  loom_serve_plan_group_query_t query = {.shapes = {1, 0}, .tokens = {3, 2}};
  EXPECT_TRUE(provider->select(provider->self, plan, &query));
  query.shapes[0] = 2;
  EXPECT_FALSE(provider->select(provider->self, plan, &query));
  query.shapes[0] = 1;
  query.tokens[1] = 0;
  EXPECT_FALSE(provider->select(provider->self, plan, &query));
  State state;
  IREE_ASSERT_OK(
      loom_serve_execution_plan_encode(plan, 1, 0, State::Emit, &state));
  EXPECT_EQ(state.order, (std::vector<std::pair<uint32_t, uint32_t>>{
                             {0, 2}, {0, 3}, {1, 0}, {1, 1}}));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_execution_plan_encode(plan, 2, 0, State::Emit, &state));
  loom_serve_execution_plan_release(plan);
}

TEST(ExecutionPlanTest, InvalidSourceDescriptorsFailBeforeEmission) {
  uint32_t stage = UINT32_MAX;
  loom_serve_plan_description_t desc = {1, 1, 1, &stage};
  loom_serve_execution_plan_t* plan = nullptr;
  // The invalid table is large enough; the first required prefix rejects it.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);
  desc.layer_count = 1025;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
}
}  // namespace
