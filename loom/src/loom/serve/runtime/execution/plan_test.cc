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
  loom_serve_plan_description_t desc = {1, 3, 9, stages, 0};
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
  const loom_serve_plan_description_t desc = {2, 1, 4, stages, 0};
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
  loom_serve_plan_description_t desc = {1, 1, 1, &stage, 0};
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
struct Dag {
  struct Node {
    uint32_t bank;
    uint32_t stage;
    std::vector<uint32_t> predecessors;
  };
  std::vector<Node> nodes;
  uint32_t reject = UINT32_MAX;
  static iree_status_t Emit(void* self, uint32_t bank, uint32_t stage,
                            iree_host_size_t count,
                            const uint32_t* predecessors, uint32_t* out_node) {
    auto& dag = *static_cast<Dag*>(self);
    if (dag.nodes.size() == dag.reject) {
      return iree_make_status(IREE_STATUS_ABORTED, "injected append failure");
    }
    // Deliberately use nonzero ordinals, as uploads precede model leaves.
    *out_node = static_cast<uint32_t>(dag.nodes.size()) + 4;
    dag.nodes.push_back({bank, stage, {predecessors, predecessors + count}});
    return iree_ok_status();
  }
  uint32_t Find(uint32_t bank, uint32_t stage) const {
    for (uint32_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].bank == bank && nodes[i].stage == stage) {
        return i + 4;
      }
    }
    return UINT32_MAX;
  }
  bool Depends(uint32_t node, uint32_t predecessor) const {
    if (node == predecessor) {
      return true;
    }
    for (uint32_t edge : nodes.at(node - 4).predecessors) {
      if (Depends(edge, predecessor)) {
        return true;
      }
    }
    return false;
  }
};

TEST(ExecutionPlanTest, FfnForksJoinBothBanksBeforeResidualAndCapture) {
  uint32_t stages[32];
  for (uint32_t i = 0; i < 32; ++i) {
    stages[i] = i;
  }
  const loom_serve_plan_description_t desc = {2, 2, 32, stages, 8};
  for (bool interleave : {false, true}) {
    loom_serve_execution_plan_t* plan = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_plan_create(
        &desc, interleave, iree_allocator_system(), &plan));
    EXPECT_EQ(loom_serve_execution_plan_node_capacity(plan), 32u);
    Dag dag;
    IREE_ASSERT_OK(
        loom_serve_execution_plan_encode_dag(plan, 0, 1, Dag::Emit, &dag));
    ASSERT_EQ(dag.nodes.size(), 32u);
    for (uint32_t bank = 0; bank < 2; ++bank) {
      for (uint32_t layer = 0; layer < 2; ++layer) {
        const auto base = bank * 16 + layer * 8;
        const auto prefix = dag.Find(bank, base);
        const auto primary = dag.Find(bank, base + 1);
        const auto secondary = dag.Find(bank, base + 2);
        const auto merge = dag.Find(bank, base + 3);
        const auto down_a = dag.Find(bank, base + 4);
        const auto down_b = dag.Find(bank, base + 5);
        const auto down_merge = dag.Find(bank, base + 6);
        const auto residual = dag.Find(bank, base + 7);
        EXPECT_TRUE(dag.Depends(primary, prefix));
        EXPECT_TRUE(dag.Depends(secondary, prefix));
        EXPECT_FALSE(dag.Depends(primary, secondary));
        EXPECT_FALSE(dag.Depends(secondary, primary));
        EXPECT_TRUE(dag.Depends(merge, primary));
        EXPECT_TRUE(dag.Depends(merge, secondary));
        EXPECT_TRUE(dag.Depends(down_a, merge));
        EXPECT_TRUE(dag.Depends(down_b, merge));
        EXPECT_FALSE(dag.Depends(down_a, down_b));
        EXPECT_FALSE(dag.Depends(down_b, down_a));
        EXPECT_TRUE(dag.Depends(down_merge, down_a));
        EXPECT_TRUE(dag.Depends(down_merge, down_b));
        EXPECT_TRUE(dag.Depends(residual, down_merge));
        if (layer) {
          EXPECT_TRUE(dag.Depends(prefix, dag.Find(bank, base - 1)));
        }
        if (bank) {
          EXPECT_TRUE(dag.Depends(prefix, dag.Find(0, layer * 8)));
        }
      }
    }
    if (interleave) {
      EXPECT_FALSE(dag.Depends(dag.Find(1, 16), dag.Find(0, 7)));
      EXPECT_LT(dag.Find(1, 16), dag.Find(0, 3));
    } else {
      EXPECT_TRUE(dag.Depends(dag.Find(1, 16), dag.Find(0, 15)));
    }
    Dag rejected;
    rejected.reject = 5;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        loom_serve_execution_plan_encode_dag(plan, 0, 1, Dag::Emit, &rejected));
    EXPECT_EQ(rejected.nodes.size(), 5u);
    loom_serve_execution_plan_release(plan);
  }
}

TEST(ExecutionPlanTest, OptionalFfnPhasesPreserveProducerDependencies) {
  const uint32_t stages[] = {0, 1,          UINT32_MAX, UINT32_MAX,
                             2, UINT32_MAX, UINT32_MAX, 3};
  loom_serve_plan_description_t desc = {1, 1, 4, stages, 8};
  loom_serve_execution_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_plan_create(
      &desc, true, iree_allocator_system(), &plan));
  Dag dag;
  IREE_ASSERT_OK(
      loom_serve_execution_plan_encode_dag(plan, 0, 0, Dag::Emit, &dag));
  EXPECT_EQ(dag.nodes.size(), 8u);
  for (uint32_t bank = 0; bank < 2; ++bank) {
    EXPECT_TRUE(dag.Depends(dag.Find(bank, 2), dag.Find(bank, 1)));
    EXPECT_TRUE(dag.Depends(dag.Find(bank, 3), dag.Find(bank, 2)));
  }
  loom_serve_execution_plan_release(plan);
}

TEST(ExecutionPlanTest, SecondaryFfnRequiresExplicitMergeAndFinalResidual) {
  uint32_t stages[] = {0, 1, 2, UINT32_MAX, 4, 5, 6, 7};
  loom_serve_plan_description_t desc = {1, 1, 8, stages, 8};
  loom_serve_execution_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  stages[3] = 3;
  stages[6] = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  stages[6] = 6;
  stages[7] = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  stages[7] = 7;
  desc.phase_count = 7;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);
}

TEST(ExecutionPlanTest, UnsupportedFfnShapesStayOnOrdinaryExecution) {
  uint32_t stages[32];
  std::fill(std::begin(stages), std::end(stages), UINT32_MAX);
  for (uint32_t i = 16; i < 32; ++i) {
    stages[i] = i - 16;
  }
  loom_serve_plan_description_t desc = {2, 2, 16, stages, 8};
  for (const auto* provider : {loom_serve_two_bank_plan_provider(),
                               loom_serve_sequential_plan_provider()}) {
    loom_serve_execution_plan_t* plan = nullptr;
    IREE_ASSERT_OK(provider->prepare(provider->self, &desc,
                                     iree_allocator_system(), &plan));
    loom_serve_plan_group_query_t query = {.shapes = {1, 1},
                                           .tokens = {1024, 1024}};
    EXPECT_TRUE(provider->select(provider->self, plan, &query));
    query.shapes[0] = 0;
    EXPECT_FALSE(provider->select(provider->self, plan, &query));
    std::swap(query.shapes[0], query.shapes[1]);
    EXPECT_FALSE(provider->select(provider->self, plan, &query));
    Dag dag;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_serve_execution_plan_encode_dag(plan, 1, 0, Dag::Emit, &dag));
    EXPECT_TRUE(dag.nodes.empty());
    IREE_ASSERT_OK(
        loom_serve_execution_plan_encode_dag(plan, 1, 1, Dag::Emit, &dag));
    EXPECT_EQ(dag.nodes.size(), 32u);
    loom_serve_execution_plan_release(plan);
  }
  // A missing prefix cannot hide an otherwise populated shape.
  stages[15] = 0;
  loom_serve_execution_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);
}

TEST(ExecutionPlanTest, ResidentProjectionsConsumeBeforeNextProjection) {
  uint32_t stages[48];
  for (uint32_t i = 0; i < 48; ++i) {
    stages[i] = i;
  }
  const loom_serve_plan_description_t desc = {2, 2, 48, stages, 12};
  for (bool interleave : {false, true}) {
    loom_serve_execution_plan_t* plan = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_plan_create(
        &desc, interleave, iree_allocator_system(), &plan));
    Dag dag;
    IREE_ASSERT_OK(
        loom_serve_execution_plan_encode_dag(plan, 0, 1, Dag::Emit, &dag));
    ASSERT_EQ(dag.nodes.size(), 48u);
    for (uint32_t bank = 0; bank < 2; ++bank) {
      for (uint32_t layer = 0; layer < 2; ++layer) {
        const uint32_t base = bank * 24 + layer * 12;
        for (uint32_t first : {1u, 4u, 8u}) {
          const auto primary = dag.Find(bank, base + first);
          const auto secondary = dag.Find(bank, base + first + 1);
          const auto consume = dag.Find(bank, base + first + 2);
          EXPECT_FALSE(dag.Depends(primary, secondary));
          EXPECT_FALSE(dag.Depends(secondary, primary));
          EXPECT_TRUE(dag.Depends(consume, primary));
          EXPECT_TRUE(dag.Depends(consume, secondary));
        }
        EXPECT_TRUE(
            dag.Depends(dag.Find(bank, base + 5), dag.Find(bank, base + 3)));
        EXPECT_TRUE(
            dag.Depends(dag.Find(bank, base + 9), dag.Find(bank, base + 7)));
        EXPECT_TRUE(
            dag.Depends(dag.Find(bank, base + 7), dag.Find(bank, base + 6)));
        EXPECT_TRUE(
            dag.Depends(dag.Find(bank, base + 11), dag.Find(bank, base + 10)));
      }
    }
    loom_serve_execution_plan_release(plan);
  }
}

TEST(ExecutionPlanTest, ResidentProjectionRequiresConsumer) {
  uint32_t stages[12];
  for (uint32_t i = 0; i < 12; ++i) {
    stages[i] = i;
  }
  const loom_serve_plan_description_t desc = {1, 1, 12, stages, 12};
  stages[6] = UINT32_MAX;
  loom_serve_execution_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_execution_plan_create(
                            &desc, true, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);
}

}  // namespace
