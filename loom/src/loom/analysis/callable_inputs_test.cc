// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/callable_inputs.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class CallableInputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  static iree_status_t Visit(void* user_data, iree_host_size_t node,
                             loom_scc_successor_callback_t successor) {
    auto* self = static_cast<CallableInputsTest*>(user_data);
    ++self->visits_;
    for (size_t callee : self->edges_[node]) {
      IREE_RETURN_IF_ERROR(successor.fn(successor.user_data, callee));
    }
    return iree_ok_status();
  }

  void Check(size_t input_count,
             const std::vector<loom_callable_input_requirement_t>& direct) {
    const loom_scc_graph_t graph = {
        edges_.size(), loom_scc_visit_successors_callback_make(Visit, this)};
    loom_scc_list_t components;
    IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &components));
    visits_ = 0;
    const auto prior_allocation_size = arena_.used_allocation_size;
    loom_callable_inputs_t inputs;
    IREE_ASSERT_OK(loom_callable_inputs_compute(
        &graph, &components, input_count, direct.data(), direct.size(), &arena_,
        &inputs));
    EXPECT_EQ(visits_, input_count ? edges_.size() : 0);
    EXPECT_EQ(inputs.words_per_node, (input_count + 63) / 64);
    if (!input_count) {
      EXPECT_EQ(inputs.words, nullptr);
      EXPECT_EQ(arena_.used_allocation_size, prior_allocation_size);
      return;
    }
    // Independent oracle: enumerate reachable nodes separately for every
    // caller, then test each key against the original direct requirements.
    for (size_t root = 0; root < edges_.size(); ++root) {
      std::vector<bool> reached(edges_.size());
      std::vector<size_t> pending{root};
      while (!pending.empty()) {
        const size_t node = pending.back();
        pending.pop_back();
        if (reached[node]) {
          continue;
        }
        reached[node] = true;
        pending.insert(pending.end(), edges_[node].begin(), edges_[node].end());
      }
      for (size_t input = 0; input < input_count; ++input) {
        bool expected = false;
        for (const auto& requirement : direct) {
          expected |= reached[requirement.node] && requirement.input == input;
        }
        EXPECT_EQ(loom_callable_inputs_contains(&inputs, root, input), expected)
            << "node=" << root << " input=" << input;
      }
    }
  }

  // Reusable allocation pool for production analysis storage.
  iree_arena_block_pool_t pool_;
  // Scratch arena for the current fixture.
  iree_arena_allocator_t arena_;
  // Valid caller-to-callee adjacency supplied to both analyses.
  std::vector<std::vector<size_t>> edges_;
  // Edge enumeration count after component construction.
  size_t visits_ = 0;
};

TEST_F(CallableInputsTest, SharedCalleesCyclesAndWordBoundaries) {
  edges_ = {{1, 2, 1}, {3}, {3, 4}, {1, 5}, {4, 5}, {}, {0}, {}};
  Check(130, {{5, 0}, {3, 63}, {1, 64}, {4, 127}, {2, 128}, {7, 129}, {5, 0}});
}

TEST_F(CallableInputsTest, EmptyRequirementsDoNotVisitTheGraph) {
  edges_ = {{1}, {0, 2}, {}};
  Check(0, {});
}

TEST_F(CallableInputsTest, DeepCallersShareEveryInput) {
  edges_.resize(4096);
  for (size_t i = 1; i < edges_.size(); ++i) {
    edges_[i - 1].push_back(i);
  }
  // Check propagation with linear work rather than a quadratic test oracle.
  const loom_scc_graph_t graph = {
      edges_.size(), loom_scc_visit_successors_callback_make(Visit, this)};
  loom_scc_list_t components;
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &components));
  const loom_callable_input_requirement_t direct[] = {{4095, 0}, {4095, 128}};
  loom_callable_inputs_t inputs;
  visits_ = 0;
  IREE_ASSERT_OK(loom_callable_inputs_compute(&graph, &components, 129, direct,
                                              IREE_ARRAYSIZE(direct), &arena_,
                                              &inputs));
  EXPECT_EQ(visits_, edges_.size());
  for (size_t node = 0; node < edges_.size(); ++node) {
    EXPECT_TRUE(loom_callable_inputs_contains(&inputs, node, 0));
    EXPECT_FALSE(loom_callable_inputs_contains(&inputs, node, 64));
    EXPECT_TRUE(loom_callable_inputs_contains(&inputs, node, 128));
  }
}

}  // namespace
}  // namespace loom
