// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/code_banks.h"

#include <algorithm>
#include <array>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

namespace {

class CodeBanksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    plan_.routing.family = loom_xdna_npu2_array_family();
    tiles_.resize(plan_.routing.family->column_count *
                  plan_.routing.family->row_count);
    plan_.tiles = tiles_.data();
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void Reserve(uint16_t column, uint16_t next_engine, uint16_t free_even,
               uint16_t free_odd) {
    const auto* memory = loom_xdna_array_tile_kind_facts(
        plan_.routing.family, LOOM_XDNA_TILE_KIND_MEMORY);
    auto& tile = tiles_[loom_xdna_array_physical_tile_index(
        plan_.routing.family, {column, memory->first_row})];
    tile.mm2s_count = next_engine;
    const uint16_t capacity = memory->dma.buffer_descriptor_count / 2;
    tile.descriptor_counts[0] = capacity - free_even;
    tile.descriptor_counts[1] = capacity - free_odd;
  }

  void Order(const std::vector<loom_aie2p_code_catalog_demand_t>& demands,
             std::vector<size_t>* result) {
    iree_host_size_t* order = nullptr;
    IREE_ASSERT_OK(loom_aie2p_code_banks_order(
        &plan_, demands.data(), demands.size(), &arena_, &order));
    result->clear();
    if (order) {
      result->assign(order, order + demands.size());
    }
  }

  // Backing blocks for this compilation.
  iree_arena_block_pool_t blocks_ = {};
  // Owns the planner's transient state and returned ordinals.
  iree_arena_allocator_t arena_ = {};
  // Retains the data path's preexisting DMA reservations.
  loom_aie2p_execution_plan_t plan_ = {};
  // One entry per physical tile, as in the pipeline execution plan.
  std::vector<loom_aie2p_execution_tile_t> tiles_;
};

TEST_F(CodeBanksTest, InterleavedColumnsKeepTheirInitializationOrdinals) {
  // Engines 2 and 3 address opposite first banks. Both columns need 5+4+4
  // descriptors in that bank and 5 in the other. A resident core uses none.
  Reserve(1, 2, 13, 7);
  Reserve(3, 3, 7, 13);
  std::vector<size_t> order;
  Order({{1, 5},
         {3, 5},
         {1, 5},
         {3, 5},
         {4, 0},
         {1, 4},
         {3, 4},
         {5, 2},
         {1, 4},
         {3, 4}},
        &order);
  EXPECT_EQ(order, (std::vector<size_t>{0, 1, 5, 6, 4, 8, 9, 7, 2, 3}));
}

TEST_F(CodeBanksTest, EveryWorkerOrderFitsTheTwoAvailableSenders) {
  Reserve(1, 4, 13, 7);
  std::array<size_t, 4> sizes = {4, 4, 5, 5};
  do {
    std::vector<size_t> order;
    Order({{1, sizes[0]}, {1, sizes[1]}, {1, sizes[2]}, {1, sizes[3]}}, &order);
    std::array<size_t, 2> remaining = {13, 7};
    size_t sender = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      const size_t size = sizes[order.empty() ? i : order[i]];
      if (size > remaining[sender]) {
        ++sender;
      }
      ASSERT_LT(sender, remaining.size());
      ASSERT_GE(remaining[sender], size);
      remaining[sender] -= size;
    }
    EXPECT_EQ(remaining, (std::array<size_t, 2>{0, 2}));
  } while (std::next_permutation(sizes.begin(), sizes.end()));
}

TEST_F(CodeBanksTest, RetainsAdmittedOrderAndRejectsImpossiblePartitions) {
  std::vector<size_t> order;
  Reserve(1, 2, 13, 7);
  Order({{1, 5}, {1, 4}, {1, 4}, {1, 5}}, &order);
  EXPECT_TRUE(order.empty());
  Order({{1, 14}, {1, 5}}, &order);  // One catalog exceeds either bank.
  EXPECT_TRUE(order.empty());
  Order({{1, 5}, {1, 5}, {1, 5}, {1, 5}}, &order);  // No 13/7 split.
  EXPECT_TRUE(order.empty());
  Reserve(1, 5, 7, 13);  // Only the last odd engine remains.
  Order({{1, 5}, {1, 5}, {1, 4}, {1, 4}}, &order);
  EXPECT_TRUE(order.empty());
}

}  // namespace
