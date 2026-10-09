// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_dataflow.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct DenseLivenessResult {
  std::vector<std::vector<uint8_t>> live_in;
  std::vector<std::vector<uint8_t>> live_out;
};

static DenseLivenessResult SolveDenseReference(
    const std::vector<std::vector<uint8_t>>& edges,
    const std::vector<std::vector<uint8_t>>& uses,
    const std::vector<std::vector<uint8_t>>& definitions) {
  const size_t block_count = edges.size();
  const size_t value_count = uses.empty() ? 0 : uses[0].size();
  DenseLivenessResult result = {
      std::vector<std::vector<uint8_t>>(block_count,
                                        std::vector<uint8_t>(value_count)),
      std::vector<std::vector<uint8_t>>(block_count,
                                        std::vector<uint8_t>(value_count)),
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t reverse_index = block_count; reverse_index > 0;
         --reverse_index) {
      const size_t block_index = reverse_index - 1u;
      for (size_t value_ordinal = 0; value_ordinal < value_count;
           ++value_ordinal) {
        uint8_t live_out = 0;
        for (size_t successor = 0; successor < block_count; ++successor) {
          live_out |= edges[block_index][successor] &&
                      result.live_in[successor][value_ordinal];
        }
        const uint8_t live_in =
            uses[block_index][value_ordinal] ||
            (live_out && !definitions[block_index][value_ordinal]);
        changed |= result.live_in[block_index][value_ordinal] != live_in ||
                   result.live_out[block_index][value_ordinal] != live_out;
        result.live_in[block_index][value_ordinal] = live_in;
        result.live_out[block_index][value_ordinal] = live_out;
      }
    }
  }
  return result;
}

static uint32_t Advance(uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static std::vector<loom_value_id_t> CopyValues(const loom_value_id_t* values,
                                               iree_host_size_t count) {
  return count == 0 ? std::vector<loom_value_id_t>()
                    : std::vector<loom_value_id_t>(values, values + count);
}

TEST(LivenessDataflowTest, MatchesDenseFixedPointAcrossGeneratedCfgs) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  for (uint32_t seed = 1; seed <= 512; ++seed) {
    uint32_t random_state = seed;
    const uint16_t block_count =
        static_cast<uint16_t>(1u + Advance(&random_state) % 8u);
    const loom_value_ordinal_t value_count = 1u + Advance(&random_state) % 12u;

    std::vector<std::vector<uint8_t>> edges(block_count,
                                            std::vector<uint8_t>(block_count));
    for (uint16_t source = 0; source < block_count; ++source) {
      for (uint16_t target = 0; target < block_count; ++target) {
        edges[source][target] =
            static_cast<uint8_t>((Advance(&random_state) >> 29) == 0);
      }
    }

    std::vector<std::vector<uint8_t>> uses(block_count,
                                           std::vector<uint8_t>(value_count));
    std::vector<std::vector<uint8_t>> definitions(
        block_count, std::vector<uint8_t>(value_count));
    for (loom_value_ordinal_t value_ordinal = 0; value_ordinal < value_count;
         ++value_ordinal) {
      const uint32_t definition_choice =
          Advance(&random_state) % (block_count + 1u);
      if (definition_choice < block_count) {
        definitions[definition_choice][value_ordinal] = 1;
      }
      for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
        uses[block_index][value_ordinal] =
            static_cast<uint8_t>((Advance(&random_state) >> 28) == 0);
      }
    }

    std::vector<std::vector<loom_value_ordinal_t>> use_ordinals(block_count);
    std::vector<std::vector<loom_value_ordinal_t>> definition_ordinals(
        block_count);
    std::vector<loom_liveness_block_transfer_t> transfers(block_count);
    for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
      for (loom_value_ordinal_t value_ordinal = 0; value_ordinal < value_count;
           ++value_ordinal) {
        if (uses[block_index][value_ordinal]) {
          use_ordinals[block_index].push_back(value_ordinal);
        }
        if (definitions[block_index][value_ordinal]) {
          definition_ordinals[block_index].push_back(value_ordinal);
        }
      }
      if ((block_index & 1u) != 0) {
        std::reverse(use_ordinals[block_index].begin(),
                     use_ordinals[block_index].end());
        std::reverse(definition_ordinals[block_index].begin(),
                     definition_ordinals[block_index].end());
      }
      transfers[block_index] = {
          use_ordinals[block_index].data(),
          use_ordinals[block_index].size(),
          definition_ordinals[block_index].data(),
          definition_ordinals[block_index].size(),
      };
    }

    std::vector<loom_cfg_block_info_t> graph_blocks(block_count);
    std::vector<uint16_t> predecessor_indices;
    for (uint16_t target = 0; target < block_count; ++target) {
      graph_blocks[target].predecessor_start = predecessor_indices.size();
      for (uint16_t source = 0; source < block_count; ++source) {
        if (edges[source][target]) {
          predecessor_indices.push_back(source);
        }
      }
      graph_blocks[target].predecessor_count =
          predecessor_indices.size() - graph_blocks[target].predecessor_start;
    }
    loom_cfg_graph_t graph = {
        .module = nullptr,
        .region = nullptr,
        .blocks = graph_blocks.data(),
        .edges = nullptr,
        .successor_indices = nullptr,
        .successor_edge_indices = nullptr,
        .predecessor_indices = predecessor_indices.data(),
        .predecessor_edge_indices = nullptr,
        .block_count = block_count,
        .edge_count = 0,
    };

    std::vector<loom_value_id_t> value_ids(value_count);
    for (loom_value_ordinal_t value_ordinal = 0; value_ordinal < value_count;
         ++value_ordinal) {
      value_ids[value_ordinal] = 17u + value_ordinal * 11u;
    }
    std::vector<loom_liveness_block_relation_t> relations(block_count);
    iree_arena_allocator_t result_arena;
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool, &result_arena);
    iree_arena_initialize(&block_pool, &scratch_arena);
    IREE_ASSERT_OK(loom_liveness_dataflow_solve(
        &graph, value_ids.data(), value_count, transfers.data(), block_count,
        relations.data(), &result_arena, &scratch_arena));

    const DenseLivenessResult expected =
        SolveDenseReference(edges, uses, definitions);
    for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
      std::vector<loom_value_id_t> expected_live_in;
      std::vector<loom_value_id_t> expected_live_out;
      for (loom_value_ordinal_t value_ordinal = 0; value_ordinal < value_count;
           ++value_ordinal) {
        if (expected.live_in[block_index][value_ordinal]) {
          expected_live_in.push_back(value_ids[value_ordinal]);
        }
        if (expected.live_out[block_index][value_ordinal]) {
          expected_live_out.push_back(value_ids[value_ordinal]);
        }
      }
      EXPECT_EQ(CopyValues(relations[block_index].live_in_values,
                           relations[block_index].live_in_count),
                expected_live_in)
          << "seed=" << seed << ", block=" << block_index;
      EXPECT_EQ(CopyValues(relations[block_index].live_out_values,
                           relations[block_index].live_out_count),
                expected_live_out)
          << "seed=" << seed << ", block=" << block_index;
    }
    iree_arena_deinitialize(&scratch_arena);
    iree_arena_deinitialize(&result_arena);
  }

  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LivenessDataflowTest, LocalBlocksPublishOnlyUpwardExposedUses) {
  const loom_value_id_t value_ids[] = {101, 303, 707};
  const loom_value_ordinal_t block0_uses[] = {2, 0};
  const loom_value_ordinal_t block1_uses[] = {1};
  const loom_liveness_block_transfer_t transfers[] = {
      {block0_uses, IREE_ARRAYSIZE(block0_uses), nullptr, 0},
      {block1_uses, IREE_ARRAYSIZE(block1_uses), nullptr, 0},
  };
  loom_liveness_block_relation_t relations[IREE_ARRAYSIZE(transfers)] = {};

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t result_arena;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool, &result_arena);
  iree_arena_initialize(&block_pool, &scratch_arena);
  IREE_ASSERT_OK(loom_liveness_dataflow_solve(
      nullptr, value_ids, IREE_ARRAYSIZE(value_ids), transfers,
      IREE_ARRAYSIZE(transfers), relations, &result_arena, &scratch_arena));

  ASSERT_EQ(relations[0].live_in_count, 2u);
  EXPECT_EQ(relations[0].live_in_values[0], value_ids[0]);
  EXPECT_EQ(relations[0].live_in_values[1], value_ids[2]);
  EXPECT_EQ(relations[0].live_out_count, 0u);
  ASSERT_EQ(relations[1].live_in_count, 1u);
  EXPECT_EQ(relations[1].live_in_values[0], value_ids[1]);
  EXPECT_EQ(relations[1].live_out_count, 0u);

  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&result_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
