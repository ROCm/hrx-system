// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_participants.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ControlParticipantsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("participants"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 16));
    for (uint8_t axis = 0; axis < coordinates_.size(); ++axis) {
      IREE_ASSERT_OK(loom_module_define_value(
          module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
          &coordinates_[axis]));
      auto facts = loom_value_facts_make(0, 63, 1);
      ASSERT_TRUE(loom_value_facts_mark_topology_domain(
          &facts,
          axis == 3 ? LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID
                    : LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID,
          static_cast<loom_value_fact_topology_axis_t>(axis)));
      IREE_ASSERT_OK(
          loom_value_fact_table_define(&facts_, coordinates_[axis], facts));
    }
    loom_symbolic_expr_context_initialize(module_, nullptr, &facts_, &arena_,
                                          &expressions_);
  }
  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_condition_integer_relation_t ZeroRelation(uint8_t axis) const {
    return {
        .relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
        .left =
            {
                .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = coordinates_[axis],
            },
        .right =
            {
                .kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
                .constant = 0,
            },
    };
  }

  // Graph and condition records are the API's production inputs. Textual IR
  // extraction and target publication are covered by the lowering goldens.
  void CheckGraph(const std::vector<std::vector<uint16_t>>& successors,
                  const std::vector<uint8_t>& guard_masks) {
    std::vector<loom_cfg_block_info_t> blocks(successors.size());
    std::vector<loom_cfg_edge_info_t> edges;
    std::vector<loom_cfg_edge_index_t> successor_edges;
    for (size_t source = 0; source < successors.size(); ++source) {
      blocks[source].successor_edge_start = edges.size();
      blocks[source].successor_count = successors[source].size();
      for (uint16_t target : successors[source]) {
        successor_edges.push_back(edges.size());
        edges.push_back({
            .source_block_index = static_cast<uint16_t>(source),
            .target_block_index = target,
        });
      }
    }
    ASSERT_EQ(edges.size(), guard_masks.size());
    // Independent oracle: remove guarded edges for each coordinate, then
    // perform ordinary reachability. The fifth traversal marks live blocks.
    std::array<std::vector<bool>, 5> reachable;
    for (uint8_t axis = 0; axis < reachable.size(); ++axis) {
      reachable[axis].resize(blocks.size());
      std::vector<uint16_t> pending;
      if (!blocks.empty()) {
        pending.push_back(0);
      }
      while (!pending.empty()) {
        const uint16_t source = pending.back();
        pending.pop_back();
        if (reachable[axis][source]) {
          continue;
        }
        reachable[axis][source] = true;
        const auto begin = blocks[source].successor_edge_start;
        for (size_t i = 0; i < successors[source].size(); ++i) {
          if ((guard_masks[begin + i] & (1u << axis)) == 0) {
            pending.push_back(successors[source][i]);
          }
        }
      }
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
      blocks[i].reachable = reachable[4][i];
    }
    std::vector<std::array<loom_condition_integer_relation_t, 4>> relations(
        edges.size());
    std::vector<loom_cfg_condition_edge_facts_t> conditions(edges.size());
    for (size_t edge = 0; edge < edges.size(); ++edge) {
      auto& facts = conditions[edge];
      facts.integer_relations = relations[edge].data();
      for (uint8_t axis = 0; axis < coordinates_.size(); ++axis) {
        if (guard_masks[edge] & (1u << axis)) {
          relations[edge][facts.integer_relation_count++] = ZeroRelation(axis);
        }
      }
    }
    const loom_cfg_graph_t graph = {
        .blocks = blocks.data(),
        .edges = edges.data(),
        .successor_edge_indices = successor_edges.data(),
        .block_count = blocks.size(),
        .edge_count = edges.size(),
    };
    loom_control_participant_zeros_t* actual = nullptr;
    IREE_ASSERT_OK(loom_control_participant_zeros_for_cfg(
        &graph, conditions.data(), &expressions_, &arena_, &actual));
    for (size_t block = 0; block < blocks.size(); ++block) {
      uint8_t expected = 0;
      if (reachable[4][block]) {
        for (uint8_t axis = 0; axis < coordinates_.size(); ++axis) {
          if (!reachable[axis][block]) {
            expected |= 1u << axis;
          }
        }
      }
      EXPECT_EQ(actual[block], expected) << "block " << block;
    }
  }

  // Pool and arenas own all production analysis storage.
  iree_arena_block_pool_t pool_;
  // Retained fact and projection storage for the current test.
  iree_arena_allocator_t arena_;
  // Type/IR context used only to define opaque scalar inputs to the API.
  loom_context_t context_;
  // Opaque typed values have no operation bodies or textual IR.
  loom_module_t* module_ = nullptr;
  // Known coordinate domains for the opaque values.
  loom_value_fact_table_t facts_;
  // Ordinary shared expression query over those facts.
  loom_symbolic_expr_context_t expressions_;
  // Workitem X/Y/Z and subgroup-lane identities.
  std::array<loom_value_id_t, 4> coordinates_;
};

TEST_F(ControlParticipantsTest, EmptyAndStraightLineGraphs) {
  CheckGraph({}, {});
  CheckGraph({{}}, {});
  CheckGraph({{1}, {2}, {}}, {0, 0});
}

TEST_F(ControlParticipantsTest, GuardedJoinsCyclesAndBypasses) {
  CheckGraph({{1, 2}, {3}, {3}, {4}, {3, 5}, {}, {6}},
             {1, 1, 0, 0, 8, 0, 0, 0});
  CheckGraph({{1, 2}, {3}, {3}, {4}, {3, 5}, {}, {6}},
             {1, 0, 0, 0, 8, 0, 0, 0});
  CheckGraph({{1, 1}, {2}, {1, 0}}, {1, 8, 15, 0, 0});
}

TEST_F(ControlParticipantsTest, AllCoordinateMasks) {
  for (uint8_t mask = 0; mask < 16; ++mask) {
    SCOPED_TRACE(mask);
    CheckGraph({{1}, {2}, {}}, {mask, static_cast<uint8_t>(15 ^ mask)});
  }
}

TEST_F(ControlParticipantsTest, CyclicGraphsMatchIndependentReachability) {
  uint32_t seed = 0x351a0297u;
  for (uint32_t trial = 0; trial < 32; ++trial) {
    SCOPED_TRACE(trial);
    std::vector<std::vector<uint16_t>> successors(16);
    std::vector<uint8_t> masks;
    for (auto& edges : successors) {
      for (uint8_t successor = 0; successor < 3; ++successor) {
        seed = seed * 1664525u + 1013904223u;
        edges.push_back((seed >> 16) % successors.size());
        masks.push_back((seed >> 24) & 15);
      }
    }
    CheckGraph(successors, masks);
  }
}

TEST_F(ControlParticipantsTest, WideFrontierCrossesWorklistSegments) {
  constexpr uint16_t kSpineBlocks = 40;
  constexpr uint16_t kLeavesPerBlock = 128;
  std::vector<std::vector<uint16_t>> successors(kSpineBlocks + 1 +
                                                kSpineBlocks * kLeavesPerBlock);
  std::vector<uint8_t> masks;
  for (uint16_t source = 0; source < kSpineBlocks; ++source) {
    for (uint16_t leaf = 0; leaf < kLeavesPerBlock; ++leaf) {
      const uint16_t target =
          kSpineBlocks + 1 + source * kLeavesPerBlock + leaf;
      successors[source].push_back(target);
      masks.push_back(target % 16);
    }
    // The last successor extends the spine while earlier leaves remain
    // queued. Every block stays within the IR's 255-successor capacity.
    successors[source].push_back(source + 1);
    masks.push_back(0);
  }
  CheckGraph(successors, masks);
}

TEST_F(ControlParticipantsTest, OnlyExactZeroEqualitiesEstablishCoordinates) {
  for (uint8_t axis = 0; axis < coordinates_.size(); ++axis) {
    SCOPED_TRACE(axis);
    auto relation = ZeroRelation(axis);
    loom_control_participant_zeros_t zeros = 0;
    IREE_ASSERT_OK(loom_control_participant_zeros_from_conditions(
        &expressions_, &relation, 1, &zeros));
    EXPECT_EQ(zeros, 1u << axis);
    std::swap(relation.left, relation.right);
    IREE_ASSERT_OK(loom_control_participant_zeros_from_conditions(
        &expressions_, &relation, 1, &zeros));
    EXPECT_EQ(zeros, 1u << axis);
    relation.left.constant = 1;
    IREE_ASSERT_OK(loom_control_participant_zeros_from_conditions(
        &expressions_, &relation, 1, &zeros));
    EXPECT_EQ(zeros, 0);
    relation = ZeroRelation(axis);
    relation.relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
    IREE_ASSERT_OK(loom_control_participant_zeros_from_conditions(
        &expressions_, &relation, 1, &zeros));
    EXPECT_EQ(zeros, 0);
  }
  std::array<loom_condition_integer_relation_t, 32> relations;
  relations.fill(ZeroRelation(0));
  for (size_t i = 1; i + 1 < relations.size(); ++i) {
    IREE_ASSERT_OK(loom_module_define_value(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        &relations[i].left.value_id));
  }
  relations.back() = ZeroRelation(3);
  loom_control_participant_zeros_t zeros = 0;
  IREE_ASSERT_OK(loom_control_participant_zeros_from_conditions(
      &expressions_, relations.data(), relations.size(), &zeros));
  EXPECT_EQ(zeros, LOOM_CONTROL_PARTICIPANT_ZERO_WORKITEM_X |
                       LOOM_CONTROL_PARTICIPANT_ZERO_SUBGROUP_LANE);
}

TEST_F(ControlParticipantsTest,
       SparseTableHandlesUnorderedSegmentsAndMissingBlocks) {
  loom_control_participant_table_t table;
  loom_control_participant_table_initialize(&arena_, &table);
  std::vector<loom_block_t> blocks(4097);
  EXPECT_EQ(loom_control_participant_table_lookup(&table, &blocks[0]), 0);
  for (size_t i = blocks.size(); i > 0; --i) {
    IREE_ASSERT_OK(loom_control_participant_table_append(
        &table, &blocks[i - 1], static_cast<uint8_t>((i - 1) % 16)));
  }
  loom_control_participant_table_publish(&table);
  for (size_t i = 0; i < blocks.size(); ++i) {
    EXPECT_EQ(loom_control_participant_table_lookup(&table, &blocks[i]),
              i % 16);
  }
  loom_block_t missing = {};
  EXPECT_EQ(loom_control_participant_table_lookup(&table, &missing), 0);
}

}  // namespace
}  // namespace loom
