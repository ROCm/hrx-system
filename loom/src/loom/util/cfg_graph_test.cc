// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_graph.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CfgGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t cfg_vtable_count = 0;
    const loom_op_vtable_t* const* cfg_vtables =
        loom_cfg_dialect_vtables(&cfg_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, cfg_vtables, (uint16_t)cfg_vtable_count));
    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                        nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op_));
    body_ = loom_func_like_body(loom_func_like_cast(module_, func_op_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;
    body_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;

    iree_arena_initialize(&block_pool_, &graph_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&graph_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_block_t* AppendBlock() {
    loom_block_t* block = nullptr;
    IREE_CHECK_OK(loom_region_append_block(module_, body_, &block));
    return block;
  }

  void SetBlock(loom_block_t* block) {
    loom_builder_set_block(&builder_, block);
    builder_.ip.parent_op = func_op_;
  }

  loom_value_id_t BuildCondition() {
    return BuildConstant(1, loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  }

  loom_value_id_t BuildConstant(int64_t value, loom_type_t type) {
    loom_op_t* condition_op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                           type, LOOM_LOCATION_UNKNOWN,
                                           &condition_op));
    return loom_test_constant_result(condition_op);
  }

  loom_value_id_t AddBlockArg(loom_block_t* block, loom_type_t type) {
    loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &arg));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, arg));
    return arg;
  }

  loom_op_t* BuildBranch(loom_block_t* dest) {
    return BuildBranchWithArgs(dest, nullptr, 0);
  }

  loom_op_t* BuildBranchWithArgs(loom_block_t* dest,
                                 const loom_value_id_t* args,
                                 iree_host_size_t arg_count) {
    loom_op_t* branch_op = nullptr;
    IREE_CHECK_OK(loom_cfg_br_build(&builder_, dest, args, arg_count,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));
    return branch_op;
  }

  loom_op_t* BuildConditionalBranch(loom_block_t* true_dest,
                                    loom_block_t* false_dest) {
    loom_value_id_t condition = BuildCondition();
    loom_op_t* branch_op = nullptr;
    IREE_CHECK_OK(loom_cfg_cond_br_build(&builder_, condition, true_dest,
                                         false_dest, LOOM_LOCATION_UNKNOWN,
                                         &branch_op));
    return branch_op;
  }

  void BuildGraph(loom_cfg_graph_t* out_graph) {
    IREE_ASSERT_OK(
        loom_cfg_graph_build(module_, body_, &graph_arena_, out_graph));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  iree_arena_allocator_t graph_arena_;
};

TEST_F(CfgGraphTest, BuildsSuccessorsAndPredecessorsForDiamond) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* then_block = AppendBlock();
  loom_block_t* else_block = AppendBlock();
  loom_block_t* merge_block = AppendBlock();

  SetBlock(entry);
  BuildConditionalBranch(then_block, else_block);
  SetBlock(then_block);
  BuildBranch(merge_block);
  SetBlock(else_block);
  BuildBranch(merge_block);

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_EQ(graph.block_count, 4u);
  EXPECT_EQ(graph.edge_count, 4u);
  EXPECT_EQ(graph.backward_edge_count, 0u);
  EXPECT_FALSE(graph.has_cycles);

  loom_cfg_block_index_span_t entry_successors =
      loom_cfg_graph_successors(&graph, 0);
  ASSERT_EQ(entry_successors.count, 2u);
  EXPECT_EQ(entry_successors.values[0], 1u);
  EXPECT_EQ(entry_successors.values[1], 2u);

  loom_cfg_block_index_span_t merge_predecessors =
      loom_cfg_graph_predecessors(&graph, 3);
  ASSERT_EQ(merge_predecessors.count, 2u);
  EXPECT_EQ(merge_predecessors.values[0], 1u);
  EXPECT_EQ(merge_predecessors.values[1], 2u);

  EXPECT_EQ(loom_cfg_graph_block_index(&graph, merge_block), 3u);
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 0));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 1));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 2));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 3));
  ASSERT_EQ(graph.reverse_postorder.count, 4u);
  EXPECT_EQ(graph.reverse_postorder.values[0], 0u);
  EXPECT_EQ(graph.reverse_postorder.values[1], 2u);
  EXPECT_EQ(graph.reverse_postorder.values[2], 1u);
  EXPECT_EQ(graph.reverse_postorder.values[3], 3u);
  EXPECT_EQ(graph.blocks[0].preorder, 0u);
  EXPECT_EQ(graph.blocks[1].preorder, 1u);
  EXPECT_EQ(graph.blocks[3].preorder, 2u);
  EXPECT_EQ(graph.blocks[2].preorder, 3u);
  EXPECT_EQ(graph.blocks[0].parent, UINT16_MAX);
  EXPECT_EQ(graph.blocks[1].parent, 0u);
  EXPECT_EQ(graph.blocks[3].parent, 1u);
  EXPECT_EQ(graph.blocks[2].parent, 0u);
}

TEST_F(CfgGraphTest, ReachabilitySkipsUnreachableBlocks) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* reachable = AppendBlock();
  loom_block_t* unreachable = AppendBlock();

  SetBlock(entry);
  BuildBranch(reachable);
  SetBlock(unreachable);
  loom_op_t* dead_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &dead_op));

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 1u);
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 0));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 1));
  EXPECT_FALSE(loom_cfg_graph_block_is_reachable(&graph, 2));
  EXPECT_EQ(graph.blocks[2].preorder, UINT16_MAX);
  EXPECT_EQ(graph.blocks[2].parent, UINT16_MAX);
  ASSERT_EQ(graph.reverse_postorder.count, 2u);
  EXPECT_EQ(graph.reverse_postorder.values[0], 0u);
  EXPECT_EQ(graph.reverse_postorder.values[1], 1u);
}

TEST_F(CfgGraphTest, TraversalRetainsLoopHeaderBeforeNonlexicalBody) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* loop_body = AppendBlock();
  loom_block_t* exit_block = AppendBlock();
  loom_block_t* header = AppendBlock();
  loom_block_t* unreachable = AppendBlock();

  SetBlock(entry);
  BuildBranch(header);
  SetBlock(header);
  BuildConditionalBranch(loop_body, exit_block);
  SetBlock(loop_body);
  BuildBranch(header);
  SetBlock(unreachable);
  BuildBranch(exit_block);

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_TRUE(graph.has_cycles);
  EXPECT_EQ(graph.block_count, 5u);
  EXPECT_EQ(graph.edge_count, 5u);
  EXPECT_FALSE(loom_cfg_graph_block_is_reachable(&graph, 4));
  ASSERT_EQ(graph.reverse_postorder.count, 4u);
  EXPECT_EQ(graph.reverse_postorder.values[0], 0u);
  EXPECT_EQ(graph.reverse_postorder.values[1], 3u);
  EXPECT_EQ(graph.reverse_postorder.values[2], 2u);
  EXPECT_EQ(graph.reverse_postorder.values[3], 1u);
  EXPECT_TRUE(graph.blocks[3].is_dfs_backedge_target);
  EXPECT_FALSE(graph.blocks[1].is_dfs_backedge_target);
}

TEST_F(CfgGraphTest, TraversalVisitsSelfLoopOnce) {
  loom_block_t* entry = loom_region_entry_block(body_);
  SetBlock(entry);
  BuildBranch(entry);

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 1u);
  ASSERT_EQ(graph.reverse_postorder.count, 1u);
  EXPECT_EQ(graph.reverse_postorder.values[0], 0u);
  EXPECT_TRUE(graph.blocks[0].component_is_cyclic);
  EXPECT_TRUE(graph.blocks[0].is_dfs_backedge_target);
  EXPECT_EQ(graph.blocks[0].reachability_root, 0u);
  EXPECT_EQ(graph.blocks[0].preorder_end, 1u);
  EXPECT_TRUE(graph.has_cycles);
}

TEST_F(CfgGraphTest, ReorderedAcyclicGraphIgnoresUnreachableCycle) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* continuation = AppendBlock();
  loom_block_t* branch = AppendBlock();
  loom_block_t* unreachable = AppendBlock();
  SetBlock(entry);
  BuildBranch(branch);
  SetBlock(branch);
  BuildBranch(continuation);
  SetBlock(unreachable);
  BuildBranch(unreachable);

  loom_cfg_graph_t graph = {};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_EQ(graph.backward_edge_count, 2u);
  EXPECT_FALSE(graph.has_cycles);
  EXPECT_FALSE(graph.blocks[3].reachable);
  ASSERT_EQ(graph.reverse_postorder.count, 3u);
  EXPECT_EQ(graph.reverse_postorder.values[0], 0u);
  EXPECT_EQ(graph.reverse_postorder.values[1], 2u);
  EXPECT_EQ(graph.reverse_postorder.values[2], 1u);
}

TEST_F(CfgGraphTest, CompletedCyclicSiblingIsNotBackedgeTarget) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* header = AppendBlock();
  loom_block_t* first = AppendBlock();
  loom_block_t* second = AppendBlock();
  SetBlock(entry);
  BuildBranch(header);
  SetBlock(header);
  BuildConditionalBranch(first, second);
  SetBlock(first);
  BuildBranch(header);
  SetBlock(second);
  BuildBranch(first);

  loom_cfg_graph_t graph = {};
  BuildGraph(&graph);

  ASSERT_FALSE(graph.malformed);
  EXPECT_TRUE(graph.blocks[1].is_dfs_backedge_target);
  // First has completed when second reaches it, although both still belong
  // to the unfinished strongly connected component rooted at header.
  for (unsigned i = 2; i < 4; ++i) {
    EXPECT_TRUE(graph.blocks[i].component_is_cyclic);
    EXPECT_FALSE(graph.blocks[i].is_dfs_backedge_target);
  }
}

TEST_F(CfgGraphTest, NestedLoopsRetainBothBackedgeTargets) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* outer_header = AppendBlock();
  loom_block_t* inner_header = AppendBlock();
  loom_block_t* inner_body = AppendBlock();
  loom_block_t* outer_latch = AppendBlock();
  loom_block_t* exit_block = AppendBlock();
  SetBlock(entry);
  BuildBranch(outer_header);
  SetBlock(outer_header);
  BuildConditionalBranch(inner_header, exit_block);
  SetBlock(inner_header);
  BuildConditionalBranch(inner_body, outer_latch);
  SetBlock(inner_body);
  BuildBranch(inner_header);
  SetBlock(outer_latch);
  BuildBranch(outer_header);

  loom_cfg_graph_t graph = {};
  BuildGraph(&graph);

  ASSERT_FALSE(graph.malformed);
  EXPECT_TRUE(graph.blocks[1].is_dfs_backedge_target);
  EXPECT_TRUE(graph.blocks[2].is_dfs_backedge_target);
  EXPECT_FALSE(graph.blocks[3].is_dfs_backedge_target);
  EXPECT_FALSE(graph.blocks[4].is_dfs_backedge_target);
}

TEST_F(CfgGraphTest, TraversalFactsForThreeBinaryBranchesAndExit) {
  loom_block_t* blocks[] = {loom_region_entry_block(body_), AppendBlock(),
                            AppendBlock(), AppendBlock()};
  loom_op_t* branches[3];
  for (unsigned i = 0; i < 3; ++i) {
    SetBlock(blocks[i]);
    branches[i] = BuildConditionalBranch(blocks[0], blocks[0]);
  }
  // All 4^6 successor assignments include irreducible cycles, self-edges,
  // parallel edges, reconvergence, entry-unreachable components, and paths
  // to the fourth block's exit. Rebuild after each retargeting, just as an
  // invalidated analysis does.
  for (unsigned topology = 0; topology < 4096; ++topology) {
    SCOPED_TRACE(topology);
    bool reaches[4][4] = {};
    unsigned remaining = topology;
    for (unsigned source = 0; source < 3; ++source) {
      for (unsigned edge = 0; edge < 2; ++edge) {
        const unsigned target = remaining % 4;
        remaining /= 4;
        loom_op_successors(branches[source])[edge] = blocks[target];
        reaches[source][target] = true;
      }
    }
    for (unsigned via = 0; via < 4; ++via) {
      for (unsigned source = 0; source < 4; ++source) {
        for (unsigned target = 0; target < 4; ++target) {
          reaches[source][target] |=
              reaches[source][via] && reaches[via][target];
        }
      }
    }
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(&graph_arena_);
    loom_cfg_graph_t graph = {};
    BuildGraph(&graph);
    ASSERT_FALSE(graph.malformed);
    unsigned reverse_postorder[4] = {};
    for (unsigned i = 0; i < graph.reverse_postorder.count; ++i) {
      reverse_postorder[graph.reverse_postorder.values[i]] = i;
    }
    bool has_reachable_cycle = false;
    for (unsigned source = 0; source < 4; ++source) {
      const auto& info = graph.blocks[source];
      EXPECT_EQ(info.reachable, source == 0 || reaches[0][source]);
      EXPECT_EQ(info.can_reach_exit,
                info.reachable && (source == 3 || reaches[source][3]));
      if (!info.reachable) {
        EXPECT_EQ(info.component, UINT16_MAX);
        EXPECT_FALSE(info.is_dfs_backedge_target);
        continue;
      }
      // A backedge target is precisely a block that does not finish before
      // every reachable predecessor. Check the retained consumer contract
      // independently of the DFS frame state used to produce the flag.
      bool finishes_before_predecessors = true;
      const auto predecessors = loom_cfg_graph_predecessors(&graph, source);
      for (unsigned i = 0; i < predecessors.count; ++i) {
        const unsigned predecessor = predecessors.values[i];
        finishes_before_predecessors &=
            !graph.blocks[predecessor].reachable ||
            reverse_postorder[source] > reverse_postorder[predecessor];
      }
      EXPECT_EQ(info.is_dfs_backedge_target, !finishes_before_predecessors);
      EXPECT_EQ(info.component_is_cyclic, reaches[source][source]);
      has_reachable_cycle |= reaches[source][source];
      unsigned earliest = source;
      for (unsigned target = 0; target < 4; ++target) {
        const auto& target_info = graph.blocks[target];
        if (reaches[source][target] &&
            target_info.preorder < graph.blocks[earliest].preorder) {
          earliest = target;
        }
        if (!target_info.reachable) {
          continue;
        }
        EXPECT_EQ(info.component == target_info.component,
                  source == target ||
                      (reaches[source][target] && reaches[target][source]));
        if (reaches[source][target]) {
          EXPECT_LE(target_info.component, info.component);
        }
        if (info.preorder <= target_info.preorder &&
            target_info.preorder < info.preorder_end) {
          EXPECT_TRUE(source == target || reaches[source][target]);
        }
      }
      EXPECT_EQ(info.reachability_root, earliest);
    }
    EXPECT_EQ(graph.has_cycles, has_reachable_cycle);
    iree_arena_checkpoint_restore(&checkpoint);
  }
}

TEST_F(CfgGraphTest, OutsideSuccessorMarksGraphMalformed) {
  loom_block_t* entry = loom_region_entry_block(body_);
  SetBlock(entry);
  BuildBranch(loom_module_block(module_));

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_TRUE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 0u);
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 0));
}

TEST_F(CfgGraphTest, SuccessorBeforeBlockEndMarksGraphMalformed) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* target = AppendBlock();
  SetBlock(entry);
  BuildBranch(target);
  loom_op_t* after_branch = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &after_branch));

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_TRUE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 1u);
}

TEST_F(CfgGraphTest, TerminatorPayloadForSuccessorUsesSingleSuccessorOperands) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* target = AppendBlock();
  AddBlockArg(target, i32);

  SetBlock(entry);
  loom_value_id_t payload = BuildConstant(42, i32);
  loom_op_t* branch_op = BuildBranchWithArgs(target, &payload, 1);

  const loom_value_id_t* args = nullptr;
  uint16_t arg_count = 0;
  EXPECT_TRUE(loom_cfg_terminator_payload_for_successor(branch_op, target,
                                                        &args, &arg_count));
  ASSERT_EQ(arg_count, 1u);
  EXPECT_EQ(args[0], payload);
}

TEST_F(CfgGraphTest,
       TerminatorPayloadForSuccessorRejectsMultiSuccessorTerminators) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* then_block = AppendBlock();
  loom_block_t* else_block = AppendBlock();

  SetBlock(entry);
  loom_op_t* branch_op = BuildConditionalBranch(then_block, else_block);

  loom_value_id_t sentinel = 123;
  const loom_value_id_t* args = &sentinel;
  uint16_t arg_count = 1;
  EXPECT_FALSE(loom_cfg_terminator_payload_for_successor(branch_op, then_block,
                                                         &args, &arg_count));
  EXPECT_EQ(args, nullptr);
  EXPECT_EQ(arg_count, 0u);
}

}  // namespace
}  // namespace loom
