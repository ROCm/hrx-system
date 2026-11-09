// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/expression_scope.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class ExpressionScopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    vtables = loom_cfg_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_CFG,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("expressions"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* Constant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_expression_cursor_t Next(loom_op_t* expected) {
    loom_expression_cursor_t cursor;
    IREE_CHECK_OK(loom_expression_walk_next(walk_, &cursor));
    EXPECT_EQ(cursor.op, expected);
    return cursor;
  }

  void Initialize(uint32_t maximum_entry_count = 0) {
    IREE_ASSERT_OK(loom_expression_walk_allocate(
        module_, module_->body,
        /*identity_policy=*/nullptr, maximum_entry_count, &arena_, &walk_));
  }

  // Shared allocator for IR and analysis; their arena lifetimes stay separate.
  iree_arena_block_pool_t pool_;
  // Analysis storage whose allocation behavior is part of the contract.
  iree_arena_allocator_t arena_;
  // Minimal registered dialect vocabulary.
  loom_context_t context_;
  // Real IR owned by the test.
  loom_module_t* module_ = nullptr;
  // Builder maintaining region ownership and use-def relationships.
  loom_builder_t builder_;
  // Active production traversal and availability index.
  loom_expression_walk_t* walk_ = nullptr;
};

TEST_F(ExpressionScopeTest,
       CollisionProbesRestoreSiblingBindingsWithoutGrowth) {
  loom_region_t* region = module_->body;
  loom_block_t* branches[3] = {};
  for (auto& block : branches) {
    IREE_ASSERT_OK(loom_region_append_block(module_, region, &block));
  }
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), &condition));
  loom_op_t* root = Constant(42);
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, condition, branches[0],
                                        branches[1], LOOM_LOCATION_UNKNOWN,
                                        &branch));
  for (int i = 0; i < 3; ++i) {
    loom_builder_set_block(&builder_, branches[i]);
    Constant(42);
    Constant(43);
    if (i < 2) {
      IREE_ASSERT_OK(loom_cfg_br_build(&builder_, branches[2], nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &branch));
    }
  }
  // A bound larger than one arena block also exercises the paged bucket tail.
  Initialize(/*maximum_entry_count=*/2048);
  loom_expression_cursor_t cursor = Next(root);
  const iree_host_size_t initial_bytes = arena_.used_allocation_size;
  // Equal hashes exercise structural equality and collision-probe rollback.
  // Hashing is a dependency of the scoped index, not the subject of this test.
  const uint32_t hash = 3000;
  int visited = 0;
  for (;;) {
    if (!cursor.op) {
      break;
    }
    if (!loom_test_constant_isa(cursor.op)) {
      IREE_ASSERT_OK(loom_expression_walk_next(walk_, &cursor));
      continue;
    }
    ++visited;
    auto probe = loom_expression_scope_probe(&cursor, hash, 0);
    auto* existing = probe.entry;
    const bool same_as_root = loom_test_constant_value(cursor.op).i64 == 42;
    if (cursor.op != root && same_as_root) {
      ASSERT_NE(existing, nullptr);
      EXPECT_EQ(existing->op, root);
    } else {
      EXPECT_EQ(existing, nullptr);
    }
    // An expired effect epoch can require a new producer even when a
    // structural equivalent dominates. Sibling bindings remain independent.
    loom_expression_scope_insert(&cursor, probe,
                                 {cursor.op, hash, cursor.epoch});
    IREE_ASSERT_OK(loom_expression_walk_next(walk_, &cursor));
  }
  EXPECT_EQ(visited, 7);
  EXPECT_EQ(arena_.used_allocation_size, initial_bytes);
}

TEST_F(ExpressionScopeTest,
       IsolatedCFGInheritsItsOwnDominatorAndRestoresOuter) {
  loom_op_t* outer = Constant(42);
  loom_op_t* isolated = nullptr;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &isolated));
  loom_region_t* body = loom_test_isolated_region_body(isolated);
  loom_block_t* exit = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &exit));
  auto saved = loom_builder_enter_region(&builder_, isolated, body);
  loom_op_t* inner = Constant(42);
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(
      loom_test_br_build(&builder_, exit, LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, exit);
  loom_op_t* dominated = Constant(42);
  loom_builder_restore(&builder_, saved);
  loom_op_t* resumed = Constant(42);
  Initialize();

  auto cursor = Next(outer);
  const uint32_t outer_epoch = cursor.epoch;
  const uint32_t hash = loom_expression_hash(module_, outer);
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {outer, hash, cursor.epoch});
  Next(isolated);
  cursor = Next(inner);
  const uint32_t inner_epoch = cursor.epoch;
  auto probe = loom_expression_scope_probe(&cursor, hash, 0);
  EXPECT_EQ(probe.entry, nullptr);
  loom_expression_scope_insert(&cursor, probe, {inner, hash, cursor.epoch});
  Next(branch);
  cursor = Next(dominated);
  auto* existing = loom_expression_scope_probe(&cursor, hash, 0).entry;
  ASSERT_NE(existing, nullptr);
  EXPECT_EQ(existing->op, inner);
  EXPECT_EQ(existing->metadata, inner_epoch);
  cursor = Next(resumed);
  existing = loom_expression_scope_probe(&cursor, hash, 0).entry;
  ASSERT_NE(existing, nullptr);
  EXPECT_EQ(existing->op, outer);
  EXPECT_EQ(existing->metadata, outer_epoch);
  Next(nullptr);
}

TEST_F(ExpressionScopeTest, ReplacingAnExpiredLocalBindingKeepsRollbackIntact) {
  loom_op_t* outer = Constant(42);
  loom_value_id_t input = loom_test_constant_result(outer);
  loom_op_t* map = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                     nullptr, 0, LOOM_LOCATION_UNKNOWN, &map));
  auto saved =
      loom_builder_enter_region(&builder_, map, loom_test_map_body(map));
  loom_op_t* first = Constant(42);
  loom_op_t* collision = Constant(43);
  loom_op_t* second = Constant(42);
  loom_op_t* third = Constant(42);
  loom_builder_restore(&builder_, saved);
  loom_op_t* resumed = Constant(42);
  loom_op_t* invisible = Constant(43);
  Initialize(/*maximum_entry_count=*/7);
  const uint32_t hash = 1;
  auto cursor = Next(outer);
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {outer, hash, cursor.epoch});
  Next(map);
  const iree_host_size_t initial_bytes = arena_.used_allocation_size;
  cursor = Next(first);
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {first, hash, cursor.epoch});
  cursor = Next(collision);
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {collision, hash, cursor.epoch});
  cursor = Next(second);
  // Epoch policy can expire a producer without removing its structural key.
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {second, hash, cursor.epoch});
  cursor = Next(third);
  auto* existing = loom_expression_scope_probe(&cursor, hash, 0).entry;
  ASSERT_NE(existing, nullptr);
  EXPECT_EQ(existing->op, second);
  EXPECT_EQ(existing->metadata, cursor.epoch - 1);
  cursor = Next(resumed);
  existing = loom_expression_scope_probe(&cursor, hash, 0).entry;
  ASSERT_NE(existing, nullptr);
  EXPECT_EQ(existing->op, outer);
  cursor = Next(invisible);
  EXPECT_EQ(loom_expression_scope_probe(&cursor, hash, 0).entry, nullptr);
  Next(nullptr);
  EXPECT_EQ(arena_.used_allocation_size, initial_bytes);
}

TEST_F(ExpressionScopeTest,
       AdaptiveIndexIncludesOperationsAfterEmptyNestedBlocks) {
  loom_op_t* outer = Constant(99);
  loom_op_t* map = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, nullptr, 0,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                     nullptr, 0, LOOM_LOCATION_UNKNOWN, &map));
  loom_region_t* body = loom_test_map_body(map);
  loom_block_t* blocks[3] = {};
  for (auto& block : blocks) {
    IREE_ASSERT_OK(loom_region_append_block(module_, body, &block));
  }
  auto saved = loom_builder_enter_region(&builder_, map, body);
  loom_builder_set_block(&builder_, blocks[0]);
  loom_op_t* empty = nullptr;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &empty));
  loom_op_t* first = Constant(100);
  loom_op_t* second = Constant(101);
  loom_op_t* third = Constant(102);
  loom_op_t* repeated = Constant(100);
  loom_builder_set_block(&builder_, blocks[2]);
  loom_op_t* remote = Constant(103);
  loom_builder_restore(&builder_, saved);
  loom_op_t* last = Constant(104);
  Initialize();
  auto cursor = Next(outer);
  const uint32_t hash = 1;
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {outer, hash, cursor.epoch});
  Next(map);
  Next(empty);
  // Deliberately collide distinct structures so the block-entry reservation
  // grows and rehashes occupied probe chains before the repeated expression.
  for (auto* op : {first, second, third}) {
    cursor = Next(op);
    loom_expression_scope_insert(&cursor,
                                 loom_expression_scope_probe(&cursor, hash, 0),
                                 {op, hash, cursor.epoch});
  }
  cursor = Next(repeated);
  auto probe = loom_expression_scope_probe(&cursor, hash, 0);
  ASSERT_NE(probe.entry, nullptr);
  EXPECT_EQ(probe.entry->op, first);
  loom_expression_scope_insert(&cursor, probe, {repeated, hash, cursor.epoch});
  cursor = Next(remote);
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {remote, hash, cursor.epoch});
  cursor = Next(last);
  loom_expression_scope_insert(&cursor,
                               loom_expression_scope_probe(&cursor, hash, 0),
                               {last, hash, cursor.epoch});
  Next(nullptr);
}

TEST_F(ExpressionScopeTest, PromotesCompactHeadsAtBindingIndexBoundary) {
  for (uint32_t i = 0; i < UINT16_MAX; ++i) {
    Constant(i);
  }
  loom_op_t* isolated = nullptr;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &isolated));
  auto saved = loom_builder_enter_region(
      &builder_, isolated, loom_test_isolated_region_body(isolated));
  Constant(UINT16_MAX);
  loom_builder_restore(&builder_, saved);

  Initialize();
  uint32_t visited = 0;
  bool all_misses = true;
  for (;;) {
    loom_expression_cursor_t cursor;
    IREE_ASSERT_OK(loom_expression_walk_next(walk_, &cursor));
    if (!cursor.op) {
      break;
    }
    if (!loom_test_constant_isa(cursor.op)) {
      continue;
    }
    const uint32_t hash = (uint32_t)loom_test_constant_value(cursor.op).i64;
    const loom_expression_probe_t probe =
        loom_expression_scope_probe(&cursor, hash, 0);
    all_misses &= probe.entry == nullptr;
    loom_expression_scope_insert(&cursor, probe,
                                 {cursor.op, hash, cursor.epoch});
    ++visited;
  }
  EXPECT_TRUE(all_misses);
  EXPECT_EQ(visited, (uint32_t)UINT16_MAX + 1);
}

}  // namespace
}  // namespace loom
