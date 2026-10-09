// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/walk.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class WalkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    // Build a test.func to provide a function body.
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    func_like_ = loom_func_like_cast(module_, func_op);
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    // Set parent_op so nested region builders inherit correct ancestry.
    builder_.ip.parent_op = func_op;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Enters a nested region's block, setting parent_op for ancestry.
  void enter_region(loom_op_t* parent_op, loom_region_t* region) {
    saved_ips_.push_back(
        loom_builder_enter_region(&builder_, parent_op, region));
  }

  void leave_region() {
    ASSERT_FALSE(saved_ips_.empty());
    loom_builder_restore(&builder_, saved_ips_.back());
    saved_ips_.pop_back();
  }

  // Finalizes the module (sets parent pointers via compute_uses).
  void finalize() { IREE_ASSERT_OK(loom_module_compute_uses(module_)); }

  uint64_t block_allocation_count() const {
    iree_arena_block_pool_statistics_t statistics;
    iree_arena_block_pool_query_statistics(&block_pool_, &statistics);
    return statistics.block_system_allocation_count;
  }

  void expect_blocks_reusable(uint64_t block_count,
                              uint64_t expected_allocation_count) {
    iree_arena_allocator_t probe_arena;
    iree_arena_initialize(&block_pool_, &probe_arena);
    for (uint64_t i = 0; i < block_count; ++i) {
      void* allocation = nullptr;
      IREE_ASSERT_OK(iree_arena_allocate(
          &probe_arena, iree_arena_block_pool_max_allocation_size(&block_pool_),
          &allocation));
    }
    EXPECT_EQ(block_allocation_count(), expected_allocation_count);
    iree_arena_deinitialize(&probe_arena);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  std::vector<loom_builder_ip_t> saved_ips_;
};

//===----------------------------------------------------------------------===//
// Walk callback helpers
//===----------------------------------------------------------------------===//

struct WalkRecord {
  // Operation visited by the walker.
  loom_op_t* op;
  // Operation owning the visited operation's region.
  loom_op_t* parent_op;
  // Region nesting depth reported by the walker.
  uint16_t depth;
};

struct WalkState {
  // Visited operations and their traversal context.
  std::vector<WalkRecord> records;
  // Operation whose nested regions are skipped, if present.
  loom_op_t* skip_op = nullptr;
  // Operation that aborts traversal, if present.
  loom_op_t* abort_op = nullptr;
  // Operation that fails traversal, if present.
  loom_op_t* fail_op = nullptr;
};

static iree_status_t record_visitor(void* user_data, loom_op_t* op,
                                    const loom_walk_context_t* context,
                                    loom_walk_result_t* out_result) {
  auto* state = static_cast<WalkState*>(user_data);
  state->records.push_back({op, context->parent_op, context->depth});
  *out_result = LOOM_WALK_CONTINUE;
  if (state->fail_op == op) {
    return iree_make_status(IREE_STATUS_ABORTED, "expected callback failure");
  }
  if (state->skip_op == op) {
    *out_result = LOOM_WALK_SKIP;
  } else if (state->abort_op == op) {
    *out_result = LOOM_WALK_ABORT;
  }
  return iree_ok_status();
}

struct NestedWalkState {
  // Module owning both traversals.
  loom_module_t* module;
  // Operation that starts the nested traversal.
  loom_op_t* trigger_op;
  // Region walked while the outer traversal remains active.
  loom_region_t* nested_region;
  // Operations visited by the outer traversal.
  WalkState outer;
  // Operations visited by the nested traversal.
  WalkState nested;
};

static iree_status_t nested_visitor(void* user_data, loom_op_t* op,
                                    const loom_walk_context_t* context,
                                    loom_walk_result_t* out_result) {
  auto* state = static_cast<NestedWalkState*>(user_data);
  IREE_RETURN_IF_ERROR(record_visitor(&state->outer, op, context, out_result));
  if (op != state->trigger_op) {
    return iree_ok_status();
  }
  loom_walk_result_t nested_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(
      loom_walk_region(state->module, state->nested_region, LOOM_WALK_PRE_ORDER,
                       {record_visitor, &state->nested}, &nested_result));
  EXPECT_EQ(nested_result, LOOM_WALK_CONTINUE);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_F(WalkTest, PreOrderFlatBlock) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  loom_op_t* c3 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), i32,
                                          LOOM_LOCATION_UNKNOWN, &c3));
  finalize();

  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;
  const uint64_t block_allocations = block_allocation_count();
  WalkState state;
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  constexpr iree_host_size_t kWalkCount = 2048;
  for (iree_host_size_t i = 0; i < kWalkCount; ++i) {
    state.records.clear();
    walk_result = LOOM_WALK_ABORT;
    IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                      {record_visitor, &state}, &walk_result));
  }

  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  ASSERT_EQ(state.records.size(), 3u);
  EXPECT_EQ(state.records[0].op, c1);
  EXPECT_EQ(state.records[1].op, c2);
  EXPECT_EQ(state.records[2].op, c3);
  // All at depth 0 (function body is the top-level region).
  for (auto& r : state.records) {
    EXPECT_EQ(r.depth, 0u);
  }
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
  EXPECT_EQ(block_allocation_count(), block_allocations);
}

TEST_F(WalkTest, PreOrderNestedRegion) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  // Build: outer_const, map(body: inner_const), after_const.
  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));
  loom_value_id_t arg = loom_test_constant_result(outer_const);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();

  loom_op_t* after_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), i32,
                                          LOOM_LOCATION_UNKNOWN, &after_const));
  finalize();

  WalkState state;
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &state}, &walk_result));

  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  ASSERT_EQ(state.records.size(), 4u);
  // Pre-order: outer, map, inner (depth 1), after.
  EXPECT_EQ(state.records[0].op, outer_const);
  EXPECT_EQ(state.records[0].depth, 0u);
  EXPECT_EQ(state.records[1].op, map_op);
  EXPECT_EQ(state.records[1].depth, 0u);
  EXPECT_EQ(state.records[2].op, inner_const);
  EXPECT_EQ(state.records[2].depth, 1u);
  EXPECT_EQ(state.records[2].parent_op, map_op);
  EXPECT_EQ(state.records[3].op, after_const);
  EXPECT_EQ(state.records[3].depth, 0u);
}

TEST_F(WalkTest, PostOrderNestedRegion) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));
  loom_value_id_t arg = loom_test_constant_result(outer_const);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();
  finalize();

  WalkState state;
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_POST_ORDER,
                                    {record_visitor, &state}, &walk_result));

  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  ASSERT_EQ(state.records.size(), 3u);
  // Post-order: outer_const (no regions, immediate), inner (child),
  // then map (parent after children).
  EXPECT_EQ(state.records[0].op, outer_const);
  EXPECT_EQ(state.records[1].op, inner_const);
  EXPECT_EQ(state.records[2].op, map_op);
}

TEST_F(WalkTest, SkipNestedRegions) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));
  loom_value_id_t arg = loom_test_constant_result(outer_const);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();
  finalize();

  WalkState state;
  state.skip_op = map_op;  // Skip map's nested regions.
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &state}, &walk_result));

  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  // Only outer_const and map_op — inner_const is skipped.
  ASSERT_EQ(state.records.size(), 2u);
  EXPECT_EQ(state.records[0].op, outer_const);
  EXPECT_EQ(state.records[1].op, map_op);
}

TEST_F(WalkTest, AbortEarly) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  loom_op_t* c3 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), i32,
                                          LOOM_LOCATION_UNKNOWN, &c3));
  finalize();

  WalkState state;
  state.abort_op = c2;  // Abort when we see c2.
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &state}, &walk_result));

  EXPECT_EQ(walk_result, LOOM_WALK_ABORT);
  // Visited c1 and c2 (abort fires after recording).
  ASSERT_EQ(state.records.size(), 2u);
  EXPECT_EQ(state.records[0].op, c1);
  EXPECT_EQ(state.records[1].op, c2);
}

TEST_F(WalkTest, EmptyFunction) {
  finalize();

  WalkState state;
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &state}, &walk_result));

  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  EXPECT_EQ(state.records.size(), 0u);
}

TEST_F(WalkTest, IsolatedRegionContext) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* iso_op = NULL;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &iso_op));

  enter_region(iso_op, loom_test_isolated_region_body(iso_op));
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();
  finalize();

  WalkState state;
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &state}, &walk_result));

  ASSERT_EQ(state.records.size(), 2u);
  // The isolated op itself is at depth 0.
  EXPECT_EQ(state.records[0].op, iso_op);
  EXPECT_EQ(state.records[0].depth, 0u);
  // The inner constant is at depth 1 with parent_op = iso_op.
  EXPECT_EQ(state.records[1].op, inner_const);
  EXPECT_EQ(state.records[1].depth, 1u);
  EXPECT_EQ(state.records[1].parent_op, iso_op);
}

TEST_F(WalkTest, DeepNesting) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  // Build enough nested regions to exercise the inline stack's spill path.
  constexpr uint16_t kMapCount = 40;
  loom_op_t* dummy = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &dummy));
  loom_value_id_t arg = loom_test_constant_result(dummy);
  std::vector<loom_op_t*> maps;
  for (uint16_t i = 0; i < kMapCount; ++i) {
    loom_op_t* map_op = NULL;
    IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &map_op));
    maps.push_back(map_op);
    loom_region_t* body = loom_test_map_body(map_op);
    enter_region(map_op, body);
    arg = loom_region_entry_arg_id(body, 0);
  }
  loom_op_t* deep_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &deep_const));
  for (uint16_t i = 0; i < kMapCount; ++i) {
    leave_region();
  }
  finalize();

  iree_arena_block_pool_trim(&block_pool_);
  const uint64_t allocations_before = block_allocation_count();
  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;
  WalkState state;
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &state}, &walk_result));

  ASSERT_EQ(state.records.size(), kMapCount + 2);
  EXPECT_EQ(state.records[0].depth, 0u);  // dummy
  for (uint16_t i = 0; i < kMapCount; ++i) {
    EXPECT_EQ(state.records[i + 1].op, maps[i]);
    EXPECT_EQ(state.records[i + 1].depth, i);
  }
  EXPECT_EQ(state.records.back().op, deep_const);
  EXPECT_EQ(state.records.back().depth, kMapCount);
  EXPECT_EQ(state.records.back().parent_op, maps.back());
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);

  const uint64_t allocations_after_walk = block_allocation_count();
  const uint64_t spill_block_count =
      allocations_after_walk - allocations_before;
  EXPECT_GT(spill_block_count, 0u);
  expect_blocks_reusable(spill_block_count, allocations_after_walk);

  WalkState post_order_state;
  walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_POST_ORDER,
                                    {record_visitor, &post_order_state},
                                    &walk_result));
  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  ASSERT_EQ(post_order_state.records.size(), kMapCount + 2);
  EXPECT_EQ(post_order_state.records[0].op, dummy);
  EXPECT_EQ(post_order_state.records[1].op, deep_const);
  EXPECT_EQ(post_order_state.records[1].depth, kMapCount);
  for (uint16_t i = 0; i < kMapCount; ++i) {
    const uint16_t map_index = kMapCount - i - 1;
    EXPECT_EQ(post_order_state.records[i + 2].op, maps[map_index]);
    EXPECT_EQ(post_order_state.records[i + 2].depth, map_index);
  }
  expect_blocks_reusable(spill_block_count, allocations_after_walk);

  WalkState abort_state;
  abort_state.abort_op = deep_const;
  walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {record_visitor, &abort_state},
                                    &walk_result));
  EXPECT_EQ(walk_result, LOOM_WALK_ABORT);
  expect_blocks_reusable(spill_block_count, allocations_after_walk);

  WalkState failure_state;
  failure_state.fail_op = deep_const;
  walk_result = LOOM_WALK_CONTINUE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                         {record_visitor, &failure_state}, &walk_result));
  expect_blocks_reusable(spill_block_count, allocations_after_walk);

  constexpr uint16_t kNestedTrigger = kMapCount / 2;
  NestedWalkState nested_state = {
      .module = module_,
      .trigger_op = maps[kNestedTrigger],
      .nested_region = loom_test_map_body(maps[kNestedTrigger]),
  };
  walk_result = LOOM_WALK_CONTINUE;
  IREE_ASSERT_OK(loom_walk_function(module_, func_like_, LOOM_WALK_PRE_ORDER,
                                    {nested_visitor, &nested_state},
                                    &walk_result));
  EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
  EXPECT_EQ(nested_state.outer.records.size(), kMapCount + 2);
  EXPECT_EQ(nested_state.nested.records.size(), kMapCount - kNestedTrigger);
  const uint64_t allocations_after_nested_walk = block_allocation_count();
  expect_blocks_reusable(allocations_after_nested_walk - allocations_before,
                         allocations_after_nested_walk);
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
}

}  // namespace
}  // namespace loom
