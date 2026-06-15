// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/dce.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

static iree_status_t InitializePassStatistics(loom_pass_t* pass,
                                              iree_arena_allocator_t* arena) {
  const loom_pass_statistic_layout_t* layout = pass->info->statistic_layout;
  if (!layout) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, layout->storage_size,
                                           (void**)&pass->statistic_storage));
  memset(pass->statistic_storage, 0, layout->storage_size);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class DCETest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    // Register test dialect so trait lookups work.
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    // Build a test.func op on the module body block to provide a real
    // func-like interface for passes under test. The builder finalizer
    // wires defining_op on the symbol automatically.
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    func_like_ = loom_func_like_cast(module_, func_op);
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Runs DCE on the test function.
  iree_status_t run_dce() {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_t pass;
    memset(&pass, 0, sizeof(pass));
    pass.info = loom_dce_pass_info();
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    iree_status_t status = InitializePassStatistics(&pass, &pass_arena);
    if (iree_status_is_ok(status)) {
      status = loom_dce_run(&pass, module_, func_like_);
    }
    iree_arena_deinitialize(&pass_arena);
    return status;
  }

  // Counts live (non-dead) ops in the function body.
  int count_live_ops() {
    int count = 0;
    loom_block_t* block = NULL;
    loom_region_for_each_block(body_, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) { ++count; }
    }
    return count;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_F(DCETest, RemovesDeadPureOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = constant (pure, unused) → should be removed.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  EXPECT_EQ(count_live_ops(), 1);

  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, PreservesUsedOps) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = constant, %b = neg %a → %a is used, both should survive.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // neg's result is unused, but neg is pure → DCE removes neg.
  // That makes %a unused → DCE removes the constant too.
  EXPECT_EQ(count_live_ops(), 2);

  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, PreservesDynamicDimProducerForLiveTypeUse) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(4), index_type, LOOM_LOCATION_UNKNOWN, &dim_op));
  loom_value_id_t dim_id = loom_test_constant_result(dim_op);
  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim_id), 0);
  loom_op_t* vector_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                          vector_type, LOOM_LOCATION_UNKNOWN,
                                          &vector_op));
  loom_value_id_t vector_id = loom_test_constant_result(vector_op);
  loom_value_id_t values[] = {vector_id};
  loom_op_t* use_op = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, values, IREE_ARRAYSIZE(values),
                                     LOOM_LOCATION_UNKNOWN, &use_op));

  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 3);
  EXPECT_FALSE(dim_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_FALSE(vector_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_FALSE(use_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_TRUE(loom_module_value_has_type_uses(module_, dim_id));
}

TEST_F(DCETest, RemovesDynamicDimProducerAfterDeadCarrierErased) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(4), index_type, LOOM_LOCATION_UNKNOWN, &dim_op));
  loom_value_id_t dim_id = loom_test_constant_result(dim_op);
  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim_id), 0);
  loom_op_t* vector_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                          vector_type, LOOM_LOCATION_UNKNOWN,
                                          &vector_op));

  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
  EXPECT_TRUE(dim_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_TRUE(vector_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_FALSE(loom_module_value_has_type_uses(module_, dim_id));
}

TEST_F(DCETest, RemovesDynamicDimProducerAfterDeadRegionBlockArgErased) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(4), index_type, LOOM_LOCATION_UNKNOWN, &dim_op));
  loom_value_id_t dim_id = loom_test_constant_result(dim_op);

  loom_op_t* lower_bound_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &lower_bound_op));
  loom_value_id_t lower_bound = loom_test_constant_result(lower_bound_op);

  loom_op_t* upper_bound_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &upper_bound_op));
  loom_value_id_t upper_bound = loom_test_constant_result(upper_bound_op);

  loom_op_t* step_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &step_op));
  loom_value_id_t step = loom_test_constant_result(step_op);

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(dim_id), 0);
  loom_op_t* init_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                          vector_type, LOOM_LOCATION_UNKNOWN,
                                          &init_op));
  loom_value_id_t init = loom_test_constant_result(init_op);

  loom_value_id_t iter_args[] = {init};
  loom_type_t result_types[] = {vector_type};
  loom_op_t* loop_op = NULL;
  IREE_ASSERT_OK(loom_test_loop_build(
      &builder_, lower_bound, upper_bound, step, iter_args,
      IREE_ARRAYSIZE(iter_args), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, LOOM_LOCATION_UNKNOWN, &loop_op));

  EXPECT_TRUE(loom_module_value_has_type_uses(module_, dim_id));
  EXPECT_EQ(count_live_ops(), 6);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
  EXPECT_TRUE(dim_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_TRUE(init_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_TRUE(loop_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_FALSE(loom_module_value_has_type_uses(module_, dim_id));
}

TEST_F(DCETest, PreservesLayoutProducerForLiveViewTypeUse) {
  loom_type_t layout_type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);

  loom_op_t* layout_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                          layout_type, LOOM_LOCATION_UNKNOWN,
                                          &layout_op));
  loom_value_id_t layout_id = loom_test_constant_result(layout_op);
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  view_type.encoding_id = (uint16_t)layout_id;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), view_type, LOOM_LOCATION_UNKNOWN, &view_op));
  loom_value_id_t view_id = loom_test_constant_result(view_op);
  loom_value_id_t values[] = {view_id};
  loom_op_t* use_op = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, values, IREE_ARRAYSIZE(values),
                                     LOOM_LOCATION_UNKNOWN, &use_op));

  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 3);
  EXPECT_FALSE(layout_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_FALSE(view_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_FALSE(use_op->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_TRUE(loom_module_value_has_type_uses(module_, layout_id));
}

TEST_F(DCETest, RemovesChain) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = constant
  // %b = neg %a
  // %c = neg %b
  // All unused, all pure. DCE should remove all three.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  loom_op_t* neg1 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg1));
  loom_value_id_t b = loom_test_neg_result(neg1);

  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, b, i32, LOOM_LOCATION_UNKNOWN, &neg2));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, PreservesTerminator) {
  // test.yield has no results and is a terminator — should never be removed.
  loom_op_t* yield_op = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  EXPECT_EQ(count_live_ops(), 1);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 1);
}

TEST_F(DCETest, PartialChainSurvives) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = constant
  // %b = neg %a
  // %c = addi %a, %a (uses %a, result unused)
  // DCE removes %c (unused pure), then %b (unused pure after %c gone).
  // %a survives? No — after %b and %c are gone, %a is also unused.
  // Actually: %c uses %a twice, %b uses %a once. After removing %c,
  // %a still has 1 use (%b). After removing %b, %a has 0 uses → removed.

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, RemovesReadOnlyOpWithUnusedResult) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Create a pool-typed block arg so we have a resource value.
  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // %tile = test.read_resource %pool (reads pool, result unused → dead).
  loom_op_t* read_op = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(
      &builder_, pool_id, i32, LOOM_LOCATION_UNKNOWN, &read_op));

  EXPECT_EQ(count_live_ops(), 1);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, PreservesWriteOp) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // Create a data value for the write.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t data = loom_test_constant_result(const_op);

  // test.write_resource %pool, %data (writes pool, no results → preserved).
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, data, LOOM_LOCATION_UNKNOWN, &write_op));

  EXPECT_EQ(count_live_ops(), 2);
  IREE_ASSERT_OK(run_dce());
  // Write op survives. Constant also survives because the write uses it.
  EXPECT_EQ(count_live_ops(), 2);
}

TEST_F(DCETest, PreservesMutateOpRemovesUnusedResult) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t data = loom_test_constant_result(const_op);

  // %old = test.mutate_resource %pool, %data (readwrite, result unused).
  // The op writes, so it must survive even though %old is unused.
  loom_op_t* mutate_op = NULL;
  IREE_ASSERT_OK(loom_test_mutate_resource_build(
      &builder_, pool_id, data, i32, LOOM_LOCATION_UNKNOWN, &mutate_op));

  EXPECT_EQ(count_live_ops(), 2);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 2);
}

//===----------------------------------------------------------------------===//
// Cascading through use patterns
//===----------------------------------------------------------------------===//

TEST_F(DCETest, DiamondUsePattern) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = const 42
  // %b = neg %a   (unused)
  // %c = neg %a   (unused)
  // All three should be removed. %a has two users, but both are dead,
  // so cascading DCE removes %b and %c first, then %a.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  loom_op_t* neg1 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg1));
  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg2));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, PartialLiveness) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // %a = const 42
  // %b = neg %a   (unused — dead)
  // write_resource %pool, %a  (uses %a, writes — alive)
  // After DCE: %b is removed, but %a and the write survive.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, a, LOOM_LOCATION_UNKNOWN, &write_op));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_dce());
  // neg removed, const and write survive.
  EXPECT_EQ(count_live_ops(), 2);
}

//===----------------------------------------------------------------------===//
// Effect-based preservation
//===----------------------------------------------------------------------===//

TEST_F(DCETest, UnknownEffectsWithResultsSurvives) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // test.invoke has UNKNOWN_EFFECTS. Even with unused results,
  // it must not be removed — the effects may be observable.
  loom_type_t result_types[] = {i32};
  loom_op_t* invoke_op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, loom_symbol_ref_null(), NULL,
                                        0, result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &invoke_op));

  EXPECT_EQ(count_live_ops(), 1);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 1);
}

TEST_F(DCETest, NonDeterministicReadWithUnusedResultIsDead) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // A read-only op with an unused result is dead — a read with no
  // observer is a no-op, even if the read itself is non-deterministic.
  loom_op_t* read_op = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(
      &builder_, pool_id, i32, LOOM_LOCATION_UNKNOWN, &read_op));

  EXPECT_EQ(count_live_ops(), 1);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

//===----------------------------------------------------------------------===//
// Nested regions
//===----------------------------------------------------------------------===//

// Helper: count live ops recursively across all blocks and nested regions.
static int count_all_live_ops_dce(loom_region_t* region) {
  int count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      ++count;
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t r = 0; r < op->region_count; ++r) {
        count += count_all_live_ops_dce(regions[r]);
      }
    }
  }
  return count;
}

TEST_F(DCETest, NestedRegionDeadOpsRemoved) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Build test.map with an unused result — the map itself is dead.
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  // Dead constant inside the map body.
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);

  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  loom_builder_restore(&builder_, saved);

  EXPECT_EQ(count_all_live_ops_dce(body_), 2);
  IREE_ASSERT_OK(run_dce());
  // Both map and inner constant removed.
  EXPECT_EQ(count_all_live_ops_dce(body_), 0);
}

TEST_F(DCETest, InnerDeadOpRemovedWhileOuterLives) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));
  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // Build test.map — keep its result alive via a write.
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  loom_value_id_t map_result = loom_test_map_result(map_op);

  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, map_result, LOOM_LOCATION_UNKNOWN, &write_op));

  // Dead constant inside the map body — unused by anything.
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);

  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  loom_builder_restore(&builder_, saved);

  // map + write + inner_const = 3 total.
  EXPECT_EQ(count_all_live_ops_dce(body_), 3);
  IREE_ASSERT_OK(run_dce());
  // inner_const removed. map and write survive (map has a live use).
  EXPECT_EQ(count_all_live_ops_dce(body_), 2);
}

TEST_F(DCETest, CascadingAcrossRegionBoundary) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Outer constant — its only use is inside a nested region.
  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));
  loom_value_id_t x = loom_test_constant_result(outer_const);

  // Map with an unused result.
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  // Inside the map: neg %x (unused). This is the only use of %x.
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);

  loom_op_t* inner_neg = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, x, i32, LOOM_LOCATION_UNKNOWN,
                                     &inner_neg));
  loom_builder_restore(&builder_, saved);

  // outer_const + map + inner_neg = 3 total.
  EXPECT_EQ(count_all_live_ops_dce(body_), 3);
  IREE_ASSERT_OK(run_dce());
  // inner_neg is dead → removed. That makes outer_const dead (no
  // more uses) → removed. map is also dead (unused result) → removed.
  EXPECT_EQ(count_all_live_ops_dce(body_), 0);
}

TEST_F(DCETest, DeepNestedDeadOpsRemoved) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Build 50 levels of nested test.map, each with a dead constant.
  static const int kDepth = 50;
  for (int depth = 0; depth < kDepth; ++depth) {
    loom_op_t* map_op = NULL;
    IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &map_op));
    loom_region_t* body = loom_test_map_body(map_op);
    loom_builder_enter_region(&builder_, map_op, body);

    loom_op_t* dead_const = NULL;
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(depth),
                                            i32, LOOM_LOCATION_UNKNOWN,
                                            &dead_const));
  }

  int before = count_all_live_ops_dce(body_);
  // 50 maps + 50 constants = 100 ops.
  EXPECT_EQ(before, kDepth * 2);
  IREE_ASSERT_OK(run_dce());
  // All dead — maps have unused results, constants are unused.
  EXPECT_EQ(count_all_live_ops_dce(body_), 0);
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(DCETest, EmptyFunctionBody) {
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, NullFunctionBody) {
  // A zero-initialized func_like is the invalid/null sentinel — DCE must
  // handle it gracefully and return OK.
  loom_func_like_t empty_func = {};

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_t pass;
  memset(&pass, 0, sizeof(pass));
  pass.info = loom_dce_pass_info();
  pass.instance_arena = &pass_arena;
  pass.arena = &pass_arena;
  IREE_ASSERT_OK(InitializePassStatistics(&pass, &pass_arena));
  IREE_EXPECT_OK(loom_dce_run(&pass, module_, empty_func));
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(DCETest, UnusedAllocationIsRemoved) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));

  // Create an index block arg for the allocation size.
  loom_value_id_t size_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), index_type, &size_id));

  // An alloc with no uses — should be eliminated by DCE.
  loom_op_t* alloc_op = NULL;
  IREE_ASSERT_OK(loom_test_alloc_build(&builder_, size_id, pool_type,
                                       LOOM_LOCATION_UNKNOWN, &alloc_op));

  EXPECT_EQ(count_live_ops(), 1);
  IREE_ASSERT_OK(run_dce());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(DCETest, UsedAllocationSurvives) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t size_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), index_type, &size_id));

  // An alloc whose pool is used by a write (which has side effects
  // and keeps the entire chain alive).
  loom_op_t* alloc_op = NULL;
  IREE_ASSERT_OK(loom_test_alloc_build(&builder_, size_id, pool_type,
                                       LOOM_LOCATION_UNKNOWN, &alloc_op));
  loom_value_id_t pool_id = loom_test_alloc_result(alloc_op);

  // Create data to write.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t data = loom_test_constant_result(const_op);

  // Write to the pool — has WRITES_MEMORY so DCE preserves it.
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, data, LOOM_LOCATION_UNKNOWN, &write_op));

  // alloc + constant + write = 3 ops.
  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_dce());
  // Write survives (side effect). That keeps pool_id alive, which
  // keeps alloc alive. Constant is also alive as write's data operand.
  EXPECT_EQ(count_live_ops(), 3);
}

}  // namespace
}  // namespace loom
