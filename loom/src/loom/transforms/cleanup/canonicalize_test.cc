// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/canonicalize.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"

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

class CanonicalizeTest : public ::testing::Test {
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
    vtables = loom_scalar_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, vtables, (uint16_t)vtable_count));
    vtables = loom_vector_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_VECTOR, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    // Build a test.func op on the module body block to provide a real
    // func-like interface for passes under test.
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

  iree_status_t run_canonicalize() {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_t pass;
    memset(&pass, 0, sizeof(pass));
    pass.info = loom_canonicalize_pass_info();
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    pass.value_facts = &value_facts;
    iree_status_t status = InitializePassStatistics(&pass, &pass_arena);
    if (iree_status_is_ok(status)) {
      status = loom_canonicalize_run(&pass, module_, func_like_);
    }
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&pass_arena);
    return status;
  }

  int count_live_ops() {
    int count = 0;
    loom_block_t* block = NULL;
    loom_region_for_each_block(body_, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) { ++count; }
    }
    return count;
  }

  iree_status_t add_symbol(iree_string_view_t name,
                           loom_symbol_ref_t* out_symbol_ref) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module_, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module_, name_id, &symbol_id));
    *out_symbol_ref = (loom_symbol_ref_t){
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    return iree_ok_status();
  }

  // Returns the integer constant value of the op defining |value_id|.
  // Assumes the defining op is a constant-like op with an i64 attr.
  int64_t constant_value(loom_value_id_t value_id) {
    loom_value_t* value = loom_module_value(module_, value_id);
    loom_op_t* def = loom_value_def_op(value);
    if (!def) return INT64_MIN;
    return loom_attr_as_i64(loom_op_attrs(def)[0]);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(CanonicalizeTest, AddiZeroRight) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %x = constant 42
  // %zero = constant 0
  // %r = addi %x, %zero → should fold to constant 42
  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* const_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_zero));
  loom_value_id_t zero = loom_test_constant_result(const_zero);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, x, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  // addi folded to a constant with value 42.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizeTest, AddiZeroLeft) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %zero = constant 0
  // %x = constant 42
  // %r = addi %zero, %x → should fold to constant 42
  loom_op_t* const_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_zero));
  loom_value_id_t zero = loom_test_constant_result(const_zero);

  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, zero, x, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizeTest, ConstantFoldAddi) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = constant 1, %b = constant 2, %r = addi %a, %b
  // Both operands are constants — fold to constant 3.
  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_value_id_t a = loom_test_constant_result(c1);

  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  loom_value_id_t b = loom_test_constant_result(c2);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, b, i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());
  // addi(1, 2) folded to constant 3.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 3);
}

TEST_F(CanonicalizeTest, ChainedFolds) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %x = constant 42
  // %zero = constant 0
  // %a = addi %x, %zero     → folds to %x
  // %b = addi %a, %zero     → folds to %x (after %a is replaced)
  // %c = neg %b              → uses %x after both folds
  //
  // This tests that the worklist propagates: when %a folds, %b's
  // operand changes, putting %b back on the worklist for another fold.
  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* const_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_zero));
  loom_value_id_t zero = loom_test_constant_result(const_zero);

  loom_op_t* addi_a = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, x, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_a));
  loom_value_id_t a = loom_test_addi_result(addi_a);

  loom_op_t* addi_b = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, a, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_b));
  loom_value_id_t b = loom_test_addi_result(addi_b);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &b, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  // Both addi ops fold to constant 42. neg uses the folded result.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizeTest, MultipleUsersRevisited) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %x = constant 5
  // %zero = constant 0
  // %a = addi %x, %zero     → folds to constant 5
  // %b = addi %a, %zero     → folds to constant 5
  // %c = addi %a, %x        → folds to constant 10 (both operands exact)
  //
  // Tests that all users are revisited when operands change.
  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(5), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* const_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_zero));
  loom_value_id_t zero = loom_test_constant_result(const_zero);

  loom_op_t* addi_a = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, x, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_a));
  loom_value_id_t a = loom_test_addi_result(addi_a);

  loom_op_t* addi_b = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, a, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_b));
  loom_value_id_t b = loom_test_addi_result(addi_b);

  loom_op_t* addi_c = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, a, x, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_c));

  loom_op_t* use_b = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &b, 1, LOOM_LOCATION_UNKNOWN, &use_b));
  loom_value_id_t c = loom_test_addi_result(addi_c);
  loom_op_t* use_c = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &c, 1, LOOM_LOCATION_UNKNOWN, &use_c));

  IREE_ASSERT_OK(run_canonicalize());

  // All addi ops fold: a=5, b=5, c=10. neg_b uses 5, neg_c uses 10.
  EXPECT_EQ(constant_value(loom_op_operands(use_b)[0]), 5);
  EXPECT_EQ(constant_value(loom_op_operands(use_c)[0]), 10);
}

TEST_F(CanonicalizeTest, NewOpsAddedToWorklist) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // test.neg has no fold — it should pass through without folding.
  // The builder callback adds new ops to the worklist, and the
  // worklist drains without crashing.
  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &x, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());
  // neg has no fold or canonicalize. constant has fold but is already a
  // constant — nothing to fold to. Both survive.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 7);
}

TEST_F(CanonicalizeTest, EmptyFunctionNoOps) {
  // Empty function body — canonicalize should succeed with nothing to do.
  IREE_ASSERT_OK(run_canonicalize());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(CanonicalizeTest, WorklistDedupPreventsRedundantWork) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %x = constant 42
  // %zero = constant 0
  // %a = addi %x, %zero
  // %b = addi %x, %zero     (identical to %a — same operands)
  // %c = addi %a, %b        (uses both)
  //
  // When %a folds, %c goes on the worklist (user of %a's result).
  // When %b folds, %c would go on the worklist again, but dedup
  // (ON_WORKLIST flag) prevents the duplicate.
  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* const_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_zero));
  loom_value_id_t zero = loom_test_constant_result(const_zero);

  loom_op_t* addi_a = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, x, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_a));
  loom_value_id_t a = loom_test_addi_result(addi_a);

  loom_op_t* addi_b = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, x, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_b));
  loom_value_id_t b = loom_test_addi_result(addi_b);

  loom_op_t* addi_c = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, a, b, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi_c));
  loom_value_id_t c = loom_test_addi_result(addi_c);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &c, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  // All three addi ops fold: a=42, b=42, c=84. neg uses 84.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 84);
}

TEST_F(CanonicalizeTest, ScalarPoisonPropagatesThroughPureOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t x = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &x));

  loom_op_t* poison = NULL;
  IREE_ASSERT_OK(
      loom_scalar_poison_build(&builder_, i32, LOOM_LOCATION_UNKNOWN, &poison));

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_scalar_addi_build(&builder_, /*instance_flags=*/0,
                                        loom_scalar_poison_result(poison), x,
                                        i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result = loom_scalar_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  loom_value_id_t observed = loom_op_operands(use)[0];
  loom_op_t* def = loom_value_def_op(loom_module_value(module_, observed));
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(loom_scalar_poison_isa(def));
  EXPECT_NE(observed, result);
}

TEST_F(CanonicalizeTest, VectorPoisonPropagatesThroughPureOp) {
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  loom_value_id_t x = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), v4f32, &x));

  loom_op_t* poison = NULL;
  IREE_ASSERT_OK(loom_vector_poison_build(&builder_, v4f32,
                                          LOOM_LOCATION_UNKNOWN, &poison));

  loom_op_t* addf = NULL;
  IREE_ASSERT_OK(loom_vector_addf_build(&builder_, /*instance_flags=*/0,
                                        loom_vector_poison_result(poison), x,
                                        v4f32, LOOM_LOCATION_UNKNOWN, &addf));
  loom_value_id_t result = loom_vector_addf_result(addf);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  loom_value_id_t observed = loom_op_operands(use)[0];
  loom_op_t* def = loom_value_def_op(loom_module_value(module_, observed));
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(loom_vector_poison_isa(def));
  EXPECT_NE(observed, result);
}

TEST_F(CanonicalizeTest, PoisonBeatsFmaiZeroMultiplier) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t c = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &c));

  loom_op_t* poison = NULL;
  IREE_ASSERT_OK(
      loom_scalar_poison_build(&builder_, i32, LOOM_LOCATION_UNKNOWN, &poison));
  loom_op_t* zero_op = NULL;
  IREE_ASSERT_OK(loom_scalar_constant_build(&builder_, loom_attr_i64(0), i32,
                                            LOOM_LOCATION_UNKNOWN, &zero_op));

  loom_op_t* fmai = NULL;
  IREE_ASSERT_OK(loom_scalar_fmai_build(&builder_, /*instance_flags=*/0,
                                        loom_scalar_poison_result(poison),
                                        loom_scalar_constant_result(zero_op), c,
                                        i32, LOOM_LOCATION_UNKNOWN, &fmai));
  loom_value_id_t result = loom_scalar_fmai_result(fmai);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  loom_value_id_t observed = loom_op_operands(use)[0];
  loom_op_t* def = loom_value_def_op(loom_module_value(module_, observed));
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(loom_scalar_poison_isa(def));
  EXPECT_NE(observed, c);
}

//===----------------------------------------------------------------------===//
// Fixed point and edge cases
//===----------------------------------------------------------------------===//

TEST_F(CanonicalizeTest, NullFunctionBody) {
  // A zero-initialized func_like is the invalid/null sentinel — canonicalize
  // must handle it gracefully and return OK.
  loom_func_like_t empty_func = {};

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_t pass;
  memset(&pass, 0, sizeof(pass));
  pass.info = loom_canonicalize_pass_info();
  pass.instance_arena = &pass_arena;
  pass.arena = &pass_arena;
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  pass.value_facts = &value_facts;
  IREE_ASSERT_OK(InitializePassStatistics(&pass, &pass_arena));
  IREE_EXPECT_OK(loom_canonicalize_run(&pass, module_, empty_func));
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CanonicalizeTest, FixedPointConvergence) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Build a foldable pattern, run canonicalize, then run again.
  // The second run should produce no changes (fixed point reached).
  loom_op_t* const_x = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_x));
  loom_value_id_t x = loom_test_constant_result(const_x);

  loom_op_t* const_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_zero));
  loom_value_id_t zero = loom_test_constant_result(const_zero);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, x, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);

  // Second run — nothing should change. The fold already happened.
  int ops_before = count_live_ops();
  IREE_ASSERT_OK(run_canonicalize());
  EXPECT_EQ(count_live_ops(), ops_before);
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizeTest, DriverAcceptsSeedFacts) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  loom_op_t* const_two = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_two));
  loom_value_id_t two = loom_test_constant_result(const_two);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, arg, two, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t addi_result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &addi_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  iree_arena_allocator_t seed_arena;
  iree_arena_initialize(&block_pool_, &seed_arena);
  loom_value_fact_table_t seed_facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&seed_facts, &seed_arena, 8));
  IREE_ASSERT_OK(loom_value_fact_table_define(&seed_facts, arg,
                                              loom_value_facts_exact_i64(40)));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(module_, &pass_arena,
                                               &value_facts, &canonicalizer));
  loom_canonicalizer_result_t result;
  loom_canonicalizer_options_t options = {
      /*.max_iterations=*/{},
      /*.seed_facts=*/&seed_facts,
  };
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, func_like_,
                                                 &options, &result));

  EXPECT_TRUE(result.changed);
  EXPECT_TRUE(result.boundary_maybe_changed);
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
  const loom_value_fact_table_t* final_facts =
      loom_canonicalizer_fact_table(&canonicalizer);
  ASSERT_NE(final_facts, nullptr);
  EXPECT_TRUE(loom_value_facts_is_exact(
      loom_value_fact_table_lookup(final_facts, arg)));
  loom_value_facts_t addi_facts =
      loom_value_fact_table_lookup(final_facts, addi_result);
  EXPECT_TRUE(loom_value_facts_is_exact(addi_facts));
  EXPECT_EQ(addi_facts.range_lo, 42);

  loom_canonicalizer_deinitialize(&canonicalizer);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
  iree_arena_deinitialize(&seed_arena);
}

TEST_F(CanonicalizeTest, RegionDriverAcceptsSeedFacts) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_builder_t module_builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &module_builder);
  loom_symbol_ref_t callee = {};
  IREE_ASSERT_OK(add_symbol(IREE_SV("projected"), &callee));
  loom_op_t* split_op = NULL;
  IREE_ASSERT_OK(loom_test_split_func_build(&module_builder, 0, 0, 0, callee,
                                            &i32, 1, LOOM_LOCATION_UNKNOWN,
                                            &split_op));
  loom_func_like_t split_func = loom_func_like_cast(module_, split_op);
  loom_region_t* config = loom_test_split_func_config(split_op);
  loom_block_t* config_entry = loom_region_entry_block(config);
  loom_value_id_t config_arg = loom_block_arg_id(config_entry, 0);

  loom_builder_t config_builder;
  loom_builder_initialize(module_, &module_->arena, config_entry,
                          &config_builder);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&config_builder, config_arg, config_arg,
                                      i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t addi_result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&config_builder, &addi_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  iree_arena_allocator_t seed_arena;
  iree_arena_initialize(&block_pool_, &seed_arena);
  loom_value_fact_table_t seed_facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&seed_facts, &seed_arena,
                                                  module_->values.capacity));
  IREE_ASSERT_OK(loom_value_fact_table_define(&seed_facts, config_arg,
                                              loom_value_facts_exact_i64(5)));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(module_, &pass_arena,
                                               &value_facts, &canonicalizer));
  loom_canonicalizer_result_t result;
  loom_canonicalizer_options_t options = {
      /*.max_iterations=*/{},
      /*.seed_facts=*/&seed_facts,
  };
  IREE_ASSERT_OK(loom_canonicalizer_run_region(
      &canonicalizer, split_func, config, split_op, &options, &result));

  EXPECT_TRUE(result.changed);
  EXPECT_TRUE(result.boundary_maybe_changed);
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 10);
  const loom_value_fact_table_t* final_facts =
      loom_canonicalizer_fact_table(&canonicalizer);
  ASSERT_NE(final_facts, nullptr);
  loom_value_facts_t addi_facts =
      loom_value_fact_table_lookup(final_facts, addi_result);
  EXPECT_TRUE(loom_value_facts_is_exact(addi_facts));
  EXPECT_EQ(addi_facts.range_lo, 10);

  loom_canonicalizer_deinitialize(&canonicalizer);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
  iree_arena_deinitialize(&seed_arena);
}

TEST_F(CanonicalizeTest, DriverResetsScratchArenaBetweenRuns) {
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(module_, &pass_arena,
                                               &value_facts, &canonicalizer));

  loom_canonicalizer_result_t result;
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, func_like_,
                                                 NULL, &result));
  EXPECT_GT(canonicalizer.scratch_arena.used_allocation_size, 0u);
  EXPECT_NE(loom_canonicalizer_fact_table(&canonicalizer), nullptr);

  loom_func_like_t empty_func = {};
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, empty_func,
                                                 NULL, &result));
  EXPECT_EQ(canonicalizer.scratch_arena.used_allocation_size, 0u);
  EXPECT_EQ(loom_canonicalizer_fact_table(&canonicalizer), nullptr);

  loom_canonicalizer_deinitialize(&canonicalizer);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CanonicalizeTest, OpWithoutCanonicalizeUntouched) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // test.neg has no canonicalize callback. It should pass through
  // the worklist without modification.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  loom_op_t* neg1 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg1));
  loom_value_id_t b = loom_test_neg_result(neg1);

  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, b, i32, LOOM_LOCATION_UNKNOWN, &neg2));
  loom_value_id_t c = loom_test_neg_result(neg2);

  // Keep the chain alive so DCE doesn't erase it.
  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &c, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());
  // test.neg has no fold or canonicalize. The chain survives
  // because test.use keeps it alive.
  EXPECT_EQ(constant_value(loom_op_operands(neg1)[0]), 7);
  EXPECT_EQ(loom_op_operands(neg2)[0], b);
}

//===----------------------------------------------------------------------===//
// Nested regions
//===----------------------------------------------------------------------===//

TEST_F(CanonicalizeTest, NestedRegionOpsCanonicalized) {
  // The rewriter's seed_function descends into nested regions so
  // foldable ops inside a test.map body are visited and canonicalized.
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  // Inside the map body: addi %arg, 0 — should fold to %arg.
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);

  loom_op_t* inner_zero = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_zero));
  loom_value_id_t zero = loom_test_constant_result(inner_zero);

  loom_op_t* inner_addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, arg, zero, i32,
                                      LOOM_LOCATION_UNKNOWN, &inner_addi));
  loom_value_id_t addi_result = loom_test_addi_result(inner_addi);

  // Keep addi result alive so the fold is observable.
  loom_op_t* inner_use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &addi_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &inner_use));

  loom_builder_restore(&builder_, saved);

  IREE_ASSERT_OK(run_canonicalize());

  // The inner addi(arg, 0) folded — use now references %arg directly.
  EXPECT_EQ(loom_op_operands(inner_use)[0], arg);
}

//===----------------------------------------------------------------------===//
// test.counter: multi-step, error, fixed point
//===----------------------------------------------------------------------===//

TEST_F(CanonicalizeTest, CounterFixedPoint) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // counter(0) — already at fixed point, no change.
  loom_op_t* counter = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, 0, i32,
                                         LOOM_LOCATION_UNKNOWN, &counter));
  loom_value_id_t result = loom_test_counter_result(counter);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  // Counter untouched. neg still uses the original result.
  EXPECT_EQ(loom_op_operands(use)[0], result);
  EXPECT_FALSE(counter->flags & LOOM_OP_FLAG_DEAD);
}

TEST_F(CanonicalizeTest, CounterSingleStep) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // counter(1) → counter(0). One step, then fixed point.
  loom_op_t* counter = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, 1, i32,
                                         LOOM_LOCATION_UNKNOWN, &counter));
  loom_value_id_t original_result = loom_test_counter_result(counter);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  // The original counter(1) was erased and replaced by counter(0).
  EXPECT_TRUE(counter->flags & LOOM_OP_FLAG_DEAD);
  // neg now uses the replacement counter(0)'s result.
  loom_value_id_t new_result = loom_op_operands(use)[0];
  EXPECT_NE(new_result, original_result);
  loom_value_t* value = loom_module_value(module_, new_result);
  loom_op_t* new_counter = loom_value_def_op(value);
  ASSERT_NE(new_counter, nullptr);
  EXPECT_EQ(loom_test_counter_value(new_counter), 0);
}

TEST_F(CanonicalizeTest, CounterMultiStep) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // counter(5) → counter(4) → counter(3) → counter(2) → counter(1) →
  // counter(0). Five steps of decrement before reaching fixed point.
  loom_op_t* counter = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, 5, i32,
                                         LOOM_LOCATION_UNKNOWN, &counter));
  loom_value_id_t original_result = loom_test_counter_result(counter);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize());

  // The final counter should have value 0.
  loom_value_id_t final_result = loom_op_operands(use)[0];
  loom_value_t* value = loom_module_value(module_, final_result);
  loom_op_t* final_counter = loom_value_def_op(value);
  ASSERT_NE(final_counter, nullptr);
  EXPECT_EQ(loom_test_counter_value(final_counter), 0);
}

TEST_F(CanonicalizeTest, CounterErrorPropagation) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // counter(-1) — canonicalize returns IREE_STATUS_INTERNAL.
  loom_op_t* counter = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, -1, i32,
                                         LOOM_LOCATION_UNKNOWN, &counter));
  loom_value_id_t result = loom_test_counter_result(counter);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  // The error should propagate through canonicalize.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, run_canonicalize());
}

TEST_F(CanonicalizeTest, CounterErrorMidWorklist) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Build a healthy op, then a failing op. The healthy op may be
  // processed first, but the failing op causes the pass to abort.
  loom_op_t* healthy = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, 1, i32,
                                         LOOM_LOCATION_UNKNOWN, &healthy));
  loom_value_id_t healthy_result = loom_test_counter_result(healthy);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &healthy_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  loom_op_t* failing = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, -1, i32,
                                         LOOM_LOCATION_UNKNOWN, &failing));
  loom_value_id_t fail_result = loom_test_counter_result(failing);

  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &fail_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &neg2));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, run_canonicalize());
}

}  // namespace
}  // namespace loom
