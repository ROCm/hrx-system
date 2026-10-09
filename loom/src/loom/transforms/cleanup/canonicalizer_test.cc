// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/canonicalizer.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"
#include "loom/transforms/cleanup/configured.h"

namespace loom {
namespace {

static const loom_cleanup_special_value_policy_t kSpecialValuePolicy = {
    .type_has_poison_materializer = loom_type_has_poison_materializer,
    .materialize_poison = loom_poison_build,
    .op_is_empty = loom_op_is_empty,
    .type_has_empty_materializer = loom_type_has_empty_materializer,
    .materialize_empty = loom_empty_build,
    .materialize_constant = loom_constant_build,
};

class CanonicalizerTest : public ::testing::Test {
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
    vtables = loom_index_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, (uint16_t)vtable_count));
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
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
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

  iree_status_t run_canonicalize(
      loom_func_like_t function,
      const loom_canonicalizer_options_t* options = nullptr,
      const loom_cleanup_special_value_policy_t* special_value_policy =
          &kSpecialValuePolicy,
      const loom_fact_refinement_policy_t* fact_refinement_policy = nullptr) {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    loom_canonicalizer_t canonicalizer;
    iree_status_t status = loom_canonicalizer_initialize(
        module_, &pass_arena, &value_facts, special_value_policy,
        fact_refinement_policy, &canonicalizer);
    if (iree_status_is_ok(status)) {
      status = loom_canonicalizer_run_function(&canonicalizer, function,
                                               options, nullptr);
    }
    loom_canonicalizer_deinitialize(&canonicalizer);
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
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }

  // Returns the integer constant value of the op defining |value_id|.
  // Assumes the defining op is a constant-like op with an i64 attr.
  int64_t constant_value(loom_value_id_t value_id) {
    loom_value_t* value = loom_module_value(module_, value_id);
    loom_op_t* def = loom_value_def_op(value);
    if (!def) {
      return INT64_MIN;
    }
    return loom_attr_as_i64(loom_test_constant_value(def));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

typedef struct CanonicalizerPhasePatternState {
  // Number of existing roots visited during ordered region initialization.
  uint32_t initialization_visit_count;
  // Root inserted by the fixed-point pattern after initialization.
  loom_op_t* inserted_op;
  // True when the fixed-point registry processes inserted_op.
  bool processed_inserted_op;
} CanonicalizerPhasePatternState;

typedef struct CanonicalizerPhasePatternData {
  // Mutable invocation state observed through this immutable pattern data.
  CanonicalizerPhasePatternState* state;
} CanonicalizerPhasePatternData;

static iree_status_t CountRegionInitializationPattern(
    const loom_rewrite_pattern_t* pattern, void*, loom_op_t*, loom_rewriter_t*,
    bool* out_changed) {
  const CanonicalizerPhasePatternData* data =
      (const CanonicalizerPhasePatternData*)pattern->user_data;
  CanonicalizerPhasePatternState* state = data->state;
  ++state->initialization_visit_count;
  *out_changed = false;
  return iree_ok_status();
}

static iree_status_t InsertAndObserveFixedPointRootPattern(
    const loom_rewrite_pattern_t* pattern, void*, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  const CanonicalizerPhasePatternData* data =
      (const CanonicalizerPhasePatternData*)pattern->user_data;
  CanonicalizerPhasePatternState* state = data->state;
  *out_changed = false;
  if (op == state->inserted_op) {
    state->processed_inserted_op = true;
    return iree_ok_status();
  }
  if (state->inserted_op != nullptr) {
    return iree_ok_status();
  }

  loom_builder_set_after(&rewriter->builder, op);
  IREE_RETURN_IF_ERROR(loom_test_use_build(
      &rewriter->builder, loom_op_operands(op), op->operand_count, op->location,
      &state->inserted_op));
  *out_changed = true;
  return iree_ok_status();
}

static void InitializeSinglePatternRegistry(
    const loom_rewrite_pattern_t* pattern,
    loom_rewrite_pattern_registry_storage_t* out_storage) {
  const loom_rewrite_pattern_provider_t provider = {
      IREE_SVL("canonicalizer-test"), pattern, 1};
  const loom_rewrite_pattern_provider_t* providers[] = {&provider};
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_storage_initialize(
      loom_rewrite_pattern_provider_list_make(providers,
                                              IREE_ARRAYSIZE(providers)),
      iree_allocator_system(), out_storage));
}

TEST_F(CanonicalizerTest, InitializesRegionOnceAndMaintainsNewRoots) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN, &constant));
  loom_value_id_t value = loom_test_constant_result(constant);
  loom_op_t* original_use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &value, 1,
                                     LOOM_LOCATION_UNKNOWN, &original_use));

  CanonicalizerPhasePatternState state = {};
  const CanonicalizerPhasePatternData pattern_data = {&state};
  const loom_rewrite_pattern_t initialization_pattern = {
      LOOM_OP_TEST_USE, CountRegionInitializationPattern, &pattern_data};
  loom_rewrite_pattern_registry_storage_t initialization_storage = {};
  InitializeSinglePatternRegistry(&initialization_pattern,
                                  &initialization_storage);

  const loom_rewrite_pattern_t pre_fold_pattern = {
      LOOM_OP_TEST_USE, InsertAndObserveFixedPointRootPattern, &pattern_data};
  loom_rewrite_pattern_registry_storage_t pre_fold_storage = {};
  InitializeSinglePatternRegistry(&pre_fold_pattern, &pre_fold_storage);

  const loom_canonicalizer_options_t options = {
      .max_iterations = 0,
      .patterns =
          {
              .region_initialization =
                  loom_rewrite_pattern_registry_storage_registry(
                      &initialization_storage),
              .pre_fold = loom_rewrite_pattern_registry_storage_registry(
                  &pre_fold_storage),
              .post_type = nullptr,
              .post_canonicalization = nullptr,
          },
      .target_facts = nullptr,
      .math_policy = nullptr,
      .seed_facts = {},
      .refine_boundary = {},
  };
  IREE_EXPECT_OK(run_canonicalize(func_like_, &options));
  EXPECT_EQ(state.initialization_visit_count, 1u);
  EXPECT_NE(state.inserted_op, nullptr);
  EXPECT_TRUE(state.processed_inserted_op);

  loom_rewrite_pattern_registry_storage_deinitialize(&pre_fold_storage);
  loom_rewrite_pattern_registry_storage_deinitialize(&initialization_storage);
}

TEST_F(CanonicalizerTest, AddiZeroRight) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // addi folded to a constant with value 42.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizerTest, AddiZeroLeft) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizerTest, ConstantFoldAddi) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));
  // addi(1, 2) folded to constant 3.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 3);
}

TEST_F(CanonicalizerTest, ChainedFolds) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // Both addi ops fold to constant 42. neg uses the folded result.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizerTest, MultipleUsersRevisited) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // All addi ops fold: a=5, b=5, c=10. neg_b uses 5, neg_c uses 10.
  EXPECT_EQ(constant_value(loom_op_operands(use_b)[0]), 5);
  EXPECT_EQ(constant_value(loom_op_operands(use_c)[0]), 10);
}

TEST_F(CanonicalizerTest, NewOpsAddedToWorklist) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));
  // neg has no fold or canonicalize. constant has fold but is already a
  // constant — nothing to fold to. Both survive.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 7);
}

TEST_F(CanonicalizerTest, FactRefinementChainUsesBoundedNames) {
  constexpr uint16_t kRelationCount = 256;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), index_type, &upper));
  loom_string_id_t upper_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("upper"), &upper_name));
  IREE_ASSERT_OK(loom_module_set_value_name(module_, upper, upper_name));
  for (uint16_t i = 0; i < kRelationCount; ++i) {
    loom_op_t* constant_op = nullptr;
    IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(i),
                                             index_type, LOOM_LOCATION_UNKNOWN,
                                             &constant_op));
    const loom_value_id_t exact = loom_index_constant_result(constant_op);
    const loom_predicate_t predicate = {
        .kind = LOOM_PREDICATE_LT,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .reserved = {},
        .args = {exact, upper, 0},
    };
    loom_op_t* assume_op = nullptr;
    IREE_ASSERT_OK(loom_index_assume_build(&builder_, &exact, 1, &predicate, 1,
                                           &index_type, 1,
                                           LOOM_LOCATION_UNKNOWN, &assume_op));
  }
  loom_op_t* use_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &upper, 1,
                                     LOOM_LOCATION_UNKNOWN, &use_op));

  const loom_fact_refinement_policy_t* refinement_policy =
      loom_cleanup_configured_pattern_provider_set()->fact_refinement_policy;
  IREE_ASSERT_OK(run_canonicalize(func_like_, nullptr, &kSpecialValuePolicy,
                                  refinement_policy));

  loom_value_id_t current = loom_op_const_operands(use_op)[0];
  for (uint16_t i = 0; i < kRelationCount; ++i) {
    loom_op_t* defining_op =
        loom_value_def_op(loom_module_value(module_, current));
    ASSERT_NE(defining_op, nullptr);
    ASSERT_TRUE(loom_index_assume_isa(defining_op));
    EXPECT_TRUE(iree_string_view_equal(loom_module_value_name(module_, current),
                                       IREE_SV("upper_refined")));
    const loom_attribute_t predicates =
        loom_index_assume_predicates(defining_op);
    ASSERT_EQ(predicates.count, 1);
    EXPECT_EQ(predicates.predicate_list[0].kind, LOOM_PREDICATE_GT);
    EXPECT_EQ(predicates.predicate_list[0].args[1], kRelationCount - i - 1);
    current = loom_index_assume_values(defining_op).values[0];
  }
  EXPECT_EQ(current, upper);
}

TEST_F(CanonicalizerTest, FactRefinementBatchesVariadicRelations) {
  constexpr uint16_t kRelationCount = 256;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t exact_values[kRelationCount];
  loom_value_id_t upper_values[kRelationCount];
  loom_type_t result_types[kRelationCount];
  loom_predicate_t predicates[kRelationCount] = {};

  loom_block_t* entry_block = loom_region_entry_block(body_);
  for (uint16_t i = 0; i < kRelationCount; ++i) {
    IREE_ASSERT_OK(loom_builder_define_block_arg(&builder_, entry_block,
                                                 index_type, &upper_values[i]));
  }
  for (uint16_t i = 0; i < kRelationCount; ++i) {
    loom_op_t* constant_op = nullptr;
    IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(i),
                                             index_type, LOOM_LOCATION_UNKNOWN,
                                             &constant_op));
    exact_values[i] = loom_index_constant_result(constant_op);
    result_types[i] = index_type;
    predicates[i] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_LT,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .reserved = {},
        .args = {exact_values[i], upper_values[i], 0},
    };
  }
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder_, exact_values, kRelationCount, predicates, kRelationCount,
      result_types, kRelationCount, LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_op_t* use_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, upper_values, kRelationCount,
                                     LOOM_LOCATION_UNKNOWN, &use_op));

  const loom_fact_refinement_policy_t* refinement_policy =
      loom_cleanup_configured_pattern_provider_set()->fact_refinement_policy;
  IREE_ASSERT_OK(run_canonicalize(func_like_, nullptr, &kSpecialValuePolicy,
                                  refinement_policy));

  for (uint16_t i = 0; i < kRelationCount; ++i) {
    const loom_value_id_t refined = loom_op_const_operands(use_op)[i];
    loom_op_t* defining_op =
        loom_value_def_op(loom_module_value(module_, refined));
    ASSERT_NE(defining_op, nullptr);
    ASSERT_TRUE(loom_index_assume_isa(defining_op));
    EXPECT_EQ(loom_op_const_operands(defining_op)[0], upper_values[i]);
    const loom_attribute_t refined_predicates =
        loom_index_assume_predicates(defining_op);
    ASSERT_EQ(refined_predicates.count, 1);
    EXPECT_EQ(refined_predicates.predicate_list[0].kind, LOOM_PREDICATE_GT);
    EXPECT_EQ(refined_predicates.predicate_list[0].arg_tags[1],
              LOOM_PRED_ARG_CONST);
    EXPECT_EQ(refined_predicates.predicate_list[0].args[1], i);
  }
}

TEST_F(CanonicalizerTest, EmptyFunctionNoOps) {
  // Empty function body — canonicalize should succeed with nothing to do.
  IREE_ASSERT_OK(run_canonicalize(func_like_));
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(CanonicalizerTest, WorklistDedupPreventsRedundantWork) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // All three addi ops fold: a=42, b=42, c=84. neg uses 84.
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 84);
}

TEST_F(CanonicalizerTest, ScalarPoisonPropagatesThroughPureOp) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  loom_value_id_t observed = loom_op_operands(use)[0];
  loom_op_t* def = loom_value_def_op(loom_module_value(module_, observed));
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(loom_scalar_poison_isa(def));
  EXPECT_NE(observed, result);
}

TEST_F(CanonicalizerTest, NullSpecialValuePolicyDisablesPoisonPropagation) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_, nullptr, nullptr));

  EXPECT_EQ(loom_op_operands(use)[0], result);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, result)), addi);
}

TEST_F(CanonicalizerTest, NullSpecialValuePolicyDisablesEmptyMaterialization) {
  loom_type_t v0f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(0), 0);

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), v0f32, &lhs));
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), v0f32, &rhs));

  loom_op_t* addf = NULL;
  IREE_ASSERT_OK(loom_vector_addf_build(&builder_, /*instance_flags=*/0, lhs,
                                        rhs, v0f32, LOOM_LOCATION_UNKNOWN,
                                        &addf));
  loom_value_id_t result = loom_vector_addf_result(addf);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize(func_like_, nullptr, nullptr));

  EXPECT_EQ(loom_op_operands(use)[0], result);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, result)), addf);
}

TEST_F(CanonicalizerTest,
       NullSpecialValuePolicyDisablesConstantMaterialization) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* one_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &one_op));
  loom_op_t* two_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &two_op));

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(
      &builder_, loom_test_constant_result(one_op),
      loom_test_constant_result(two_op), i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result = loom_test_addi_result(addi);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize(func_like_, nullptr, nullptr));

  EXPECT_EQ(loom_op_operands(use)[0], result);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, result)), addi);
}

TEST_F(CanonicalizerTest, VectorPoisonPropagatesThroughPureOp) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  loom_value_id_t observed = loom_op_operands(use)[0];
  loom_op_t* def = loom_value_def_op(loom_module_value(module_, observed));
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(loom_vector_poison_isa(def));
  EXPECT_NE(observed, result);
}

TEST_F(CanonicalizerTest, PoisonBeatsFmaiZeroMultiplier) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  loom_value_id_t observed = loom_op_operands(use)[0];
  loom_op_t* def = loom_value_def_op(loom_module_value(module_, observed));
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(loom_scalar_poison_isa(def));
  EXPECT_NE(observed, c);
}

//===----------------------------------------------------------------------===//
// Fixed point and edge cases
//===----------------------------------------------------------------------===//

TEST_F(CanonicalizerTest, NullFunctionBody) {
  // The empty function sentinel has no regions or worklist to process.
  IREE_EXPECT_OK(run_canonicalize({}));
}

TEST_F(CanonicalizerTest, FixedPointConvergence) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);

  // Second run — nothing should change. The fold already happened.
  int ops_before = count_live_ops();
  IREE_ASSERT_OK(run_canonicalize(func_like_));
  EXPECT_EQ(count_live_ops(), ops_before);
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
}

TEST_F(CanonicalizerTest, DriverReusesFactsAcrossNoopRuns) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constant = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &constant));
  loom_value_id_t value = loom_test_constant_result(constant);
  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &value, 1, LOOM_LOCATION_UNKNOWN, &use));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_pass_value_fact_lifecycle_counts_t counts = {};
  value_facts.lifecycle_counts = &counts;
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(
      module_, &pass_arena, &value_facts, &kSpecialValuePolicy,
      /*fact_refinement_policy=*/nullptr, &canonicalizer));

  loom_canonicalizer_result_t result;
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, func_like_,
                                                 NULL, &result));
  EXPECT_FALSE(result.changed);
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, func_like_,
                                                 NULL, &result));
  EXPECT_FALSE(result.changed);

  EXPECT_EQ(counts.acquisition_count, 2u);
  EXPECT_EQ(counts.cache_hit_count, 1u);
  EXPECT_EQ(counts.recomputation_count, 1u);
  EXPECT_EQ(counts.preparation_count, 0u);
  EXPECT_EQ(counts.invalidation_count, 0u);
  EXPECT_EQ(counts.scope_clear_count, 0u);
  EXPECT_GT(counts.computed_value_count, 0u);
  EXPECT_EQ(value_facts.active_scope.kind, LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION);

  loom_canonicalizer_deinitialize(&canonicalizer);
  EXPECT_EQ(value_facts.active_scope.kind, LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION);
  value_facts.lifecycle_counts = nullptr;
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CanonicalizerTest, DriverFactsMatchFreshRecomputationAfterChanges) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_forty = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(40), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_forty));
  loom_value_id_t forty = loom_test_constant_result(const_forty);
  loom_op_t* keep_forty = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &forty, 1,
                                     LOOM_LOCATION_UNKNOWN, &keep_forty));

  loom_op_t* const_two = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_two));
  loom_value_id_t two = loom_test_constant_result(const_two);
  loom_op_t* keep_two = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &two, 1, LOOM_LOCATION_UNKNOWN,
                                     &keep_two));

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, forty, two, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t sum = loom_test_addi_result(addi);
  loom_op_t* use_sum = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &sum, 1, LOOM_LOCATION_UNKNOWN, &use_sum));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(
      module_, &pass_arena, &value_facts, &kSpecialValuePolicy,
      /*fact_refinement_policy=*/nullptr, &canonicalizer));
  loom_canonicalizer_result_t result;
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, func_like_,
                                                 NULL, &result));
  EXPECT_TRUE(result.changed);

  const loom_value_fact_table_t* incremental_facts =
      loom_canonicalizer_fact_table(&canonicalizer);
  ASSERT_NE(incremental_facts, nullptr);
  loom_value_id_t folded_sum = loom_op_operands(use_sum)[0];
  EXPECT_NE(folded_sum, sum);
  loom_value_facts_t incremental_sum_facts =
      loom_value_fact_table_lookup(incremental_facts, folded_sum);
  EXPECT_TRUE(loom_value_facts_is_exact(incremental_sum_facts));
  EXPECT_EQ(incremental_sum_facts.range_lo, 42);

  loom_pass_value_fact_owner_t fresh_owner = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
  loom_value_fact_table_t* fresh_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fresh_owner, module_, loom_pass_value_fact_scope_function(func_like_),
      &fresh_facts));
  ASSERT_NE(fresh_facts, nullptr);
  const loom_value_id_t live_values[] = {forty, two, folded_sum};
  for (loom_value_id_t live_value : live_values) {
    loom_type_t type = loom_module_value_type(module_, live_value);
    EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
        module_, type, incremental_facts,
        loom_value_fact_table_lookup(incremental_facts, live_value),
        fresh_facts, loom_value_fact_table_lookup(fresh_facts, live_value)));
  }

  loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  loom_canonicalizer_deinitialize(&canonicalizer);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CanonicalizerTest, DriverAcceptsSeedFacts) {
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

  loom_builder_t module_builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &module_builder);
  loom_symbol_ref_t other_callee = {};
  IREE_ASSERT_OK(add_symbol(IREE_SV("other"), &other_callee));
  loom_op_t* other_op = nullptr;
  IREE_ASSERT_OK(loom_test_split_func_build(&module_builder, 0, 0, 0,
                                            other_callee, &i32, 1,
                                            LOOM_LOCATION_UNKNOWN, &other_op));
  const loom_value_id_t other_arg = loom_block_arg_id(
      loom_region_entry_block(loom_test_split_func_config(other_op)), 0);

  iree_arena_allocator_t seed_arena;
  iree_arena_initialize(&block_pool_, &seed_arena);
  loom_value_fact_table_t seed_facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&seed_facts, &seed_arena, 8));
  IREE_ASSERT_OK(loom_value_fact_table_define(&seed_facts, arg,
                                              loom_value_facts_exact_i64(40)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&seed_facts, other_arg,
                                              loom_value_facts_exact_i64(99)));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(
      module_, &pass_arena, &value_facts, &kSpecialValuePolicy,
      /*fact_refinement_policy=*/nullptr, &canonicalizer));
  loom_canonicalizer_result_t result;
  loom_canonicalizer_options_t options = {.seed_facts = {&seed_facts, &arg, 1}};
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, func_like_,
                                                 &options, &result));

  EXPECT_TRUE(result.changed);
  EXPECT_TRUE(result.boundary_maybe_changed);
  EXPECT_EQ(constant_value(loom_op_operands(use)[0]), 42);
  const loom_value_fact_table_t* final_facts =
      loom_canonicalizer_fact_table(&canonicalizer);
  ASSERT_NE(final_facts, nullptr);
  EXPECT_FALSE(loom_value_fact_table_has_entry(final_facts, other_arg));
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

TEST_F(CanonicalizerTest, DriverPreservesExplicitTargetFactsAcrossSideRegions) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_builder_t module_builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &module_builder);
  loom_symbol_ref_t callee = {};
  IREE_ASSERT_OK(add_symbol(IREE_SV("target_scoped"), &callee));
  loom_op_t* split_op = NULL;
  IREE_ASSERT_OK(loom_test_split_func_build(&module_builder, 0, 0, 0, callee,
                                            &i32, 1, LOOM_LOCATION_UNKNOWN,
                                            &split_op));
  loom_func_like_t split_func = loom_func_like_cast(module_, split_op);

  loom_target_snapshot_t snapshot = {.name = IREE_SVL("target-context-test")};
  loom_target_export_plan_t export_plan = {.name =
                                               IREE_SVL("target-context-test")};
  loom_target_config_t config = {.name = IREE_SVL("target-context-test")};
  loom_target_bundle_t bundle = {
      .name = IREE_SVL("target-context-test"),
      .snapshot = &snapshot,
      .export_plan = &export_plan,
      .config = &config,
  };
  const loom_target_fact_type_t target_fact_type = {
      .name = IREE_SVL("test"),
      .storage_size = sizeof(loom_target_facts_t),
  };
  loom_target_facts_t target_facts = {
      .fact_type = &target_fact_type,
      .selector = 0,
      .explicit_fields = 0,
      .storage =
          {
              .snapshot = snapshot,
              .export_plan = export_plan,
              .config = config,
              .bundle = bundle,
          },
  };
  loom_target_bundle_storage_rebind(&target_facts.storage);

  iree_arena_allocator_t seed_arena;
  iree_arena_initialize(&block_pool_, &seed_arena);
  loom_value_fact_table_t seed_facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &seed_facts, &seed_arena, loom_value_table_capacity(&module_->values)));
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_pass_value_fact_lifecycle_counts_t counts = {};
  value_facts.lifecycle_counts = &counts;
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(
      module_, &pass_arena, &value_facts, &kSpecialValuePolicy,
      /*fact_refinement_policy=*/nullptr, &canonicalizer));
  loom_canonicalizer_result_t result;
  loom_canonicalizer_options_t options = {
      .target_facts = &target_facts, .seed_facts = {&seed_facts, nullptr, 0}};
  IREE_ASSERT_OK(loom_canonicalizer_run_function(&canonicalizer, split_func,
                                                 &options, &result));

  const loom_value_fact_table_t* final_facts =
      loom_canonicalizer_fact_table(&canonicalizer);
  ASSERT_NE(final_facts, nullptr);
  EXPECT_EQ(final_facts->context.target_facts, &target_facts);
  EXPECT_EQ(counts.acquisition_count, 0u);
  EXPECT_EQ(counts.recomputation_count, 0u);
  EXPECT_EQ(counts.preparation_count, 1u);
  EXPECT_EQ(counts.invalidation_count, 0u);
  EXPECT_EQ(counts.scope_clear_count, 0u);

  loom_canonicalizer_deinitialize(&canonicalizer);
  value_facts.lifecycle_counts = nullptr;
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
  iree_arena_deinitialize(&seed_arena);
}

TEST_F(CanonicalizerTest, RegionDriverAcceptsSeedFacts) {
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
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &seed_facts, &seed_arena, loom_value_table_capacity(&module_->values)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&seed_facts, config_arg,
                                              loom_value_facts_exact_i64(5)));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(
      module_, &pass_arena, &value_facts, &kSpecialValuePolicy,
      /*fact_refinement_policy=*/nullptr, &canonicalizer));
  loom_canonicalizer_result_t result;
  loom_canonicalizer_options_t options = {
      .seed_facts = {&seed_facts, &config_arg, 1}};
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

TEST_F(CanonicalizerTest, DriverResetsScratchArenaBetweenRuns) {
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_ASSERT_OK(loom_canonicalizer_initialize(
      module_, &pass_arena, &value_facts, &kSpecialValuePolicy,
      /*fact_refinement_policy=*/nullptr, &canonicalizer));

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

TEST_F(CanonicalizerTest, OpWithoutCanonicalizeUntouched) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));
  // test.neg has no fold or canonicalize. The chain survives
  // because test.use keeps it alive.
  EXPECT_EQ(constant_value(loom_op_operands(neg1)[0]), 7);
  EXPECT_EQ(loom_op_operands(neg2)[0], b);
}

//===----------------------------------------------------------------------===//
// Nested regions
//===----------------------------------------------------------------------===//

TEST_F(CanonicalizerTest, NestedRegionOpsCanonicalized) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // The inner addi(arg, 0) folded — use now references %arg directly.
  EXPECT_EQ(loom_op_operands(inner_use)[0], arg);
}

//===----------------------------------------------------------------------===//
// test.counter: multi-step, error, fixed point
//===----------------------------------------------------------------------===//

TEST_F(CanonicalizerTest, CounterFixedPoint) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // counter(0) — already at fixed point, no change.
  loom_op_t* counter = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, 0, i32,
                                         LOOM_LOCATION_UNKNOWN, &counter));
  loom_value_id_t result = loom_test_counter_result(counter);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // Counter untouched. neg still uses the original result.
  EXPECT_EQ(loom_op_operands(use)[0], result);
  EXPECT_FALSE(counter->flags & LOOM_OP_FLAG_DEAD);
}

TEST_F(CanonicalizerTest, CounterSingleStep) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // counter(1) → counter(0). One step, then fixed point.
  loom_op_t* counter = NULL;
  IREE_ASSERT_OK(loom_test_counter_build(&builder_, 1, i32,
                                         LOOM_LOCATION_UNKNOWN, &counter));
  loom_value_id_t original_result = loom_test_counter_result(counter);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  IREE_ASSERT_OK(run_canonicalize(func_like_));

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

TEST_F(CanonicalizerTest, CounterMultiStep) {
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

  IREE_ASSERT_OK(run_canonicalize(func_like_));

  // The final counter should have value 0.
  loom_value_id_t final_result = loom_op_operands(use)[0];
  loom_value_t* value = loom_module_value(module_, final_result);
  loom_op_t* final_counter = loom_value_def_op(value);
  ASSERT_NE(final_counter, nullptr);
  EXPECT_EQ(loom_test_counter_value(final_counter), 0);
}

TEST_F(CanonicalizerTest, CounterErrorPropagation) {
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, run_canonicalize(func_like_));
}

TEST_F(CanonicalizerTest, CounterErrorMidWorklist) {
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

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, run_canonicalize(func_like_));
}

}  // namespace
}  // namespace loom
