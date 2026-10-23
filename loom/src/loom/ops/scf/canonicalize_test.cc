// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class ScfCanonicalizeTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* tables = loom_scf_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCF,
                                                 tables, (uint16_t)count));
    tables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 tables, (uint16_t)count));
    if (GetParam()) {
      tables = loom_scalar_dialect_vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(
          &context_, LOOM_DIALECT_SCALAR, tables, (uint16_t)count));
    }
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("canonicalize"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&facts_, &module_->arena, 0));
    loom_rewriter_initialize(&rewriter_, module_, &module_->arena);
    loom_rewriter_attach_value_facts(&rewriter_, &facts_);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t Constant(bool value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder_, loom_attr_bool(value), loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        LOOM_LOCATION_UNKNOWN, &op));
    IREE_CHECK_OK(loom_value_fact_table_compute_op(&facts_, module_, op));
    return loom_test_constant_result(op);
  }

  // Shared backing storage for the module and rewrite facts.
  iree_arena_block_pool_t pool_;
  // Immutable dialect vocabulary, optionally including scalar arithmetic.
  loom_context_t context_;
  // Module owning the API-built operations.
  loom_module_t* module_ = nullptr;
  // Builder positioned at the module's input block.
  loom_builder_t builder_ = {};
  // Facts computed by the registered producer callbacks.
  loom_value_fact_table_t facts_ = {};
  // Mutation owner borrowing the producer facts.
  loom_rewriter_t rewriter_ = {};
};

TEST_P(ScfCanonicalizeTest, ComplementUsesRegisteredScalarVocabulary) {
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), type, &condition));
  const auto false_value = Constant(false);
  const auto true_value = Constant(true);
  loom_op_t* select = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, false_value,
                                       true_value, type, LOOM_LOCATION_UNKNOWN,
                                       &select));

  IREE_ASSERT_OK(loom_scf_select_canonicalize(select, &rewriter_));
  auto* last = loom_module_block(module_)->last_op;
  if (GetParam()) {
    ASSERT_TRUE(loom_scalar_xori_isa(last));
    EXPECT_EQ(loom_scalar_xori_lhs(last), condition);
    EXPECT_EQ(loom_scalar_xori_rhs(last), true_value);
    EXPECT_TRUE(select->flags & LOOM_OP_FLAG_DEAD);
    EXPECT_EQ(rewriter_.created_op_count, 1u);
    EXPECT_EQ(rewriter_.erased_op_count, 1u);
  } else {
    EXPECT_EQ(last, select);
    EXPECT_FALSE(select->flags & LOOM_OP_FLAG_DEAD);
    EXPECT_EQ(rewriter_.created_op_count, 0u);
    EXPECT_EQ(rewriter_.erased_op_count, 0u);
  }
}

TEST_P(ScfCanonicalizeTest, EmptyThenInversionPreservesGuardedEffects) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), &condition));
  loom_op_t* conditional = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &conditional));
  auto saved = loom_builder_enter_region(&builder_, conditional,
                                         loom_scf_if_then_region(conditional));
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);
  saved = loom_builder_enter_region(&builder_, conditional,
                                    loom_scf_if_else_region(conditional));
  loom_op_t* effect = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &condition, 1,
                                     LOOM_LOCATION_UNKNOWN, &effect));
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);
  rewriter_.constant_materializer = {
      .supports =
          [](loom_value_facts_t facts, loom_type_t type) {
            int64_t value = 0;
            return loom_type_is_scalar(type) &&
                   loom_value_facts_as_exact_i64(facts, &value);
          },
      .materialize =
          [](loom_builder_t* builder, loom_value_facts_t facts,
             loom_type_t type, loom_location_id_t location,
             loom_value_id_t* out_value) {
            loom_op_t* constant = nullptr;
            IREE_RETURN_IF_ERROR(
                loom_test_constant_build(builder, loom_attr_i64(facts.range_lo),
                                         type, location, &constant));
            *out_value = loom_test_constant_result(constant);
            return iree_ok_status();
          },
  };

  IREE_ASSERT_OK(loom_scf_if_canonicalize(conditional, &rewriter_));
  auto* last = loom_module_block(module_)->last_op;
  ASSERT_TRUE(loom_scf_if_isa(last));
  if (GetParam()) {
    EXPECT_TRUE(conditional->flags & LOOM_OP_FLAG_DEAD);
    EXPECT_EQ(loom_scf_if_else_region(last), nullptr);
    EXPECT_EQ(loom_region_entry_block(loom_scf_if_then_region(last))->first_op,
              effect);
    auto* inverse = loom_value_def_op(
        loom_module_value(module_, loom_scf_if_condition(last)));
    ASSERT_TRUE(loom_scalar_xori_isa(inverse));
    EXPECT_EQ(loom_scalar_xori_lhs(inverse), condition);
  } else {
    EXPECT_EQ(last, conditional);
    EXPECT_FALSE(conditional->flags & LOOM_OP_FLAG_DEAD);
    EXPECT_EQ(loom_region_entry_block(loom_scf_if_else_region(last))->first_op,
              effect);
    EXPECT_EQ(rewriter_.created_op_count, 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(ScalarRegistration, ScfCanonicalizeTest,
                         ::testing::Bool());

}  // namespace
}  // namespace loom
