// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_SYMBOLIC_EXPR_TEST_FIXTURE_H_
#define LOOM_ANALYSIS_SYMBOLIC_EXPR_TEST_FIXTURE_H_

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/fact_table.h"

namespace loom {

// Keeps borrowed facts and cached expressions within their active test scope,
// including when a fatal assertion returns from the test early.
class ScopedConditionFacts {
 public:
  ScopedConditionFacts(loom_symbolic_expr_context_t* context,
                       const loom_condition_fact_set_t* facts)
      : context_(*context), previous_scope_(context->condition_scope) {
    derivation_.integer_facts = *facts;
    loom_condition_fact_scope_initialize_local(previous_scope_, &derivation_,
                                               &scope_);
    context_.condition_scope = &scope_;
    loom_symbolic_expr_context_reset(&context_);
  }

  ~ScopedConditionFacts() {
    context_.condition_scope = previous_scope_;
    loom_symbolic_expr_context_reset(&context_);
  }

  ScopedConditionFacts(const ScopedConditionFacts&) = delete;
  ScopedConditionFacts& operator=(const ScopedConditionFacts&) = delete;

 private:
  // Fixture context borrowing the active facts.
  loom_symbolic_expr_context_t& context_;
  // Local derivation adapting the test's flat fact storage.
  loom_condition_derivation_t derivation_ = {};
  // Local scope installed while the test facts are active.
  loom_condition_fact_scope_t scope_ = {};
  // Enclosing scope restored before the local facts leave their lifetime.
  const loom_condition_fact_scope_t* previous_scope_;
};

inline loom_condition_integer_relation_t ValueRelation(
    loom_symbolic_integer_relation_t relation, loom_value_id_t left,
    loom_value_id_t right) {
  loom_condition_integer_relation_t result = {.relation = relation};
  result.left.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
  result.left.value_id = left;
  result.right.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
  result.right.value_id = right;
  return result;
}

inline bool EvaluateIntegerRelation(loom_symbolic_integer_relation_t relation,
                                    int64_t left, int64_t right) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return left == right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      return left != right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return left < right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return left <= right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return left > right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return left >= right;
  }
  return false;
}

inline loom_condition_fact_set_t SingleRelationFacts(
    loom_condition_integer_relation_t* relation) {
  loom_condition_fact_set_t facts = {
      .integer_relations = relation,
      .integer_relation_count = 1,
      .integer_relation_capacity = 1,
  };
  return facts;
}

inline loom_predicate_t ValuePredicate(loom_predicate_kind_t kind,
                                       loom_value_id_t left,
                                       loom_value_id_t right) {
  loom_predicate_t predicate = {.kind = kind, .arg_count = 2};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = left;
  predicate.args[1] = right;
  return predicate;
}

class SymbolicExprTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t index_vtable_count = 0;
    const loom_op_vtable_t* const* index_vtables =
        loom_index_dialect_vtables(&index_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 index_vtables,
                                                 (uint16_t)index_vtable_count));
    iree_host_size_t scalar_vtable_count = 0;
    const loom_op_vtable_t* const* scalar_vtables =
        loom_scalar_dialect_vtables(&scalar_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, scalar_vtables,
        (uint16_t)scalar_vtable_count));
    iree_host_size_t scf_vtable_count = 0;
    const loom_op_vtable_t* const* scf_vtables =
        loom_scf_dialect_vtables(&scf_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, scf_vtables, (uint16_t)scf_vtable_count));
    iree_host_size_t sanitizer_vtable_count = 0;
    const loom_op_vtable_t* const* sanitizer_vtables =
        loom_sanitizer_dialect_vtables(&sanitizer_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SANITIZER, sanitizer_vtables,
        (uint16_t)sanitizer_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 16));
    loom_symbolic_expr_context_initialize(module_, /*value_domain=*/nullptr,
                                          &fact_table_, &analysis_arena_,
                                          &expression_context_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t DefineIndexValue() {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
    return value_id;
  }

  loom_value_id_t DefineI64Value() {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I64), &value_id));
    return value_id;
  }

  loom_value_id_t DefineI1Value() {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I1), &value_id));
    return value_id;
  }

  void DefineFacts(loom_value_id_t value_id, loom_value_facts_t facts) {
    IREE_CHECK_OK(loom_value_fact_table_define(&fact_table_, value_id, facts));
    loom_symbolic_expr_context_reset(&expression_context_);
  }

  void ComputeFacts(loom_op_t* op) {
    IREE_CHECK_OK(loom_value_fact_table_compute_op(&fact_table_, module_, op));
    loom_symbolic_expr_context_reset(&expression_context_);
  }

  loom_op_t* BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_value_id_t BuildScalarAndI(loom_value_id_t lhs, loom_value_id_t rhs) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_andi_build(&builder_, lhs, rhs,
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                                         LOOM_LOCATION_UNKNOWN, &op));
    return loom_scalar_andi_result(op);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
  loom_symbolic_expr_context_t expression_context_;
};

}  // namespace loom

#endif  // LOOM_ANALYSIS_SYMBOLIC_EXPR_TEST_FIXTURE_H_
