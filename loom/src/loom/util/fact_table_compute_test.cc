// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_edge_projection.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactTableComputeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, static_cast<uint16_t>(count)));
    vtables = loom_cfg_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, vtables, static_cast<uint16_t>(count)));
    vtables = loom_scf_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, static_cast<uint16_t>(count)));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, vtables, static_cast<uint16_t>(count)));
    vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, static_cast<uint16_t>(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("facts"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 4));
    for (loom_value_id_t& input : inputs_) {
      IREE_ASSERT_OK(loom_builder_define_value(&builder_, type_, &input));
      IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                                  loom_value_facts_unknown()));
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t BuildIdentity(loom_value_id_t input, loom_op_t** out_op) {
    return loom_index_assume_build(&builder_, &input, 1, nullptr, 0, &type_, 1,
                                   LOOM_LOCATION_UNKNOWN, out_op);
  }

  loom_value_id_t DefineValue(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(&builder_, type, &value));
    IREE_CHECK_OK(loom_value_fact_table_define(&table_, value,
                                               loom_value_facts_unknown()));
    return value;
  }

  loom_op_t* BuildSelect(loom_value_id_t condition, loom_value_id_t true_value,
                         loom_value_id_t false_value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scf_select_build(&builder_, condition, true_value,
                                        false_value, type_,
                                        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  static std::vector<loom_value_id_t> SelectDependencies(
      const loom_value_fact_table_t& table, loom_value_id_t value) {
    loom_value_set_cursor_t cursor;
    loom_value_fact_table_select_dependencies_begin(&table, value, &cursor);
    std::vector<loom_value_id_t> dependencies;
    for (loom_value_id_t dependency = loom_value_set_cursor_next(&cursor);
         dependency != LOOM_VALUE_ID_INVALID;
         dependency = loom_value_set_cursor_next(&cursor)) {
      dependencies.push_back(dependency);
    }
    return dependencies;
  }

  static loom_value_set_id_t SelectDependencyRoot(
      const loom_value_fact_table_t& table, loom_value_id_t value) {
    loom_value_set_cursor_t cursor;
    return loom_value_fact_table_select_dependencies_begin(&table, value,
                                                           &cursor);
  }

  static std::vector<loom_op_t*> PendingExactRelations(
      const loom_value_fact_table_t& table) {
    loom_op_t* const* ops = nullptr;
    iree_host_size_t op_count = 0;
    loom_value_fact_table_pending_exact_relations(&table, &ops, &op_count);
    return op_count ? std::vector<loom_op_t*>(ops, ops + op_count)
                    : std::vector<loom_op_t*>();
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t table_;
  const loom_type_t type_ = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t inputs_[2];
};

TEST_F(FactTableComputeTest, IdentityOnlyMutationReportsChangedFacts) {
  EXPECT_EQ(table_.identities.entries, nullptr);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, inputs_[0]),
            inputs_[0]);
  EXPECT_EQ(loom_value_fact_table_query_identity(nullptr, inputs_[0]),
            inputs_[0]);
  loom_op_t* first = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &first));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             first, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(table_.select_dependencies.index, nullptr);
  EXPECT_EQ(table_.select_dependencies.roots, nullptr);
  const loom_value_id_t first_result = loom_op_const_results(first)[0];
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(BuildIdentity(first_result, &second));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  const loom_value_id_t second_result = loom_op_const_results(second)[0];
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, second_result),
            inputs_[0]);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             first, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(table_.exact_relations.ops, nullptr);

  IREE_ASSERT_OK(loom_op_set_operand(module_, first, 0, inputs_[1]));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             first, &changed));
  EXPECT_TRUE(changed);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             second, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, second_result),
            inputs_[1]);
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(&table_, second_result)));
}

TEST_F(FactTableComputeTest, OpaqueDefinitionsRetainSelectedInputs) {
  const loom_value_id_t storage = DefineValue(loom_type_pool());
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, storage, type_,
                                               LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, storage, type_,
                                               LOOM_LOCATION_UNKNOWN, &second));
  const loom_value_id_t first_result = loom_test_read_resource_result(first);
  const loom_value_id_t second_result = loom_test_read_resource_result(second);

  loom_value_fact_table_t inputs;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&inputs, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &inputs, first_result, loom_value_facts_make(16, 64, 16)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&inputs, second_result,
                                              loom_value_facts_exact_i64(32)));
  IREE_ASSERT_OK(loom_value_fact_table_seed_values(
      &table_, {&inputs, &first_result, 1}, module_));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  auto facts = loom_value_fact_table_lookup(&table_, first_result);
  EXPECT_EQ(facts.range_lo, 16);
  EXPECT_EQ(facts.range_hi, 64);
  EXPECT_EQ(facts.known_divisor, 16);
  EXPECT_FALSE(loom_value_facts_is_exact(
      loom_value_fact_table_lookup(&table_, second_result)));

  // Cyclic inference may undefine local entries before computing them again.
  loom_value_fact_table_undefine(&table_, first_result);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  facts = loom_value_fact_table_lookup(&table_, first_result);
  EXPECT_EQ(facts.range_lo, 16);
  EXPECT_EQ(facts.range_hi, 64);
  EXPECT_EQ(facts.known_divisor, 16);

  loom_value_fact_table_clear_scope(&table_);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  EXPECT_NE(loom_value_fact_table_lookup(&table_, first_result).range_lo, 16);
  IREE_ASSERT_OK(loom_value_fact_table_define(&inputs, first_result,
                                              loom_value_facts_exact_i64(48)));
  IREE_ASSERT_OK(loom_value_fact_table_seed_values(
      &table_, {&inputs, &first_result, 1}, module_));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  facts = loom_value_fact_table_lookup(&table_, first_result);
  EXPECT_TRUE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, 48);
}

TEST_F(FactTableComputeTest, BooleanBranchTruthIsRetainedByRegion) {
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(
      &builder_, inputs_[0], /*result_types=*/nullptr, /*result_count=*/0,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &branch));
  loom_region_branch_t branch_interface =
      loom_region_branch_cast(module_, branch);
  for (uint8_t i = 0; i < branch->region_count; ++i) {
    loom_region_t* region =
        loom_region_branch_region(module_, branch_interface, i);
    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, branch, region);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, /*values=*/nullptr,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
  }

  EXPECT_FALSE(table_.has_boolean_branch_regions);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, branch));
  EXPECT_TRUE(table_.has_boolean_branch_regions);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                &table_, loom_test_branch_then_region(branch)),
            LOOM_REGION_BRANCH_TRUTH_TRUE);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                &table_, loom_test_branch_else_region(branch)),
            LOOM_REGION_BRANCH_TRUTH_FALSE);

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_FALSE(table_.has_boolean_branch_regions);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                &table_, loom_test_branch_then_region(branch)),
            LOOM_REGION_BRANCH_TRUTH_UNKNOWN);
}

TEST_F(FactTableComputeTest, SelectorBranchRegionsRemainTruthUnknown) {
  int64_t case_keys[] = {0, 1};
  loom_op_t* table_op = nullptr;
  IREE_ASSERT_OK(loom_test_region_table_build(
      &builder_, inputs_[0], case_keys, IREE_ARRAYSIZE(case_keys),
      LOOM_LOCATION_UNKNOWN, &table_op));
  loom_region_branch_t branch = loom_region_branch_cast(module_, table_op);
  for (uint8_t i = 0; i < table_op->region_count; ++i) {
    loom_region_t* region = loom_region_branch_region(module_, branch, i);
    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, table_op, region);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, /*values=*/nullptr,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
  }

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, table_op));
  EXPECT_FALSE(table_.has_boolean_branch_regions);
  for (uint8_t i = 0; i < table_op->region_count; ++i) {
    EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                  &table_, loom_region_branch_region(module_, branch, i)),
              LOOM_REGION_BRANCH_TRUTH_UNKNOWN);
  }
}

TEST_F(FactTableComputeTest, ExactDynamicRelationsAreRetainedLazily) {
  EXPECT_EQ(table_.exact_relations.ops, nullptr);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(5)));
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1]},
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(PendingExactRelations(table_), std::vector<loom_op_t*>({assume}));

  // A stable incremental refresh performs no candidate work and does not
  // duplicate the pending observation.
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(PendingExactRelations(table_), std::vector<loom_op_t*>({assume}));

  loom_value_fact_table_clear_pending_exact_relations(&table_);
  EXPECT_TRUE(PendingExactRelations(table_).empty());
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.args[1] = 10;
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(PendingExactRelations(table_).empty());

  predicate.kind = LOOM_PREDICATE_POWER_OF_TWO;
  predicate.arg_count = 1;
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = inputs_[1];
  loom_op_t* unary_assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &unary_assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
      &table_, module_, unary_assume, &changed));
  EXPECT_EQ(PendingExactRelations(table_),
            std::vector<loom_op_t*>({unary_assume}));

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.exact_relations.ops, nullptr);
}

TEST_F(FactTableComputeTest,
       ExactDynamicRelationsAreRetainedWithoutChangeReporting) {
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(5)));
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1]},
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume));

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));

  EXPECT_EQ(PendingExactRelations(table_), std::vector<loom_op_t*>({assume}));
}

TEST_F(FactTableComputeTest, ExactRelationRetentionGrowsWithCandidates) {
  constexpr iree_host_size_t kAssumeCount = 64;
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(5)));
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1]},
  };
  std::vector<loom_op_t*> assumes;
  assumes.reserve(kAssumeCount);
  for (iree_host_size_t i = 0; i < kAssumeCount; ++i) {
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1,
                                           &predicate, 1, &type_, 1,
                                           LOOM_LOCATION_UNKNOWN, &assume));
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
        &table_, module_, assume, &changed));
    EXPECT_TRUE(changed);
    assumes.push_back(assume);
  }
  EXPECT_EQ(PendingExactRelations(table_), assumes);
}

TEST_F(FactTableComputeTest, UnknownFloatResultsRetainTheirTypeDomain) {
  const loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t input = DefineValue(f32_type);
  loom_op_t* negation = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32_type,
                                     LOOM_LOCATION_UNKNOWN, &negation));

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, negation));

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(&table_, loom_test_neg_result(negation));
  EXPECT_TRUE(loom_value_facts_is_float(facts));
  EXPECT_FALSE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, INT64_MIN);
  EXPECT_EQ(facts.range_hi, INT64_MAX);
}

TEST_F(FactTableComputeTest,
       SelectDependenciesPropagateCloneAndTrackOperandMutation) {
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t first_condition = DefineValue(i1_type);
  const loom_value_id_t second_condition = DefineValue(i1_type);
  loom_op_t* select = BuildSelect(first_condition, inputs_[0], inputs_[1]);
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             select, &changed));
  EXPECT_TRUE(changed);
  const loom_value_id_t selected = loom_scf_select_result(select);
  EXPECT_EQ(SelectDependencies(table_, selected),
            std::vector<loom_value_id_t>({first_condition}));

  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, selected, inputs_[0], type_,
                                      LOOM_LOCATION_UNKNOWN, &add));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, add));
  const loom_value_id_t sum = loom_index_add_result(add);
  EXPECT_EQ(SelectDependencies(table_, sum),
            std::vector<loom_value_id_t>({first_condition}));

  IREE_ASSERT_OK(loom_op_set_operand(module_, select, 0, second_condition));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             select, &changed));
  EXPECT_TRUE(changed);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             add, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(SelectDependencies(table_, sum),
            std::vector<loom_value_id_t>({second_condition}));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             add, &changed));
  EXPECT_FALSE(changed);

  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_values(&clone, {&table_, &sum, 1}, module_));
  EXPECT_EQ(SelectDependencies(clone, sum),
            std::vector<loom_value_id_t>({second_condition}));

  loom_value_fact_table_undefine(&table_, sum);
  EXPECT_EQ(SelectDependencies(table_, sum),
            std::vector<loom_value_id_t>({second_condition}));
  loom_value_fact_table_t undefined_clone = {};
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&undefined_clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &undefined_clone, {&table_, &sum, 1}, module_));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&undefined_clone, sum));
  EXPECT_EQ(SelectDependencies(undefined_clone, sum),
            std::vector<loom_value_id_t>({second_condition}));
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_TRUE(SelectDependencies(table_, sum).empty());
  EXPECT_EQ(table_.select_dependencies.index, nullptr);
  EXPECT_EQ(table_.select_dependencies.roots, nullptr);
}

TEST_F(FactTableComputeTest,
       FactIdentityPredicateReferencesContributeDependencies) {
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t first_condition = DefineValue(i1_type);
  const loom_value_id_t second_condition = DefineValue(i1_type);
  loom_op_t* first_select =
      BuildSelect(first_condition, inputs_[0], inputs_[1]);
  loom_op_t* second_select =
      BuildSelect(second_condition, inputs_[0], inputs_[1]);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, first_select));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, second_select));

  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_GE,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], loom_scf_select_result(first_select)},
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];
  EXPECT_EQ(SelectDependencies(table_, result),
            std::vector<loom_value_id_t>({first_condition}));

  predicate.args[1] = loom_scf_select_result(second_select);
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(SelectDependencies(table_, result),
            std::vector<loom_value_id_t>({second_condition}));

  predicate.args[1] = inputs_[1];
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(SelectDependencies(table_, result).empty());
}

TEST_F(FactTableComputeTest, NonBooleanSelectorDoesNotCreateDependency) {
  const int64_t case_key = 0;
  const loom_value_id_t values[] = {inputs_[0], inputs_[1]};
  loom_op_t* lookup = nullptr;
  IREE_ASSERT_OK(loom_scf_lookup_build(
      &builder_, inputs_[0], &case_key, 1, values, IREE_ARRAYSIZE(values),
      &type_, 1, /*tied_results=*/nullptr, /*tied_result_count=*/0,
      LOOM_LOCATION_UNKNOWN, &lookup));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, lookup));
  EXPECT_TRUE(
      SelectDependencies(table_, loom_scf_lookup_results(lookup).values[0])
          .empty());
  EXPECT_EQ(table_.select_dependencies.index, nullptr);
  EXPECT_EQ(table_.select_dependencies.roots, nullptr);
}

TEST_F(FactTableComputeTest,
       VariadicMultiResultProducerSharesCompleteDependencySet) {
  constexpr uint16_t kCaseCount = 47;
  constexpr uint16_t kConditionCount = 96;
  static_assert(kConditionCount == (kCaseCount + 1) * 2);
  std::vector<loom_value_id_t> conditions;
  std::vector<loom_value_id_t> selected_values;
  conditions.reserve(kConditionCount);
  selected_values.reserve(kConditionCount);
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  for (uint16_t i = 0; i < kConditionCount; ++i) {
    const loom_value_id_t condition = DefineValue(i1_type);
    conditions.push_back(condition);
    loom_op_t* select = BuildSelect(condition, inputs_[0], inputs_[1]);
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, select));
    selected_values.push_back(loom_scf_select_result(select));
  }

  int64_t case_keys[kCaseCount];
  for (uint16_t i = 0; i < kCaseCount; ++i) {
    case_keys[i] = i;
  }
  const loom_type_t result_types[] = {type_, type_};
  loom_op_t* lookup = nullptr;
  IREE_ASSERT_OK(loom_scf_lookup_build(
      &builder_, inputs_[0], case_keys, IREE_ARRAYSIZE(case_keys),
      selected_values.data(), selected_values.size(), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &lookup));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, lookup));
  const loom_value_id_t* results = loom_op_const_results(lookup);
  EXPECT_EQ(SelectDependencies(table_, results[0]), conditions);
  EXPECT_EQ(SelectDependencies(table_, results[1]), conditions);
  const loom_value_set_id_t shared_root =
      SelectDependencyRoot(table_, results[0]);
  EXPECT_EQ(SelectDependencyRoot(table_, results[1]), shared_root);

  loom_op_t* shared = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, results[0], results[0], type_,
                                      LOOM_LOCATION_UNKNOWN, &shared));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, shared));
  const loom_value_id_t shared_result = loom_index_add_result(shared);
  EXPECT_EQ(SelectDependencyRoot(table_, shared_result), shared_root);
  EXPECT_EQ(SelectDependencies(table_, shared_result), conditions);
}

TEST_F(FactTableComputeTest, IdentityCloneUndefineAndScopeReuse) {
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &alias));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, alias));
  const loom_value_id_t result = loom_op_const_results(alias)[0];
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table_, &result, 1}, module_));
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, result), inputs_[0]);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&clone, inputs_[0]));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&clone, inputs_[1]));
  loom_value_fact_table_undefine(&table_, result);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), result);
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             alias, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), inputs_[0]);

  const auto* storage = table_.identities.entries;
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.identities.entries, storage);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), result);
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, result), inputs_[0]);
  IREE_ASSERT_OK(loom_op_set_operand(module_, alias, 0, inputs_[1]));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, alias));
  EXPECT_EQ(table_.identities.entries, storage);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), inputs_[1]);
}

TEST_F(FactTableComputeTest, NumericEqualityDoesNotCreateSSAIdentity) {
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(7), type_,
                                           LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(7), type_,
                                           LOOM_LOCATION_UNKNOWN, &second));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  const loom_value_id_t first_result = loom_op_const_results(first)[0];
  const loom_value_id_t second_result = loom_op_const_results(second)[0];
  EXPECT_NE(loom_value_fact_table_query_identity(&table_, first_result),
            loom_value_fact_table_query_identity(&table_, second_result));
  EXPECT_EQ(table_.identities.entries, nullptr);
}

TEST_F(FactTableComputeTest, DependentResultsPublishOnlyFinalChanges) {
  const loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  const loom_value_id_t input = DefineValue(tensor_type);
  for (uint16_t extent_index = 0; extent_index < 2; ++extent_index) {
    SCOPED_TRACE(extent_index);
    const uint16_t tensor_index = 1 - extent_index;
    loom_type_t result_types[] = {type_, type_, type_};
    result_types[tensor_index] = tensor_type;
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_deflate_build(
        &builder_, input, result_types, IREE_ARRAYSIZE(result_types),
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &op));
    const loom_value_id_t* results = loom_op_const_results(op);
    IREE_ASSERT_OK(loom_module_set_value_type(
        module_, results[tensor_index],
        loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(results[extent_index]), 0)));

    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(loom_value_facts_is_non_negative(
        loom_value_fact_table_lookup(&table_, results[extent_index])));
    EXPECT_TRUE(loom_value_fact_table_has_entry(&table_, results[2]));
    EXPECT_TRUE(loom_value_facts_is_unknown(
        loom_value_fact_table_lookup(&table_, results[2])));

    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(loom_value_facts_is_non_negative(
        loom_value_fact_table_lookup(&table_, results[extent_index])));

    // Type edits update the canonical use index before producer inference.
    // Removing the shaped use withdraws its domain without a separate cache.
    IREE_ASSERT_OK(loom_module_set_value_type(module_, results[tensor_index],
                                              tensor_type));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(loom_value_facts_is_unknown(
        loom_value_fact_table_lookup(&table_, results[extent_index])));
  }
}

TEST_F(FactTableComputeTest, ConditionLoopRetainsAndReplacesBodyEntryFacts) {
  loom_op_t* bound_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, inputs_[1], inputs_[1], type_,
                                      LOOM_LOCATION_UNKNOWN, &bound_op));
  const loom_value_id_t bound = loom_index_add_result(bound_op);
  // Keep the dynamic upper bound in the signed carrier range so the true edge
  // of the unsigned comparison also proves that its lower operand is
  // nonnegative.
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, bound, loom_value_facts_make(0, 100, 1)));

  loom_op_t* loop = nullptr;
  const loom_type_t result_types[] = {
      type_,
      type_,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1),
  };
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, &inputs_[0], 1, /*iter_args_types=*/nullptr, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &loop));
  loom_region_t* before = loom_scf_while_before(loop);
  loom_region_t* body = loom_scf_while_after(loop);
  const loom_value_id_t before_argument = loom_region_entry_arg_id(before, 0);
  const loom_value_id_t body_arguments[] = {
      loom_region_entry_arg_id(body, 0),
      loom_region_entry_arg_id(body, 1),
  };
  const loom_value_id_t body_condition = loom_region_entry_arg_id(body, 2);

  const loom_value_id_t opaque_condition =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, loop, before);
  loom_op_t* compare = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_ULT,
                                      before_argument, bound,
                                      LOOM_LOCATION_UNKNOWN, &compare));
  loom_op_t* disjunction = nullptr;
  IREE_ASSERT_OK(loom_scalar_ori_build(
      &builder_, loom_index_cmp_result(compare), opaque_condition,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN,
      &disjunction));
  const loom_value_id_t forwarded[] = {
      before_argument,
      before_argument,
      loom_index_cmp_result(compare),
  };
  loom_op_t* condition = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder_, loom_index_cmp_result(compare), forwarded,
      IREE_ARRAYSIZE(forwarded), LOOM_LOCATION_UNKNOWN, &condition));
  loom_builder_restore(&builder_, saved);

  saved = loom_builder_enter_region(&builder_, loop, body);
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, body_arguments, 1,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  const loom_condition_edge_projection_t* projection =
      loom_value_fact_table_lookup_region_condition_projection(&table_, body);
  ASSERT_NE(projection, nullptr);
  EXPECT_EQ(table_.condition_integer_projection_count, 1u);

  auto expect_relation = [&](loom_symbolic_integer_relation_t relation,
                             loom_value_id_t left,
                             loom_condition_integer_operand_t right,
                             bool expected_proven, bool expected_result) {
    const loom_condition_integer_relation_t queried = {
        .relation = relation,
        .left =
            {
                .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = left,
                .constant = 0,
            },
        .right = right,
    };
    bool result = false;
    EXPECT_EQ(loom_condition_edge_projection_proves_integer_relation(
                  projection, &table_, &queried, &result),
              expected_proven);
    if (expected_proven) {
      EXPECT_EQ(result, expected_result);
    }
  };
  const loom_condition_integer_operand_t upper_bound = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = bound,
      .constant = 0,
  };
  const loom_condition_integer_operand_t zero = {
      /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
      /*.value_id=*/LOOM_VALUE_ID_INVALID,
      /*.constant=*/0,
  };
  for (loom_value_id_t body_argument : body_arguments) {
    expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, body_argument,
                    upper_bound, /*expected_proven=*/true,
                    /*expected_result=*/true);
    expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_GE, body_argument, zero,
                    /*expected_proven=*/true, /*expected_result=*/true);
  }
  bool projected_condition = false;
  EXPECT_TRUE(loom_condition_edge_projection_query_boolean(
      projection, body_condition, &projected_condition));
  EXPECT_TRUE(projected_condition);

  // Replacing one duplicate payload must replace its projected relation. The
  // retained object is updated in place so consumers never observe stale facts
  // and repeated rewrites reuse its high-water storage.
  const loom_value_id_t* target_sources = projection->target_sources;
  const loom_condition_edge_mapping_t* mappings = projection->mappings;
  const loom_condition_integer_relation_t* relations =
      projection->source_derivation.integer_facts.integer_relations;
  const loom_condition_boolean_fact_t* boolean_facts =
      projection->source_derivation.boolean_facts;
  IREE_ASSERT_OK(loom_op_set_operand(module_, condition, 2, bound));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  EXPECT_EQ(
      loom_value_fact_table_lookup_region_condition_projection(&table_, body),
      projection);
  EXPECT_EQ(projection->target_sources, target_sources);
  EXPECT_EQ(projection->mappings, mappings);
  EXPECT_EQ(projection->source_derivation.integer_facts.integer_relations,
            relations);
  EXPECT_EQ(projection->source_derivation.boolean_facts, boolean_facts);
  EXPECT_EQ(table_.condition_integer_projection_count, 1u);
  expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, body_arguments[0],
                  upper_bound, /*expected_proven=*/true,
                  /*expected_result=*/true);
  expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, body_arguments[1],
                  upper_bound, /*expected_proven=*/false,
                  /*expected_result=*/false);

  // A true disjunction does not prove the comparison, so replacing the exact
  // guard withdraws the table summary without scanning retained region
  // entries. Restoring it republishes the same projection object and slot.
  IREE_ASSERT_OK(loom_op_set_operand(module_, condition, 0,
                                     loom_scalar_ori_result(disjunction)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  EXPECT_TRUE(loom_condition_edge_projection_is_empty(projection));
  EXPECT_EQ(table_.condition_integer_projection_count, 0u);
  IREE_ASSERT_OK(loom_op_set_operand(module_, condition, 0,
                                     loom_index_cmp_result(compare)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  EXPECT_FALSE(loom_condition_edge_projection_is_empty(projection));
  EXPECT_EQ(table_.condition_integer_projection_count, 1u);

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(
      loom_value_fact_table_lookup_region_condition_projection(&table_, body),
      nullptr);
  EXPECT_EQ(table_.condition_integer_projection_count, 0u);
  EXPECT_EQ(table_.scratch.condition, nullptr);
}

TEST_F(FactTableComputeTest, ConditionLoopFactorsDuplicatePayloadRelations) {
  const loom_type_t result_types[] = {type_, type_, type_, type_};
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, inputs_, IREE_ARRAYSIZE(inputs_),
      /*iter_args_types=*/nullptr, result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &loop));
  loom_region_t* before = loom_scf_while_before(loop);
  loom_region_t* body = loom_scf_while_after(loop);
  const loom_value_id_t before_arguments[] = {
      loom_region_entry_arg_id(before, 0),
      loom_region_entry_arg_id(before, 1),
  };
  const loom_value_id_t body_arguments[] = {
      loom_region_entry_arg_id(body, 0),
      loom_region_entry_arg_id(body, 1),
      loom_region_entry_arg_id(body, 2),
      loom_region_entry_arg_id(body, 3),
  };

  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, loop, before);
  loom_op_t* compare = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_SLT,
                                      before_arguments[0], before_arguments[1],
                                      LOOM_LOCATION_UNKNOWN, &compare));
  const loom_value_id_t forwarded[] = {
      before_arguments[0],
      before_arguments[0],
      before_arguments[1],
      before_arguments[1],
  };
  loom_op_t* condition = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder_, loom_index_cmp_result(compare), forwarded,
      IREE_ARRAYSIZE(forwarded), LOOM_LOCATION_UNKNOWN, &condition));
  loom_builder_restore(&builder_, saved);

  saved = loom_builder_enter_region(&builder_, loop, body);
  const loom_value_id_t yielded[] = {body_arguments[0], body_arguments[2]};
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, yielded,
                                      IREE_ARRAYSIZE(yielded),
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  const loom_condition_edge_projection_t* projection =
      loom_value_fact_table_lookup_region_condition_projection(&table_, body);
  ASSERT_NE(projection, nullptr);
  EXPECT_EQ(projection->mapping_count, IREE_ARRAYSIZE(forwarded));
  // The source relation remains singular instead of expanding into every
  // pair of duplicate body arguments.
  EXPECT_EQ(projection->source_derivation.integer_facts.integer_relation_count,
            1u);
  for (iree_host_size_t left = 0; left < 2; ++left) {
    for (iree_host_size_t right = 2; right < 4; ++right) {
      const loom_condition_integer_relation_t query = {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
          .left =
              {
                  .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                  .value_id = body_arguments[left],
              },
          .right =
              {
                  .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                  .value_id = body_arguments[right],
              },
      };
      bool result = false;
      EXPECT_TRUE(loom_condition_edge_projection_proves_integer_relation(
          projection, &table_, &query, &result));
      EXPECT_TRUE(result);
    }
  }
}

class FactTableCfgRecomputeTest
    : public FactTableComputeTest,
      public ::testing::WithParamInterface<iree_host_size_t> {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<FactTableCfgRecomputeTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected snapshot allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    FactTableComputeTest::SetUp();
    iree_arena_block_pool_initialize(GetParam(), {this, Allocate},
                                     &scratch_pool_);
    iree_arena_initialize(&scratch_pool_, &scratch_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&scratch_pool_);
    FactTableComputeTest::TearDown();
  }

  // A captured initializer feeds a straight-line chain in a CFG cycle. Its
  // length crosses storage boundaries without changing the dataflow shape.
  void BuildCycle(iree_host_size_t count) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("cycle"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t callee = {0, symbol_id};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                        nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &function_op));
    loom_func_like_t function = loom_func_like_cast(module_, function_op);
    loom_region_t* body = loom_func_like_body(function);
    body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_builder_set_block(&builder_, loom_region_entry_block(body));
    builder_.ip.parent_op = function_op;
    IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(5), type_,
                                             LOOM_LOCATION_UNKNOWN, &initial_));
    loom_block_t* loop = nullptr;
    IREE_ASSERT_OK(loom_region_append_block(module_, body, &loop));
    loom_op_t* branch = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, loop, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    loom_builder_set_block(&builder_, loop);
    IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(1), type_,
                                             LOOM_LOCATION_UNKNOWN, &step_));
    loom_value_id_t previous = loom_index_constant_result(initial_);
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_op_t* add = nullptr;
      IREE_ASSERT_OK(loom_index_add_build(&builder_, previous,
                                          loom_index_constant_result(step_),
                                          type_, LOOM_LOCATION_UNKNOWN, &add));
      previous = loom_index_add_result(add);
      results_.push_back(previous);
    }
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, loop, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&table_, module_, function));
    structure_ = loom_value_fact_table_lookup_cfg_region(&table_, body);
    ASSERT_NE(structure_, nullptr);
    component_ =
        &structure_->control_flow.components
             .values[structure_->graph.blocks[loop->region_index].component];
    ASSERT_TRUE(component_->is_cycle);
  }

  void ChangeInitializer() {
    IREE_ASSERT_OK(loom_op_set_attr(module_, initial_, 0, loom_attr_i64(9)));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, initial_));
  }

  iree_status_t Recompute() {
    return loom_value_fact_table_recompute_cfg_component(
        &table_, module_, structure_, component_, &scratch_arena_,
        [](void* self, loom_value_id_t value) {
          static_cast<FactTableCfgRecomputeTest*>(self)->changed_.push_back(
              value);
          return iree_ok_status();
        },
        this);
  }

  void ExpectExact(loom_value_id_t value, int64_t expected) {
    ASSERT_TRUE(loom_value_fact_table_has_entry(&table_, value));
    const auto facts = loom_value_fact_table_lookup(&table_, value);
    EXPECT_EQ(facts.range_lo, expected);
    EXPECT_EQ(facts.range_hi, expected);
  }

  void CheckRecompute(iree_host_size_t result_count) {
    ASSERT_NO_FATAL_FAILURE(BuildCycle(result_count));
    for (iree_host_size_t i = 0; i < results_.size(); ++i) {
      ExpectExact(results_[i], 6 + i);
    }
    ASSERT_NO_FATAL_FAILURE(ChangeInitializer());
    IREE_ASSERT_OK(Recompute());
    EXPECT_EQ(changed_, results_);
    EXPECT_EQ(scratch_arena_.allocation_head, nullptr);
    ASSERT_GT(allocation_count_, 0u);
    const iree_host_size_t allocation_count = allocation_count_;
    const iree_host_size_t used_bytes = scratch_arena_.used_allocation_size;
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_initialize(&scratch_pool_, &scratch_arena_);
    changed_.clear();
    IREE_ASSERT_OK(Recompute());
    EXPECT_TRUE(changed_.empty());
    EXPECT_EQ(scratch_arena_.allocation_head, nullptr);
    EXPECT_EQ(allocation_count_, allocation_count);
    EXPECT_EQ(scratch_arena_.used_allocation_size, used_bytes);
    ExpectExact(loom_index_constant_result(initial_), 9);
    ExpectExact(loom_index_constant_result(step_), 1);
    for (iree_host_size_t i = 0; i < results_.size(); ++i) {
      ExpectExact(results_[i], 10 + i);
    }
  }

  void CheckAllocationFailure(iree_host_size_t failure_index) {
    // Both pool sizes require several allocations for this component.
    ASSERT_NO_FATAL_FAILURE(BuildCycle(4096));
    ASSERT_NO_FATAL_FAILURE(ChangeInitializer());
    failure_index_ = failure_index;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Recompute());
    EXPECT_EQ(allocation_count_, failure_index + 1);
    EXPECT_TRUE(changed_.empty());
    EXPECT_EQ(scratch_arena_.allocation_head, nullptr);
    ExpectExact(loom_index_constant_result(initial_), 9);
    bool reached_unsaved = false;
    iree_host_size_t undefined_count = 0;
    for (iree_host_size_t i = 0; i < results_.size(); ++i) {
      if (loom_value_fact_table_has_entry(&table_, results_[i])) {
        reached_unsaved = true;
        ExpectExact(results_[i], 6 + i);
      } else {
        EXPECT_FALSE(reached_unsaved);
        ++undefined_count;
      }
    }
    EXPECT_LT(undefined_count, results_.size());
    if (failure_index == 0) {
      EXPECT_EQ(undefined_count, 0u);
      ExpectExact(loom_index_constant_result(step_), 1);
    } else {
      EXPECT_GT(undefined_count, 0u);
    }
  }

  // Pool backing only the component's temporary snapshot stream.
  iree_arena_block_pool_t scratch_pool_ = {};
  // Reinitialized between updates, matching the rewriter's scratch lifetime.
  iree_arena_allocator_t scratch_arena_ = {};
  // Number of attempted scratch backing allocations.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation to fail, or SIZE_MAX for unrestricted allocation.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Definition outside the component, edited to invalidate its dependents.
  loom_op_t* initial_ = nullptr;
  // Unchanged definition within the component.
  loom_op_t* step_ = nullptr;
  // Component results in definition order.
  std::vector<loom_value_id_t> results_;
  // Actual changes reported by the recomputation API.
  std::vector<loom_value_id_t> changed_;
  // Retained region analysis produced by the initial whole-function solve.
  const loom_value_fact_cfg_region_t* structure_ = nullptr;
  // The loop's component in the retained region analysis.
  const loom_scc_t* component_ = nullptr;
};

TEST_P(FactTableCfgRecomputeTest,
       ChangedResultsStreamThroughReusablePoolBlocks) {
  CheckRecompute(2050);
}

TEST_P(FactTableCfgRecomputeTest, ExactFullSnapshotTailReportsEveryChange) {
  // The unchanged step and these results fill sixteen 128-row chunks.
  CheckRecompute(2047);
}

TEST_P(FactTableCfgRecomputeTest, FirstSnapshotFailurePreservesAllFacts) {
  CheckAllocationFailure(0);
}

TEST_P(FactTableCfgRecomputeTest, LaterSnapshotFailurePreservesUnsavedFacts) {
  CheckAllocationFailure(1);
}

INSTANTIATE_TEST_SUITE_P(PoolSizes, FactTableCfgRecomputeTest,
                         ::testing::Values(32 * 1024, 128 * 1024));

}  // namespace
}  // namespace loom
