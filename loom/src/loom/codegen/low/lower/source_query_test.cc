// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_query.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/source_dataflow_verify.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/lower/source_representation.h"
#include "loom/target/test/target_records.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class LowLowerSourceQueryTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    iree_arena_initialize(&block_pool_, &query_scope_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source_query_test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    target_facts_.fact_type = &loom_test_target_fact_type;
    target_facts_.storage.bundle = *loom_test_target_bundles.values[1];
    BuildFunction();
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));
    options_ = {};
    options_.target_facts = &target_facts_;
    options_.descriptor_registry = &descriptor_registry_.registry;
    options_.policy = loom_test_low_lower_policy();
    options_.fact_table = &fact_table_;
  }

  void TearDown() override {
    loom_low_lower_source_query_scope_deinitialize(query_scope_);
    iree_arena_deinitialize(&query_scope_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&module_builder, IREE_SV("add"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t argument_types[] = {i32_type, i32_type};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, symbol, argument_types,
        IREE_ARRAYSIZE(argument_types), &i32_type, 1, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);

    loom_region_t* body = loom_func_like_body(function_);
    loom_block_t* entry_block = loom_region_entry_block(body);
    argument_ids_[0] = loom_block_arg_id(entry_block, 0);
    argument_ids_[1] = loom_block_arg_id(entry_block, 1);
    loom_builder_t body_builder;
    loom_builder_initialize(module_, &module_->arena, entry_block,
                            &body_builder);
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, argument_ids_[0],
                                          argument_ids_[1], i32_type,
                                          LOOM_LOCATION_UNKNOWN, &source_op_));
    result_id_ = loom_scalar_addi_result(source_op_);
    loom_op_t* yield_op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&body_builder, &result_id_, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield_op));
  }

  void CreateQueryScope() {
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_source_lowering(
        &descriptor_registry_.registry,
        loom_target_facts_bundle(&target_facts_), &descriptor_set_));
    IREE_ASSERT_OK(loom_low_lower_source_query_scope_create(
        module_, function_, &options_, descriptor_set_, &query_scope_arena_,
        &query_scope_));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  iree_arena_allocator_t query_scope_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_op_t* source_op_ = nullptr;
  loom_value_id_t argument_ids_[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  loom_value_id_t result_id_ = LOOM_VALUE_ID_INVALID;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_options_t options_ = {};
  loom_low_lower_source_query_scope_t* query_scope_ = nullptr;
};

TEST_F(LowLowerSourceQueryTest, OwnsFunctionAnalysesForScopeLifetime) {
  CreateQueryScope();

  const loom_local_value_domain_t* value_domain =
      loom_low_lower_source_query_scope_value_domain(query_scope_);
  ASSERT_NE(value_domain, nullptr);
  EXPECT_NE(loom_local_value_domain_try_ordinal(value_domain, argument_ids_[0]),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_NE(loom_local_value_domain_try_ordinal(value_domain, argument_ids_[1]),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_NE(loom_local_value_domain_try_ordinal(value_domain, result_id_),
            LOOM_VALUE_ORDINAL_INVALID);

  const loom_view_region_table_t* first_view_regions = nullptr;
  IREE_ASSERT_OK(loom_low_lower_source_query_scope_view_regions(
      query_scope_, &first_view_regions));
  ASSERT_NE(first_view_regions, nullptr);
  const loom_view_region_table_t* second_view_regions = nullptr;
  IREE_ASSERT_OK(loom_low_lower_source_query_scope_view_regions(
      query_scope_, &second_view_regions));
  EXPECT_EQ(second_view_regions, first_view_regions);
}

TEST_F(LowLowerSourceQueryTest, RetainsTestTargetSourceDataflow) {
  IREE_ASSERT_OK(
      loom_source_dataflow_provider_verify(&loom_test_low_source_dataflow));
  CreateQueryScope();
  const loom_source_dataflow_result_t* dataflow =
      loom_low_lower_source_query_scope_dataflow(query_scope_);
  ASSERT_NE(dataflow, nullptr);
  EXPECT_EQ(dataflow->provider, &loom_test_low_source_dataflow);
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      dataflow, result_id_,
      LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED |
          LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED |
          LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED));
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      dataflow, argument_ids_[0],
      LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED));
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      dataflow, argument_ids_[1],
      LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED));
  EXPECT_EQ(dataflow->statistics.value_seed_invocation_count,
            dataflow->state_count);
  EXPECT_EQ(dataflow->statistics.predicate_invocation_count, 0u);
}

TEST_F(LowLowerSourceQueryTest, RetainsTestTargetSourceRepresentations) {
  CreateQueryScope();
  const loom_low_source_representation_plan_t* representation_plan =
      loom_low_lower_source_query_scope_representation_plan(query_scope_);
  ASSERT_NE(representation_plan, nullptr);
  EXPECT_EQ(representation_plan->provider,
            &loom_test_low_source_representation_provider);
  EXPECT_GT(representation_plan->statistics.candidate_group_count, 0u);
}

TEST_F(LowLowerSourceQueryTest, SelectsGeneratedTargetContract) {
  CreateQueryScope();
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_source_lowering(
      &descriptor_registry_.registry, loom_target_facts_bundle(&target_facts_),
      &descriptor_set));

  loom_target_contract_query_environment_t environment = {};
  environment.module = module_;
  environment.function = function_;
  environment.target_facts = &target_facts_;
  environment.descriptor_set = descriptor_set;
  environment.fact_table = &fact_table_;
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  const loom_target_contract_query_callback_t callback =
      loom_low_lower_source_query_scope_callback(query_scope_);
  IREE_ASSERT_OK(
      callback.fn(callback.user_data, &environment, source_op_, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  ASSERT_NE(result.selected_descriptor, nullptr);
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      descriptor_set, result.selected_descriptor->semantic_tag_string_offset);
  EXPECT_TRUE(iree_string_view_equal(semantic_tag, IREE_SV("integer.add.i32")));
}

}  // namespace
}  // namespace loom
