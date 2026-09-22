// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_query.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
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
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
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
    mapping_context_.module = module_;
    mapping_context_.source_function = function_;
    mapping_context_.options = &options_;
    mapping_context_.policy = options_.policy;
    mapping_context_.result = &mapping_result_;
    mapping_context_.lowering.fact_table = &fact_table_;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_source_lowering(
        options_.descriptor_registry, loom_target_facts_bundle(&target_facts_),
        &mapping_context_.descriptor_set));
    iree_arena_initialize(&block_pool_, &mapping_context_.function_arena);
  }

  void TearDown() override {
    loom_local_value_domain_release(&mapping_context_.lowering.value_domain);
    loom_low_lower_result_deinitialize(&mapping_result_);
    iree_arena_deinitialize(&mapping_context_.function_arena);
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
    loom_op_t* constant_op = nullptr;
    IREE_ASSERT_OK(
        loom_scalar_constant_build(&body_builder, loom_attr_f64(1.5),
                                   loom_type_scalar(LOOM_SCALAR_TYPE_F64),
                                   LOOM_LOCATION_UNKNOWN, &constant_op));
    unsupported_value_id_ = loom_scalar_constant_result(constant_op);
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, argument_ids_[0],
                                          argument_ids_[1], i32_type,
                                          LOOM_LOCATION_UNKNOWN, &source_op_));
    result_id_ = loom_scalar_addi_result(source_op_);
    const loom_type_t vector_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
    const loom_value_id_t elements[] = {argument_ids_[0], argument_ids_[1],
                                        argument_ids_[0], argument_ids_[1]};
    loom_op_t* vector_op = nullptr;
    IREE_ASSERT_OK(loom_vector_from_elements_build(
        &body_builder, elements, IREE_ARRAYSIZE(elements), vector_type,
        LOOM_LOCATION_UNKNOWN, &vector_op));
    const loom_value_id_t vector = loom_vector_from_elements_result(vector_op);
    IREE_ASSERT_OK(loom_vector_subi_build(&body_builder, 0, vector, vector,
                                          vector_type, LOOM_LOCATION_UNKNOWN,
                                          &mapped_source_op_));
    loom_op_t* yield_op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&body_builder, &result_id_, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield_op));
  }

  void CreateQueryScope() {
    IREE_ASSERT_OK(loom_low_lower_source_query_scope_create(
        module_, function_, &options_, &query_scope_arena_, &query_scope_));
  }

  iree_status_t QueryContract(const loom_op_t* source_op,
                              loom_target_contract_query_result_t* out_result) {
    loom_target_contract_query_environment_t environment = {};
    environment.module = module_;
    environment.function = function_;
    environment.target_facts = &target_facts_;
    environment.descriptor_set = mapping_context_.descriptor_set;
    environment.fact_table = &fact_table_;
    const loom_target_contract_query_callback_t callback =
        loom_low_lower_source_query_scope_callback(query_scope_);
    return callback.fn(callback.user_data, &environment, source_op, out_result);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  iree_arena_allocator_t query_scope_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_op_t* source_op_ = nullptr;
  // Generated vector rule whose guards require native register metadata.
  loom_op_t* mapped_source_op_ = nullptr;
  loom_value_id_t argument_ids_[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  loom_value_id_t result_id_ = LOOM_VALUE_ID_INVALID;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_options_t options_ = {};
  loom_low_lower_source_query_scope_t* query_scope_ = nullptr;
  // Native mapping context sharing the fixture's selected target contract.
  loom_low_lower_context_t mapping_context_ = {};
  // Diagnostics emitted only when lowering requires a native mapping.
  loom_low_lower_result_t mapping_result_ = {};
  // Valid source constant whose f64 type has no test-target representation.
  loom_value_id_t unsupported_value_id_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(LowLowerSourceQueryTest, NativeMappingAgreesWithRequiredLowering) {
  loom_type_t queried_type = loom_type_none();
  IREE_ASSERT_OK(loom_low_lower_query_value(&mapping_context_, source_op_,
                                            argument_ids_[0], &queried_type));
  ASSERT_TRUE(loom_type_is_register(queried_type));
  EXPECT_EQ(mapping_result_.error_count, 0u);

  loom_type_t required_type = loom_type_none();
  IREE_ASSERT_OK(loom_low_lower_map_value(&mapping_context_, source_op_,
                                          argument_ids_[0], &required_type));
  EXPECT_TRUE(loom_type_equal(queried_type, required_type));
  EXPECT_EQ(mapping_result_.error_count, 0u);
}

TEST_F(LowLowerSourceQueryTest, ContractQueriesRetainSourceScopeState) {
  static const uint8_t state_key = 0;
  struct MappingProbe {
    // Arena owned by the source query scope.
    iree_arena_allocator_t* expected_arena = nullptr;
    // State allocated before issuing a contract query.
    void* expected_state = nullptr;
    // Whether native value queries observed the scope's state and arena.
    bool observed_scope = false;
    // Whether a supplied foreign allocator was used.
    bool used_foreign_allocator = false;
  } probe;
  loom_low_lower_policy_t policy = *options_.policy;
  policy.map_contract_value = {
      /*.fn=*/[](void* user_data,
                 const loom_target_contract_query_environment_t* environment,
                 const loom_op_t*, loom_value_id_t,
                 loom_low_lower_rule_mapped_value_t* out_value)
                  -> iree_status_t {
        auto* probe = static_cast<MappingProbe*>(user_data);
        void* state = nullptr;
        IREE_RETURN_IF_ERROR(
            loom_target_contract_query_get_or_allocate_target_state(
                environment, &state_key, sizeof(uint32_t), &state));
        probe->observed_scope = state == probe->expected_state &&
                                environment->arena == probe->expected_arena &&
                                *static_cast<uint32_t*>(state) == 42;
        *out_value = loom_low_lower_rule_mapped_value_none();
        return iree_ok_status();
      },
      /*.user_data=*/&probe,
  };
  options_.policy = &policy;
  CreateQueryScope();
  loom_target_contract_query_environment_t environment = {};
  IREE_ASSERT_OK(loom_low_lower_source_query_scope_environment_initialize(
      query_scope_, &environment));
  probe.expected_arena = environment.arena;
  IREE_ASSERT_OK(loom_target_contract_query_get_or_allocate_target_state(
      &environment, &state_key, sizeof(uint32_t), &probe.expected_state));
  *static_cast<uint32_t*>(probe.expected_state) = 42;

  environment.arena = &analysis_arena_;
  environment.target_state_allocator = {
      /*.fn=*/[](void* user_data, const void*, iree_host_size_t,
                 void** out_data) -> iree_status_t {
        auto* probe = static_cast<MappingProbe*>(user_data);
        probe->used_foreign_allocator = true;
        *out_data = probe->expected_state;
        return iree_ok_status();
      },
      /*.user_data=*/&probe,
  };
  const auto callback =
      loom_low_lower_source_query_scope_callback(query_scope_);
  loom_target_contract_query_result_t result = {};
  IREE_ASSERT_OK(callback.fn(callback.user_data, &environment,
                             mapped_source_op_, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_TRUE(probe.observed_scope);
  EXPECT_FALSE(probe.used_foreign_allocator);
}

TEST_F(LowLowerSourceQueryTest, AbsentNativeMappingDoesNotEmitDiagnostic) {
  const loom_op_t* constant =
      loom_value_def_op(loom_module_value(module_, unsupported_value_id_));
  loom_type_t queried_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(loom_low_lower_query_value(
      &mapping_context_, constant, unsupported_value_id_, &queried_type));
  EXPECT_EQ(loom_type_kind(queried_type), LOOM_TYPE_NONE);
  EXPECT_EQ(mapping_result_.error_count, 0u);

  IREE_ASSERT_OK(loom_low_lower_map_value(
      &mapping_context_, constant, unsupported_value_id_, &queried_type));
  EXPECT_EQ(loom_type_kind(queried_type), LOOM_TYPE_NONE);
  EXPECT_EQ(mapping_result_.error_count, 1u);
}

TEST_F(LowLowerSourceQueryTest, RejectedNativeCandidateAllowsFollowingRule) {
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
      module_, loom_func_like_body(function_), &mapping_context_.function_arena,
      &mapping_context_.lowering.value_domain));
  const loom_op_t* constant =
      loom_value_def_op(loom_module_value(module_, unsupported_value_id_));
  loom_low_lower_value_ref_t value_ref = {};
  value_ref.kind = LOOM_LOW_LOWER_VALUE_REF_RESULT;
  loom_low_lower_guard_t guard = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT;
  guard.payload.u64 = 1;
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_rule_t rules[2] = {};
  rules[0].source_op_kind = constant->kind;
  rules[0].guard_count = 1;
  rules[1].source_op_kind = constant->kind;
  const loom_low_lower_rule_span_t span = {constant->kind, 0, 2};
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = rules;
  rule_set.rule_count = IREE_ARRAYSIZE(rules);
  rule_set.value_refs = &value_ref;
  rule_set.value_ref_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.guard_refs = &guard_ref;
  rule_set.guard_ref_count = 1;

  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select(&mapping_context_, &rule_set,
                                                constant, &selection));
  EXPECT_EQ(selection.rule, &rules[1]);
  EXPECT_EQ(selection.rule_index, 1u);
  EXPECT_EQ(mapping_result_.error_count, 0u);
}

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

TEST_F(LowLowerSourceQueryTest, SelectsGeneratedTargetContract) {
  CreateQueryScope();
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(QueryContract(source_op_, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  ASSERT_NE(result.selected_descriptor, nullptr);
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      mapping_context_.descriptor_set,
      result.selected_descriptor->semantic_tag_string_ref);
  EXPECT_TRUE(iree_string_view_equal(semantic_tag, IREE_SV("integer.add.i32")));
}

TEST_F(LowLowerSourceQueryTest, NativeContractWithoutMetadataCallback) {
  loom_low_lower_policy_t policy = *options_.policy;
  policy.map_contract_value = {};
  options_.policy = &policy;
  CreateQueryScope();

  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(QueryContract(mapped_source_op_, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_NE(result.selected_descriptor, nullptr);
}

}  // namespace
}  // namespace loom
