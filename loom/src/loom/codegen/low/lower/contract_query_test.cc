// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/contract_query.h"

#include <cstdint>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

namespace {

constexpr loom_op_kind_t kSourceOpKind = LOOM_OP_KIND(7, 3);
constexpr uint64_t kDescriptorId = UINT64_C(0x123456789abcdef0);

const loom_low_descriptor_t kDescriptor = {
    /*.key_string_offset=*/0,
    /*.stable_id=*/kDescriptorId,
};

iree_status_t ResolveTestDescriptorRef(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  (void)user_data;
  (void)context;
  (void)rule_set;
  *out_descriptor = descriptor_ref == 0 ? &kDescriptor : nullptr;
  return iree_ok_status();
}

const loom_target_config_t kTargetConfig = {
    /*.name=*/IREE_SV("test-config"),
    /*.contract_set_key=*/{},
    /*.contract_feature_bits=*/0,
};

const loom_target_export_plan_t kTargetExportPlan = {
    /*.name=*/IREE_SV("test-export"),
};

const loom_target_bundle_t kTargetBundle = {
    /*.name=*/IREE_SV("test-target"),
    /*.snapshot=*/nullptr,
    /*.export_plan=*/&kTargetExportPlan,
    /*.config=*/&kTargetConfig,
};

loom_target_contract_fragment_t MakeContractFragment(
    loom_target_contract_op_entry_t* op_entries,
    loom_target_contract_dialect_table_t* dialects,
    loom_target_contract_fragment_case_t* cases,
    loom_target_contract_descriptor_rule_t* descriptor_rules) {
  op_entries[loom_op_dialect_index(kSourceOpKind)].case_start = 0;
  op_entries[loom_op_dialect_index(kSourceOpKind)].case_count = 1;
  dialects[0].op_count = 4;
  dialects[0].op_entries = op_entries;
  cases[0].system = LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE;
  cases[0].row_index = 0;
  descriptor_rules[0].rule_index = 0;

  loom_target_contract_fragment_t fragment = {};
  fragment.dialect_base_id = loom_op_dialect_id(kSourceOpKind);
  fragment.dialect_count = 1;
  fragment.flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY;
  fragment.dialects = dialects;
  fragment.case_count = 1;
  fragment.cases = cases;
  fragment.descriptor_rule_count = 1;
  fragment.descriptor_rules = descriptor_rules;
  return fragment;
}

loom_target_contract_fragment_t MakeContractFragmentForOp(
    loom_op_kind_t source_op_kind, loom_target_contract_op_entry_t* op_entries,
    uint16_t op_entry_count, loom_target_contract_dialect_table_t* dialects,
    loom_target_contract_fragment_case_t* cases,
    loom_target_contract_descriptor_rule_t* descriptor_rules) {
  op_entries[loom_op_dialect_index(source_op_kind)].case_start = 0;
  op_entries[loom_op_dialect_index(source_op_kind)].case_count = 1;
  dialects[0].op_count = op_entry_count;
  dialects[0].op_entries = op_entries;
  cases[0].system = LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE;
  cases[0].row_index = 0;
  descriptor_rules[0].rule_index = 0;

  loom_target_contract_fragment_t fragment = {};
  fragment.dialect_base_id = loom_op_dialect_id(source_op_kind);
  fragment.dialect_count = 1;
  fragment.flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY;
  fragment.dialects = dialects;
  fragment.case_count = 1;
  fragment.cases = cases;
  fragment.descriptor_rule_count = 1;
  fragment.descriptor_rules = descriptor_rules;
  return fragment;
}

class LowContractQuerySourceMemoryTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    BuildFunction();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &module_builder, IREE_SV("source_memory"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, symbol, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
  }

  loom_value_id_t DefineBufferArg() {
    loom_value_id_t buffer = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_buffer(), &buffer));
    return buffer;
  }

  loom_value_id_t BuildDenseLayout() {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_encoding_layout_dense_build(
        &builder_,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_dense_result(op);
  }

  loom_value_id_t BuildOffsetConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  loom_value_id_t BuildIndexAssumeRange(loom_value_id_t source,
                                        int64_t minimum_value,
                                        int64_t maximum_value) {
    loom_predicate_t* predicate = nullptr;
    IREE_CHECK_OK(iree_arena_allocate_array(
        &module_->arena, 1, sizeof(*predicate), (void**)&predicate));
    *predicate = (loom_predicate_t){
        /*.kind=*/LOOM_PREDICATE_RANGE,
        /*.arg_count=*/3,
        /*.arg_tags=*/
        {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
        /*.reserved=*/{},
        /*.args=*/{source, minimum_value, maximum_value},
    };
    const loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_assume_build(&builder_, &source, 1, predicate, 1,
                                          &result_type, 1,
                                          LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_assume_results(op).values[0];
  }

  loom_type_t ViewType1D(int64_t extent, loom_value_id_t layout) {
    loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                           loom_dim_pack_static(extent), 0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return type;
  }

  loom_type_t VectorType1D(int64_t extent) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                               loom_dim_pack_static(extent), 0);
  }

  void ComputeFacts(loom_value_fact_table_t* out_facts) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(out_facts, &analysis_arena_,
                                                    module_->values.count));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(out_facts, module_, function_));
  }

  loom_context_t context_;
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_builder_t builder_;
};

TEST(LowContractQueryTest, ContractIndexDescriptorRuleSelectsLegalCase) {
  loom_low_lower_rule_descriptor_ref_t descriptor_ref = {
      /*.key=*/IREE_SV("test.descriptor"),
  };
  loom_low_lower_emit_t emit = {};
  emit.kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP;
  emit.descriptor_ref = 0;
  loom_low_lower_rule_t rule = {};
  rule.source_op_kind = kSourceOpKind;
  rule.emit_count = 1;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.descriptor_refs = &descriptor_ref;
  rule_set.descriptor_ref_count = 1;
  rule_set.emits = &emit;
  rule_set.emit_count = 1;
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  loom_target_contract_op_entry_t op_entries[4] = {};
  loom_target_contract_dialect_table_t dialects[1] = {};
  loom_target_contract_fragment_case_t cases[1] = {};
  loom_target_contract_descriptor_rule_t descriptor_rules[1] = {};
  loom_target_contract_fragment_t fragment =
      MakeContractFragment(op_entries, dialects, cases, descriptor_rules);
  const loom_target_contract_binding_t bindings[] = {
      {
          &fragment,
          0,
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena));

  const loom_low_lower_contract_query_options_t options = {
      /*.contract_index=*/&index,
      /*.rule_sets=*/
      {
          /*.count=*/IREE_ARRAYSIZE(rule_sets),
          /*.values=*/rule_sets,
      },
      /*.map_value=*/{},
      /*.can_materialize=*/{},
      /*.descriptor_ref=*/
      {
          /*.fn=*/ResolveTestDescriptorRef,
          /*.user_data=*/nullptr,
      },
  };
  const loom_target_contract_query_environment_t environment = {
      /*.module=*/nullptr,
      /*.function=*/{},
      /*.bundle=*/&kTargetBundle,
      /*.target_data=*/nullptr,
      /*.target_ref=*/{},
      /*.descriptor_set=*/nullptr,
      /*.fact_table=*/nullptr,
      /*.view_regions=*/nullptr,
      /*.arena=*/nullptr,
  };
  loom_op_t op = {};
  op.kind = kSourceOpKind;
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      &op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(result.binding_index, 0);
  EXPECT_EQ(result.case_index, 0);
  EXPECT_EQ(result.rule_set_index, 0);
  EXPECT_EQ(result.rule_index, 0);
  EXPECT_EQ(result.selected_descriptor, &kDescriptor);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowContractQueryTest, ContractIndexDescriptorRuleReportsRejectedCase) {
  loom_low_lower_guard_t guard = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND;
  guard.attr_kind = LOOM_ATTR_I64;
  guard.diagnostic_index = 0;
  loom_low_lower_diagnostic_param_t diagnostic_params[] = {
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("test.source"),
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("field"),
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("value"),
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("attr_kind"),
      },
  };
  loom_low_lower_diagnostic_t diagnostic = {};
  diagnostic.error_ref = LOOM_ERR_TARGET_003_REF;
  diagnostic.param_count = IREE_ARRAYSIZE(diagnostic_params);
  loom_low_lower_rule_t rule = {};
  rule.source_op_kind = kSourceOpKind;
  rule.guard_count = 1;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.diagnostic_params = diagnostic_params;
  rule_set.diagnostic_param_count = IREE_ARRAYSIZE(diagnostic_params);
  rule_set.diagnostics = &diagnostic;
  rule_set.diagnostic_count = 1;
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  loom_target_contract_op_entry_t op_entries[4] = {};
  loom_target_contract_dialect_table_t dialects[1] = {};
  loom_target_contract_fragment_case_t cases[1] = {};
  loom_target_contract_descriptor_rule_t descriptor_rules[1] = {};
  loom_target_contract_fragment_t fragment =
      MakeContractFragment(op_entries, dialects, cases, descriptor_rules);
  const loom_target_contract_binding_t bindings[] = {
      {
          &fragment,
          0,
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena));

  const loom_low_lower_contract_query_options_t options = {
      /*.contract_index=*/&index,
      /*.rule_sets=*/
      {
          /*.count=*/IREE_ARRAYSIZE(rule_sets),
          /*.values=*/rule_sets,
      },
  };
  const loom_target_contract_query_environment_t environment = {
      /*.module=*/nullptr,
      /*.function=*/{},
      /*.bundle=*/&kTargetBundle,
      /*.target_data=*/nullptr,
      /*.target_ref=*/{},
      /*.descriptor_set=*/nullptr,
      /*.fact_table=*/nullptr,
      /*.view_regions=*/nullptr,
      /*.arena=*/&arena,
  };
  loom_op_t op = {};
  op.kind = kSourceOpKind;
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      &op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(result.binding_index, 0);
  EXPECT_EQ(result.case_index, 0);
  EXPECT_EQ(result.rule_set_index, 0);
  EXPECT_EQ(result.rule_index, UINT16_MAX);
  EXPECT_EQ(result.diagnostic_index, 0);
  ASSERT_NE(result.rejection, nullptr);
  EXPECT_EQ(result.rejection->error_ref, LOOM_ERR_TARGET_003_REF);
  ASSERT_EQ(result.rejection->param_count, 8);
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[4].string,
                                     IREE_SV("test.source")));
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[5].string,
                                     IREE_SV("field")));
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[6].string,
                                     IREE_SV("value")));
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[7].string,
                                     IREE_SV("attr_kind")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST_F(LowContractQuerySourceMemoryTest,
       ValueDynamicIndexAcceptsWorkitemProvenance) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset = BuildOffsetConstant(0);

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(64, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  loom_value_id_t assumed_workitem =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(workitem_op), 0, 63);
  const loom_value_id_t dynamic_indices[] = {
      assumed_workitem,
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  const loom_low_lower_source_memory_t source_memory = {
      /*.flags=*/0,
      /*.operation_kind=*/LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD,
      /*.root_kind=*/LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_BLOCK_ARGUMENT,
      /*.memory_space_mask=*/LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_UNKNOWN |
          LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_GENERIC |
          LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_GLOBAL |
          LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_WORKGROUP |
          LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_PRIVATE |
          LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_CONSTANT |
          LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_DESCRIPTOR,
      /*.element_byte_count=*/4,
      /*.vector_lane_count=*/1,
      /*.vector_lane_byte_stride=*/4,
      /*.static_byte_offset_minimum=*/INT64_MIN,
      /*.static_byte_offset_maximum=*/INT64_MAX,
      /*.minimum_alignment=*/0,
      /*.dynamic_term_count=*/1,
      /*.dynamic_index_source=*/
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
      /*.dynamic_byte_stride=*/4,
      /*.dynamic_offset_unsigned_bit_count=*/0,
      /*.dynamic_offset_diagnostic_index=*/0,
      /*.cache_policy_build_flags=*/0,
      /*.diagnostic_index=*/0,
  };
  const loom_low_lower_emit_t emit = {
      /*.kind=*/LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
      /*.flags=*/0,
      /*.descriptor_ref=*/LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE,
      /*.operand_ref_start=*/0,
      /*.operand_ref_count=*/0,
      /*.copy_operand_mask=*/0,
      /*.accumulator_operand_index=*/0,
      /*.result_ref_start=*/0,
      /*.result_type_pattern_start=*/0,
      /*.result_ref_count=*/0,
      /*.result_bind_ref_start=*/0,
      /*.attr_copy_start=*/0,
      /*.attr_copy_count=*/0,
      /*.tied_result_start=*/0,
      /*.tied_result_count=*/0,
      /*.source_memory_ordinal=*/1,
  };
  const loom_low_lower_rule_t rule = {
      /*.source_op_kind=*/LOOM_OP_VECTOR_LOAD,
      /*.temporary_count=*/0,
      /*.guard_start=*/0,
      /*.guard_count=*/0,
      /*.emit_start=*/0,
      /*.emit_count=*/1,
      /*.alias_ref_start=*/0,
      /*.alias_ref_count=*/0,
      /*.elide_ref_start=*/0,
      /*.elide_ref_count=*/0,
  };
  const loom_low_lower_rule_set_t rule_set = {
      /*.flags=*/0,
      /*.spans=*/nullptr,
      /*.span_count=*/0,
      /*.rules=*/&rule,
      /*.rule_count=*/1,
      /*.type_patterns=*/nullptr,
      /*.type_pattern_count=*/0,
      /*.value_refs=*/nullptr,
      /*.value_ref_count=*/0,
      /*.materializers=*/nullptr,
      /*.materializer_count=*/0,
      /*.source_memories=*/&source_memory,
      /*.source_memory_count=*/1,
      /*.descriptor_refs=*/nullptr,
      /*.descriptor_ref_count=*/0,
      /*.diagnostic_params=*/nullptr,
      /*.diagnostic_param_count=*/0,
      /*.guards=*/nullptr,
      /*.guard_count=*/0,
      /*.attr_copies=*/nullptr,
      /*.attr_copy_count=*/0,
      /*.tied_results=*/nullptr,
      /*.tied_result_count=*/0,
      /*.emits=*/&emit,
      /*.emit_count=*/1,
      /*.diagnostics=*/nullptr,
      /*.diagnostic_count=*/0,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  loom_target_contract_op_entry_t op_entries[LOOM_OP_VECTOR_COUNT_] = {};
  loom_target_contract_dialect_table_t dialects[1] = {};
  loom_target_contract_fragment_case_t cases[1] = {};
  loom_target_contract_descriptor_rule_t descriptor_rules[1] = {};
  loom_target_contract_fragment_t fragment = MakeContractFragmentForOp(
      LOOM_OP_VECTOR_LOAD, op_entries, IREE_ARRAYSIZE(op_entries), dialects,
      cases, descriptor_rules);
  const loom_target_contract_binding_t bindings[] = {
      {
          &fragment,
          0,
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena));

  const loom_low_lower_contract_query_options_t options = {
      /*.contract_index=*/&index,
      /*.rule_sets=*/
      {
          /*.count=*/IREE_ARRAYSIZE(rule_sets),
          /*.values=*/rule_sets,
      },
  };
  const loom_target_contract_query_environment_t environment = {
      /*.module=*/module_,
      /*.function=*/function_,
      /*.bundle=*/&kTargetBundle,
      /*.target_data=*/nullptr,
      /*.target_ref=*/{},
      /*.descriptor_set=*/nullptr,
      /*.fact_table=*/&facts,
      /*.view_regions=*/nullptr,
      /*.arena=*/&arena,
  };
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      load_op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(result.rule_index, 0);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
