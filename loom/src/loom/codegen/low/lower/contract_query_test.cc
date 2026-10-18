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
#include "loom/analysis/view_regions.h"
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

const char kDescriptorStringData[] = "test.descriptor";
const char kRuleStringData[] = "test.sourcefieldvalueattr_kind";

enum : loom_string_ref_t {
  kRuleStringDescriptor = LOOM_STRING_REF(0, 0, 15),
  kRuleStringSource = LOOM_STRING_REF(1, 0, 11),
  kRuleStringField = LOOM_STRING_REF(1, 11, 5),
  kRuleStringValue = LOOM_STRING_REF(1, 16, 5),
  kRuleStringAttrKind = LOOM_STRING_REF(1, 21, 9),
};

const loom_string_pool_t kRuleStringPool = {
    .data = {kDescriptorStringData, kRuleStringData},
    .data_length = {sizeof(kDescriptorStringData) - 1,
                    sizeof(kRuleStringData) - 1},
};

const loom_low_descriptor_t kDescriptor = {
    .stable_id = kDescriptorId,
    .key_string_ref = kRuleStringDescriptor,
};

const loom_low_descriptor_layout_t kDescriptorLayout = {};
const loom_low_descriptor_set_t kDescriptorSet = {
    .descriptors = &kDescriptor,
    .descriptor_layouts = &kDescriptorLayout,
    .descriptor_layout_count = 1,
    .descriptor_count = 1,
    .descriptor_ordinal_count = 1,
    .string_pool = kRuleStringPool,
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

iree_status_t ResolveDescriptorFromArray(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  (void)context;
  (void)rule_set;
  const auto* descriptors =
      static_cast<const loom_low_descriptor_t*>(user_data);
  *out_descriptor = &descriptors[descriptor_ref];
  return iree_ok_status();
}

const loom_target_config_t kTargetConfig = {
    .name = IREE_SV("test-config"),
    .contract_set_key = {},
    .contract_feature_bits = 0,
};

const loom_target_export_plan_t kTargetExportPlan = {
    .name = IREE_SV("test-export"),
};

const loom_target_bundle_t kTargetBundle = {
    .name = IREE_SV("test-target"),
    .snapshot = nullptr,
    .export_plan = &kTargetExportPlan,
    .config = &kTargetConfig,
};

loom_target_facts_t MakeTargetFacts() {
  loom_target_facts_t facts = {};
  facts.storage.bundle = kTargetBundle;
  return facts;
}

const loom_target_contract_descriptor_rule_t kContractDescriptorRules[] = {{0}};
const loom_target_contract_fragment_t kContractFragment = {
    .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
    .descriptor_rule_count = 1,
    .descriptor_rules = kContractDescriptorRules,
};
const loom_target_contract_binding_t kContractBindings[] = {
    {&kContractFragment, 0}};
const loom_target_contract_case_t kContractCases[] = {
    {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 0},
};

template <loom_op_kind_t OpKind>
class SingleOpContract {
 public:
  SingleOpContract(
      const loom_target_contract_case_t* cases = kContractCases,
      uint16_t case_count = IREE_ARRAYSIZE(kContractCases),
      const loom_target_contract_binding_t* bindings = kContractBindings,
      uint8_t binding_count = IREE_ARRAYSIZE(kContractBindings),
      const loom_op_kind_t* source_graph_root_op_kinds = nullptr,
      uint16_t source_graph_root_op_kind_count = 0) {
    entries_[loom_op_dialect_index(OpKind)] = {0, case_count};
    dialect_ = {IREE_ARRAYSIZE(entries_), entries_};
    index_ = {
        .dialect_base_id = OpKind >> 8,
        .dialect_count = 1,
        .dialects = &dialect_,
        .case_count = case_count,
        .cases = cases,
        .binding_count = binding_count,
        .source_graph_root_op_kind_count = source_graph_root_op_kind_count,
        .bindings = bindings,
        .selection_data = nullptr,
        .source_graph_root_op_kinds = source_graph_root_op_kinds,
    };
    for (uint16_t i = 0; i < case_count; ++i) {
      uint16_t rule_index = UINT16_MAX;
      if (loom_low_lower_contract_case_queries_target(&index_, &cases[i]) &&
          loom_low_lower_contract_case_lower_rule_index(&index_, &cases[i],
                                                        &rule_index)) {
        ++entries_[loom_op_dialect_index(OpKind)].query_rule_count;
      }
    }
  }
  const loom_target_contract_index_t* index() const { return &index_; }

 private:
  // One populated op span in this fixture's source dialect.
  loom_target_contract_op_entry_t entries_[(OpKind & 0xFF) + 1] = {};
  // Dense dialect table borrowed by the index.
  loom_target_contract_dialect_table_t dialect_ = {};
  // Query-ready index, with the same representation as generated policy tables.
  loom_target_contract_index_t index_ = {};
};

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
        .module_id = 0,
        .symbol_id = symbol_id,
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
    *predicate = loom_predicate_t{
        .kind = LOOM_PREDICATE_RANGE,
        .arg_count = 3,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_CONST},
        .reserved = {},
        .args = {source, minimum_value, maximum_value},
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

TEST_F(LowContractQuerySourceMemoryTest,
       MaterializesImplicitTargetContextDiagnosticParams) {
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &source_op));
  const loom_low_lower_diagnostic_param_kind_t diagnostic_param_kinds[] = {
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
  };
  const loom_low_lower_diagnostic_value_t diagnostic_param_values[] = {
      {.string_value_ref = kRuleStringField},
      {.string_value_ref = kRuleStringValue},
      {.string_value_ref = kRuleStringAttrKind},
  };
  const loom_low_lower_diagnostic_param_ref_t diagnostic_param_refs[] = {0, 1,
                                                                         2};
  const loom_low_lower_diagnostic_t diagnostic = {
      .error_ref = LOOM_ERR_TARGET_003_REF,
      .param_start = 0,
      .param_count = 8,
      .flags = LOOM_LOW_LOWER_DIAGNOSTIC_FLAG_IMPLICIT_TARGET_CONTEXT,
  };
  loom_low_lower_rule_set_t rule_set = {
      .diagnostic_params =
          {
              .kinds = diagnostic_param_kinds,
              .values = diagnostic_param_values,
              .count = IREE_ARRAYSIZE(diagnostic_param_kinds),
          },
      .diagnostic_param_refs = diagnostic_param_refs,
      .diagnostic_param_ref_count = IREE_ARRAYSIZE(diagnostic_param_refs),
      .string_pool = kRuleStringPool,
  };
  loom_low_lower_rule_match_context_t match_context = {
      .module = module_,
      .function = function_,
      .bundle = &kTargetBundle,
  };
  loom_diagnostic_param_t params[8] = {};

  IREE_ASSERT_OK(loom_low_lower_rule_materialize_diagnostic_params(
      &match_context, &rule_set, source_op, &diagnostic, params));

  EXPECT_TRUE(iree_string_view_equal(params[0].string, IREE_SV("test-target")));
  EXPECT_TRUE(iree_string_view_equal(params[1].string, IREE_SV("test-export")));
  EXPECT_TRUE(iree_string_view_equal(params[2].string, IREE_SV("test-config")));
  EXPECT_TRUE(
      iree_string_view_equal(params[3].string, IREE_SV("source_memory")));
  EXPECT_TRUE(
      iree_string_view_equal(params[4].string, IREE_SV("index.constant")));
  EXPECT_TRUE(iree_string_view_equal(params[5].string, IREE_SV("field")));
  EXPECT_TRUE(iree_string_view_equal(params[6].string, IREE_SV("value")));
  EXPECT_TRUE(iree_string_view_equal(params[7].string, IREE_SV("attr_kind")));
}

TEST_F(LowContractQuerySourceMemoryTest,
       MaterializesNativeValuesThroughSharedParameterOrdinals) {
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &source_op));
  const loom_low_lower_diagnostic_param_kind_t kinds[] = {
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U64_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U32_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_VALUE_TYPE,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME,
  };
  const loom_low_lower_diagnostic_value_t values[] = {
      {.i64_value = INT64_MIN},
      {.i64_value = INT64_MAX},
      {.u64_value = UINT64_MAX},
      {.u32_value = UINT32_MAX},
      {.bool_value = false},
      {.bool_value = true},
      {.string_value_ref = kRuleStringField},
      {.value_ref_index = 0},
      {},
  };
  const loom_low_lower_diagnostic_param_ref_t refs[] = {
      2, 0, 3, 1, 5, 4, 6, 7, 8, 2, 0,
  };
  const loom_low_lower_value_ref_t value_ref = {
      .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
      .index = 0,
  };
  const loom_low_lower_diagnostic_t diagnostic = {
      .param_count = IREE_ARRAYSIZE(refs),
  };
  const loom_low_lower_rule_set_t rule_set = {
      .value_refs = &value_ref,
      .value_ref_count = 1,
      .diagnostic_params =
          {
              .kinds = kinds,
              .values = values,
              .count = IREE_ARRAYSIZE(kinds),
          },
      .diagnostic_param_refs = refs,
      .diagnostic_param_ref_count = IREE_ARRAYSIZE(refs),
      .string_pool = kRuleStringPool,
  };
  const loom_low_lower_rule_match_context_t match_context = {
      .module = module_,
      .function = function_,
      .bundle = &kTargetBundle,
  };
  loom_diagnostic_param_t params[IREE_ARRAYSIZE(refs)] = {};
  IREE_ASSERT_OK(loom_low_lower_rule_materialize_diagnostic_params(
      &match_context, &rule_set, source_op, &diagnostic, params));

  EXPECT_EQ(params[0].kind, LOOM_PARAM_U64);
  EXPECT_EQ(params[0].u64, UINT64_MAX);
  EXPECT_EQ(params[1].kind, LOOM_PARAM_I64);
  EXPECT_EQ(params[1].i64, INT64_MIN);
  EXPECT_EQ(params[2].kind, LOOM_PARAM_U32);
  EXPECT_EQ(params[2].u32, UINT32_MAX);
  EXPECT_EQ(params[3].kind, LOOM_PARAM_I64);
  EXPECT_EQ(params[3].i64, INT64_MAX);
  EXPECT_EQ(params[4].kind, LOOM_PARAM_BOOL);
  EXPECT_TRUE(params[4].boolean);
  EXPECT_EQ(params[5].kind, LOOM_PARAM_BOOL);
  EXPECT_FALSE(params[5].boolean);
  EXPECT_EQ(params[6].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(params[6].string, IREE_SV("field")));
  EXPECT_EQ(params[7].kind, LOOM_PARAM_TYPE);
  EXPECT_TRUE(loom_type_equal(params[7].type,
                              loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)));
  EXPECT_EQ(params[8].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(
      iree_string_view_equal(params[8].string, IREE_SV("index.constant")));
  EXPECT_EQ(params[9].kind, params[0].kind);
  EXPECT_EQ(params[9].u64, params[0].u64);
  EXPECT_EQ(params[10].kind, params[1].kind);
  EXPECT_EQ(params[10].i64, params[1].i64);
}

TEST(LowContractQueryTest, ContractIndexDescriptorRuleSelectsLegalCase) {
  loom_low_lower_rule_descriptor_ref_t descriptor_ref = {
      .stable_id = kDescriptorId,
  };
  loom_low_lower_emit_t emit = {
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
  };
  const loom_low_lower_emit_ref_t emit_ref = {
      .descriptor_ref = 0,
      .emit_index = 0,
  };
  loom_low_lower_rule_t rule = {.emit_count = 1};
  loom_low_lower_rule_set_t rule_set = {
      .rules = &rule,
      .rule_count = 1,
      .descriptor_refs = &descriptor_ref,
      .descriptor_ref_count = 1,
      .emit_refs = &emit_ref,
      .emit_ref_count = 1,
      .emits = &emit,
      .emit_count = 1,
      .string_pool = kRuleStringPool,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  SingleOpContract<kSourceOpKind> contract;

  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
      .map_value = {},
      .can_materialize = {},
      .descriptor_ref =
          {
              .fn = ResolveTestDescriptorRef,
              .user_data = nullptr,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .target_facts = &target_facts,
      .descriptor_set = &kDescriptorSet,
  };
  loom_op_t op = {.kind = kSourceOpKind};
  for (bool defer_diagnostics : {false, true}) {
    environment.defer_rejection_diagnostics = defer_diagnostics;
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
  }
}

TEST(LowContractQueryTest, LoweringVisitorIncludesNonQueryCases) {
  loom_low_lower_rule_t rule = {};
  loom_low_lower_rule_set_t rule_set = {
      .rules = &rule,
      .rule_count = 1,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set, &rule_set};

  const loom_target_contract_descriptor_rule_t descriptor_rule = {
      .rule_index = 0,
  };
  const loom_target_contract_fragment_t fragments[] = {
      {},
      {
          .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
          .descriptor_rule_count = 1,
          .descriptor_rules = &descriptor_rule,
      },
  };
  const loom_target_contract_binding_t bindings[] = {
      {.fragment = &fragments[0], .rule_set_index = 0},
      {.fragment = &fragments[1], .rule_set_index = 1},
  };
  const loom_target_contract_case_t contract_cases[] = {
      {
          .system = LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE,
          .binding_index = 0,
          .row_index = 0,
      },
      {
          .system = LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE_LOWERING_ONLY,
          .binding_index = 1,
          .row_index = 0,
      },
  };
  const SingleOpContract<kSourceOpKind> contract(
      contract_cases, IREE_ARRAYSIZE(contract_cases), bindings,
      IREE_ARRAYSIZE(bindings));
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  const loom_target_contract_query_environment_t environment = {
      .module = nullptr,
      .function = {},
      .target_facts = &target_facts,
  };
  loom_low_lower_rule_match_context_t match_context = {
      .bundle = &kTargetBundle,
  };
  loom_op_t op = {
      .kind = kSourceOpKind,
  };

  struct VisitState {
    const loom_low_lower_rule_t* expected_rule;
    int count;
  } lowering_state = {.expected_rule = &rule, .count = 0},
    target_query_state = {.expected_rule = &rule, .count = 0};
  const loom_low_lower_contract_case_callback_t callback = {
      .fn = [](void* user_data, uint16_t case_index,
               const loom_low_lower_contract_case_match_t* match,
               bool* out_stop) -> iree_status_t {
        auto* state = static_cast<VisitState*>(user_data);
        EXPECT_LT(case_index, 2);
        EXPECT_EQ(match->kind, LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE);
        EXPECT_EQ(match->selection.rule, state->expected_rule);
        ++state->count;
        *out_stop = false;
        return iree_ok_status();
      },
      .user_data = &lowering_state,
  };
  IREE_ASSERT_OK(loom_low_lower_visit_lowering_contract_cases(
      &environment, &options, &op,
      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_ALL, &match_context, callback,
      /*out_used_candidate_index=*/nullptr));
  EXPECT_EQ(lowering_state.count, 2);

  loom_low_lower_contract_case_callback_t target_query_callback = callback;
  target_query_callback.user_data = &target_query_state;
  IREE_ASSERT_OK(loom_low_lower_visit_target_contract_cases(
      &environment, &options, &op,
      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_ALL, &match_context,
      target_query_callback, /*out_used_candidate_index=*/nullptr));
  EXPECT_EQ(target_query_state.count, 0);

  for (bool defer_diagnostics : {false, true}) {
    loom_target_contract_query_environment_t query_environment = environment;
    query_environment.defer_rejection_diagnostics = defer_diagnostics;
    loom_target_contract_query_result_t result;
    IREE_ASSERT_OK(loom_low_lower_query_target_contract(
        &query_environment, &options, &op, &result));
    EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNHANDLED);
    EXPECT_EQ(result.rejection, nullptr);
  }
}

TEST(LowContractQueryTest, DeferredRejectionPreservesOwnershipDeclines) {
  const loom_low_lower_guard_payload_t guard_payload = {.u64 = 1};
  const loom_low_lower_guard_t guard = {
      .kind = LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
      .payload_ordinal = 1,
  };
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_rule_t rules[2] = {};
  const loom_low_lower_rule_set_t rule_set = {
      .rules = rules,
      .rule_count = IREE_ARRAYSIZE(rules),
      .guard_payloads = &guard_payload,
      .guard_payload_count = 1,
      .guards = &guard,
      .guard_count = 1,
      .guard_refs = &guard_ref,
      .guard_ref_count = 1,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};
  const loom_target_contract_descriptor_rule_t descriptor_rules[] = {{0}, {1}};
  const loom_target_contract_fragment_t fragment = {
      .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
      .descriptor_rule_count = IREE_ARRAYSIZE(descriptor_rules),
      .descriptor_rules = descriptor_rules,
  };
  const loom_target_contract_binding_t binding = {&fragment, 0};
  const loom_target_contract_case_t cases[] = {
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 0},
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 1},
      {LOOM_TARGET_CONTRACT_SYSTEM_UNSUPPORTED, 0,
       LOOM_TARGET_CONTRACT_ROW_NONE},
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .target_facts = &target_facts,
  };
  loom_op_t op = {.kind = kSourceOpKind};
  for (bool has_marker : {false, true}) {
    const SingleOpContract<kSourceOpKind> contract(cases, has_marker ? 3 : 2,
                                                   &binding, 1);
    const loom_low_lower_contract_query_options_t options = {
        .contract_index = contract.index(),
        .rule_sets = {.count = 1, .values = rule_sets},
        .accept_rule =
            {
                .fn = [](void*, const loom_low_lower_rule_selection_t*,
                         bool* out_accepted) -> iree_status_t {
                  *out_accepted = false;
                  return iree_ok_status();
                },
            },
    };
    for (bool second_rule_fails : {false, true}) {
      rules[1].guard_count = second_rule_fails ? 1 : 0;
      for (bool defer_diagnostics : {false, true}) {
        environment.defer_rejection_diagnostics = defer_diagnostics;
        loom_target_contract_query_result_t result;
        IREE_ASSERT_OK(loom_low_lower_query_target_contract(
            &environment, &options, &op, &result));
        EXPECT_EQ(result.outcome, has_marker || second_rule_fails
                                      ? LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED
                                      : LOOM_TARGET_CONTRACT_QUERY_UNHANDLED);
        EXPECT_EQ(result.rejection, nullptr);
      }
    }
  }
}

TEST(LowContractQueryTest, SingletonScopeSkipsGraphCasesBeforeMatching) {
  loom_low_lower_rule_t rules[2] = {};
  rules[1].source_node_span = LOOM_LOW_LOWER_SOURCE_NODE_SPAN(0, 1);
  const loom_low_lower_rule_set_t rule_set = {
      .rules = rules,
      .rule_count = IREE_ARRAYSIZE(rules),
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  const loom_target_contract_fragment_t fragment = {};
  const loom_target_contract_binding_t binding = {
      .fragment = &fragment,
      .rule_set_index = 0,
  };
  const loom_target_contract_case_t contract_cases[] = {
      {
          .system = LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE,
          .binding_index = 0,
          .row_index = 0,
      },
      {
          .system = LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE,
          .binding_index = 0,
          .row_index = 1,
      },
  };
  const SingleOpContract<kSourceOpKind> contract(
      contract_cases, IREE_ARRAYSIZE(contract_cases), &binding, 1,
      &kSourceOpKind, 1);
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  const loom_target_contract_query_environment_t environment = {
      .module = nullptr,
      .function = {},
      .target_facts = &target_facts,
  };
  loom_low_lower_rule_match_context_t match_context = {
      .bundle = &kTargetBundle,
  };
  loom_op_t op = {
      .kind = kSourceOpKind,
  };

  struct VisitState {
    uint16_t expected_case_index;
    const loom_low_lower_rule_t* expected_rule;
    int count;
  } state = {};
  const loom_low_lower_contract_case_callback_t callback = {
      .fn = [](void* user_data, uint16_t case_index,
               const loom_low_lower_contract_case_match_t* match,
               bool* out_stop) -> iree_status_t {
        auto* state = static_cast<VisitState*>(user_data);
        EXPECT_EQ(case_index, state->expected_case_index);
        EXPECT_EQ(match->kind, LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE);
        EXPECT_EQ(match->selection.rule, state->expected_rule);
        ++state->count;
        *out_stop = false;
        return iree_ok_status();
      },
      .user_data = &state,
  };

  state.expected_case_index = 0;
  state.expected_rule = &rules[0];
  IREE_ASSERT_OK(loom_low_lower_visit_lowering_contract_cases(
      &environment, &options, &op,
      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SINGLETON, &match_context, callback,
      /*out_used_candidate_index=*/nullptr));
  EXPECT_EQ(state.count, 1);
}

TEST_F(LowContractQuerySourceMemoryTest,
       SourceGraphScopeSkipsCaseWhenFirstEdgeCannotResolve) {
  const loom_value_id_t source_value = BuildOffsetConstant(0);
  const loom_op_t* source_op =
      loom_value_def_op(loom_module_value(module_, source_value));
  const loom_low_lower_value_ref_t value_refs[] = {
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
          .source_node_index = 0,
      },
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
          .source_node_index = 1,
      },
  };
  const loom_low_lower_source_node_t source_node = {
      .relation = LOOM_LOW_LOWER_SOURCE_NODE_EXCLUSIVE_USER,
      .parent_node_index = 0,
      .source_op_kind = LOOM_OP_INDEX_ASSUME,
      .parent_value_ref_index = 0,
      .node_value_ref_index = 1,
  };
  const loom_low_lower_rule_t rule = {
      .source_node_span = LOOM_LOW_LOWER_SOURCE_NODE_SPAN(0, 1),
  };
  const loom_low_lower_rule_set_t rule_set = {
      .rules = &rule,
      .rule_count = 1,
      .value_refs = value_refs,
      .value_ref_count = IREE_ARRAYSIZE(value_refs),
      .source_nodes = &source_node,
      .source_node_count = 1,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};
  const loom_target_contract_fragment_t fragment = {};
  const loom_target_contract_binding_t binding = {
      .fragment = &fragment,
      .rule_set_index = 0,
  };
  const loom_target_contract_case_t contract_case = {
      .system = LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE,
      .binding_index = 0,
      .row_index = 0,
  };
  const loom_op_kind_t root_op_kind = LOOM_OP_INDEX_CONSTANT;
  const SingleOpContract<LOOM_OP_INDEX_CONSTANT> contract(
      &contract_case, 1, &binding, 1, &root_op_kind, 1);
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  const loom_target_contract_query_environment_t environment = {
      .module = module_,
      .function = function_,
      .target_facts = &target_facts,
  };
  loom_low_lower_rule_match_context_t match_context = {
      .module = module_,
      .function = function_,
      .bundle = &kTargetBundle,
  };
  int visit_count = 0;
  const loom_low_lower_contract_case_callback_t callback = {
      .fn = [](void* user_data, uint16_t case_index,
               const loom_low_lower_contract_case_match_t* match,
               bool* out_stop) -> iree_status_t {
        (void)case_index;
        (void)match;
        ++*static_cast<int*>(user_data);
        *out_stop = false;
        return iree_ok_status();
      },
      .user_data = &visit_count,
  };

  IREE_ASSERT_OK(loom_low_lower_visit_lowering_contract_cases(
      &environment, &options, source_op,
      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SOURCE_GRAPH, &match_context, callback,
      /*out_used_candidate_index=*/nullptr));
  EXPECT_EQ(visit_count, 0);
}

TEST(LowContractQueryTest,
     ContractIndexDefersUnavailableDescriptorAcrossBindings) {
  const uint64_t feature_mask_words[] = {1};
  loom_low_descriptor_t descriptors[2] = {};
  loom_low_descriptor_layout_t layouts[] = {
      {.feature_mask_word_count = 1},
      {},
  };
  descriptors[1].layout_index = 1;
  loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = layouts,
      .descriptor_layout_count = IREE_ARRAYSIZE(layouts),
      .feature_mask_words = feature_mask_words,
      .feature_mask_word_count = IREE_ARRAYSIZE(feature_mask_words),
  };

  const loom_low_lower_emit_t emit = {
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
  };
  const loom_low_lower_emit_ref_t emit_refs[] = {
      {.descriptor_ref = 0, .emit_index = 0},
      {.descriptor_ref = 1, .emit_index = 0},
  };
  loom_low_lower_rule_t rules[2] = {};
  rules[0].emit_count = 1;
  rules[1].emit_count = 1;
  loom_low_lower_rule_set_t rule_set_storage[2] = {};
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(rule_set_storage); ++i) {
    rule_set_storage[i].rules = &rules[i];
    rule_set_storage[i].rule_count = 1;
    rule_set_storage[i].descriptor_ref_count = IREE_ARRAYSIZE(descriptors);
    rule_set_storage[i].emit_refs = &emit_refs[i];
    rule_set_storage[i].emit_ref_count = 1;
    rule_set_storage[i].emits = &emit;
    rule_set_storage[i].emit_count = 1;
  }
  const loom_low_lower_rule_set_t* rule_sets[] = {
      &rule_set_storage[0],
      &rule_set_storage[1],
  };

  const loom_target_contract_descriptor_rule_t descriptor_rule = {0};
  const loom_target_contract_fragment_t fragments[] = {
      {
          .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
          .descriptor_rule_count = 1,
          .descriptor_rules = &descriptor_rule,
      },
      {
          .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
          .descriptor_rule_count = 1,
          .descriptor_rules = &descriptor_rule,
      },
  };
  const loom_target_contract_binding_t bindings[] = {
      {&fragments[0], 0},
      {&fragments[1], 1},
  };
  const loom_target_contract_case_t cases[] = {
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 0},
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 1, 0},
  };
  loom_target_contract_op_entry_t entries[(kSourceOpKind & 0xFF) + 1] = {};
  entries[loom_op_dialect_index(kSourceOpKind)] = {
      0,
      IREE_ARRAYSIZE(cases),
      IREE_ARRAYSIZE(cases),
  };
  const loom_target_contract_dialect_table_t dialect = {IREE_ARRAYSIZE(entries),
                                                        entries};
  const loom_target_contract_index_t index = {
      kSourceOpKind >> 8,
      1,
      &dialect,
      IREE_ARRAYSIZE(cases),
      cases,
      IREE_ARRAYSIZE(bindings),
      0,
      bindings,
      nullptr,
      nullptr,
  };
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = &index,
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
      .map_value = {},
      .can_materialize = {},
      .descriptor_ref =
          {
              .fn = ResolveDescriptorFromArray,
              .user_data = descriptors,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .target_facts = &target_facts,
      .descriptor_set = &descriptor_set,
  };
  loom_op_t op = {
      .kind = kSourceOpKind,
  };

  for (bool defer_diagnostics : {false, true}) {
    environment.defer_rejection_diagnostics = defer_diagnostics;
    layouts[1].feature_mask_word_count = 0;
    loom_target_contract_query_result_t result;
    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_query_target_contract_with_selection(
        &environment, &options, &op, &result, &selection));
    EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
    EXPECT_EQ(result.case_index, 1);
    EXPECT_EQ(result.selected_descriptor, &descriptors[1]);
    EXPECT_EQ(selection.rule, &rules[1]);

    layouts[1].feature_mask_word_count = 1;
    IREE_ASSERT_OK(loom_low_lower_query_target_contract_with_selection(
        &environment, &options, &op, &result, &selection));
    EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
    EXPECT_EQ(result.case_index, 0);
    EXPECT_EQ(result.selected_descriptor, &descriptors[0]);
    EXPECT_EQ(selection.rule, &rules[0]);
  }
}

TEST(LowContractQueryTest, IndexedMissReplaysCompleteOrderForBestRejection) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_low_lower_guard_payload_t guard_payload = {};
  guard_payload.u64 = 1;
  loom_low_lower_guard_t guards[2] = {};
  guards[0].kind = LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL;
  guards[0].diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  guards[0].payload_ordinal = 1;
  guards[1] = guards[0];
  guards[1].diagnostic_index = 0;
  const loom_low_lower_guard_ref_t guard_refs[] = {0, 1};
  loom_low_lower_rule_t rules[2] = {};
  rules[0].guard_count = 1;
  rules[1].guard_start = 1;
  rules[1].guard_count = 1;
  const loom_low_lower_diagnostic_t diagnostic = {
      .error_ref = LOOM_ERR_TARGET_003_REF,
  };
  loom_low_lower_rule_set_t rule_set = {
      .rules = rules,
      .rule_count = IREE_ARRAYSIZE(rules),
      .guard_payloads = &guard_payload,
      .guard_payload_count = 1,
      .guards = guards,
      .guard_count = IREE_ARRAYSIZE(guards),
      .guard_refs = guard_refs,
      .guard_ref_count = IREE_ARRAYSIZE(guard_refs),
      .diagnostics = &diagnostic,
      .diagnostic_count = 1,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  const loom_target_contract_descriptor_rule_t descriptor_rules[] = {{0}, {1}};
  const loom_target_contract_fragment_t fragment = {
      .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
      .descriptor_rule_count = IREE_ARRAYSIZE(descriptor_rules),
      .descriptor_rules = descriptor_rules,
  };
  const loom_target_contract_binding_t binding = {&fragment, 0};
  const loom_target_contract_case_t cases[] = {
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 0},
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 1},
  };
  loom_target_contract_op_entry_t entries[(kSourceOpKind & 0xFF) + 1] = {};
  entries[loom_op_dialect_index(kSourceOpKind)] = {
      0,
      2,
      2,
  };
  const loom_target_contract_dialect_table_t dialect = {IREE_ARRAYSIZE(entries),
                                                        entries};
  // The generated candidate row visits only case 0. Both cases reject, so the
  // query must replay the complete authored order and retain case 1's better
  // diagnostic.
  const uint32_t selection_data[] = {
      1, 2, 1, 0, 1u | (static_cast<uint32_t>(kSourceOpKind) << 16), 0,
  };
  const loom_target_contract_index_t index = {
      kSourceOpKind >> 8,
      1,
      &dialect,
      IREE_ARRAYSIZE(cases),
      cases,
      1,
      0,
      &binding,
      selection_data,
      nullptr,
  };
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = &index,
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .target_facts = &target_facts,
      .arena = &arena,
  };
  loom_op_t op = {.kind = kSourceOpKind};
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();

  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      &op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(result.case_index, 1);
  EXPECT_EQ(result.rule_set_index, 0);
  EXPECT_EQ(result.diagnostic_index, 0);
  EXPECT_NE(result.rejection, nullptr);
  if (result.rejection != nullptr) {
    EXPECT_EQ(result.rejection->error_ref, LOOM_ERR_TARGET_003_REF);
  }

  environment.defer_rejection_diagnostics = true;
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      &op, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(result.case_index, UINT16_MAX);
  EXPECT_EQ(result.diagnostic_index, UINT16_MAX);
  EXPECT_EQ(result.rejection, nullptr);

  // Even an empty exact-key candidate list preserves the rejected-family
  // outcome without evaluating excluded rules to choose their diagnostics.
  const uint32_t empty_selection_data[] = {
      1, 2, 1, 0, static_cast<uint32_t>(kSourceOpKind) << 16,
  };
  loom_target_contract_index_t empty_index = index;
  empty_index.selection_data = empty_selection_data;
  loom_low_lower_contract_query_options_t empty_options = options;
  empty_options.contract_index = &empty_index;
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(
      &environment, &empty_options, &op, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(result.rejection, nullptr);

  environment.defer_rejection_diagnostics = false;
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(
      &environment, &empty_options, &op, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(result.case_index, 1);
  EXPECT_EQ(result.diagnostic_index, 0);
  EXPECT_NE(result.rejection, nullptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST_F(LowContractQuerySourceMemoryTest,
       ScopedValueTypesRetainProjectedProducerClosureRequirement) {
  const loom_type_t authored_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(64), 0);
  loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(loom_func_like_body(function_)),
      authored_type, &argument));
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_vector_negf_build(&builder_, /*instance_flags=*/0,
                                        argument, authored_type,
                                        LOOM_LOCATION_UNKNOWN, &source_op));

  const loom_low_lower_value_ref_t value_refs[] = {
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
          .source_node_index = 0,
          .index = 0,
      },
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND,
          .source_node_index = 0,
          .index = 0,
      },
  };
  loom_low_lower_type_pattern_t type_patterns[2] = {};
  for (loom_low_lower_type_pattern_t& type_pattern : type_patterns) {
    type_pattern.flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                         LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                         LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                         LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0;
    type_pattern.type_kind = LOOM_TYPE_VECTOR;
    type_pattern.rank = 1;
    type_pattern.element_type_mask =
        LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32);
  }
  type_patterns[0].shape.exact.dim0 = 64;
  type_patterns[1].shape.exact.dim0 = 16;
  loom_low_lower_guard_t guards[2] = {};
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(guards); ++i) {
    guards[i].kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE;
    guards[i].diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    guards[i].selector.value.value_ref_index = i;
    guards[i].selector.value.parameter_index = i;
  }
  const loom_low_lower_guard_ref_t guard_refs[] = {0, 1};
  loom_low_lower_rule_t rules[2] = {};
  rules[0].guard_count = 1;
  rules[1].guard_start = 1;
  rules[1].guard_count = 1;
  const uint64_t rule_root_producer_closure_bits[] = {UINT64_C(1) << 1};
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_VECTOR_NEGF,
      .rule_start = 0,
      .rule_count = 2,
  };
  loom_low_lower_rule_set_t rule_set = {
      .spans = &span,
      .span_count = 1,
      .rules = rules,
      .rule_root_producer_closure_bits = rule_root_producer_closure_bits,
      .rule_count = IREE_ARRAYSIZE(rules),
      .type_patterns = type_patterns,
      .type_pattern_count = IREE_ARRAYSIZE(type_patterns),
      .value_refs = value_refs,
      .value_ref_count = IREE_ARRAYSIZE(value_refs),
      .guards = guards,
      .guard_count = IREE_ARRAYSIZE(guards),
      .guard_refs = guard_refs,
      .guard_ref_count = IREE_ARRAYSIZE(guard_refs),
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};
  const loom_target_contract_descriptor_rule_t descriptor_rules[] = {{0}, {1}};
  const loom_target_contract_fragment_t fragment = {
      .flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY,
      .descriptor_rule_count = IREE_ARRAYSIZE(descriptor_rules),
      .descriptor_rules = descriptor_rules,
  };
  const loom_target_contract_binding_t binding = {&fragment, 0};
  const loom_target_contract_case_t cases[] = {
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 0},
      {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 1},
  };
  loom_target_contract_op_entry_t entries[(LOOM_OP_VECTOR_NEGF & 0xFF) + 1] =
      {};
  entries[loom_op_dialect_index(LOOM_OP_VECTOR_NEGF)] = {0, 2, 2};
  const loom_target_contract_dialect_table_t dialect = {IREE_ARRAYSIZE(entries),
                                                        entries};
  const uint32_t selection_data[] = {
      1u | (2u << 16),
      1u,
      3u,
      2u,
      static_cast<uint32_t>(LOOM_OP_VECTOR_NEGF) << 16,
      (2u << 28) | (LOOM_SCALAR_TYPE_F32 << 16) | 16u,
      1u << 16,
      (2u << 28) | (LOOM_SCALAR_TYPE_F32 << 16) | 64u,
      1u | (1u << 16),
      1u,
      0u,
  };
  const loom_target_contract_index_t contract_index = {
      LOOM_OP_VECTOR_NEGF >> 8,
      1,
      &dialect,
      IREE_ARRAYSIZE(cases),
      cases,
      1,
      0,
      &binding,
      selection_data,
      nullptr,
  };
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = &contract_index,
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &facts, argument,
      (loom_value_fact_exact_lane_origin_t){
          .source_value_id = argument,
          .source_lane_offset = 0,
          .source_lane_stride = 1,
      }));
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .module = module_,
      .function = function_,
      .target_facts = &target_facts,
      .fact_table = &facts,
  };

  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      source_op, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(result.case_index, 0);
  EXPECT_EQ(result.projection_flags, LOOM_TARGET_CONTRACT_PROJECTION_FLAG_NONE);

  environment.vector_lane_projection = {
      .source_lane_count = 64,
      .projected_lane_count = 16,
  };
  result = loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      source_op, &result));
  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(result.case_index, 1);
  EXPECT_TRUE(iree_all_bits_set(
      result.projection_flags,
      LOOM_TARGET_CONTRACT_PROJECTION_FLAG_ROOT_PRODUCER_CLOSURE));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, loom_vector_negf_result(source_op)),
      authored_type));
}

TEST(LowContractQueryTest, TargetSubgroupSizeRangeRequiresKnownInRangeSize) {
  loom_low_lower_rule_descriptor_ref_t descriptor_ref = {
      .stable_id = kDescriptorId,
  };
  loom_low_lower_guard_t guard = {};
  loom_low_lower_guard_payload_t guard_payload = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_TARGET_SUBGROUP_SIZE_RANGE;
  guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  guard.payload_ordinal = 1;
  guard_payload.i64_range.minimum = 1;
  guard_payload.i64_range.maximum = 32;
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_emit_t emit = {
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
  };
  const loom_low_lower_emit_ref_t emit_ref = {
      .descriptor_ref = 0,
      .emit_index = 0,
  };
  loom_low_lower_rule_t rule = {
      .emit_count = 1,
      .guard_count = 1,
  };
  loom_low_lower_rule_set_t rule_set = {
      .rules = &rule,
      .rule_count = 1,
      .descriptor_refs = &descriptor_ref,
      .descriptor_ref_count = 1,
      .guard_payloads = &guard_payload,
      .guard_payload_count = 1,
      .guards = &guard,
      .guard_count = 1,
      .guard_refs = &guard_ref,
      .guard_ref_count = 1,
      .emit_refs = &emit_ref,
      .emit_ref_count = 1,
      .emits = &emit,
      .emit_count = 1,
      .string_pool = kRuleStringPool,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  SingleOpContract<kSourceOpKind> contract;
  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
      .map_value = {},
      .can_materialize = {},
      .descriptor_ref =
          {
              .fn = ResolveTestDescriptorRef,
              .user_data = nullptr,
          },
  };
  loom_op_t op = {.kind = kSourceOpKind};

  const auto query = [&](uint32_t subgroup_size) {
    loom_target_facts_t target_facts = MakeTargetFacts();
    loom_target_snapshot_t snapshot = {.subgroup_size = subgroup_size};
    target_facts.storage.bundle.snapshot = &snapshot;
    loom_target_contract_query_environment_t environment = {
        .target_facts = &target_facts,
        .descriptor_set = &kDescriptorSet,
    };
    loom_target_contract_query_result_t result =
        loom_target_contract_query_result_empty();
    IREE_EXPECT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                        &op, &result));
    return result.outcome;
  };

  EXPECT_EQ(query(0), LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(query(1), LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(query(32), LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(query(33), LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
}

TEST(LowContractQueryTest, ContractIndexDescriptorRuleReportsRejectedCase) {
  loom_low_lower_guard_t guard = {
      .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
      .attr_kind = LOOM_ATTR_I64,
      .diagnostic_index = 0,
  };
  const loom_low_lower_diagnostic_param_kind_t diagnostic_param_kinds[] = {
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
      LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
  };
  const loom_low_lower_diagnostic_value_t diagnostic_param_values[] = {
      {},
      {},
      {},
      {},
      {.string_value_ref = kRuleStringSource},
      {.string_value_ref = kRuleStringField},
      {.string_value_ref = kRuleStringValue},
      {.string_value_ref = kRuleStringAttrKind},
  };
  const loom_low_lower_diagnostic_param_ref_t diagnostic_param_refs[] = {
      0, 1, 2, 3, 4, 5, 6, 7};
  const loom_low_lower_guard_ref_t guard_refs[] = {0};
  loom_low_lower_diagnostic_t diagnostic = {
      .error_ref = LOOM_ERR_TARGET_003_REF,
      .param_count = IREE_ARRAYSIZE(diagnostic_param_kinds),
  };
  loom_low_lower_rule_t rule = {.guard_count = 1};
  loom_low_lower_rule_set_t rule_set = {
      .rules = &rule,
      .rule_count = 1,
      .diagnostic_params =
          {
              .kinds = diagnostic_param_kinds,
              .values = diagnostic_param_values,
              .count = IREE_ARRAYSIZE(diagnostic_param_kinds),
          },
      .diagnostic_param_refs = diagnostic_param_refs,
      .diagnostic_param_ref_count = IREE_ARRAYSIZE(diagnostic_param_refs),
      .guards = &guard,
      .guard_count = 1,
      .guard_refs = guard_refs,
      .guard_ref_count = IREE_ARRAYSIZE(guard_refs),
      .diagnostics = &diagnostic,
      .diagnostic_count = 1,
      .string_pool = kRuleStringPool,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  SingleOpContract<kSourceOpKind> contract;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .target_facts = &target_facts,
      .arena = &arena,
  };
  loom_op_t op = {.kind = kSourceOpKind};
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, loom_func_like_body(function_), &analysis_arena_,
      &value_domain));
  loom_symbolic_expr_context_t expression_context = {};
  loom_symbolic_expr_context_initialize(module_, &value_domain, &facts,
                                        &analysis_arena_, &expression_context);
  loom_view_region_table_t view_regions = {};
  IREE_ASSERT_OK(loom_view_region_table_initialize(
      &value_domain, &expression_context, &view_regions));
  IREE_ASSERT_OK(loom_view_region_table_analyze(&view_regions));
  loom_low_lower_source_memory_t source_memory = {
      .operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD,
      .root_kind = LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_BLOCK_ARGUMENT,
      .dynamic_term_count = 1,
      .dynamic_index_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
      .memory_space_mask = LOOM_LOW_LOWER_MEMORY_SPACE_UNKNOWN |
                           LOOM_LOW_LOWER_MEMORY_SPACE_GENERIC |
                           LOOM_LOW_LOWER_MEMORY_SPACE_GLOBAL |
                           LOOM_LOW_LOWER_MEMORY_SPACE_WORKGROUP |
                           LOOM_LOW_LOWER_MEMORY_SPACE_PRIVATE |
                           LOOM_LOW_LOWER_MEMORY_SPACE_CONSTANT |
                           LOOM_LOW_LOWER_MEMORY_SPACE_DESCRIPTOR,
      .vector_lane_count = 1,
      .element_byte_count = 4,
  };
  loom_low_lower_source_memory_shape_t source_memory_shape = {
      .vector_lane_byte_stride = 4,
      .static_byte_offset_minimum = INT64_MIN,
      .static_byte_offset_maximum = INT64_MAX,
      .dynamic_byte_stride = 4,
  };
  loom_low_lower_source_memory_diagnostics_t source_memory_diagnostics = {};
  for (uint16_t& diagnostic_index :
       source_memory_diagnostics.rejection_diagnostic_indices) {
    diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  }
  loom_low_lower_emit_t emit = {
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
      .source_memory_ordinal = 1,
  };
  const loom_low_lower_emit_ref_t emit_ref = {
      .descriptor_ref = LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE,
      .emit_index = 0,
  };
  loom_low_lower_rule_t rule = {.emit_count = 1};

  loom_low_lower_rule_set_t rule_set = {
      .rules = &rule,
      .rule_count = 1,
      .source_memories = &source_memory,
      .source_memory_count = 1,
      .source_memory_shapes = &source_memory_shape,
      .source_memory_shape_count = 1,
      .source_memory_diagnostics = &source_memory_diagnostics,
      .source_memory_diagnostic_count = 1,
      .emit_refs = &emit_ref,
      .emit_ref_count = 1,
      .emits = &emit,
      .emit_count = 1,
  };
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  SingleOpContract<LOOM_OP_VECTOR_LOAD> contract;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_lower_contract_query_options_t options = {
      .contract_index = contract.index(),
      .rule_sets =
          {
              .count = IREE_ARRAYSIZE(rule_sets),
              .values = rule_sets,
          },
  };
  const loom_target_facts_t target_facts = MakeTargetFacts();
  loom_target_contract_query_environment_t environment = {
      .module = module_,
      .function = function_,
      .target_facts = &target_facts,
      .fact_table = &facts,
      .value_domain = &value_domain,
      .view_regions = &view_regions,
      .arena = &arena,
  };
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      load_op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(result.rule_index, 0);

  loom_local_value_domain_release(&value_domain);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
