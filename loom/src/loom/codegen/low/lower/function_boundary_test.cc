// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/function_boundary.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class LowLowerFunctionBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("function_boundary_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));

    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    target_facts_.fact_type = &loom_test_target_fact_type;
    target_facts_.storage.bundle = *loom_test_target_bundles.values[1];
    policy_ = *loom_test_low_lower_policy();
    policy_.import_decl_kind = LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE;
    options_.target_ref = loom_symbol_ref_null();
    options_.target_facts = &target_facts_;
    options_.descriptor_registry = &descriptor_registry_.registry;
    options_.policy = &policy_;
    options_.fact_table = &fact_table_;
    mapping_context_.module = module_;
    mapping_context_.options = &options_;
    mapping_context_.policy = &policy_;
    mapping_context_.result = &result_;
    mapping_context_.lowering.fact_table = &fact_table_;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_source_lowering(
        options_.descriptor_registry, loom_target_facts_bundle(&target_facts_),
        &mapping_context_.descriptor_set));
    iree_arena_initialize(&block_pool_, &mapping_context_.function_arena);
    iree_arena_initialize(&block_pool_, &mapping_context_.emission_arena);
  }

  void TearDown() override {
    iree_arena_deinitialize(&mapping_context_.emission_arena);
    iree_arena_deinitialize(&mapping_context_.function_arena);
    loom_low_lower_result_deinitialize(&result_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_EXPECT_OK(loom_module_intern_string(module_, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_EXPECT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t symbol = {0, symbol_id};
    return symbol;
  }

  loom_string_id_t InternString(iree_string_view_t value) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_EXPECT_OK(loom_module_intern_string(module_, value, &string_id));
    return string_id;
  }

  void SetValueName(loom_value_id_t value_id, iree_string_view_t name) {
    IREE_ASSERT_OK(
        loom_module_set_value_name(module_, value_id, InternString(name)));
  }

  void ComputeFacts(loom_func_like_t function) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function));
  }

  void SetArgumentPredicate(loom_func_like_t function,
                            loom_value_id_t argument) {
    loom_predicate_t* predicate = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(&module_->arena, sizeof(*predicate),
                                       reinterpret_cast<void**>(&predicate)));
    *predicate = {};
    predicate->kind = LOOM_PREDICATE_GE;
    predicate->arg_count = 2;
    predicate->arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
    predicate->args[0] = argument;
    predicate->args[1] = 0;
    IREE_ASSERT_OK(loom_op_set_attr(module_, function.op,
                                    function.vtable->predicates_attr_index,
                                    loom_attr_predicate_list(predicate, 1)));
  }

  void ExpectRegister(loom_value_id_t value_id, uint16_t register_class_id) {
    const loom_type_t type = loom_module_value_type(module_, value_id);
    ASSERT_TRUE(loom_low_type_is_register(type));
    EXPECT_EQ(loom_low_register_type_descriptor_set_stable_id(type),
              result_.descriptor_set->stable_id);
    EXPECT_EQ(loom_low_register_type_class_id(type), register_class_id);
    EXPECT_EQ(loom_low_register_type_unit_count(type), 1u);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_policy_t policy_ = {};
  loom_low_lower_options_t options_ = {};
  loom_low_lower_result_t result_ = {};
  loom_low_lower_context_t mapping_context_ = {};
};

TEST_F(LowLowerFunctionBoundaryTest,
       DefinitionSeparatesDirectAndResourceArguments) {
  loom_builder_t module_builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &module_builder);
  const loom_symbol_ref_t symbol = AddSymbol(IREE_SV("identity"));
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t argument_types[] = {loom_type_buffer(), i32_type};
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
      /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
      argument_types, IREE_ARRAYSIZE(argument_types), &i32_type, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &source_op));
  const loom_func_like_t source_function =
      loom_func_like_cast(module_, source_op);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_function, &source_argument_count);
  ASSERT_EQ(source_argument_count, 2u);
  SetValueName(source_arguments[0], IREE_SV("storage"));
  SetValueName(source_arguments[1], IREE_SV("value"));
  SetValueName(loom_op_const_results(source_op)[0], IREE_SV("result"));
  SetArgumentPredicate(source_function, source_arguments[1]);

  loom_builder_t body_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_func_like_body(source_function)),
      &body_builder);
  body_builder.ip.parent_op = source_op;
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &source_arguments[1], 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  ComputeFacts(source_function);
  mapping_context_.source_function = source_function;
  loom_low_lower_abi_argument_t resource_argument = {};
  resource_argument.resource_build_flags =
      LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT;
  resource_argument.resource_extent = 64;
  IREE_ASSERT_OK(loom_low_lower_query_argument(
      &mapping_context_, 0, source_arguments[0], &resource_argument));
  EXPECT_EQ(resource_argument.kind, LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE);
  EXPECT_EQ(resource_argument.resource_import_kind,
            LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER);
  EXPECT_EQ(resource_argument.resource_index, 0);
  EXPECT_EQ(resource_argument.resource_build_flags, 0u);
  EXPECT_TRUE(loom_type_is_buffer(resource_argument.resource_source_type));
  loom_low_lower_abi_argument_t direct_argument = {};
  IREE_ASSERT_OK(loom_low_lower_query_argument(
      &mapping_context_, 1, source_arguments[1], &direct_argument));
  EXPECT_EQ(direct_argument.kind, LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT);
  EXPECT_TRUE(loom_low_type_is_register(direct_argument.abi_type));
  EXPECT_EQ(result_.error_count, 0u);

  IREE_ASSERT_OK(
      loom_low_lower_function(module_, source_function, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  ASSERT_NE(result_.descriptor_set, nullptr);
  ASSERT_NE(result_.low_func_op, nullptr);
  ASSERT_TRUE(loom_low_func_def_isa(result_.low_func_op));

  const loom_func_like_t low_function =
      loom_func_like_cast(module_, result_.low_func_op);
  uint16_t low_argument_count = 0;
  const loom_value_id_t* low_arguments =
      loom_func_like_arg_ids(low_function, &low_argument_count);
  ASSERT_EQ(low_argument_count, 1u);
  ExpectRegister(low_arguments[0], TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, low_arguments[0]), IREE_SV("value")));

  ASSERT_EQ(result_.low_func_op->result_count, 1u);
  const loom_value_id_t low_result =
      loom_op_const_results(result_.low_func_op)[0];
  ExpectRegister(low_result, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, low_result), IREE_SV("result")));

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(low_function, &predicate_count);
  ASSERT_EQ(predicate_count, 1u);
  ASSERT_NE(predicates, nullptr);
  EXPECT_EQ(predicates[0].args[0], low_arguments[0]);
  EXPECT_TRUE(loom_value_has_attribute_uses(
      loom_module_value(module_, low_arguments[0])));

  loom_op_t* resource_op = nullptr;
  iree_host_size_t resource_count = 0;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(
      loom_region_entry_block(loom_func_like_body(low_function)), op) {
    if (loom_low_resource_isa(op)) {
      resource_op = op;
      ++resource_count;
    }
  }
  ASSERT_EQ(resource_count, 1u);
  ASSERT_NE(resource_op, nullptr);
  EXPECT_EQ(loom_low_resource_import_kind(resource_op),
            LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER);
  EXPECT_EQ(loom_low_resource_index(resource_op), 0);
  const loom_type_id_t source_type_id =
      loom_low_resource_source_type(resource_op);
  ASSERT_LT(source_type_id, module_->types.count);
  EXPECT_TRUE(loom_type_is_buffer(
      loom_type_table_get(&module_->types, source_type_id)));
  const loom_value_id_t resource_result = loom_low_resource_result(resource_op);
  ExpectRegister(resource_result, TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR);
  EXPECT_TRUE(
      loom_type_equal(resource_argument.abi_type,
                      loom_module_value_type(module_, resource_result)));
  EXPECT_TRUE(
      loom_type_equal(direct_argument.abi_type,
                      loom_module_value_type(module_, low_arguments[0])));
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, resource_result), IREE_SV("storage")));

  EXPECT_EQ(result_.low_func_ref.module_id, symbol.module_id);
  EXPECT_EQ(result_.low_func_ref.symbol_id, symbol.symbol_id);
  EXPECT_EQ(module_->symbols.entries[symbol.symbol_id].defining_op,
            result_.low_func_op);
}

TEST_F(LowLowerFunctionBoundaryTest,
       ImportDeclarationPreservesCallableContract) {
  loom_builder_t module_builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &module_builder);
  const loom_symbol_ref_t symbol = AddSymbol(IREE_SV("external"));
  const loom_string_id_t import_module = InternString(IREE_SV("kernel_lib"));
  const loom_string_id_t import_symbol = InternString(IREE_SV("extern_f"));
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_func_decl_build(
      &module_builder,
      LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE |
          LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_SYMBOL,
      /*visibility=*/0, /*retain=*/0, import_module, import_symbol,
      /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, &i32_type,
      1, &i32_type, 1, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
      &source_op));
  const loom_func_like_t source_declaration =
      loom_func_like_cast(module_, source_op);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_declaration, &source_argument_count);
  ASSERT_EQ(source_argument_count, 1u);
  SetValueName(source_arguments[0], IREE_SV("input"));
  SetValueName(loom_op_const_results(source_op)[0], IREE_SV("output"));
  SetArgumentPredicate(source_declaration, source_arguments[0]);

  IREE_ASSERT_OK(loom_low_lower_declaration(module_, source_declaration,
                                            &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  ASSERT_NE(result_.descriptor_set, nullptr);
  ASSERT_NE(result_.low_func_op, nullptr);
  ASSERT_TRUE(loom_low_func_decl_isa(result_.low_func_op));

  const loom_value_slice_t low_arguments =
      loom_low_func_decl_args(result_.low_func_op);
  const loom_value_slice_t low_results =
      loom_low_func_decl_results(result_.low_func_op);
  ASSERT_EQ(low_arguments.count, 1u);
  ASSERT_EQ(low_results.count, 1u);
  ExpectRegister(low_arguments.values[0], TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  ExpectRegister(low_results.values[0], TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, low_arguments.values[0]),
      IREE_SV("input")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, low_results.values[0]),
      IREE_SV("output")));

  const loom_func_like_t low_declaration =
      loom_func_like_cast(module_, result_.low_func_op);
  EXPECT_EQ(loom_func_like_import_module(low_declaration), import_module);
  EXPECT_EQ(loom_func_like_import_symbol(low_declaration),
            loom_low_func_decl_code_symbol(result_.low_func_op));
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(low_declaration, &predicate_count);
  ASSERT_EQ(predicate_count, 1u);
  ASSERT_NE(predicates, nullptr);
  EXPECT_EQ(predicates[0].args[0], low_arguments.values[0]);
  EXPECT_TRUE(loom_value_has_attribute_uses(
      loom_module_value(module_, low_arguments.values[0])));

  EXPECT_EQ(loom_low_func_decl_import_kind(result_.low_func_op),
            LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE);
  const loom_string_id_t code_symbol =
      loom_low_func_decl_code_symbol(result_.low_func_op);
  ASSERT_LT(code_symbol, module_->strings.count);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&module_->strings, code_symbol),
      IREE_SV("extern_f")));
  const loom_string_id_t descriptor_set =
      loom_low_func_decl_descriptor_set(result_.low_func_op);
  ASSERT_LT(descriptor_set, module_->strings.count);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&module_->strings, descriptor_set),
      IREE_SV("test.low.core")));

  const loom_symbol_ref_t low_callee =
      loom_low_func_decl_callee(result_.low_func_op);
  EXPECT_EQ(low_callee.module_id, symbol.module_id);
  EXPECT_EQ(low_callee.symbol_id, symbol.symbol_id);
  EXPECT_EQ(module_->symbols.entries[symbol.symbol_id].defining_op,
            result_.low_func_op);
}

class LowLowerResultMappingTest : public LowLowerFunctionBoundaryTest,
                                  public ::testing::WithParamInterface<bool> {};

TEST_P(LowLowerResultMappingTest, DefinitionConsumesPreparedResultTypes) {
  const uint16_t result_count = GetParam() ? 2 : 0;
  const loom_type_t argument_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  const loom_type_t result_types[] = {argument_types[1], argument_types[0]};
  const loom_symbol_ref_t symbol = AddSymbol(IREE_SV("permute"));
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
      /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
      argument_types, IREE_ARRAYSIZE(argument_types), result_types,
      result_count, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &source_op));
  mapping_context_.source_function = loom_func_like_cast(module_, source_op);
  loom_region_t* body = loom_func_like_body(mapping_context_.source_function);
  loom_block_t* entry = loom_region_entry_block(body);
  const loom_value_id_t returned_values[] = {entry->arg_ids[1],
                                             entry->arg_ids[0]};
  loom_builder_initialize(module_, &module_->arena, entry, &builder);
  builder.ip.parent_op = source_op;
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, returned_values, result_count,
                                        LOOM_LOCATION_UNKNOWN, &return_op));
  ComputeFacts(mapping_context_.source_function);

  uint32_t result_query_count = 0;
  policy_.map_value = {
      +[](void* user_data, loom_low_lower_context_t* context,
          const loom_op_t* source_op, loom_value_id_t source_value,
          loom_type_t source_type, loom_type_t* out_low_type) -> iree_status_t {
        (void)source_value;
        if (loom_func_return_isa(source_op)) {
          ++*static_cast<uint32_t*>(user_data);
        }
        return context->policy->map_type.fn(context->policy->map_type.user_data,
                                            context, source_op, source_type,
                                            out_low_type);
      },
      &result_query_count,
  };
  IREE_ASSERT_OK(loom_low_lower_function_boundary_validate(&mapping_context_));
  IREE_ASSERT_OK(loom_low_lower_function_boundary_observe_return(
      &mapping_context_, return_op));
  IREE_ASSERT_OK(loom_low_lower_function_boundary_finalize(&mapping_context_));
  ASSERT_EQ(result_.error_count, 0u);
  EXPECT_EQ(result_query_count, result_count);
  EXPECT_EQ(mapping_context_.lowering.result_types != nullptr, GetParam());

  loom_low_lower_emission_scope_begin(&mapping_context_);
  IREE_ASSERT_OK(
      loom_low_lower_function_boundary_create(&mapping_context_, body, symbol));
  loom_low_lower_emission_scope_end(&mapping_context_);
  ASSERT_EQ(result_.error_count, 0u);
  EXPECT_EQ(result_query_count, result_count);
  ASSERT_EQ(result_.low_func_op->result_count, result_count);
  for (uint16_t i = 0; i < result_count; ++i) {
    EXPECT_TRUE(loom_type_equal(
        mapping_context_.lowering.result_types[i],
        loom_module_value_type(module_,
                               loom_op_const_results(result_.low_func_op)[i])));
  }
}

INSTANTIATE_TEST_SUITE_P(EmptyAndMultiple, LowLowerResultMappingTest,
                         ::testing::Bool());

class LowLowerArgumentQueryTest : public LowLowerFunctionBoundaryTest,
                                  public ::testing::WithParamInterface<bool> {};

TEST_P(LowLowerArgumentQueryTest, OnlyRequiredArgumentsEmitDiagnostics) {
  if (!GetParam()) {
    policy_.map_argument = {};
  }
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  const loom_type_t argument_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_F64),
  };
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
      /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
      AddSymbol(IREE_SV("arguments")), argument_types,
      IREE_ARRAYSIZE(argument_types), nullptr, 0, nullptr, 0, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &source_op));
  const loom_func_like_t function = loom_func_like_cast(module_, source_op);
  loom_region_t* body = loom_func_like_body(function);
  loom_builder_initialize(module_, &module_->arena,
                          loom_region_entry_block(body), &builder);
  builder.ip.parent_op = source_op;
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));
  ComputeFacts(function);
  mapping_context_.source_function = function;
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  ASSERT_EQ(argument_count, 2u);

  loom_low_lower_abi_argument_t native_argument = {};
  IREE_ASSERT_OK(loom_low_lower_query_argument(&mapping_context_, 0,
                                               arguments[0], &native_argument));
  EXPECT_TRUE(loom_low_type_is_register(native_argument.abi_type));
  loom_low_lower_abi_argument_t absent_argument = native_argument;
  IREE_ASSERT_OK(loom_low_lower_query_argument(&mapping_context_, 1,
                                               arguments[1], &absent_argument));
  EXPECT_EQ(loom_type_kind(absent_argument.abi_type), LOOM_TYPE_NONE);
  EXPECT_EQ(result_.error_count, 0u);
  EXPECT_EQ(mapping_context_.lowering.argument_map, nullptr);

  IREE_ASSERT_OK(loom_low_lower_function_boundary_validate(&mapping_context_));
  EXPECT_EQ(result_.error_count, 1u);
  EXPECT_TRUE(
      loom_type_equal(native_argument.abi_type,
                      mapping_context_.lowering.argument_map[0].abi_type));
  EXPECT_EQ(loom_type_kind(mapping_context_.lowering.argument_map[1].abi_type),
            LOOM_TYPE_NONE);
}

INSTANTIATE_TEST_SUITE_P(DefaultAndTargetMapping, LowLowerArgumentQueryTest,
                         ::testing::Bool());

}  // namespace
}  // namespace loom
