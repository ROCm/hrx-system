// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/function_boundary.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
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
  }

  void TearDown() override {
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
    loom_op_attrs(function.op)[function.vtable->predicates_attr_index] =
        loom_attr_predicate_list(predicate, 1);
    IREE_ASSERT_OK(
        loom_module_note_op_attribute_value_refs(module_, function.op));
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
  EXPECT_TRUE(loom_module_value_has_predicate_attribute_uses(module_,
                                                             low_arguments[0]));

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
  EXPECT_TRUE(loom_type_is_buffer(module_->types.entries[source_type_id]));
  const loom_value_id_t resource_result = loom_low_resource_result(resource_op);
  ExpectRegister(resource_result, TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR);
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

  IREE_ASSERT_OK(loom_low_lower_import_declaration(module_, source_declaration,
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
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(low_declaration, &predicate_count);
  ASSERT_EQ(predicate_count, 1u);
  ASSERT_NE(predicates, nullptr);
  EXPECT_EQ(predicates[0].args[0], low_arguments.values[0]);
  EXPECT_TRUE(loom_module_value_has_predicate_attribute_uses(
      module_, low_arguments.values[0]));

  EXPECT_EQ(loom_low_func_decl_import_kind(result_.low_func_op),
            LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE);
  const loom_string_id_t code_symbol =
      loom_low_func_decl_code_symbol(result_.low_func_op);
  ASSERT_LT(code_symbol, module_->strings.count);
  EXPECT_TRUE(iree_string_view_equal(module_->strings.entries[code_symbol],
                                     IREE_SV("extern_f")));
  const loom_string_id_t descriptor_set =
      loom_low_func_decl_descriptor_set(result_.low_func_op);
  ASSERT_LT(descriptor_set, module_->strings.count);
  EXPECT_TRUE(iree_string_view_equal(module_->strings.entries[descriptor_set],
                                     IREE_SV("test.low.core")));

  const loom_symbol_ref_t low_callee =
      loom_low_func_decl_callee(result_.low_func_op);
  EXPECT_EQ(low_callee.module_id, symbol.module_id);
  EXPECT_EQ(low_callee.symbol_id, symbol.symbol_id);
  EXPECT_EQ(module_->symbols.entries[symbol.symbol_id].defining_op,
            result_.low_func_op);
}

}  // namespace
}  // namespace loom
