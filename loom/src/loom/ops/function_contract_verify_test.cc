// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"

#include <utility>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;

class FunctionContractVerifyTest : public ::testing::Test {
 protected:
  static iree_status_t FailAllocation(void* self,
                                      iree_allocator_command_t command,
                                      const void* parameters, void** pointer) {
    (void)self;
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    const iree_allocator_t allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void AddSymbol(iree_string_view_t name, loom_symbol_ref_t* out_symbol) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    *out_symbol = loom_symbol_ref_t{/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  void AddIndexDeclaration(iree_string_view_t name, loom_op_t** out_op) {
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    AddSymbol(name, &symbol);
    const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    const loom_type_t arguments[] = {index, index, index, index, index};
    const loom_type_t results[] = {index, index, index, index};
    IREE_ASSERT_OK(loom_func_decl_build(
        &builder_, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        LOOM_STRING_ID_INVALID, LOOM_STRING_ID_INVALID, /*cc=*/0,
        /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        arguments, IREE_ARRAYSIZE(arguments), results, IREE_ARRAYSIZE(results),
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        out_op));
  }

  loom_symbol_ref_t AddDeclaration(iree_string_view_t name,
                                   const loom_type_t* argument_types,
                                   iree_host_size_t argument_count,
                                   const loom_type_t* result_types,
                                   iree_host_size_t result_count,
                                   loom_op_t** out_op) {
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    AddSymbol(name, &symbol);
    IREE_CHECK_OK(loom_func_decl_build(
        &builder_, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        LOOM_STRING_ID_INVALID, LOOM_STRING_ID_INVALID, /*cc=*/0,
        /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        argument_types, argument_count, result_types, result_count,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        out_op));
    return symbol;
  }

  loom_type_t AddFunctionType(loom_type_t argument_type) {
    loom_type_t type = {};
    IREE_CHECK_OK(loom_module_intern_function_type(module_, &argument_type, 1,
                                                   nullptr, 0, &type));
    return type;
  }

  loom_value_id_t AddBlockArgument(loom_type_t type) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &value_id));
    IREE_CHECK_OK(
        loom_block_add_arg(module_, loom_module_block(module_), value_id));
    return value_id;
  }

  loom_op_t* AddCall(loom_symbol_ref_t callee, const loom_value_id_t* operands,
                     iree_host_size_t operand_count) {
    loom_op_t* call_op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(
        &builder_, /*build_flags=*/0, /*purity=*/0, /*temperature=*/0,
        /*inline_policy=*/0, callee, operands, operand_count,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &call_op));
    return call_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_ = {};
};

TEST_F(FunctionContractVerifyTest,
       IndexedDeclarationSliceKeepsDefinitionDomains) {
  loom_op_t* declaration = nullptr;
  loom_op_t* other_declaration = nullptr;
  AddIndexDeclaration(IREE_SV("source"), &declaration);
  AddIndexDeclaration(IREE_SV("other"), &other_declaration);
  ASSERT_NE(declaration, nullptr);
  ASSERT_NE(other_declaration, nullptr);
  const auto arguments = loom_func_decl_args(declaration);
  const auto results = loom_func_decl_results(declaration);
  const auto other_arguments = loom_func_decl_args(other_declaration);
  loom_value_id_t targets[6] = {};
  for (loom_value_id_t& target : targets) {
    IREE_ASSERT_OK(loom_module_define_value(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &target));
    IREE_ASSERT_OK(
        loom_block_add_arg(module_, loom_module_block(module_), target));
  }
  const loom_type_value_remap_t result_remap = {
      /*.source_values=*/results.values,
      /*.target_values=*/&targets[3],
      /*.count=*/3,
      /*.flags=*/LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  const loom_type_value_remap_t argument_remap = {
      /*.source_values=*/&arguments.values[1],
      /*.target_values=*/targets,
      /*.count=*/3,
      /*.flags=*/LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
      /*.next=*/&result_remap,
  };

  // The slice begins at operand one. Results, other declarations and block
  // arguments do not become members merely by sharing an in-slice index.
  const std::pair<loom_value_id_t, loom_value_id_t> mappings[] = {
      {arguments.values[0], arguments.values[0]},
      {arguments.values[1], targets[0]},
      {arguments.values[2], targets[1]},
      {arguments.values[3], targets[2]},
      {arguments.values[4], arguments.values[4]},
      {results.values[0], targets[3]},
      {results.values[1], targets[4]},
      {results.values[2], targets[5]},
      {results.values[3], results.values[3]},
      {other_arguments.values[1], other_arguments.values[1]},
      {targets[1], targets[1]},
  };
  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;
  for (const auto& [source_value, expected_value] : mappings) {
    SCOPED_TRACE(source_value);
    const loom_type_t source =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(source_value), 0);
    const loom_type_t expected =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(expected_value), 0);
    EXPECT_TRUE(loom_type_equal_after_value_remap(module_, source, expected,
                                                  &argument_remap));
    EXPECT_EQ(
        loom_type_hash_after_value_remap(module_, source, &argument_remap),
        loom_type_hash_after_value_remap(module_, expected, nullptr));
  }
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
}

TEST_F(FunctionContractVerifyTest, RejectsPredicateValueOutsideSignature) {
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t foreign_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &foreign_value));

  loom_predicate_t predicate = {};
  predicate.kind = LOOM_PREDICATE_EQ;
  predicate.arg_count = 2;
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = foreign_value;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.args[1] = 4;

  loom_symbol_ref_t function_symbol = loom_symbol_ref_null();
  AddSymbol(IREE_SV("invalid"), &function_symbol);
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_PREDICATES,
      /*visibility=*/0, /*retain=*/0, /*cc=*/0, /*purity=*/0,
      /*temperature=*/0, /*inline_policy=*/0, loom_symbol_ref_null(),
      /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), function_symbol, &i32, 1,
      /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, &predicate, 1, LOOM_LOCATION_UNKNOWN,
      &function_op));

  DiagnosticEmissionCapture capture;
  IREE_EXPECT_OK(
      loom_function_contract_verify(module_, function_op, capture.emitter()));
  ASSERT_EQ(capture.emissions.size(), 1u);
  const auto& emission = capture.emissions.front();
  EXPECT_EQ(emission.error, LOOM_ERR_STRUCTURE_032);
  ASSERT_EQ(emission.string_params.size(), 3u);
  EXPECT_EQ(emission.string_params[0], "func.def");
  EXPECT_EQ(emission.string_params[1], "predicates[0].arg[0]");
  EXPECT_EQ(emission.string_params[2], "a function argument or result");
}

TEST_F(FunctionContractVerifyTest, ReportsEveryMismatchAfterEnteringExactMode) {
  const loom_type_t expected_inner =
      AddFunctionType(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  const loom_type_t actual_inner =
      AddFunctionType(loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  const loom_type_t expected_outer = AddFunctionType(expected_inner);
  const loom_type_t actual_outer = AddFunctionType(actual_inner);
  const loom_type_t expected_types[] = {expected_inner, expected_outer};
  loom_op_t* declaration = nullptr;
  const loom_symbol_ref_t callee = AddDeclaration(
      IREE_SV("mismatches"), expected_types, IREE_ARRAYSIZE(expected_types),
      /*result_types=*/nullptr, /*result_count=*/0, &declaration);
  ASSERT_NE(declaration, nullptr);
  const loom_value_id_t operands[] = {
      AddBlockArgument(actual_inner),
      AddBlockArgument(actual_outer),
  };
  loom_op_t* call = AddCall(callee, operands, IREE_ARRAYSIZE(operands));

  DiagnosticEmissionCapture capture;
  IREE_EXPECT_OK(loom_function_call_contract_verify(
      module_, call, callee, loom_func_call_operands(call),
      loom_func_call_results(call), capture.emitter()));
  ASSERT_EQ(capture.emissions.size(), 2u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_TYPE_001);
  EXPECT_EQ(capture.emissions[1].error, LOOM_ERR_TYPE_001);
}

TEST_F(FunctionContractVerifyTest,
       ScratchAllocationFailurePrecedesMismatchDiagnostics) {
  const loom_type_t compound =
      AddFunctionType(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  loom_op_t* declaration = nullptr;
  const loom_symbol_ref_t callee = AddDeclaration(
      IREE_SV("allocation_failure"), &compound, 1,
      /*result_types=*/nullptr, /*result_count=*/0, &declaration);
  ASSERT_NE(declaration, nullptr);
  const loom_value_id_t operand = AddBlockArgument(compound);
  loom_op_t* call = AddCall(callee, &operand, 1);

  iree_arena_block_pool_trim(&block_pool_);
  const iree_allocator_t system_allocator = block_pool_.block_allocator;
  block_pool_.block_allocator = {/*.self=*/nullptr,
                                 /*.ctl=*/FailAllocation};
  DiagnosticEmissionCapture capture;
  iree_status_t status = loom_function_call_contract_verify(
      module_, call, callee, loom_func_call_operands(call),
      loom_func_call_results(call), capture.emitter());
  block_pool_.block_allocator = system_allocator;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_TRUE(capture.emissions.empty());
}

}  // namespace
}  // namespace loom
