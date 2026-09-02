// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"

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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_ = {};
};

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

}  // namespace
}  // namespace loom
