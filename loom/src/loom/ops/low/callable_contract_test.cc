// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/callable.h"

namespace loom {
namespace {

class CallableContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
    iree_arena_initialize(&block_pool_, &rewriter_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&rewriter_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_string_id_t InternString(iree_string_view_t value) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&module_builder_, value, &string_id));
    return string_id;
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    const loom_string_id_t name_id = InternString(name);
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return loom_symbol_ref_t{
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  loom_op_t* BuildLowFunction(loom_op_t** out_fence_op,
                              loom_op_t** out_return_op) {
    const loom_symbol_ref_t function_ref = AddSymbol(IREE_SV("source"));
    const loom_string_id_t contract_id = InternString(IREE_SV("test.low.core"));
    loom_op_t* function_op = nullptr;
    IREE_CHECK_OK(loom_low_func_def_build(
        &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*allocation=*/0, /*schedule=*/0, contract_id,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), function_ref,
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));

    loom_builder_t body_builder = {};
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_low_func_def_body(function_op)),
        &body_builder);
    body_builder.ip.parent_op = function_op;

    IREE_CHECK_OK(loom_low_schedule_fence_build(
        &body_builder, LOOM_LOCATION_UNKNOWN, out_fence_op));
    IREE_CHECK_OK(loom_low_return_build(&body_builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, out_return_op));
    return function_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
  iree_arena_allocator_t rewriter_arena_;
};

TEST_F(CallableContractTest, OutlineRejectsRepresentationBoundRange) {
  loom_op_t* fence_op = nullptr;
  loom_op_t* return_op = nullptr;
  loom_op_t* function_op = BuildLowFunction(&fence_op, &return_op);
  const loom_symbol_ref_t outlined_ref = AddSymbol(IREE_SV("outlined"));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  loom_callable_outline_result_t result = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_callable_outline_range(&rewriter, fence_op, return_op, outlined_ref,
                                  &result));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(loom_func_like_isa(result.outlined));
  EXPECT_EQ(result.call_op, nullptr);
  EXPECT_EQ(module_->symbols.entries[outlined_ref.symbol_id].defining_op,
            nullptr);
  EXPECT_FALSE(iree_any_bit_set(fence_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(return_op->flags, LOOM_OP_FLAG_DEAD));
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function_op));
  EXPECT_EQ(body->op_count, 2u);
}

}  // namespace
}  // namespace loom
