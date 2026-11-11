// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_specialization.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class KernelSpecializationTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &scratch_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    for (auto& module : modules_) {
      IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("binding"), &pool_,
                                          nullptr, iree_allocator_system(),
                                          &module));
    }
  }

  void TearDown() override {
    for (auto* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Verify(loom_module_t* module) {
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  void Bind(loom_module_t* module, int unrelated_value_count,
            iree_host_size_t* out_scratch_bytes) {
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name;
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("kernel"), &name));
    IREE_ASSERT_OK(loom_module_add_symbol(module, name, &symbol));
    const auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    const loom_type_t binding_types[] = {
        index_type,
        loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(1), 0),
    };
    loom_op_t* definition = nullptr;
    IREE_ASSERT_OK(loom_pipeline_def_build(
        &builder, 0, 0, 0, 0, loom_symbol_ref_null(), {0, symbol}, &index_type,
        1, binding_types, 2, nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
    auto* body = loom_pipeline_def_body(definition);
    auto* block = loom_region_entry_block(body);
    const auto staged = block->arg_ids[0];
    const auto invocation = block->arg_ids[1];
    const auto view = block->arg_ids[2];
    IREE_ASSERT_OK(loom_module_set_value_type(
        module, view,
        loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_dynamic(staged), 0)));
    loom_predicate_t predicates[] = {
        {
            .kind = LOOM_PREDICATE_MULTIPLE_OF,
            .arg_count = 2,
            .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
            .reserved = {},
            .args = {staged, 2},
        },
        {
            .kind = LOOM_PREDICATE_GE,
            .arg_count = 2,
            .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
            .reserved = {},
            .args = {invocation, staged},
        },
    };
    if (GetParam()) {
      IREE_ASSERT_OK(loom_pipeline_def_set_predicates(
          module, definition, loom_attr_predicate_list(predicates, 2)));
    }
    loom_builder_enter_region(&builder, definition, body);
    loom_op_t* add = nullptr;
    IREE_ASSERT_OK(loom_index_add_build(
        &builder, staged, invocation, index_type, LOOM_LOCATION_UNKNOWN, &add));
    // Valid unrelated values separate the formal IDs from their replacements.
    // Their storage belongs to the module, outside the binding scratch arena.
    for (int i = 0; i < unrelated_value_count; ++i) {
      loom_op_t* constant = nullptr;
      IREE_ASSERT_OK(
          loom_index_constant_build(&builder, loom_attr_i64(i), index_type,
                                    LOOM_LOCATION_UNKNOWN, &constant));
    }
    loom_op_t* finish = nullptr;
    IREE_ASSERT_OK(
        loom_pipeline_finish_build(&builder, LOOM_LOCATION_UNKNOWN, &finish));
    ASSERT_NO_FATAL_FAILURE(Verify(module));

    iree_arena_reset(&scratch_);
    loom_rewriter_t rewriter;
    loom_rewriter_initialize(&rewriter, module, &scratch_);
    const loom_kernel_specialization_value_t specialization = {
        .type = index_type,
        .facts = loom_value_facts_exact_i64(16),
    };
    IREE_ASSERT_OK(loom_kernel_specializations_apply(
        module, &specialization, 1, loom_func_like_cast(module, definition),
        &rewriter, &scratch_));
    *out_scratch_bytes = scratch_.total_allocation_size;
    loom_rewriter_deinitialize(&rewriter);
    ASSERT_NO_FATAL_FAILURE(Verify(module));

    ASSERT_EQ(block->arg_count, 2);
    EXPECT_EQ(block->arg_ids[0], invocation);
    EXPECT_EQ(block->arg_ids[1], view);
    EXPECT_EQ(loom_pipeline_def_specialization_count(definition), 0);
    EXPECT_EQ(
        loom_type_dim_static_size_at(loom_module_value_type(module, view), 0),
        16);
    EXPECT_EQ(loom_index_add_rhs(add), invocation);
    const auto* replacement =
        loom_value_def_op(loom_module_value(module, loom_index_add_lhs(add)));
    ASSERT_TRUE(loom_index_constant_isa(replacement));
    EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(replacement)), 16);
    uint16_t predicate_count = 0;
    const auto* projected = loom_func_like_predicates(
        loom_func_like_cast(module, definition), &predicate_count);
    ASSERT_EQ(predicate_count, GetParam() ? 1 : 0);
    if (GetParam()) {
      EXPECT_EQ(projected[0].kind, LOOM_PREDICATE_GE);
      EXPECT_EQ(projected[0].arg_tags[0], LOOM_PRED_ARG_VALUE);
      EXPECT_EQ(projected[0].args[0], invocation);
      EXPECT_EQ(projected[0].arg_tags[1], LOOM_PRED_ARG_CONST);
      EXPECT_EQ(projected[0].args[1], 16);
    }
  }

  // Shared block cache for independently owned module and scratch arenas.
  iree_arena_block_pool_t pool_;
  // Temporary memory used only by staged binding.
  iree_arena_allocator_t scratch_;
  // Registered production dialects used by both modules.
  loom_context_t context_;
  // Independent short and long SSA namespaces with identical bindings.
  loom_module_t* modules_[2] = {};
};

TEST_P(KernelSpecializationTest, ScratchDoesNotSpanUnrelatedValues) {
  iree_host_size_t short_bytes = 0;
  iree_host_size_t long_bytes = 0;
  ASSERT_NO_FATAL_FAILURE(Bind(modules_[0], 0, &short_bytes));
  ASSERT_NO_FATAL_FAILURE(Bind(modules_[1], 4096, &long_bytes));
  // Allow one block of segment-boundary slack, but no growth with the body.
  EXPECT_LE(long_bytes, short_bytes + pool_.total_block_size);
}

INSTANTIATE_TEST_SUITE_P(SignaturePredicates, KernelSpecializationTest,
                         ::testing::Bool());

}  // namespace
}  // namespace loom
