// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/worker_materialization.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/verify/verify.h"

namespace {

class WorkerMaterializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&context_));
    Register(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables,
             loom_cfg_dialect_op_semantics);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
             loom_func_dialect_op_semantics);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables,
             loom_index_dialect_op_semantics);
    Register(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables,
             loom_kernel_dialect_op_semantics);
    Register(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables,
             loom_scalar_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("worker_clone"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    auto* builder = &rewriter_.builder;
    loom_builder_set_block(builder, loom_module_block(module_));
    const auto view = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                          loom_dim_pack_static(4), 0);
    const loom_type_t arguments[] = {view, view,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                     loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)};
    loom_op_t* function;
    IREE_ASSERT_OK(loom_func_def_build(
        builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), Symbol("copy"),
        arguments, IREE_ARRAYSIZE(arguments), nullptr, 0, nullptr, 0, nullptr,
        0, LOOM_LOCATION_UNKNOWN, &function));
    source_.function = loom_func_like_cast(module_, function);
    auto* region = loom_func_def_body(function);
    auto* entry = loom_region_entry_block(region);
    loom_builder_enter_region(builder, function, region);
    loom_block_t* continuation;
    IREE_ASSERT_OK(loom_region_append_block(module_, region, &continuation));
    auto token_type = Opaque("kernel.async.token");
    auto group_type = Opaque("kernel.async.group");
    for (size_t i = 0; i < 2; ++i) {
      auto* block = i == 0 ? entry : continuation;
      loom_builder_set_block(builder, block);
      loom_op_t* copy;
      IREE_ASSERT_OK(loom_kernel_async_copy_build(
          builder, entry->arg_ids[0], entry->arg_ids[1],
          LOOM_CACHE_SCOPE_WORKGROUP, LOOM_CACHE_TEMPORAL_REGULAR,
          LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP, token_type,
          LOOM_LOCATION_UNKNOWN, &copy));
      const auto token = loom_op_results(copy)[0];
      loom_op_t* group;
      IREE_ASSERT_OK(loom_kernel_async_group_build(
          builder, &token, 1, group_type, LOOM_LOCATION_UNKNOWN, &group));
      IREE_ASSERT_OK(
          loom_kernel_async_wait_build(builder, loom_op_results(group)[0], 0,
                                       LOOM_LOCATION_UNKNOWN, &waits_[i]));
      transfers_[i].request.op = copy;
      transfers_[i].request.kind = LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY;
      groups_[i] = {group, 0, 1, waits_[i]};
      streams_[i] = {i == 0 ? &streams_[1] : nullptr,
                     block,
                     &transfers_[i],
                     1,
                     &groups_[i],
                     1,
                     &waits_[i],
                     1};
      loom_op_t* terminator;
      if (i == 0) {
        IREE_ASSERT_OK(loom_cfg_br_build(builder, continuation, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));
      } else {
        IREE_ASSERT_OK(loom_func_return_build(
            builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &terminator));
      }
    }
    source_.asynchronous.streams = streams_;
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, region, &arena_, &source_.value_domain));
    loom_channel_plan_rejection_t rejection;
    IREE_ASSERT_OK(loom_channel_plan_build(&source_.value_domain, nullptr, 0,
                                           nullptr, &arena_, &source_.channels,
                                           &rejection));
    ASSERT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
    IREE_ASSERT_OK(loom_cfg_graph_build(module_, region, &arena_, &graph_));
    source_.graph = &graph_;
    loom_builder_set_block(builder, loom_module_block(module_));
    builder->ip.parent_op = nullptr;
    IREE_ASSERT_OK(loom_pipeline_worker_materialization_clone(
        &rewriter_, &source_, Symbol("first"), &instances_[0]));
    IREE_ASSERT_OK(loom_pipeline_worker_materialization_clone(
        &rewriter_, &source_, Symbol("second"), &instances_[1]));
    loom_local_value_domain_release(&source_.value_domain);
  }

  void TearDown() override {
    if (loom_local_value_domain_is_acquired(&source_.value_domain)) {
      loom_local_value_domain_release(&source_.value_domain);
    }
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Register(
      uint8_t dialect, const loom_op_vtable_t* const* (*get)(iree_host_size_t*),
      const loom_op_semantics_t* (*get_semantics)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
    const auto* semantics = get_semantics(&count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(&context_, dialect,
                                                           semantics, count));
  }

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t string;
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &string));
    IREE_CHECK_OK(loom_module_add_symbol(module_, string, &symbol));
    return {0, symbol};
  }

  loom_type_t Opaque(const char* name) {
    loom_string_id_t string;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &string));
    return loom_type_dialect_opaque(string);
  }

  void Verify() {
    const loom_verify_options_t options = {
        .sink = {.fn = loom_diagnostic_stderr_sink},
    };
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  // Common allocation lifetime for source and independent consuming copies.
  iree_arena_block_pool_t pool_;
  // Retained plans and remapping tables.
  iree_arena_allocator_t arena_;
  // Source operation vocabulary.
  loom_context_t context_;
  // Module owning source and cloned definitions.
  loom_module_t* module_ = nullptr;
  // Clone and offset emission frame.
  loom_rewriter_t rewriter_;
  // Immutable analysis input for the cloning API.
  loom_pipeline_worker_t source_ = {};
  // Canonical control-flow graph shared through the source occurrence.
  loom_cfg_graph_t graph_;
  // One retained transfer per straight-line block.
  loom_kernel_async_transfer_t transfers_[2] = {};
  // Each group has one transfer and one explicit completion.
  loom_kernel_async_group_t groups_[2] = {};
  // Source-ordered selected asynchronous sites.
  loom_kernel_async_stream_t streams_[2] = {};
  // Explicit waits projected by the clone observer.
  loom_op_t* waits_[2] = {};
  // Separate consuming instances sharing the source plans.
  loom_pipeline_worker_materialization_t instances_[2] = {};
};

TEST_F(WorkerMaterializationTest, AsyncSitesAndOperandsBelongToEachInstance) {
  const auto* source_entry =
      loom_region_entry_block(loom_func_like_body(source_.function));
  for (auto& instance : instances_) {
    ASSERT_EQ(instance.operation_count, 6u);
    auto* body = loom_func_like_body(instance.channels.function);
    auto* entry = loom_region_entry_block(body);
    for (size_t i = 0; i < 2; ++i) {
      const loom_op_t* sites[] = {transfers_[i].request.op, groups_[i].op,
                                  waits_[i]};
      auto* actual = body->blocks[i]->first_op;
      for (const auto* site : sites) {
        EXPECT_EQ(loom_pipeline_worker_materialization_op(&instance, site),
                  actual);
        EXPECT_NE(actual, site);
        actual = actual->next_op;
      }
      auto* copy = body->blocks[i]->first_op;
      EXPECT_EQ(loom_kernel_async_copy_source(copy), entry->arg_ids[0]);
      EXPECT_EQ(loom_kernel_async_copy_dest(copy), entry->arg_ids[1]);
    }
    for (size_t i = 0; i < entry->arg_count; ++i) {
      EXPECT_EQ(loom_pipeline_worker_materialization_value(
                    &instance, source_entry->arg_ids[i]),
                entry->arg_ids[i]);
      EXPECT_NE(entry->arg_ids[i], source_entry->arg_ids[i]);
    }
  }
  Verify();
}

TEST_F(WorkerMaterializationTest, OffsetRecipesRetainLocalOperandsAndAnchors) {
  const auto* source_entry =
      loom_region_entry_block(loom_func_like_body(source_.function));
  const loom_symbolic_term_t term = {1, source_entry->arg_ids[2],
                                     source_entry->arg_ids[2]};
  const loom_symbolic_expr_t affine = {
      .terms = &term,
      .term_count = 1,
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
  const loom_symbolic_expr_t retained = {
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_RELATION_VALUE,
      .relation_value_id = source_entry->arg_ids[3],
  };
  for (auto& instance : instances_) {
    auto* entry = loom_region_entry_block(
        loom_func_like_body(instance.channels.function));
    loom_value_id_t offset;
    IREE_ASSERT_OK(loom_pipeline_worker_materialization_offset(
        &instance, &rewriter_.builder, &retained, source_entry->arg_ids[0],
        &offset));
    EXPECT_EQ(offset, entry->arg_ids[3]);
    IREE_ASSERT_OK(loom_pipeline_worker_materialization_offset(
        &instance, &rewriter_.builder, &affine, source_entry->arg_ids[0],
        &offset));
    auto* cast = entry->first_op;
    ASSERT_TRUE(loom_index_cast_isa(cast));
    EXPECT_EQ(loom_op_const_operands(cast)[0], entry->arg_ids[2]);
    EXPECT_EQ(
        loom_value_def_op(loom_module_value(module_, offset))->parent_block,
        entry);
  }
  EXPECT_EQ(term.value_id, source_entry->arg_ids[2]);
  EXPECT_EQ(term.relation_value_id, source_entry->arg_ids[2]);
  EXPECT_EQ(source_entry->first_op, transfers_[0].request.op);
  Verify();
}

}  // namespace
