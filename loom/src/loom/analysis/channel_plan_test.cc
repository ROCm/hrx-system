// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/channel_plan.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/type_registry.h"

namespace loom {
namespace {

class ChannelPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    Register(LOOM_DIALECT_CHANNEL, loom_channel_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("channel_plan"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_type_id_t payload;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_,
        loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(4), 0),
        &payload));
    IREE_ASSERT_OK(loom_channel_type_make(module_, payload, &channel_type_));
    IREE_ASSERT_OK(loom_read_type_make(module_, 0, payload,
                                       (loom_read_type_mode_t)0, &read_type_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    if (loom_local_value_domain_is_acquired(&domain_)) {
      loom_local_value_domain_release(&domain_);
    }
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Register(uint8_t id,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, id, vtables, (uint16_t)count));
  }

  loom_op_t* Function(const char* name,
                      const std::vector<loom_type_t>& arguments,
                      const std::vector<loom_type_t>& results = {}) {
    loom_string_id_t name_id;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol));
    loom_builder_set_block(&builder_, loom_module_block(module_));
    builder_.ip.parent_op = nullptr;
    loom_op_t* function;
    IREE_CHECK_OK(loom_func_def_build(
        &builder_, 0, 0, /*linkage=*/0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), {0, symbol}, arguments.data(),
        arguments.size(), results.data(), results.size(), nullptr, 0, nullptr,
        0, LOOM_LOCATION_UNKNOWN, &function));
    loom_builder_enter_region(&builder_, function,
                              loom_func_def_body(function));
    return function;
  }

  loom_channel_plan_t Analyze(
      loom_op_t* function,
      const std::vector<loom_channel_plan_binding_t>& bindings,
      const loom_channel_plan_callable_t* const* callables = nullptr) {
    IREE_CHECK_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_def_body(function), &arena_, &domain_));
    loom_channel_plan_t plan;
    loom_channel_plan_rejection_t rejection;
    IREE_CHECK_OK(loom_channel_plan_build(&domain_, bindings.data(),
                                          bindings.size(), callables, &arena_,
                                          &plan, &rejection));
    EXPECT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
    return plan;
  }

  // Shared backing for source and analysis allocations.
  iree_arena_block_pool_t pool_;
  // Analysis storage whose summaries survive individual domain releases.
  iree_arena_allocator_t arena_;
  // Registered source operation vocabulary.
  loom_context_t context_;
  // Source module owned by this fixture.
  loom_module_t* module_ = nullptr;
  // Insertion point for each source function under construction.
  loom_builder_t builder_ = {};
  // Acquired local correspondence for the current analyzed function.
  loom_local_value_domain_t domain_ = {};
  // Channel carrying four i32 elements per record.
  loom_type_t channel_type_;
  // Immutable read obligation for the same record payload.
  loom_type_t read_type_;
};

TEST_F(ChannelPlanTest, SharedStorageDoesNotMergeChannelBindings) {
  const loom_type_t storage_type =
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(2), loom_dim_pack_static(4), 0);
  loom_op_t* function = Function("bindings", {storage_type});
  const loom_value_id_t storage =
      loom_region_entry_block(loom_func_def_body(function))->arg_ids[0];
  loom_op_t* capacity;
  IREE_ASSERT_OK(loom_index_constant_build(
      &builder_, loom_attr_i64(2), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &capacity));
  loom_op_t* first;
  loom_op_t* second;
  IREE_ASSERT_OK(
      loom_channel_bind_build(&builder_, LOOM_CHANNEL_BIND_DISCIPLINE_FIFO,
                              storage, loom_index_constant_result(capacity),
                              channel_type_, LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(
      loom_channel_bind_build(&builder_, LOOM_CHANNEL_BIND_DISCIPLINE_FIFO,
                              storage, loom_index_constant_result(capacity),
                              channel_type_, LOOM_LOCATION_UNKNOWN, &second));
  loom_op_t* exit;
  IREE_ASSERT_OK(loom_func_return_build(&builder_, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &exit));
  const auto plan = Analyze(function, {});
  const auto* first_identity =
      loom_channel_plan_channel(&plan, loom_channel_bind_result(first));
  const auto* second_identity =
      loom_channel_plan_channel(&plan, loom_channel_bind_result(second));
  EXPECT_EQ(first_identity->value_id, loom_channel_bind_result(first));
  EXPECT_EQ(second_identity->value_id, loom_channel_bind_result(second));
  EXPECT_NE(first_identity, second_identity);

  loom_local_value_domain_release(&domain_);
  const auto repeated = Analyze(function, {});
  const auto* repeated_identity =
      loom_channel_plan_channel(&repeated, loom_channel_bind_result(first));
  EXPECT_EQ(repeated_identity->value_id, first_identity->value_id);
  EXPECT_NE(repeated_identity, first_identity);
}

TEST_F(ChannelPlanTest, CallableSummaryTracksNewReadsFromFormalChannels) {
  loom_op_t* callee = Function("accept_record", {channel_type_}, {read_type_});
  const loom_value_id_t formal =
      loom_region_entry_block(loom_func_def_body(callee))->arg_ids[0];
  loom_op_t* accept;
  IREE_ASSERT_OK(loom_channel_accept_build(&builder_, 0, 0, formal, read_type_,
                                           LOOM_LOCATION_UNKNOWN, &accept));
  const loom_value_id_t read = loom_channel_accept_read(accept);
  loom_op_t* exit;
  IREE_ASSERT_OK(loom_func_return_build(&builder_, &read, 1,
                                        LOOM_LOCATION_UNKNOWN, &exit));
  const loom_channel_identity_t formal_identity = {formal};
  const auto formal_plan = Analyze(callee, {{formal, &formal_identity}});
  loom_channel_plan_callable_t summary;
  loom_channel_plan_rejection_t rejection;
  IREE_ASSERT_OK(loom_channel_plan_summarize(&formal_plan, callee, &arena_,
                                             &summary, &rejection));
  ASSERT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
  EXPECT_EQ(summary.result_arguments[0], 0);
  loom_local_value_domain_release(&domain_);

  loom_op_t* caller =
      Function("consume_record", {channel_type_, channel_type_});
  const auto* actuals =
      loom_region_entry_block(loom_func_def_body(caller))->arg_ids;
  const loom_value_id_t operands[] = {actuals[0], actuals[1], actuals[0]};
  loom_value_id_t reads[3];
  for (size_t i = 0; i < 3; ++i) {
    loom_op_t* call;
    IREE_ASSERT_OK(loom_func_call_build(
        &builder_, 0, 0, 0, 0, loom_func_def_callee(callee), &operands[i], 1,
        &read_type_, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &call));
    reads[i] = loom_op_results(call)[0];
    loom_op_t* release;
    IREE_ASSERT_OK(loom_channel_release_build(&builder_, reads[i],
                                              LOOM_LOCATION_UNKNOWN, &release));
  }
  IREE_ASSERT_OK(loom_func_return_build(&builder_, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &exit));
  std::vector<const loom_channel_plan_callable_t*> summaries(
      module_->symbols.count, nullptr);
  summaries[loom_func_def_callee(callee).symbol_id] = &summary;
  // Two occurrences of the same source binding remain independent through
  // one reusable helper. Calling it again with the first channel forwards
  // that original identity rather than creating a third protocol.
  const loom_channel_identity_t instances[] = {{formal}, {formal}};
  const auto plan = Analyze(
      caller, {{actuals[0], &instances[0]}, {actuals[1], &instances[1]}},
      summaries.data());
  ASSERT_EQ(plan.action_count, 6u);
  for (size_t i = 0; i < 3; ++i) {
    const auto* expected = &instances[i % 2];
    EXPECT_EQ(loom_channel_plan_channel(&plan, reads[i]), expected);
    EXPECT_EQ(plan.actions[2 * i].callable, &summary);
    EXPECT_EQ(plan.actions[2 * i + 1].channel, expected);
  }
}

}  // namespace
}  // namespace loom
