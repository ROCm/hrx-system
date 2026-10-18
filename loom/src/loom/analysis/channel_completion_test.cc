// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/channel_completion.h"

#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class ChannelCompletionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    Register(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    Register(LOOM_DIALECT_CHANNEL, loom_channel_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("completion"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_type_id_t payload;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_,
        loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(1), 0),
        &payload));
    loom_type_t channel_type;
    IREE_ASSERT_OK(loom_channel_type_make(module_, payload, &channel_type));
    IREE_ASSERT_OK(loom_read_type_make(module_, 0, payload,
                                       (loom_read_type_mode_t)0, &read_type_));
    IREE_ASSERT_OK(loom_write_type_make(module_, payload, &write_type_));
    view_type_ = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                     loom_dim_pack_static(1), 0);
    loom_string_id_t name;
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("worker"), &name));
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t arguments[] = {channel_type, channel_type,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I1)};
    IREE_ASSERT_OK(loom_func_def_build(
        &builder_, 0, 0, /*linkage=*/0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), {0, symbol}, arguments,
        IREE_ARRAYSIZE(arguments), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function_));
    region_ = loom_func_def_body(function_);
    loom_builder_enter_region(&builder_, function_, region_);
    const auto* argument_ids = loom_region_entry_block(region_)->arg_ids;
    first_ = argument_ids[0];
    second_ = argument_ids[1];
    condition_ = argument_ids[2];
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

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
  }

  loom_block_t* Block() {
    loom_block_t* block;
    IREE_CHECK_OK(loom_region_append_block(module_, region_, &block));
    return block;
  }

  void At(loom_block_t* block) {
    loom_builder_set_block(&builder_, block);
    builder_.ip.parent_op = function_;
  }

  void Branch(loom_block_t* destination) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, destination, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));
  }

  void Choose(loom_block_t* yes, loom_block_t* no) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, condition_, yes, no,
                                          LOOM_LOCATION_UNKNOWN, &op));
  }

  void Return() {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_func_return_build(&builder_, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &op));
  }

  loom_value_id_t Reserve(loom_value_id_t channel) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_channel_reserve_build(&builder_, channel, write_type_,
                                             view_type_, LOOM_LOCATION_UNKNOWN,
                                             &op));
    return loom_channel_reserve_write(op);
  }

  void Publish(loom_value_id_t write) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_channel_publish_build(&builder_, write,
                                              LOOM_LOCATION_UNKNOWN, &op));
  }

  loom_value_id_t Acquire(loom_value_id_t channel) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_channel_acquire_build(&builder_, 0, 0, channel,
                                             read_type_, view_type_,
                                             LOOM_LOCATION_UNKNOWN, &op));
    return loom_channel_acquire_read(op);
  }

  void Wait(loom_value_id_t read) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_channel_wait_build(&builder_, read, view_type_,
                                           LOOM_LOCATION_UNKNOWN, &op));
  }

  void Release(loom_value_id_t read) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_channel_release_build(&builder_, read,
                                              LOOM_LOCATION_UNKNOWN, &op));
  }

  void ExpectActions(
      std::initializer_list<loom_channel_completion_action_t> expected) {
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t verified;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &verified));
    ASSERT_EQ(verified.error_count, 0u);
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, region_, &arena_, &domain_));
    const loom_channel_identity_t identities[] = {{first_}, {second_}};
    const loom_channel_plan_binding_t bindings[] = {{first_, &identities[0]},
                                                    {second_, &identities[1]}};
    loom_channel_plan_t plan;
    loom_channel_plan_rejection_t rejection;
    IREE_ASSERT_OK(loom_channel_plan_build(&domain_, bindings,
                                           IREE_ARRAYSIZE(bindings), nullptr,
                                           &arena_, &plan, &rejection));
    ASSERT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
    loom_cfg_graph_t graph;
    IREE_ASSERT_OK(loom_cfg_graph_build(module_, region_, &arena_, &graph));
    loom_channel_completion_t completion;
    IREE_ASSERT_OK(loom_channel_completion_analyze(&plan, &graph, nullptr,
                                                   &arena_, &completion));
    ASSERT_EQ(completion.requirement, LOOM_CHANNEL_COMPLETION_REQUIREMENT_NONE);
    ASSERT_EQ(plan.action_count, expected.size());
    if (expected.size() == 0) {
      EXPECT_EQ(completion.actions, nullptr);
    }
    size_t index = 0;
    for (const auto& action : expected) {
      SCOPED_TRACE(index);
      EXPECT_EQ(completion.actions[index].credits, action.credits);
      EXPECT_EQ(completion.actions[index].maximum_admissions,
                action.maximum_admissions);
      ++index;
    }
  }

  // Shared backing for source and analysis allocations.
  iree_arena_block_pool_t pool_;
  // Storage for the plan, CFG, and completion results.
  iree_arena_allocator_t arena_;
  // Registered source operation vocabulary.
  loom_context_t context_;
  // Source module owned by this fixture.
  loom_module_t* module_ = nullptr;
  // Worker owning the analyzed actions.
  loom_op_t* function_ = nullptr;
  // Worker's CFG region.
  loom_region_t* region_ = nullptr;
  // Source insertion point.
  loom_builder_t builder_ = {};
  // Acquired worker-local value correspondence.
  loom_local_value_domain_t domain_ = {};
  // First independently bound channel.
  loom_value_id_t first_;
  // Second independently bound channel.
  loom_value_id_t second_;
  // Runtime branch condition.
  loom_value_id_t condition_;
  // Immutable read of one i32 record.
  loom_type_t read_type_;
  // Reservation of one i32 record.
  loom_type_t write_type_;
  // Borrowed record payload.
  loom_type_t view_type_;
};

TEST_F(ChannelCompletionTest, NoActionsHaveNoResultTable) {
  Return();
  ExpectActions({});
}

TEST_F(ChannelCompletionTest, SingleAdmissionsRemainFixedAcrossControlEdges) {
  const auto write = Reserve(first_);
  const auto read = Acquire(second_);
  auto* finish = Block();
  Branch(finish);
  At(finish);
  Publish(write);
  Release(read);
  Return();
  ExpectActions({{0, 1}, {0, 1}, {1, 1}, {1, 1}});
}

TEST_F(ChannelCompletionTest, RepeatedAdmissionsCountEachDirectionSeparately) {
  Publish(Reserve(first_));
  Publish(Reserve(first_));
  Release(Acquire(first_));
  Release(Acquire(first_));
  Release(Acquire(second_));
  Return();
  ExpectActions({{0, 2},
                 {1, 2},
                 {0, 2},
                 {1, 2},
                 {0, 2},
                 {1, 2},
                 {0, 2},
                 {1, 2},
                 {0, 1},
                 {1, 1}});
}

TEST_F(ChannelCompletionTest, SingleCyclicAdmissionIsUnbounded) {
  auto* body = Block();
  auto* finish = Block();
  Branch(body);
  At(body);
  Publish(Reserve(first_));
  Release(Acquire(second_));
  Choose(body, finish);
  At(finish);
  Return();
  ExpectActions(
      {{0, UINT32_MAX}, {1, UINT32_MAX}, {0, UINT32_MAX}, {1, UINT32_MAX}});
}

TEST_F(ChannelCompletionTest, WaitLoopDoesNotReadmitRetainedRecord) {
  const auto read = Acquire(first_);
  auto* body = Block();
  auto* finish = Block();
  Branch(body);
  At(body);
  Wait(read);
  Choose(body, finish);
  At(finish);
  Release(read);
  Return();
  ExpectActions({{0, 1}, {0, 1}, {1, 1}});
}

TEST_F(ChannelCompletionTest, CyclicWriterDoesNotMakeReaderUnbounded) {
  const auto read = Acquire(first_);
  auto* body = Block();
  auto* finish = Block();
  Branch(body);
  At(body);
  Publish(Reserve(first_));
  Choose(body, finish);
  At(finish);
  Release(read);
  Return();
  ExpectActions({{0, 1}, {0, UINT32_MAX}, {1, UINT32_MAX}, {1, 1}});
}

TEST_F(ChannelCompletionTest, DeferredPrefixPreservesTotalAdmissions) {
  const auto earlier = Reserve(first_);
  const auto later = Reserve(first_);
  Publish(later);
  Publish(earlier);
  Return();
  ExpectActions({{0, 2}, {0, 2}, {0, 2}, {2, 2}});
}

}  // namespace
}  // namespace loom
