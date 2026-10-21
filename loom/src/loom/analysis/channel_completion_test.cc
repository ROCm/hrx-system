// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/channel_completion.h"

#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_table.h"
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
    Register(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
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

  void Branch(loom_block_t* destination,
              std::initializer_list<loom_value_id_t> arguments) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, destination, arguments.begin(),
                                     arguments.size(), LOOM_LOCATION_UNKNOWN,
                                     &op));
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

  loom_value_id_t BlockArgument(loom_block_t* block, loom_type_t type) {
    loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_define_block_arg(&builder_, block, type, &argument));
    return argument;
  }

  loom_value_id_t Reserve(loom_value_id_t channel,
                          loom_value_id_t* out_payload = nullptr) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_channel_reserve_build(&builder_, channel, write_type_,
                                             view_type_, LOOM_LOCATION_UNKNOWN,
                                             &op));
    if (out_payload) {
      *out_payload = loom_channel_reserve_view(op);
    }
    return loom_channel_reserve_write(op);
  }

  void Publish(loom_value_id_t write) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_channel_publish_build(&builder_, write,
                                              LOOM_LOCATION_UNKNOWN, &op));
  }

  loom_value_id_t Acquire(loom_value_id_t channel,
                          loom_value_id_t* out_payload = nullptr) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_channel_acquire_build(&builder_, 0, 0, channel,
                                             read_type_, view_type_,
                                             LOOM_LOCATION_UNKNOWN, &op));
    if (out_payload) {
      *out_payload = loom_channel_acquire_view(op);
    }
    return loom_channel_acquire_read(op);
  }

  loom_value_id_t Wait(loom_value_id_t read) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_channel_wait_build(&builder_, read, view_type_,
                                          LOOM_LOCATION_UNKNOWN, &op));
    return loom_channel_wait_view(op);
  }

  loom_value_id_t Project(loom_value_id_t source) {
    const int64_t zero[] = {0};
    loom_op_t* op;
    IREE_CHECK_OK(loom_view_subview_build(&builder_, source, nullptr, 0, zero,
                                          IREE_ARRAYSIZE(zero), view_type_,
                                          LOOM_LOCATION_UNKNOWN, &op));
    return loom_view_subview_result(op);
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
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 0));
    loom_type_registry_configure_fact_context(&facts_.context);
    IREE_ASSERT_OK(loom_value_fact_table_compute(
        &facts_, module_, loom_func_like_cast(module_, function_)));
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
    loom_symbolic_expr_context_initialize(module_, &domain_, &facts_, &arena_,
                                          &expression_context_);
    IREE_ASSERT_OK(loom_view_region_table_initialize(
        &domain_, &expression_context_, &view_regions_));
    IREE_ASSERT_OK(loom_view_region_table_analyze(&view_regions_));
    loom_cfg_graph_t graph;
    IREE_ASSERT_OK(loom_cfg_graph_build(module_, region_, &arena_, &graph));
    IREE_ASSERT_OK(loom_channel_completion_analyze(
        &plan, &graph, &view_regions_, nullptr, &arena_, &completion_));
    ASSERT_EQ(completion_.requirement,
              LOOM_CHANNEL_COMPLETION_REQUIREMENT_NONE);
    ASSERT_EQ(plan.action_count, expected.size());
    if (expected.size() == 0) {
      EXPECT_EQ(completion_.actions, nullptr);
      EXPECT_EQ(completion_.payload_records, nullptr);
    }
    size_t index = 0;
    for (const auto& action : expected) {
      SCOPED_TRACE(index);
      EXPECT_EQ(completion_.actions[index].credits, action.credits);
      EXPECT_EQ(completion_.actions[index].maximum_admissions,
                action.maximum_admissions);
      ++index;
    }
  }

  void ExpectPayloadRecord(loom_value_id_t payload, loom_value_id_t record,
                           uint32_t maximum_admissions) {
    loom_channel_payload_record_t actual;
    ASSERT_TRUE(loom_channel_completion_try_lookup_payload_record(
        &completion_, payload, &actual));
    EXPECT_EQ(actual.value_id, record);
    EXPECT_EQ(actual.maximum_admissions, maximum_admissions);
  }

  void ExpectNoPayloadRecord(loom_value_id_t payload) {
    loom_channel_payload_record_t actual;
    EXPECT_FALSE(loom_channel_completion_try_lookup_payload_record(
        &completion_, payload, &actual));
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
  // Facts supplying the production view-region analysis.
  loom_value_fact_table_t facts_ = {};
  // Symbolic expressions shared by view-region analysis.
  loom_symbolic_expr_context_t expression_context_ = {};
  // Canonical view bases used during CFG payload projection.
  loom_view_region_table_t view_regions_ = {};
  // Retained result of the most recent analysis.
  loom_channel_completion_t completion_ = {};
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

TEST_F(ChannelCompletionTest, DirectPayloadsRetainTheirOwnedRecords) {
  loom_value_id_t reserved_payload;
  const auto write = Reserve(first_, &reserved_payload);
  loom_value_id_t acquired_payload;
  const auto read = Acquire(second_, &acquired_payload);
  const auto waited_payload = Wait(read);
  Publish(write);
  Release(read);
  Return();
  ExpectActions({{0, 1}, {0, 1}, {0, 1}, {1, 1}, {1, 1}});
  ExpectPayloadRecord(reserved_payload, write, 1);
  ExpectPayloadRecord(acquired_payload, read, 1);
  ExpectPayloadRecord(waited_payload, read, 1);
}

TEST_F(ChannelCompletionTest, LoopCarriesPayloadWithCurrentOwnedRecord) {
  loom_value_id_t first_payload;
  const auto first_read = Acquire(first_, &first_payload);
  const auto first_projection = Project(first_payload);
  auto* header = Block();
  const auto current_read = BlockArgument(header, read_type_);
  const auto current_payload = BlockArgument(header, view_type_);
  auto* body = Block();
  auto* finish = Block();
  Branch(header, {first_read, first_projection});
  At(header);
  Choose(body, finish);
  At(body);
  const auto current_projection = Project(current_payload);
  Release(current_read);
  loom_value_id_t next_payload;
  const auto next_read = Acquire(first_, &next_payload);
  const auto next_projection = Project(next_payload);
  Branch(header, {next_read, next_projection});
  At(finish);
  Release(current_read);
  Return();
  ExpectActions(
      {{0, UINT32_MAX}, {1, UINT32_MAX}, {0, UINT32_MAX}, {1, UINT32_MAX}});
  ExpectPayloadRecord(first_payload, first_read, UINT32_MAX);
  ExpectPayloadRecord(first_projection, first_read, UINT32_MAX);
  ExpectPayloadRecord(current_payload, current_read, UINT32_MAX);
  ExpectPayloadRecord(current_projection, current_read, UINT32_MAX);
  ExpectPayloadRecord(next_payload, next_read, UINT32_MAX);
  ExpectPayloadRecord(next_projection, next_read, UINT32_MAX);
}

TEST_F(ChannelCompletionTest, AmbiguousJoinInvalidatesForwardedPayload) {
  loom_value_id_t first_payload;
  const auto first_read = Acquire(first_, &first_payload);
  loom_value_id_t second_payload;
  const auto second_read = Acquire(first_, &second_payload);
  auto* left = Block();
  auto* delay = Block();
  auto* right = Block();
  auto* join = Block();
  const auto selected_payload = BlockArgument(join, view_type_);
  auto* finish = Block();
  const auto forwarded_payload = BlockArgument(finish, view_type_);
  Choose(left, delay);
  At(left);
  Branch(join, {first_payload});
  At(delay);
  Branch(right);
  At(right);
  Branch(join, {second_payload});
  At(join);
  Branch(finish, {selected_payload});
  At(finish);
  Release(first_read);
  Release(second_read);
  Return();
  ExpectActions({{0, 2}, {0, 2}, {1, 2}, {1, 2}});
  ExpectPayloadRecord(first_payload, first_read, 2);
  ExpectPayloadRecord(second_payload, second_read, 2);
  ExpectNoPayloadRecord(selected_payload);
  ExpectNoPayloadRecord(forwarded_payload);
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
