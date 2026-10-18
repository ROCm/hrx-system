// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/channel_materialization.h"

#include <vector>

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

// Exercise the consuming rewrite with selected mechanics expressed as ordinary
// calls. Admission returns the next cursor; publication/copy use the reserved
// cursor. These fixtures inspect SSA edges and effect order, not target code.
class ChannelMaterializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    Register(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    Register(LOOM_DIALECT_CHANNEL, loom_channel_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("materialization"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    const auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    admission_ = Declare("admit", {i32}, {i32});
    publication_ = Declare("publish", {i32}, {});
    submission_ = Declare("submit", {i32, i32}, {});
    completion_ = Declare("complete", {i32}, {});

    loom_type_id_t payload;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_,
        loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(1), 0),
        &payload));
    loom_type_t channel;
    IREE_ASSERT_OK(loom_channel_type_make(module_, payload, &channel));
    IREE_ASSERT_OK(loom_read_type_make(module_, 0, payload,
                                       (loom_read_type_mode_t)0, &read_type_));
    IREE_ASSERT_OK(loom_write_type_make(module_, payload, &write_type_));
    view_type_ = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                     loom_dim_pack_static(1), 0);
    const loom_type_t arguments[] = {
        channel, loom_type_scalar(LOOM_SCALAR_TYPE_I1), i32};
    IREE_ASSERT_OK(loom_func_def_build(
        &builder_, 0, 0, /*linkage=*/0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), Symbol("worker"), arguments,
        IREE_ARRAYSIZE(arguments), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function_));
    region_ = loom_func_def_body(function_);
    loom_builder_enter_region(&builder_, function_, region_);
    entry_ = loom_region_entry_block(region_);
    channel_ = entry_->arg_ids[0];
    condition_ = entry_->arg_ids[1];
    initial_ = entry_->arg_ids[2];
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

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t string;
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &string));
    IREE_CHECK_OK(loom_module_add_symbol(module_, string, &symbol));
    return {0, symbol};
  }

  loom_symbol_ref_t Declare(const char* name,
                            const std::vector<loom_type_t>& arguments,
                            const std::vector<loom_type_t>& results) {
    const auto symbol = Symbol(name);
    loom_op_t* declaration;
    IREE_CHECK_OK(loom_func_decl_build(
        &builder_, 0, 0, 0, LOOM_STRING_ID_INVALID, LOOM_STRING_ID_INVALID, 0,
        0, 0, 0, loom_symbol_ref_null(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        arguments.data(), arguments.size(), results.data(), results.size(),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &declaration));
    return symbol;
  }

  loom_block_t* Block() {
    loom_block_t* block;
    IREE_CHECK_OK(loom_region_append_block(module_, region_, &block));
    return block;
  }

  loom_value_id_t Argument(loom_type_t type) {
    loom_value_id_t argument;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &argument));
    IREE_CHECK_OK(loom_block_add_arg(module_, entry_, argument));
    return argument;
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

  loom_value_id_t Reserve() {
    loom_op_t* op;
    IREE_CHECK_OK(loom_channel_reserve_build(&builder_, channel_, write_type_,
                                             view_type_, LOOM_LOCATION_UNKNOWN,
                                             &op));
    return loom_channel_reserve_write(op);
  }

  void Publish(loom_value_id_t write) {
    loom_op_t* op;
    IREE_ASSERT_OK(loom_channel_publish_build(&builder_, write,
                                              LOOM_LOCATION_UNKNOWN, &op));
  }

  enum class Boundary { Helper, Execution };

  void Verify() {
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  void Materialize(Boundary boundary) {
    Verify();
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, region_, &arena_, &domain_));
    const loom_channel_identity_t output_identity = {channel_};
    const loom_channel_identity_t input_identity = {input_};
    const loom_channel_identity_t notification_identity = {notification_};
    std::vector<loom_channel_plan_binding_t> bindings = {
        {channel_, &output_identity}};
    if (input_ != LOOM_VALUE_ID_INVALID) {
      bindings.push_back({input_, &input_identity});
      bindings.push_back({notification_, &notification_identity});
    }
    loom_channel_plan_t plan;
    loom_channel_plan_rejection_t rejection;
    IREE_ASSERT_OK(loom_channel_plan_build(&domain_, bindings.data(),
                                           bindings.size(), nullptr, &arena_,
                                           &plan, &rejection));
    ASSERT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
    MaterializePlan(plan, initial_, boundary);
    loom_local_value_domain_release(&domain_);
    Verify();
  }

  void MaterializePlan(const loom_channel_plan_t& plan, loom_value_id_t initial,
                       Boundary boundary) {
    std::vector<loom_type_t> carriers(plan.value_count);
    std::vector<uint8_t> erased(plan.value_count, 0);
    for (loom_value_ordinal_t i = 0; i < plan.value_count; ++i) {
      const auto type =
          loom_module_value_type(module_, plan.value_domain->value_ids[i]);
      if (loom_read_type_isa(type) || loom_write_type_isa(type)) {
        carriers[i] = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
      }
    }
    for (iree_host_size_t i = 0; i < plan.action_count; ++i) {
      if (loom_channel_reserve_isa(plan.actions[i].op)) {
        erased[loom_local_value_domain_ordinal(
            plan.value_domain, loom_channel_reserve_view(plan.actions[i].op))] =
            1;
      }
    }
    const loom_channel_materialization_options_t options = {
        carriers.data(),
        erased.data(),
        &initial,
        1,
        {Emit, boundary == Boundary::Execution ? Exit : nullptr, this}};
    loom_rewriter_t rewriter;
    loom_rewriter_initialize(&rewriter, module_, &arena_);
    loom_cfg_graph_t graph;
    IREE_ASSERT_OK(loom_cfg_graph_build(module_, plan.value_domain->region,
                                        &arena_, &graph));
    IREE_ASSERT_OK(
        loom_channel_materialize(&rewriter, &plan, &graph, &options));
    loom_rewriter_deinitialize(&rewriter);
  }

  static iree_status_t Emit(void* user_data, loom_rewriter_t* rewriter,
                            const loom_channel_plan_action_t* action,
                            loom_value_id_t* state,
                            iree_host_size_t state_count,
                            loom_value_id_t* results) {
    auto& self = *static_cast<ChannelMaterializationTest*>(user_data);
    EXPECT_EQ(state_count, 1u);
    const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* call;
    if (loom_channel_reserve_isa(action->op)) {
      results[0] = state[0];
      IREE_RETURN_IF_ERROR(loom_func_call_build(
          &rewriter->builder, 0, 0, 0, 0, self.admission_, state, 1, &type, 1,
          nullptr, 0, action->op->location, &call));
      state[0] = loom_op_results(call)[0];
    } else {
      const auto symbol = loom_channel_publish_isa(action->op)
                              ? self.publication_
                              : self.submission_;
      IREE_RETURN_IF_ERROR(loom_func_call_build(
          &rewriter->builder, 0, 0, 0, 0, symbol, loom_op_operands(action->op),
          action->op->operand_count, nullptr, 0, nullptr, 0,
          action->op->location, &call));
    }
    return iree_ok_status();
  }

  static iree_status_t Exit(void* user_data, loom_rewriter_t* rewriter,
                            const loom_op_t* terminator,
                            const loom_value_id_t* state,
                            iree_host_size_t state_count) {
    auto& self = *static_cast<ChannelMaterializationTest*>(user_data);
    EXPECT_TRUE(loom_func_return_isa(terminator));
    EXPECT_EQ(state_count, 1u);
    loom_op_t* call;
    return loom_func_call_build(&rewriter->builder, 0, 0, 0, 0,
                                self.completion_, state, state_count, nullptr,
                                0, nullptr, 0, terminator->location, &call);
  }

  void ExpectCall(const loom_op_t* op, loom_symbol_ref_t symbol,
                  const std::vector<loom_value_id_t>& arguments) {
    ASSERT_TRUE(loom_func_call_isa(op));
    EXPECT_EQ(loom_func_call_callee(op).symbol_id, symbol.symbol_id);
    ASSERT_EQ(op->operand_count, arguments.size());
    for (size_t i = 0; i < arguments.size(); ++i) {
      EXPECT_EQ(loom_op_const_operands(op)[i], arguments[i]);
    }
  }

  void ExpectBranch(const loom_block_t* block, loom_block_t* destination,
                    loom_value_id_t state) {
    const auto* branch = block->last_op;
    ASSERT_TRUE(loom_cfg_br_isa(branch));
    EXPECT_EQ(loom_cfg_br_dest(branch), destination);
    ASSERT_EQ(branch->operand_count, 1u);
    EXPECT_EQ(loom_op_const_operands(branch)[0], state);
  }

  // Source and transform allocations share this backing pool.
  iree_arena_block_pool_t pool_;
  // Plan and mutation storage retained through each assertion.
  iree_arena_allocator_t arena_;
  // Registered source dialects used by the API fixtures.
  loom_context_t context_;
  // Owned source module.
  loom_module_t* module_ = nullptr;
  // Current source insertion point.
  loom_builder_t builder_ = {};
  // Captured original value ordinals retained through materialization.
  loom_local_value_domain_t domain_ = {};
  // Ordinary callable whose channel actions are materialized.
  loom_op_t* function_;
  // Original flat execution region.
  loom_region_t* region_;
  // Entry holding incoming protocol values and the initial cursor.
  loom_block_t* entry_;
  // Source destination channel.
  loom_value_id_t channel_;
  // Incoming read transferred by the copy fixture.
  loom_value_id_t input_ = LOOM_VALUE_ID_INVALID;
  // Independent write exposing reservation/publication effect order.
  loom_value_id_t notification_ = LOOM_VALUE_ID_INVALID;
  // Runtime condition controlling the CFG fixtures.
  loom_value_id_t condition_;
  // Caller-supplied initial cursor, deliberately not an assumed zero.
  loom_value_id_t initial_;
  // Owned write for one i32 record.
  loom_type_t write_type_;
  // Owned read for the copy fixture's incoming record.
  loom_type_t read_type_;
  // Borrowed record view erased when no source use needs it.
  loom_type_t view_type_;
  // Selected admission call that returns the next cursor.
  loom_symbol_ref_t admission_;
  // Selected publication call consuming the reserved cursor.
  loom_symbol_ref_t publication_;
  // Selected copy call consuming a read and a reservation.
  loom_symbol_ref_t submission_;
  // Selected invocation completion, absent on an ordinary helper return.
  loom_symbol_ref_t completion_;
};

TEST_F(ChannelMaterializationTest, AdmissionPrecedesIndependentPublication) {
  input_ = Argument(read_type_);
  notification_ = Argument(write_type_);
  const auto write = Reserve();
  Publish(notification_);
  loom_op_t* copy;
  IREE_ASSERT_OK(loom_channel_copy_build(&builder_, input_, write,
                                         LOOM_LOCATION_UNKNOWN, &copy));
  Return();
  Materialize(Boundary::Execution);

  const auto* admit = entry_->first_op;
  ExpectCall(admit, admission_, {initial_});
  const auto next = loom_op_const_results(admit)[0];
  ExpectCall(admit->next_op, publication_, {notification_});
  ExpectCall(admit->next_op->next_op, submission_, {input_, initial_});
  ExpectCall(admit->next_op->next_op->next_op, completion_, {next});
  EXPECT_TRUE(loom_func_return_isa(entry_->last_op));
}

TEST_F(ChannelMaterializationTest, HelperReturnDoesNotDrain) {
  Publish(Reserve());
  Return();
  Materialize(Boundary::Helper);
  const auto* admit = entry_->first_op;
  ExpectCall(admit, admission_, {initial_});
  ExpectCall(admit->next_op, publication_, {initial_});
  EXPECT_EQ(admit->next_op->next_op, entry_->last_op);
  EXPECT_TRUE(loom_func_return_isa(entry_->last_op));
}

TEST_F(ChannelMaterializationTest, ClonedOccurrencesKeepLoopOwnershipSeparate) {
  const auto width = Argument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  loom_type_id_t payload;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_,
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_dynamic(width), 0),
      &payload));
  loom_type_t channel;
  IREE_ASSERT_OK(loom_channel_type_make(module_, payload, &channel));
  IREE_ASSERT_OK(loom_module_set_value_type(module_, channel_, channel));
  IREE_ASSERT_OK(loom_write_type_make(module_, payload, &write_type_));
  view_type_ = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                   loom_dim_pack_dynamic(width), 0);
  const auto first_write = Reserve();
  auto* loop = Block();
  auto* repeat = Block();
  auto* exit = Block();
  loom_value_id_t carried_write;
  IREE_ASSERT_OK(
      loom_module_define_value(module_, write_type_, &carried_write));
  IREE_ASSERT_OK(loom_block_add_arg(module_, loop, carried_write));
  loom_op_t* branch;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, loop, &first_write, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  At(loop);
  Choose(repeat, exit);
  At(repeat);
  Publish(carried_write);
  const auto next_write = Reserve();
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, loop, &next_write, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  At(exit);
  Publish(carried_write);
  Return();
  Verify();

  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(module_, region_,
                                                            &arena_, &domain_));
  const auto initial_ordinal =
      loom_local_value_domain_ordinal(&domain_, initial_);
  const auto write_ordinal =
      loom_local_value_domain_ordinal(&domain_, carried_write);
  // The same source argument is bound by two different constructions. Their
  // identities remain distinct even when the callable and storage could alias.
  const loom_channel_identity_t identities[] = {{channel_}, {channel_}};
  loom_channel_materialization_instance_t instances[2];
  loom_channel_plan_t plans[2];
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &arena_);
  loom_builder_set_block(&rewriter.builder, loom_module_block(module_));
  rewriter.builder.ip.parent_op = nullptr;
  const char* names[] = {"first_worker", "second_worker"};
  for (size_t i = 0; i < 2; ++i) {
    const loom_channel_plan_binding_t binding = {channel_, &identities[i]};
    loom_channel_plan_rejection_t rejection;
    IREE_ASSERT_OK(loom_channel_plan_build(&domain_, &binding, 1, nullptr,
                                           &arena_, &plans[i], &rejection));
    ASSERT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
    IREE_ASSERT_OK(loom_channel_materialization_clone(
        &rewriter, loom_func_like_cast(module_, function_), &plans[i],
        Symbol(names[i]), &instances[i]));
    EXPECT_TRUE(loom_local_value_domain_is_acquired(&domain_));
    EXPECT_FALSE(
        loom_local_value_domain_is_acquired(&instances[i].value_domain));
    ASSERT_EQ(instances[i].plan.action_count, 4u);
    ASSERT_EQ(instances[i].plan.return_count, 1u);
    EXPECT_EQ(instances[i].plan.returns[0],
              instances[i].value_domain.region->blocks[3]->last_op);
    for (size_t action = 0; action < plans[i].action_count; ++action) {
      EXPECT_NE(instances[i].plan.actions[action].op,
                plans[i].actions[action].op);
      EXPECT_EQ(instances[i].plan.actions[action].channel, &identities[i]);
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_local_value_domain_release(&domain_);

  // Consume in the opposite order to construction. Each loop carries its own
  // physical write and cursor; neither rewrite can alter the shared definition.
  for (size_t i : {1u, 0u}) {
    auto& instance = instances[i];
    auto* body = loom_func_like_body(instance.function);
    auto* entry = loom_region_entry_block(body);
    const auto initial = instance.value_domain.value_ids[initial_ordinal];
    const auto write = instance.value_domain.value_ids[write_ordinal];
    const auto channel_type =
        loom_module_value_type(module_, entry->arg_ids[0]);
    const auto payload_type = loom_type_table_get(
        &module_->types, loom_channel_type_payload(channel_type));
    EXPECT_EQ(loom_dim_value_id(loom_type_dim(payload_type, 0)),
              entry->arg_ids[3]);
    EXPECT_NE(entry->arg_ids[3], width);
    EXPECT_NE(initial, initial_);
    EXPECT_EQ(write, body->blocks[1]->arg_ids[0]);
    loom_local_value_domain_restore(&instance.value_domain);
    EXPECT_EQ(loom_channel_plan_channel(&instance.plan, write), &identities[i]);
    MaterializePlan(instance.plan, initial, Boundary::Execution);
    loom_local_value_domain_release(&instance.value_domain);
    ExpectCall(entry->first_op, admission_, {initial});
    ExpectCall(body->blocks[2]->first_op, publication_, {write});
    const auto* final_publication = body->blocks[3]->first_op;
    ExpectCall(final_publication, publication_, {write});
    ExpectCall(final_publication->next_op, completion_,
               {body->blocks[1]->arg_ids[1]});
    EXPECT_TRUE(loom_channel_reserve_isa(entry_->first_op));
    EXPECT_TRUE(loom_channel_publish_isa(repeat->first_op));
    EXPECT_EQ(loop->arg_ids[0], carried_write);
    Verify();
  }
  loom_local_value_domain_restore(&domain_);
  EXPECT_EQ(loom_channel_plan_channel(&plans[0], carried_write),
            &identities[0]);
  EXPECT_EQ(loom_channel_plan_channel(&plans[1], carried_write),
            &identities[1]);
}

TEST_F(ChannelMaterializationTest, ReconvergenceRetainsEachPathsCursor) {
  auto* producing = Block();
  auto* bypass = Block();
  auto* exit = Block();
  Choose(producing, bypass);
  At(producing);
  Publish(Reserve());
  Branch(exit);
  At(bypass);
  Branch(exit);
  At(exit);
  Return();
  Materialize(Boundary::Execution);

  const auto* decision = entry_->last_op;
  ASSERT_TRUE(loom_cfg_cond_br_isa(decision));
  EXPECT_EQ(loom_op_const_successors(decision)[0], producing);
  EXPECT_EQ(loom_op_const_successors(decision)[1], bypass);
  EXPECT_EQ(producing->arg_count, 0u);
  EXPECT_EQ(bypass->arg_count, 0u);
  ExpectCall(producing->first_op, admission_, {initial_});
  ExpectCall(producing->first_op->next_op, publication_, {initial_});
  ExpectBranch(producing, exit, loom_op_const_results(producing->first_op)[0]);
  ExpectBranch(bypass, exit, initial_);
  ExpectCall(exit->first_op, completion_, {exit->arg_ids[0]});
}

TEST_F(ChannelMaterializationTest, LoopExitUsesTheCarriedCursor) {
  auto* header = Block();
  auto* body = Block();
  auto* exit = Block();
  Branch(header);
  At(header);
  Choose(body, exit);
  At(body);
  Publish(Reserve());
  Branch(header);
  At(exit);
  Return();
  Materialize(Boundary::Execution);

  ExpectBranch(entry_, header, initial_);
  const auto* decision = header->last_op;
  ASSERT_TRUE(loom_cfg_cond_br_isa(decision));
  EXPECT_EQ(loom_op_const_successors(decision)[0], body);
  EXPECT_EQ(loom_op_const_successors(decision)[1], exit);
  EXPECT_EQ(body->arg_count, 0u);
  EXPECT_EQ(exit->arg_count, 0u);
  ExpectCall(body->first_op, admission_, {header->arg_ids[0]});
  ExpectCall(body->first_op->next_op, publication_, {header->arg_ids[0]});
  ExpectBranch(body, header, loom_op_const_results(body->first_op)[0]);
  ExpectCall(exit->first_op, completion_, {header->arg_ids[0]});
}

TEST_F(ChannelMaterializationTest, PhysicalBlockOrderDoesNotChangeCursorFlow) {
  auto* exit = Block();
  auto* producing = Block();
  Branch(producing);
  At(producing);
  Publish(Reserve());
  Branch(exit);
  At(exit);
  Return();
  Materialize(Boundary::Execution);

  EXPECT_EQ(producing->arg_count, 0u);
  EXPECT_EQ(exit->arg_count, 0u);
  ExpectCall(producing->first_op, admission_, {initial_});
  ExpectCall(producing->first_op->next_op, publication_, {initial_});
  ExpectCall(exit->first_op, completion_,
             {loom_op_const_results(producing->first_op)[0]});
}

TEST_F(ChannelMaterializationTest, ConditionalBackedgeThreadsUpdatedCursor) {
  auto* header = Block();
  auto* exit = Block();
  Branch(header);
  At(header);
  Publish(Reserve());
  Choose(header, exit);
  At(exit);
  Return();
  Materialize(Boundary::Execution);

  ExpectBranch(entry_, header, initial_);
  ExpectCall(header->first_op, admission_, {header->arg_ids[0]});
  ExpectCall(header->first_op->next_op, publication_, {header->arg_ids[0]});
  const auto next = loom_op_const_results(header->first_op)[0];
  const auto* decision = header->last_op;
  ASSERT_TRUE(loom_cfg_cond_br_isa(decision));
  ExpectBranch(loom_op_const_successors(decision)[0], header, next);
  EXPECT_EQ(loom_op_const_successors(decision)[1], exit);
  EXPECT_EQ(exit->arg_count, 0u);
  ExpectCall(exit->first_op, completion_, {next});
}

TEST_F(ChannelMaterializationTest, UnreachableCycleKeepsItsOwnCursor) {
  Return();
  auto* header = Block();
  auto* exit = Block();
  At(header);
  Publish(Reserve());
  Choose(header, exit);
  At(exit);
  Return();
  Materialize(Boundary::Execution);

  ExpectCall(entry_->first_op, completion_, {initial_});
  ExpectCall(header->first_op, admission_, {header->arg_ids[0]});
  ExpectCall(header->first_op->next_op, publication_, {header->arg_ids[0]});
  const auto next = loom_op_const_results(header->first_op)[0];
  const auto* decision = header->last_op;
  ASSERT_TRUE(loom_cfg_cond_br_isa(decision));
  ExpectBranch(loom_op_const_successors(decision)[0], header, next);
  ExpectBranch(loom_op_const_successors(decision)[1], exit, next);
  ExpectCall(exit->first_op, completion_, {exit->arg_ids[0]});
}

}  // namespace
}  // namespace loom
