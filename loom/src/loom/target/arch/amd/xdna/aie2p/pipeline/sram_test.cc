// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sram.h"

#include <bitset>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"

namespace {

class SramRequestTest
    : public ::testing::TestWithParam<std::tuple<bool, uint16_t, uint32_t>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &registry_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&registry_));
    iree_host_size_t count;
    const auto* tables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&registry_, LOOM_DIALECT_LOW,
                                                 tables, count));
    IREE_ASSERT_OK(loom_context_finalize(&registry_));
    IREE_ASSERT_OK(loom_module_allocate(&registry_, IREE_SV("sram_request"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &context_.code));
    realization_.worker_versions = versions_;
    context_.code.realization = &realization_;
    context_.family = loom_xdna_npu2_array_family();
    context_.tiles = &tile_;
    context_.pool_tiles = pools_;
    context_.workers = &worker_;
    context_.core_engines = &engine_;
    context_.execution.workers = &engine_;
    tile_.coordinate = {7, 1};
    worker_.control_packets = &packet_;
    worker_.control_replies = &reply_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&registry_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  uint32_t Integer(loom_value_id_t value) {
    const auto* op = loom_value_def_op(loom_module_value(module_, value));
    const auto attrs = loom_low_const_attrs(op);
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (attrs.entries[i].name_id == context_.code.integer_name) {
        return static_cast<uint32_t>(attrs.entries[i].value.i64);
      }
    }
    ADD_FAILURE() << "missing integer attribute";
    return 0;
  }

  bool Is(const loom_op_t* op, uint32_t descriptor) {
    return loom_low_op_isa(op) &&
           loom_low_op_descriptor(op) ==
               loom_low_descriptor_set_descriptor_ordinal(
                   context_.code.descriptors,
                   &context_.code.descriptors->descriptors[descriptor]);
  }

  // Module and native emitter storage backing.
  iree_arena_block_pool_t blocks_;
  // Transient native emission state.
  iree_arena_allocator_t arena_;
  // Ordinary Low operation registry.
  loom_context_t registry_;
  // Owns the emitted callable and its values.
  loom_module_t* module_ = nullptr;
  // Selected worker, route, and memory-tile facts.
  loom_aie2p_native_context_t context_ = {};
  // Standalone helper emission has no optional function version.
  loom_pipeline_realization_t realization_ = {};
  // One absent function-version owner.
  loom_target_function_version_t* versions_[1] = {};
  // Memory tile in a nonzero column, whose origin is not part of the request.
  loom_aie2p_native_tile_t tile_ = {};
  // The channel's selected backing pool.
  loom_aie2p_native_tile_t* pools_[1] = {&tile_};
  // Core owning the reply stream.
  loom_aie2p_native_worker_t worker_ = {};
  // Physical engine and representative worker.
  iree_host_size_t engine_ = 0;
  // Selected request route.
  uint8_t packet_ = 17;
  // Selected reply route.
  uint8_t reply_ = 23;
};

TEST_P(SramRequestTest, ExactRequestConsumesHeaderThenSuccessBeforeRetry) {
  const auto [acquire, lock, credits] = GetParam();
  const loom_aie2p_native_channel_t channel = {};
  loom_symbol_ref_t symbol;
  IREE_ASSERT_OK(loom_aie2p_native_sram_lock_helper(&context_, &channel, lock,
                                                    acquire, credits, &symbol));
  const auto* function = module_->symbols.entries[symbol.symbol_id].defining_op;
  auto* body = loom_low_func_def_body(function);
  const auto* entry = loom_region_entry_block(body);
  ASSERT_TRUE(loom_low_br_isa(entry->last_op));
  const auto* request = loom_low_br_dest(entry->last_op);
  std::vector<const loom_op_t*> stream;
  for (const auto* op = request->first_op; op; op = op->next_op) {
    if (Is(op, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32) ||
        Is(op, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32) ||
        Is(op, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32)) {
      stream.push_back(op);
    }
  }
  ASSERT_EQ(stream.size(), 4u);
  EXPECT_TRUE(Is(stream[0], AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32));
  EXPECT_TRUE(Is(stream[1], AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32));
  EXPECT_TRUE(Is(stream[2], AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32));
  EXPECT_TRUE(Is(stream[3], AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32));
  const uint32_t route = Integer(loom_op_operands(stream[0])[0]);
  EXPECT_EQ(route & UINT32_C(0x7fffffff), packet_);
  EXPECT_EQ(std::bitset<32>(route).count() % 2, 1u);
  const uint32_t command = Integer(loom_op_operands(stream[1])[0]);
  const int32_t delta =
      acquire ? -static_cast<int32_t>(credits) : static_cast<int32_t>(credits);
  const uint32_t address = 0xd0000u | (lock << 10) |
                           (static_cast<uint32_t>(acquire) << 9) |
                           ((static_cast<uint32_t>(delta) & 127) << 2);
  // One exact word, including low address bits. Prefix reads would perform
  // additional semaphore operations and corrupt credits.
  EXPECT_EQ(command & 0xfffffu, address);
  EXPECT_EQ((command >> 20) & 3, 0u);
  EXPECT_EQ((command >> 22) & 3, 1u);
  EXPECT_EQ((command >> 24) & 31, reply_);
  EXPECT_EQ(std::bitset<32>(command).count() % 2, 1u);
  const auto* branch = request->last_op;
  ASSERT_TRUE(loom_low_cond_br_isa(branch));
  const auto* pending = loom_value_def_op(
      loom_module_value(module_, loom_low_cond_br_condition(branch)));
  ASSERT_TRUE(Is(pending, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32));
  EXPECT_EQ(loom_op_operands(pending)[0], loom_op_results(stream[3])[0]);
  EXPECT_EQ(Integer(loom_op_operands(pending)[1]), 0u);
  EXPECT_EQ(loom_low_cond_br_true_dest(branch), request);
  const auto* complete = loom_low_cond_br_false_dest(branch);
  EXPECT_TRUE(loom_low_return_isa(complete->first_op));
}

INSTANTIATE_TEST_SUITE_P(
    OperationLockAndCredit, SramRequestTest,
    ::testing::Combine(::testing::Bool(), ::testing::Values<uint16_t>(0, 63),
                       ::testing::Values<uint32_t>(1, 63)));

class SramReplyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    invocations_.channels = channels_;
    invocations_.channel_count = 3;
    invocations_.channel_order = order_;
    worker_.execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE;
    worker_.channels.remote_count = 1;
    worker_.channels.count = 3;
    worker_.channels.accesses = accesses_;
    worker_.channels.indices = indices_;
    worker_.transfer_tasks = tasks_;
    source_.channels.actions = actions_;
    source_.channels.action_count = 1;
    source_.completion.actions = completions_;
    block_.region_index = 1;
    for (size_t i = 0; i < 3; ++i) {
      channels_[i].source = &resources_[i];
      actions_[i].op = &operations_[i];
      actions_[i].channel = &resources_[0].identity;
      actions_[i].kind = LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE;
      operations_[i].parent_block = &block_;
      operations_[i].block_ordinal = 3 + i * 4;
      issues_[i].parent_block = waits_[i].parent_block = &block_;
      issues_[i].block_ordinal = 2 + i * 4;
      waits_[i].block_ordinal = 4 + i * 4;
      copies_[i].request.op = &issues_[i];
      transfers_[i].source = &copies_[i];
      transfers_[i].completion = &waits_[i];
      transfers_[i].worker_index = 7;
      tasks_[i].first_transfer = &transfers_[i];
      tasks_[i].path = &path_;
    }
    path_.completion_tokens = 2;
    accesses_[1].window = loom_xdna_array_load_window(
        loom_xdna_npu2_array_family(), {0, 2}, {0, 2});
    accesses_[2].batch_publication = true;
  }

  const loom_op_t* Conflict() {
    return loom_aie2p_native_sram_reply_conflict(&invocations_, &source_,
                                                 &worker_, 7);
  }

  // Retained source-order block, with no groups live across its exit.
  loom_block_t block_ = {};
  // Explicit source actions, ordered independently of DMA task grouping.
  loom_op_t operations_[3] = {};
  // Issue sites opening the reply stream's live intervals.
  loom_op_t issues_[3] = {};
  // Authored waits closing those intervals.
  loom_op_t waits_[3] = {};
  // Producer-classified channel actions.
  loom_channel_plan_action_t actions_[3] = {};
  // Retained number of credits actually returned by each action.
  loom_channel_completion_action_t completions_[3] = {};
  // Distinct protocol identities in stable storage.
  loom_pipeline_resource_channel_t resources_[3] = {};
  // Construction-owned channel inventory.
  loom_pipeline_invocation_channel_t channels_[3] = {};
  // Identity addresses follow their contiguous source-array order.
  iree_host_size_t order_[3] = {0, 1, 2};
  // Worker-local access ordinals for each construction channel.
  uint32_t indices_[3] = {0, 1, 2};
  // Ordinary SRAM, direct compute memory, and complete-batch publication.
  loom_aie2p_native_channel_access_t accesses_[3] = {};
  // Canonical invocation identities supplied by shared analysis.
  loom_pipeline_invocations_t invocations_ = {};
  // Retained action/completion information for this occurrence.
  loom_pipeline_worker_t source_ = {};
  // Physical core and its selected task sequence.
  loom_aie2p_native_worker_t worker_ = {};
  // Shared selected path whose DMA completions consume scalar replies.
  loom_aie2p_native_dma_path_t path_ = {};
  // Transfer issue identities supplied by async analysis.
  loom_kernel_async_transfer_t copies_[3] = {};
  // Retained occurrence-qualified completion intervals.
  loom_aie2p_native_transfer_t transfers_[3] = {};
  // Ordered tasks; tests link the participating prefix explicitly.
  loom_aie2p_native_transfer_task_t tasks_[3] = {};
};

TEST_F(SramReplyTest, AdmissionRequiresPreviousDmaResponseToRetire) {
  for (const auto kind : {LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE,
                          LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE}) {
    actions_[0].kind = kind;
    EXPECT_EQ(Conflict(), &operations_[0]);
    operations_[0].block_ordinal = 1;
    EXPECT_EQ(Conflict(), nullptr);
    operations_[0].block_ordinal = 5;
    EXPECT_EQ(Conflict(), nullptr);
    operations_[0].block_ordinal = 3;
  }
}

TEST_F(SramReplyTest, OnlyActionsProducingRepliesClaimTheStream) {
  for (const auto kind : {LOOM_CHANNEL_PLAN_ACTION_KIND_PUBLISH,
                          LOOM_CHANNEL_PLAN_ACTION_KIND_RELEASE}) {
    actions_[0].kind = kind;
    EXPECT_EQ(Conflict(), nullptr);
    completions_[0].credits = 2;
    EXPECT_EQ(Conflict(), &operations_[0]);
    completions_[0].credits = 0;
  }
  actions_[0].kind = LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE;
  for (size_t channel : {1u, 2u}) {
    actions_[0].channel = &resources_[channel].identity;
    EXPECT_EQ(Conflict(), nullptr);
  }
  actions_[0].channel = &resources_[0].identity;
  path_.completion_tokens = 0;
  EXPECT_EQ(Conflict(), nullptr);
}

TEST_F(SramReplyTest, LaterTaskRetainsThePendingActionFrontier) {
  tasks_[0].next = &tasks_[1];
  operations_[0].block_ordinal = 7;
  EXPECT_EQ(Conflict(), &operations_[0]);
  tasks_[1].next = &tasks_[2];
  source_.channels.action_count = 3;
  operations_[0].block_ordinal = 5;
  operations_[1].block_ordinal = 9;
  EXPECT_EQ(Conflict(), &operations_[2]);
  operations_[2].block_ordinal = 13;
  EXPECT_EQ(Conflict(), nullptr);
}

TEST_F(SramReplyTest, BlockRetirementKeepsSeparateControlPathsDisjoint) {
  loom_block_t preceding = {}, following = {};
  preceding.region_index = 0;
  following.region_index = 2;
  operations_[0].parent_block = &preceding;
  EXPECT_EQ(Conflict(), nullptr);
  operations_[0].parent_block = &following;
  EXPECT_EQ(Conflict(), nullptr);
  tasks_[0].next = &tasks_[1];
  issues_[1].parent_block = waits_[1].parent_block = &following;
  operations_[0].block_ordinal = 7;
  EXPECT_EQ(Conflict(), &operations_[0]);
}

}  // namespace
