// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_progress.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

enum SyntheticProgressClass {
  kSyntheticProgressPipe = 7,
  kSyntheticProgressScoreboard = UINT16_MAX - 1,
};

iree_string_view_t SyntheticProgressClassName(uint16_t progress_class_id) {
  switch (progress_class_id) {
    case kSyntheticProgressPipe:
      return IREE_SV("synthetic.pipe");
    case kSyntheticProgressScoreboard:
      return IREE_SV("synthetic.scoreboard");
    default:
      return IREE_SV("unknown");
  }
}

struct PacketProgressTestState {
  loom_low_descriptor_t descriptor = {};
  loom_low_descriptor_set_t descriptor_set = {};
  loom_module_t module = {};
  loom_op_t function_op = {};
  loom_region_t region = {};
  loom_block_t block = {};
  loom_block_t* region_blocks[1] = {};
  loom_low_schedule_block_t blocks[1] = {};
  // Stable node storage for the requested packet domain.
  std::vector<loom_low_schedule_node_t> nodes;
  // Final packet order over |nodes|.
  std::vector<uint32_t> scheduled_node_indices;
  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
};

class LowPacketProgressTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<LowPacketProgressTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected progress allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializeStorage(iree_host_size_t block_size = 4096) {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate},
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void DeinitializeStorage() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void SetUp() override {
    InitializeStorage();
    InitializePacketProgressTestState(&state_);
  }

  void TearDown() override { DeinitializeStorage(); }

  static void InitializePacketProgressTestState(PacketProgressTestState* state,
                                                uint32_t packet_count = 2) {
    state->nodes.resize(packet_count);
    state->scheduled_node_indices.resize(packet_count);
    state->descriptor_set.descriptors = &state->descriptor;
    state->descriptor_set.descriptor_count = 1;
    state->descriptor_set.descriptor_ordinal_count = 1;

    state->region_blocks[0] = &state->block;
    state->region.block_count = IREE_ARRAYSIZE(state->region_blocks);
    state->region.block_capacity = IREE_ARRAYSIZE(state->region_blocks);
    state->region.blocks = state->region_blocks;
    state->block.parent_region = &state->region;
    state->block.region_index = 0;

    state->blocks[0].block = &state->block;
    state->blocks[0].node_start = 0;
    state->blocks[0].node_count = packet_count;
    state->blocks[0].scheduled_node_start = 0;
    state->blocks[0].scheduled_node_count = packet_count;

    for (uint32_t i = 0; i < packet_count; ++i) {
      state->nodes[i].block_index = 0;
      state->nodes[i].scheduled_ordinal = i;
      state->nodes[i].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
      state->nodes[i].descriptor = &state->descriptor;
      state->scheduled_node_indices[i] = i;
    }
    state->nodes.back().kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL;
    state->nodes.back().descriptor = nullptr;

    state->schedule.module = &state->module;
    state->schedule.function_op = &state->function_op;
    state->schedule.target.descriptor_set = &state->descriptor_set;
    state->schedule.blocks = state->blocks;
    state->schedule.block_count = IREE_ARRAYSIZE(state->blocks);
    state->schedule.nodes = state->nodes.data();
    state->schedule.node_count = packet_count;
    state->schedule.scheduled_node_indices =
        state->scheduled_node_indices.data();
    state->schedule.scheduled_node_count = packet_count;

    state->allocation.module = &state->module;
    state->allocation.function_op = &state->function_op;
    state->allocation.target.descriptor_set = &state->descriptor_set;
  }

  // Reusable backing for published progress records and query indexes.
  iree_arena_block_pool_t block_pool_;
  // Progress and index storage for one compilation epoch.
  iree_arena_allocator_t arena_;
  // Trusted schedule and allocation supplied to the progress builder.
  PacketProgressTestState state_;
  // Attempted backing allocations, excluding frees.
  iree_host_size_t allocation_count_ = 0;
  // Allocation selected for injected failure, or SIZE_MAX.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

void EmitEvent(loom_low_packet_progress_emit_fn_t emit, void* emit_user_data,
               uint16_t progress_class_id,
               loom_low_packet_progress_action_t action, uint32_t units) {
  const loom_low_packet_progress_event_t event = {
      .progress_class_id = progress_class_id,
      .action = action,
      .units = units,
  };
  emit(emit_user_data, &event);
}

void SyntheticProgressQuery(void* user_data,
                            const loom_low_schedule_table_t* schedule,
                            const loom_low_allocation_table_t* allocation,
                            const loom_low_packet_view_t* packet,
                            loom_low_packet_progress_emit_fn_t emit,
                            void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  if (packet->node_index == 0) {
    EmitEvent(emit, emit_user_data, kSyntheticProgressPipe,
              LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 2);
    return;
  }
  EmitEvent(emit, emit_user_data, kSyntheticProgressScoreboard,
            LOOM_LOW_PACKET_PROGRESS_ACTION_RESET, 0);
  EmitEvent(emit, emit_user_data, kSyntheticProgressPipe,
            LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 1);
}

void EmptyProgressQuery(void* user_data,
                        const loom_low_schedule_table_t* schedule,
                        const loom_low_allocation_table_t* allocation,
                        const loom_low_packet_view_t* packet,
                        loom_low_packet_progress_emit_fn_t emit,
                        void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  (void)packet;
  (void)emit;
  (void)emit_user_data;
}

struct ProgressQueryAudit {
  // Number of packets visited by the provider.
  iree_host_size_t query_count = 0;
  // Next ordinal expected by the single-pass provider.
  iree_host_size_t next_packet_index = 0;
  // Visited packet bits for the two-packet empty-provider test.
  uint32_t queried_packet_mask = 0;
};

void SequentialProgressQuery(void* user_data,
                             const loom_low_schedule_table_t* schedule,
                             const loom_low_allocation_table_t* allocation,
                             const loom_low_packet_view_t* packet,
                             loom_low_packet_progress_emit_fn_t emit,
                             void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  auto* audit = static_cast<ProgressQueryAudit*>(user_data);
  EXPECT_EQ(packet->packet_index, audit->next_packet_index++);
  ++audit->query_count;
  EmitEvent(emit, emit_user_data, kSyntheticProgressPipe,
            LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
            static_cast<uint32_t>(packet->packet_index + 1));
}

void AuditEmptyProgressQuery(void* user_data,
                             const loom_low_schedule_table_t* schedule,
                             const loom_low_allocation_table_t* allocation,
                             const loom_low_packet_view_t* packet,
                             loom_low_packet_progress_emit_fn_t emit,
                             void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)emit;
  (void)emit_user_data;
  ProgressQueryAudit* audit = static_cast<ProgressQueryAudit*>(user_data);
  EXPECT_EQ(packet->packet_index, audit->next_packet_index);
  ++audit->query_count;
  ++audit->next_packet_index;
  audit->queried_packet_mask |= 1u << packet->packet_index;
}

TEST_F(LowPacketProgressTest, BuildsSyntheticTargetProgressRecords) {
  const loom_low_packet_progress_provider_t provider = {
      .user_data = {},
      .event_count = 3,
      .query = SyntheticProgressQuery,
      .class_name = SyntheticProgressClassName,
  };
  loom_low_packet_progress_table_t table = {};
  IREE_ASSERT_OK(loom_low_packet_progress_build(
      &state_.schedule, &state_.allocation, &provider, &arena_, &table));

  ASSERT_EQ(table.schedule, &state_.schedule);
  ASSERT_EQ(table.allocation, &state_.allocation);
  ASSERT_EQ(table.record_count, 3u);
  ASSERT_NE(table.records, nullptr);
  ASSERT_EQ(table.class_name, &SyntheticProgressClassName);

  EXPECT_EQ(table.records[0].packet_index, 0u);
  EXPECT_EQ(table.records[0].progress_class_id, kSyntheticProgressPipe);
  EXPECT_TRUE(iree_string_view_equal(
      table.class_name(table.records[0].progress_class_id),
      IREE_SV("synthetic.pipe")));
  EXPECT_EQ(table.records[0].action, LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE);
  EXPECT_EQ(table.records[0].units, 2u);

  EXPECT_EQ(table.records[1].packet_index, 1u);
  EXPECT_EQ(table.records[1].progress_class_id, kSyntheticProgressScoreboard);
  EXPECT_TRUE(iree_string_view_equal(
      table.class_name(table.records[1].progress_class_id),
      IREE_SV("synthetic.scoreboard")));
  EXPECT_EQ(table.records[1].action, LOOM_LOW_PACKET_PROGRESS_ACTION_RESET);
  EXPECT_EQ(table.records[1].units, 0u);

  EXPECT_EQ(table.records[2].packet_index, 1u);
  EXPECT_EQ(table.records[2].progress_class_id, kSyntheticProgressPipe);
  EXPECT_TRUE(iree_string_view_equal(
      table.class_name(table.records[2].progress_class_id),
      IREE_SV("synthetic.pipe")));
  EXPECT_EQ(table.records[2].action, LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE);
  EXPECT_EQ(table.records[2].units, 1u);
}

TEST_F(LowPacketProgressTest, BuildsEmptyProgressTable) {
  const loom_low_packet_progress_provider_t provider = {
      .user_data = {},
      .event_count = 0,
      .query = EmptyProgressQuery,
      .class_name = SyntheticProgressClassName,
  };
  loom_low_packet_progress_table_t table = {};
  IREE_ASSERT_OK(loom_low_packet_progress_build(
      &state_.schedule, &state_.allocation, &provider, &arena_, &table));
  EXPECT_EQ(table.record_count, 0u);
  EXPECT_EQ(table.records, nullptr);
}

TEST_F(LowPacketProgressTest, QueriesProviderExactlyOncePerPacket) {
  ProgressQueryAudit audit;
  const loom_low_packet_progress_provider_t provider = {
      .user_data = &audit,
      .event_count = 0,
      .query = AuditEmptyProgressQuery,
      .class_name = SyntheticProgressClassName,
  };
  loom_low_packet_progress_table_t table = {};
  IREE_ASSERT_OK(loom_low_packet_progress_build(
      &state_.schedule, &state_.allocation, &provider, &arena_, &table));

  EXPECT_EQ(audit.query_count, state_.schedule.scheduled_node_count);
  EXPECT_EQ(audit.next_packet_index, state_.schedule.scheduled_node_count);
  EXPECT_EQ(audit.queried_packet_mask, 0b11u);
}

TEST_F(LowPacketProgressTest, RecordsReusePoolBlocksAcrossEpochs) {
  constexpr uint32_t kPacketCount = 1500;
  InitializePacketProgressTestState(&state_, kPacketCount);
  DeinitializeStorage();
  InitializeStorage(32 * 1024);
  iree_host_size_t first_allocation_count = 0;
  for (int epoch = 0; epoch < 3; ++epoch) {
    SCOPED_TRACE(epoch);
    iree_arena_reset(&arena_);
    loom_low_packet_progress_table_t table = {};
    {
      ProgressQueryAudit audit;
      const loom_low_packet_progress_provider_t provider = {
          .user_data = &audit,
          .event_count = kPacketCount,
          .query = SequentialProgressQuery,
          .class_name = SyntheticProgressClassName,
      };
      IREE_ASSERT_OK(loom_low_packet_progress_build(
          &state_.schedule, &state_.allocation, &provider, &arena_, &table));
      EXPECT_EQ(audit.query_count, kPacketCount);
    }
    ASSERT_EQ(table.record_count, kPacketCount);
    ASSERT_EQ(table.class_name, &SyntheticProgressClassName);
    EXPECT_TRUE(iree_string_view_equal(table.class_name(kSyntheticProgressPipe),
                                       IREE_SV("synthetic.pipe")));
    for (uint32_t i = 0; i < kPacketCount; ++i) {
      const auto& record = table.records[i];
      EXPECT_EQ(record.packet_index, i);
      EXPECT_EQ(record.progress_class_id, kSyntheticProgressPipe);
      EXPECT_EQ(record.action, LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE);
      EXPECT_EQ(record.units, i + 1);
    }
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&block_pool_, &statistics);
    EXPECT_EQ(statistics.oversized_allocation_count, 0u);
    if (epoch == 0) {
      first_allocation_count = allocation_count_;
    }
    EXPECT_EQ(allocation_count_, first_allocation_count);
  }
}

TEST_F(LowPacketProgressTest,
       BackingFailureDoesNotVisitProviderOrPublishTable) {
  ProgressQueryAudit audit;
  const loom_low_packet_progress_provider_t provider = {
      .user_data = &audit,
      .event_count = 2,
      .query = SequentialProgressQuery,
      .class_name = SyntheticProgressClassName,
  };
  loom_low_packet_progress_table_t table = {
      .schedule = &state_.schedule,
      .record_count = 1,
  };
  failure_index_ = 0;
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &provider, &arena_, &table));
  EXPECT_EQ(audit.query_count, 0u);
  EXPECT_EQ(table.schedule, nullptr);
  EXPECT_EQ(table.allocation, nullptr);
  EXPECT_EQ(table.records, nullptr);
  EXPECT_EQ(table.record_count, 0u);
  EXPECT_EQ(table.class_name, nullptr);
  failure_index_ = SIZE_MAX;
  IREE_ASSERT_OK(loom_low_packet_progress_build(
      &state_.schedule, &state_.allocation, &provider, &arena_, &table));
  EXPECT_EQ(audit.query_count, 2u);
  EXPECT_EQ(table.record_count, 2u);
}

}  // namespace
}  // namespace loom
