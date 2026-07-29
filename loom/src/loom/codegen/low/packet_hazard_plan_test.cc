// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_hazard_plan.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

enum SyntheticProgressClass {
  kSyntheticProgressPipe = 9,
};

enum SyntheticHazardReason {
  kSyntheticHazardLatency = 3,
  kSyntheticHazardMissingData = 4,
  kSyntheticHazardRequiresAllocation = 5,
  kSyntheticHazardImpossible = 6,
  kSyntheticHazardStorageRelease = 7,
};

enum SyntheticHazardAction {
  kSyntheticHazardActionPadding = 1,
  kSyntheticHazardActionReleaseStorage = 2,
};

struct PacketHazardPlanTestState {
  loom_low_descriptor_t descriptors[2] = {};
  loom_low_descriptor_set_t descriptor_set = {};
  loom_module_t module = {};
  loom_op_t function_op = {};
  loom_region_t region = {};
  loom_block_t block = {};
  loom_block_t* region_blocks[1] = {};
  loom_low_schedule_block_t blocks[1] = {};
  loom_low_schedule_node_t nodes[3] = {};
  uint32_t scheduled_node_indices[3] = {};
  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
};

class LowPacketHazardPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    InitializePacketHazardPlanTestState(&state_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static void InitializePacketHazardPlanTestState(
      PacketHazardPlanTestState* state) {
    state->descriptor_set.descriptors = state->descriptors;
    state->descriptor_set.descriptor_count = IREE_ARRAYSIZE(state->descriptors);

    state->region_blocks[0] = &state->block;
    state->region.block_count = IREE_ARRAYSIZE(state->region_blocks);
    state->region.block_capacity = IREE_ARRAYSIZE(state->region_blocks);
    state->region.blocks = state->region_blocks;
    state->block.parent_region = &state->region;
    state->block.region_index = 0;

    state->blocks[0].block = &state->block;
    state->blocks[0].node_start = 0;
    state->blocks[0].node_count = IREE_ARRAYSIZE(state->nodes);
    state->blocks[0].scheduled_node_start = 0;
    state->blocks[0].scheduled_node_count =
        IREE_ARRAYSIZE(state->scheduled_node_indices);

    for (uint32_t i = 0; i < IREE_ARRAYSIZE(state->nodes); ++i) {
      state->nodes[i].block = &state->block;
      state->nodes[i].block_index = 0;
      state->nodes[i].source_ordinal = i;
      state->nodes[i].scheduled_ordinal = i;
      state->scheduled_node_indices[i] = i;
    }
    state->nodes[0].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    state->nodes[0].descriptor = &state->descriptors[0];
    state->nodes[1].kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL;
    state->nodes[2].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    state->nodes[2].descriptor = &state->descriptors[1];

    state->schedule.module = &state->module;
    state->schedule.function_op = &state->function_op;
    state->schedule.target.descriptor_set = &state->descriptor_set;
    state->schedule.blocks = state->blocks;
    state->schedule.block_count = IREE_ARRAYSIZE(state->blocks);
    state->schedule.nodes = state->nodes;
    state->schedule.node_count = IREE_ARRAYSIZE(state->nodes);
    state->schedule.scheduled_node_indices = state->scheduled_node_indices;
    state->schedule.scheduled_node_count =
        IREE_ARRAYSIZE(state->scheduled_node_indices);

    state->allocation.module = &state->module;
    state->allocation.function_op = &state->function_op;
    state->allocation.target.descriptor_set = &state->descriptor_set;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  PacketHazardPlanTestState state_;
};

void EmitHazardEvent(loom_low_packet_hazard_plan_emit_fn_t emit,
                     void* emit_user_data,
                     loom_low_packet_hazard_plan_record_kind_t kind,
                     uint16_t reason_id, iree_string_view_t reason_name,
                     uint32_t producer_node_index, uint16_t progress_class_id,
                     iree_string_view_t progress_class_name,
                     uint32_t required_progress, uint32_t observed_progress,
                     uint32_t residual_progress) {
  const uint16_t action_id =
      kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION
          ? static_cast<uint16_t>(kSyntheticHazardActionPadding)
          : LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE;
  const iree_string_view_t action_name =
      kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION
          ? IREE_SV("synthetic.padding")
          : iree_string_view_empty();
  const loom_low_packet_hazard_plan_event_t event = {
      /*.kind=*/kind,
      /*.action_id=*/action_id,
      /*.action_name=*/action_name,
      /*.reason_id=*/reason_id,
      /*.reason_name=*/reason_name,
      /*.producer_node_index=*/producer_node_index,
      /*.progress_class_id=*/progress_class_id,
      /*.progress_class_name=*/progress_class_name,
      /*.required_progress=*/required_progress,
      /*.observed_progress=*/observed_progress,
      /*.residual_progress=*/residual_progress,
  };
  emit(emit_user_data, &event);
}

void EmitProgressEvent(loom_low_packet_progress_emit_fn_t emit,
                       void* emit_user_data, uint32_t units) {
  const loom_low_packet_progress_event_t event = {
      /*.progress_class_id=*/kSyntheticProgressPipe,
      /*.progress_class_name=*/IREE_SV("synthetic.pipe"),
      /*.action=*/LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
      /*.units=*/units,
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
  if (packet->node_index == 1) {
    EmitProgressEvent(emit, emit_user_data, 1);
  }
}

void EmptyResidualHazardQuery(void* user_data,
                              const loom_low_schedule_table_t* schedule,
                              const loom_low_allocation_table_t* allocation,
                              const loom_low_packet_progress_table_t* progress,
                              const loom_low_packet_view_t* packet,
                              loom_low_packet_hazard_plan_emit_fn_t emit,
                              void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  (void)progress;
  (void)packet;
  (void)emit;
  (void)emit_user_data;
}

struct HazardQueryAudit {
  iree_host_size_t query_count = 0;
  iree_host_size_t next_packet_index = 0;
  uint32_t queried_packet_mask = 0;
};

void AuditEmptyResidualHazardQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  (void)emit;
  (void)emit_user_data;
  HazardQueryAudit* audit = static_cast<HazardQueryAudit*>(user_data);
  EXPECT_EQ(packet->packet_index, audit->next_packet_index);
  ++audit->query_count;
  ++audit->next_packet_index;
  audit->queried_packet_mask |= 1u << packet->packet_index;
}

void SyntheticResidualHazardQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  if (packet->node_index != 2) {
    return;
  }
  uint32_t observed_progress = 0;
  for (iree_host_size_t i = 0; i < progress->record_count; ++i) {
    const loom_low_packet_progress_record_t* record = &progress->records[i];
    if (record->packet_index > 0 &&
        record->packet_index < packet->packet_index &&
        record->progress_class_id == kSyntheticProgressPipe &&
        record->action == LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE) {
      observed_progress += record->units;
    }
  }
  const uint32_t required_progress = 3;
  EmitHazardEvent(emit, emit_user_data,
                  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
                  kSyntheticHazardLatency, IREE_SV("synthetic.latency"),
                  /*producer_node_index=*/0, kSyntheticProgressPipe,
                  IREE_SV("synthetic.pipe"), required_progress,
                  observed_progress, required_progress - observed_progress);
}

TEST_F(LowPacketHazardPlanTest, RecordsResidualActionsWithPacketIdentity) {
  const loom_low_packet_progress_provider_t progress_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticProgressQuery,
  };
  loom_low_packet_progress_table_t progress = {};
  IREE_ASSERT_OK(
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &progress_provider, &arena_, &progress));

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticResidualHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, &state_.allocation, &progress, &hazard_provider,
      &arena_, &plan));

  ASSERT_EQ(plan.schedule, &state_.schedule);
  ASSERT_EQ(plan.allocation, &state_.allocation);
  ASSERT_EQ(plan.progress, &progress);
  ASSERT_EQ(plan.record_count, 1u);
  const loom_low_packet_hazard_plan_record_t& record = plan.records[0];
  EXPECT_EQ(record.kind, LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION);
  EXPECT_EQ(record.action_id, kSyntheticHazardActionPadding);
  EXPECT_TRUE(
      iree_string_view_equal(record.action_name, IREE_SV("synthetic.padding")));
  EXPECT_EQ(record.reason_id, kSyntheticHazardLatency);
  EXPECT_TRUE(
      iree_string_view_equal(record.reason_name, IREE_SV("synthetic.latency")));
  EXPECT_EQ(record.producer_node_index, 0u);
  EXPECT_EQ(record.producer_packet_index, 0u);
  EXPECT_EQ(record.producer_scheduled_ordinal, 0u);
  EXPECT_EQ(record.consumer_node_index, 2u);
  EXPECT_EQ(record.insertion_packet_index, 2u);
  EXPECT_EQ(record.block_index, 0u);
  EXPECT_EQ(record.scheduled_ordinal, 2u);
  EXPECT_EQ(record.progress_class_id, kSyntheticProgressPipe);
  EXPECT_TRUE(iree_string_view_equal(record.progress_class_name,
                                     IREE_SV("synthetic.pipe")));
  EXPECT_EQ(record.required_progress, 3u);
  EXPECT_EQ(record.observed_progress, 1u);
  EXPECT_EQ(record.residual_progress, 2u);
}

TEST_F(LowPacketHazardPlanTest,
       EmitsAllocatorStorageReleaseActionsWithObservedProgress) {
  const loom_low_packet_progress_provider_t progress_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticProgressQuery,
  };
  loom_low_packet_progress_table_t progress = {};
  IREE_ASSERT_OK(
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &progress_provider, &arena_, &progress));

  const loom_low_storage_lease_record_t storage_leases[1] = {
      {
          /*.packet_index=*/0,
          /*.node_index=*/0,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/0,
          /*.kind=*/{},
          /*.attachment=*/{},
          /*.attachment_index=*/{},
          /*.unit_offset=*/{},
          /*.unit_count=*/{},
          /*.release_scope=*/{},
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
      },
  };
  const loom_low_storage_release_action_t storage_release_actions[1] = {
      {
          /*.insertion_packet_index=*/2,
          /*.insertion_node_index=*/2,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/2,
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
          /*.required_progress=*/3,
          /*.lease_record_index=*/0,
      },
  };
  state_.allocation.storage_leases = {
      /*.schedule=*/&state_.schedule,
      /*.records=*/storage_leases,
      /*.record_count=*/IREE_ARRAYSIZE(storage_leases),
  };
  state_.allocation.storage_release_actions = storage_release_actions;
  state_.allocation.storage_release_action_count =
      IREE_ARRAYSIZE(storage_release_actions);

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/0,
      /*.query=*/EmptyResidualHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, &state_.allocation, &progress, &hazard_provider,
      &arena_, &plan));

  ASSERT_EQ(plan.record_count, 1u);
  const loom_low_packet_hazard_plan_record_t& record = plan.records[0];
  EXPECT_EQ(record.kind, LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION);
  EXPECT_EQ(record.action_id, kSyntheticHazardActionReleaseStorage);
  EXPECT_TRUE(iree_string_view_equal(record.action_name,
                                     IREE_SV("synthetic.release-storage")));
  EXPECT_EQ(record.reason_id, kSyntheticHazardStorageRelease);
  EXPECT_TRUE(iree_string_view_equal(record.reason_name,
                                     IREE_SV("synthetic.storage-release")));
  EXPECT_EQ(record.producer_node_index, 0u);
  EXPECT_EQ(record.producer_packet_index, 0u);
  EXPECT_EQ(record.consumer_node_index, 2u);
  EXPECT_EQ(record.insertion_packet_index, 2u);
  EXPECT_EQ(record.progress_class_id, kSyntheticProgressPipe);
  EXPECT_EQ(record.required_progress, 3u);
  EXPECT_EQ(record.observed_progress, 1u);
  EXPECT_EQ(record.residual_progress, 2u);
}

TEST_F(LowPacketHazardPlanTest, SatisfiedStorageReleaseRetainsNoPlanStorage) {
  const loom_low_packet_progress_provider_t progress_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticProgressQuery,
  };
  loom_low_packet_progress_table_t progress = {};
  IREE_ASSERT_OK(
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &progress_provider, &arena_, &progress));

  const loom_low_storage_lease_record_t storage_leases[1] = {
      {
          /*.packet_index=*/0,
          /*.node_index=*/0,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/0,
          /*.kind=*/{},
          /*.attachment=*/{},
          /*.attachment_index=*/{},
          /*.unit_offset=*/{},
          /*.unit_count=*/{},
          /*.release_scope=*/{},
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
      },
  };
  const loom_low_storage_release_action_t storage_release_actions[1] = {
      {
          /*.insertion_packet_index=*/2,
          /*.insertion_node_index=*/2,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/2,
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
          /*.required_progress=*/1,
          /*.lease_record_index=*/0,
      },
  };
  state_.allocation.storage_leases = {
      /*.schedule=*/&state_.schedule,
      /*.records=*/storage_leases,
      /*.record_count=*/IREE_ARRAYSIZE(storage_leases),
  };
  state_.allocation.storage_release_actions = storage_release_actions;
  state_.allocation.storage_release_action_count =
      IREE_ARRAYSIZE(storage_release_actions);

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/0,
      /*.query=*/EmptyResidualHazardQuery,
  };
  const iree_host_size_t retained_used_bytes_before =
      arena_.used_allocation_size;
  const iree_host_size_t retained_owned_bytes_before =
      arena_.total_allocation_size;
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, &state_.allocation, &progress, &hazard_provider,
      &arena_, &plan));

  EXPECT_EQ(plan.record_count, 0u);
  EXPECT_EQ(plan.records, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, retained_used_bytes_before);
  EXPECT_EQ(arena_.total_allocation_size, retained_owned_bytes_before);
}

TEST_F(LowPacketHazardPlanTest,
       PreservesStorageReleaseRowsWhenRangeIndexAmortizes) {
  const loom_low_packet_progress_provider_t progress_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticProgressQuery,
  };
  loom_low_packet_progress_table_t progress = {};
  IREE_ASSERT_OK(
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &progress_provider, &arena_, &progress));

  const loom_low_storage_lease_record_t storage_leases[1] = {
      {
          /*.packet_index=*/0,
          /*.node_index=*/0,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/0,
          /*.kind=*/{},
          /*.attachment=*/{},
          /*.attachment_index=*/{},
          /*.unit_offset=*/{},
          /*.unit_count=*/{},
          /*.release_scope=*/{},
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
      },
  };
  loom_low_storage_release_action_t storage_release_actions[6] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(storage_release_actions); ++i) {
    storage_release_actions[i] = {
        /*.insertion_packet_index=*/2,
        /*.insertion_node_index=*/2,
        /*.block_index=*/0,
        /*.scheduled_ordinal=*/2,
        /*.release_class_id=*/kSyntheticProgressPipe,
        /*.release_class_name=*/IREE_SV("synthetic.pipe"),
        /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
        /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
        /*.release_reason_id=*/kSyntheticHazardStorageRelease,
        /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
        /*.required_progress=*/i + 2,
        /*.lease_record_index=*/0,
    };
  }
  state_.allocation.storage_leases = {
      /*.schedule=*/&state_.schedule,
      /*.records=*/storage_leases,
      /*.record_count=*/IREE_ARRAYSIZE(storage_leases),
  };
  state_.allocation.storage_release_actions = storage_release_actions;
  state_.allocation.storage_release_action_count =
      IREE_ARRAYSIZE(storage_release_actions);

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/0,
      /*.query=*/EmptyResidualHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, &state_.allocation, &progress, &hazard_provider,
      &arena_, &plan));

  ASSERT_EQ(plan.record_count, IREE_ARRAYSIZE(storage_release_actions));
  for (uint32_t i = 0; i < plan.record_count; ++i) {
    EXPECT_EQ(plan.records[i].consumer_node_index, 2u);
    EXPECT_EQ(plan.records[i].insertion_packet_index, 2u);
    EXPECT_EQ(plan.records[i].required_progress, i + 2);
    EXPECT_EQ(plan.records[i].observed_progress, 1u);
    EXPECT_EQ(plan.records[i].residual_progress, i + 1);
  }
}

TEST_F(LowPacketHazardPlanTest, RejectsCombinedEventCountOverflow) {
  const loom_low_storage_lease_record_t storage_leases[1] = {
      {
          /*.packet_index=*/0,
          /*.node_index=*/0,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/0,
          /*.kind=*/{},
          /*.attachment=*/{},
          /*.attachment_index=*/{},
          /*.unit_offset=*/{},
          /*.unit_count=*/{},
          /*.release_scope=*/{},
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
      },
  };
  const loom_low_storage_release_action_t storage_release_actions[1] = {
      {
          /*.insertion_packet_index=*/2,
          /*.insertion_node_index=*/2,
          /*.block_index=*/0,
          /*.scheduled_ordinal=*/2,
          /*.release_class_id=*/kSyntheticProgressPipe,
          /*.release_class_name=*/IREE_SV("synthetic.pipe"),
          /*.release_action_id=*/kSyntheticHazardActionReleaseStorage,
          /*.release_action_name=*/IREE_SV("synthetic.release-storage"),
          /*.release_reason_id=*/kSyntheticHazardStorageRelease,
          /*.release_reason_name=*/IREE_SV("synthetic.storage-release"),
          /*.required_progress=*/1,
          /*.lease_record_index=*/0,
      },
  };
  state_.allocation.storage_leases = {
      /*.schedule=*/&state_.schedule,
      /*.records=*/storage_leases,
      /*.record_count=*/IREE_ARRAYSIZE(storage_leases),
  };
  state_.allocation.storage_release_actions = storage_release_actions;
  state_.allocation.storage_release_action_count =
      IREE_ARRAYSIZE(storage_release_actions);

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/IREE_HOST_SIZE_MAX,
      /*.query=*/EmptyResidualHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_low_packet_hazard_plan_build(&state_.schedule, &state_.allocation,
                                        /*progress=*/nullptr, &hazard_provider,
                                        &arena_, &plan));
}

void SyntheticAggregateResidualHazardQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  if (packet->node_index != 1) {
    return;
  }
  uint32_t observed_progress = 0;
  for (iree_host_size_t i = 0; i < progress->record_count; ++i) {
    const loom_low_packet_progress_record_t* record = &progress->records[i];
    if (record->packet_index < packet->packet_index &&
        record->progress_class_id == kSyntheticProgressPipe &&
        record->action == LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE) {
      observed_progress += record->units;
    }
  }
  const uint32_t required_progress = 3;
  EmitHazardEvent(emit, emit_user_data,
                  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
                  kSyntheticHazardLatency, IREE_SV("synthetic.latency"),
                  LOOM_LOW_SCHEDULE_NODE_NONE, kSyntheticProgressPipe,
                  IREE_SV("synthetic.pipe"), required_progress,
                  observed_progress, required_progress - observed_progress);
}

TEST_F(LowPacketHazardPlanTest,
       RecordsAggregateResidualActionsWithoutProducerIdentity) {
  const loom_low_packet_progress_provider_t progress_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticProgressQuery,
  };
  loom_low_packet_progress_table_t progress = {};
  IREE_ASSERT_OK(
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &progress_provider, &arena_, &progress));

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/SyntheticAggregateResidualHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, &state_.allocation, &progress, &hazard_provider,
      &arena_, &plan));

  ASSERT_EQ(plan.record_count, 1u);
  const loom_low_packet_hazard_plan_record_t& record = plan.records[0];
  EXPECT_EQ(record.kind, LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION);
  EXPECT_EQ(record.action_id, kSyntheticHazardActionPadding);
  EXPECT_TRUE(
      iree_string_view_equal(record.action_name, IREE_SV("synthetic.padding")));
  EXPECT_EQ(record.producer_node_index, LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(record.producer_packet_index,
            LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE);
  EXPECT_EQ(record.producer_scheduled_ordinal,
            LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE);
  EXPECT_EQ(record.consumer_node_index, 1u);
  EXPECT_EQ(record.insertion_packet_index, 1u);
  EXPECT_EQ(record.progress_class_id, kSyntheticProgressPipe);
  EXPECT_EQ(record.required_progress, 3u);
  EXPECT_EQ(record.observed_progress, 0u);
  EXPECT_EQ(record.residual_progress, 3u);
}

void SyntheticScheduleOnlyDiagnosticQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)progress;
  if (packet->node_index == 0) {
    EmitHazardEvent(
        emit, emit_user_data,
        LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA,
        kSyntheticHazardMissingData, IREE_SV("synthetic.missing-data"),
        LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_LOW_PACKET_PROGRESS_CLASS_NONE,
        iree_string_view_empty(), 0, 0, 0);
    return;
  }
  if (packet->node_index == 1 && allocation == NULL) {
    EmitHazardEvent(
        emit, emit_user_data,
        LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNSUPPORTED_PRE_ALLOCATION,
        kSyntheticHazardRequiresAllocation,
        IREE_SV("synthetic.requires-allocation"), LOOM_LOW_SCHEDULE_NODE_NONE,
        LOOM_LOW_PACKET_PROGRESS_CLASS_NONE, iree_string_view_empty(), 0, 0, 0);
    return;
  }
  if (packet->node_index == 2) {
    EmitHazardEvent(emit, emit_user_data,
                    LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION,
                    kSyntheticHazardImpossible, IREE_SV("synthetic.impossible"),
                    /*producer_node_index=*/0, kSyntheticProgressPipe,
                    IREE_SV("synthetic.pipe"), 4, 1, 3);
  }
}

TEST_F(LowPacketHazardPlanTest, SupportsScheduleOnlyDiagnostics) {
  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/3,
      /*.query=*/SyntheticScheduleOnlyDiagnosticQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, /*allocation=*/nullptr, /*progress=*/nullptr,
      &hazard_provider, &arena_, &plan));

  ASSERT_EQ(plan.allocation, nullptr);
  ASSERT_EQ(plan.record_count, 3u);
  EXPECT_EQ(plan.records[0].kind,
            LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA);
  EXPECT_EQ(plan.records[0].reason_id, kSyntheticHazardMissingData);
  EXPECT_TRUE(iree_string_view_equal(plan.records[0].reason_name,
                                     IREE_SV("synthetic.missing-data")));
  EXPECT_EQ(plan.records[1].kind,
            LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNSUPPORTED_PRE_ALLOCATION);
  EXPECT_EQ(plan.records[1].reason_id, kSyntheticHazardRequiresAllocation);
  EXPECT_TRUE(iree_string_view_equal(plan.records[1].reason_name,
                                     IREE_SV("synthetic.requires-allocation")));
  EXPECT_EQ(plan.records[2].kind,
            LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION);
  EXPECT_EQ(plan.records[2].reason_id, kSyntheticHazardImpossible);
  EXPECT_TRUE(iree_string_view_equal(plan.records[2].reason_name,
                                     IREE_SV("synthetic.impossible")));
  EXPECT_EQ(plan.records[2].producer_packet_index, 0u);
  EXPECT_EQ(plan.records[2].residual_progress, 3u);
}

TEST_F(LowPacketHazardPlanTest, QueriesProviderExactlyOncePerPacket) {
  HazardQueryAudit audit;
  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/&audit,
      /*.event_count=*/0,
      /*.query=*/AuditEmptyResidualHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, /*allocation=*/nullptr, /*progress=*/nullptr,
      &hazard_provider, &arena_, &plan));

  EXPECT_EQ(audit.query_count, state_.schedule.scheduled_node_count);
  EXPECT_EQ(audit.next_packet_index, state_.schedule.scheduled_node_count);
  EXPECT_EQ(audit.queried_packet_mask, 0b111u);
}

void LoopCarriedProducerHazardQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  (void)progress;
  if (packet->node_index != 1) {
    return;
  }
  EmitHazardEvent(emit, emit_user_data,
                  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
                  kSyntheticHazardLatency, IREE_SV("synthetic.latency"),
                  /*producer_node_index=*/2, kSyntheticProgressPipe,
                  IREE_SV("synthetic.pipe"), 3, 1, 2);
}

TEST_F(LowPacketHazardPlanTest, RecordsLoopCarriedProducerAfterInsertion) {
  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      /*.user_data=*/{},
      /*.event_count=*/1,
      /*.query=*/LoopCarriedProducerHazardQuery,
  };
  loom_low_packet_hazard_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_packet_hazard_plan_build(
      &state_.schedule, &state_.allocation, /*progress=*/nullptr,
      &hazard_provider, &arena_, &plan));

  ASSERT_EQ(plan.record_count, 1u);
  const loom_low_packet_hazard_plan_record_t& record = plan.records[0];
  EXPECT_EQ(record.kind, LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION);
  EXPECT_EQ(record.action_id, kSyntheticHazardActionPadding);
  EXPECT_EQ(record.producer_node_index, 2u);
  EXPECT_EQ(record.producer_packet_index, 2u);
  EXPECT_EQ(record.consumer_node_index, 1u);
  EXPECT_EQ(record.insertion_packet_index, 1u);
  EXPECT_EQ(record.required_progress, 3u);
  EXPECT_EQ(record.observed_progress, 1u);
  EXPECT_EQ(record.residual_progress, 2u);
}

}  // namespace
}  // namespace loom
