// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_lease.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

enum SyntheticReleaseClass {
  kSyntheticReleaseStore = 11,
  kSyntheticReleaseRead = 12,
};

enum SyntheticReleaseAction {
  kSyntheticReleaseActionWait = 1,
};

enum SyntheticReleaseReason {
  kSyntheticReleaseReasonStorage = 21,
};

struct StorageLeaseTestState {
  loom_module_t module = {};
  loom_op_t function_op = {};
  loom_region_t region = {};
  loom_block_t block = {};
  loom_block_t* region_blocks[1] = {};
  loom_low_schedule_block_t blocks[1] = {};
  loom_low_schedule_node_t nodes[2] = {};
  uint32_t scheduled_node_indices[2] = {};
  loom_low_schedule_table_t schedule = {};
};

class LowStorageLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    InitializeStorageLeaseTestState(&state_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static void InitializeStorageLeaseTestState(StorageLeaseTestState* state) {
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

    state->nodes[0].block = &state->block;
    state->nodes[0].block_index = 0;
    state->nodes[0].source_ordinal = 0;
    state->nodes[0].scheduled_ordinal = 0;
    state->nodes[0].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    state->nodes[0].operand_count = 2;
    state->nodes[0].result_count = 1;
    state->nodes[1].block = &state->block;
    state->nodes[1].block_index = 0;
    state->nodes[1].source_ordinal = 1;
    state->nodes[1].scheduled_ordinal = 1;
    state->nodes[1].kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL;
    state->nodes[1].operand_count = 1;
    state->nodes[1].result_count = 0;

    state->scheduled_node_indices[0] = 0;
    state->scheduled_node_indices[1] = 1;

    state->schedule.module = &state->module;
    state->schedule.function_op = &state->function_op;
    state->schedule.blocks = state->blocks;
    state->schedule.block_count = IREE_ARRAYSIZE(state->blocks);
    state->schedule.nodes = state->nodes;
    state->schedule.node_count = IREE_ARRAYSIZE(state->nodes);
    state->schedule.scheduled_node_indices = state->scheduled_node_indices;
    state->schedule.scheduled_node_count =
        IREE_ARRAYSIZE(state->scheduled_node_indices);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  StorageLeaseTestState state_;
};

iree_status_t EmitEvent(loom_low_storage_lease_emit_fn_t emit,
                        void* emit_user_data,
                        loom_low_storage_lease_kind_t kind,
                        loom_low_storage_lease_attachment_t attachment,
                        uint16_t attachment_index, uint32_t unit_offset,
                        uint32_t unit_count, uint16_t release_class_id,
                        iree_string_view_t release_class_name,
                        loom_low_storage_lease_flags_t flags) {
  const loom_low_storage_lease_event_t event = {
      /*.kind=*/kind,
      /*.attachment=*/attachment,
      /*.attachment_index=*/attachment_index,
      /*.unit_offset=*/unit_offset,
      /*.unit_count=*/unit_count,
      /*.release_scope=*/LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
      /*.release_class_id=*/release_class_id,
      /*.release_class_name=*/release_class_name,
      /*.release_action_id=*/kSyntheticReleaseActionWait,
      /*.release_action_name=*/IREE_SV("synthetic.wait"),
      /*.release_reason_id=*/kSyntheticReleaseReasonStorage,
      /*.release_reason_name=*/IREE_SV("synthetic.storage-reuse"),
      /*.flags=*/flags,
  };
  return emit(emit_user_data, &event);
}

iree_status_t SyntheticLeaseQuery(void* user_data,
                                  const loom_low_schedule_table_t* schedule,
                                  const loom_low_schedule_node_t* node,
                                  loom_low_storage_lease_emit_fn_t emit,
                                  void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  if (node->source_ordinal == 0) {
    IREE_RETURN_IF_ERROR(
        EmitEvent(emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
                  LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND, 1, 0, 2,
                  kSyntheticReleaseStore, IREE_SV("synthetic.store"),
                  LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE |
                      LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY));
    return EmitEvent(emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_RESULT_WRITE,
                     LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT, 0, 3, 1,
                     kSyntheticReleaseRead, IREE_SV("synthetic.read"),
                     LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE);
  }
  return EmitEvent(emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
                   LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND, 0, 0, 1,
                   kSyntheticReleaseStore, IREE_SV("synthetic.store"),
                   LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE);
}

iree_status_t EmptyLeaseQuery(void* user_data,
                              const loom_low_schedule_table_t* schedule,
                              const loom_low_schedule_node_t* node,
                              loom_low_storage_lease_emit_fn_t emit,
                              void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)node;
  (void)emit;
  (void)emit_user_data;
  return iree_ok_status();
}

TEST_F(LowStorageLeaseTest, BuildsSyntheticTargetLeaseRecords) {
  const loom_low_storage_lease_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/SyntheticLeaseQuery,
  };
  loom_low_storage_lease_table_t table = {};
  IREE_ASSERT_OK(loom_low_storage_lease_build(&state_.schedule, &provider,
                                              &arena_, &table));

  ASSERT_EQ(table.schedule, &state_.schedule);
  ASSERT_EQ(table.record_count, 3u);
  ASSERT_NE(table.records, nullptr);

  EXPECT_EQ(table.records[0].packet_index, 0u);
  EXPECT_EQ(table.records[0].node_index, 0u);
  EXPECT_EQ(table.records[0].block_index, 0u);
  EXPECT_EQ(table.records[0].scheduled_ordinal, 0u);
  EXPECT_EQ(table.records[0].kind, LOOM_LOW_STORAGE_LEASE_SOURCE_READ);
  EXPECT_EQ(table.records[0].attachment,
            LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND);
  EXPECT_EQ(table.records[0].attachment_index, 1u);
  EXPECT_EQ(table.records[0].unit_offset, 0u);
  EXPECT_EQ(table.records[0].unit_count, 2u);
  EXPECT_EQ(table.records[0].release_scope,
            LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS);
  EXPECT_EQ(table.records[0].release_class_id, kSyntheticReleaseStore);
  EXPECT_TRUE(iree_string_view_equal(table.records[0].release_class_name,
                                     IREE_SV("synthetic.store")));
  EXPECT_EQ(table.records[0].release_action_id, kSyntheticReleaseActionWait);
  EXPECT_TRUE(iree_string_view_equal(table.records[0].release_action_name,
                                     IREE_SV("synthetic.wait")));
  EXPECT_EQ(table.records[0].release_reason_id, kSyntheticReleaseReasonStorage);
  EXPECT_TRUE(iree_string_view_equal(table.records[0].release_reason_name,
                                     IREE_SV("synthetic.storage-reuse")));
  EXPECT_EQ(table.records[0].flags,
            LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE |
                LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY);

  EXPECT_EQ(table.records[1].packet_index, 0u);
  EXPECT_EQ(table.records[1].node_index, 0u);
  EXPECT_EQ(table.records[1].kind, LOOM_LOW_STORAGE_LEASE_RESULT_WRITE);
  EXPECT_EQ(table.records[1].attachment,
            LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT);
  EXPECT_EQ(table.records[1].attachment_index, 0u);
  EXPECT_EQ(table.records[1].unit_offset, 3u);
  EXPECT_EQ(table.records[1].unit_count, 1u);
  EXPECT_EQ(table.records[1].release_class_id, kSyntheticReleaseRead);

  EXPECT_EQ(table.records[2].packet_index, 1u);
  EXPECT_EQ(table.records[2].node_index, 1u);
  EXPECT_EQ(table.records[2].scheduled_ordinal, 1u);
}

TEST_F(LowStorageLeaseTest, BuildsEmptyLeaseTable) {
  const loom_low_storage_lease_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/EmptyLeaseQuery,
  };
  loom_low_storage_lease_table_t table = {};
  IREE_ASSERT_OK(loom_low_storage_lease_build(&state_.schedule, &provider,
                                              &arena_, &table));
  EXPECT_EQ(table.schedule, &state_.schedule);
  EXPECT_EQ(table.record_count, 0u);
  EXPECT_EQ(table.records, nullptr);
}

iree_status_t InvalidAttachmentLeaseQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, loom_low_storage_lease_emit_fn_t emit,
    void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)node;
  return EmitEvent(emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
                   LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND, 2, 0, 1,
                   kSyntheticReleaseStore, IREE_SV("synthetic.store"),
                   LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE);
}

TEST_F(LowStorageLeaseTest, RejectsInvalidAttachmentIndex) {
  const loom_low_storage_lease_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/InvalidAttachmentLeaseQuery,
  };
  loom_low_storage_lease_table_t table = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_storage_lease_build(
                            &state_.schedule, &provider, &arena_, &table));
}

iree_status_t InvalidFlagsLeaseQuery(void* user_data,
                                     const loom_low_schedule_table_t* schedule,
                                     const loom_low_schedule_node_t* node,
                                     loom_low_storage_lease_emit_fn_t emit,
                                     void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)node;
  return EmitEvent(emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
                   LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND, 0, 0, 1,
                   kSyntheticReleaseStore, IREE_SV("synthetic.store"),
                   LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY |
                       LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY);
}

TEST_F(LowStorageLeaseTest, RejectsContradictoryBoundaryFlags) {
  const loom_low_storage_lease_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/InvalidFlagsLeaseQuery,
  };
  loom_low_storage_lease_table_t table = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_storage_lease_build(
                            &state_.schedule, &provider, &arena_, &table));
}

}  // namespace
}  // namespace loom
