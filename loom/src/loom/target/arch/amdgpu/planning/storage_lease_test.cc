// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/storage_lease.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

namespace loom {
namespace {

struct ScheduleFixture {
  loom_low_schedule_block_t block = {};
  loom_low_schedule_node_t node = {};
  uint32_t scheduled_node_indices[1] = {};
  loom_low_schedule_table_t schedule = {};
};

uint16_t PacketOperandCount(const loom_low_descriptor_set_t* descriptor_set,
                            const loom_low_descriptor_t* descriptor) {
  uint16_t packet_operand_count = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (loom_low_operand_role_is_packet_operand(operand->role)) {
      ++packet_operand_count;
    }
  }
  return packet_operand_count;
}

void InitializeScheduleFixture(const loom_low_descriptor_set_t* descriptor_set,
                               const loom_low_descriptor_t* descriptor,
                               ScheduleFixture* fixture) {
  fixture->block = {
      /*.block=*/{},
      /*.node_start=*/0,
      /*.node_count=*/1,
      /*.scheduled_node_start=*/0,
      /*.scheduled_node_count=*/1,
  };
  fixture->node = {
      /*.op=*/{},
      /*.block=*/{},
      /*.block_index=*/0,
      /*.source_ordinal=*/0,
      /*.scheduled_ordinal=*/0,
      /*.kind=*/LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
      /*.traits=*/{},
      /*.descriptor=*/descriptor,
      /*.memory_access_record_index=*/{},
      /*.schedule_class_id=*/{},
      /*.schedule_class_name=*/{},
      /*.latency_cycles=*/{},
      /*.latency_kind=*/{},
      /*.model_quality=*/{},
      /*.issue_use_count=*/{},
      /*.hazard_count=*/{},
      /*.effect_count=*/{},
      /*.operand_count=*/PacketOperandCount(descriptor_set, descriptor),
      /*.result_count=*/descriptor->result_count,
  };
  fixture->scheduled_node_indices[0] = 0;
  fixture->schedule = {
      /*.module=*/{},
      /*.function_op=*/{},
      /*.target=*/
      {/*.target_symbol=*/{}, /*.target_op=*/{}, /*.bundle_storage=*/{},
       /*.target_name=*/{}, /*.descriptor_set_key=*/{}, /*.feature_bits=*/{},
       /*.descriptor_set=*/descriptor_set},
      /*.memory_access_table=*/{},
      /*.liveness=*/{},
      /*.blocks=*/&fixture->block,
      /*.block_count=*/1,
      /*.nodes=*/&fixture->node,
      /*.node_count=*/1,
      /*.dependencies=*/{},
      /*.dependency_count=*/{},
      /*.scheduled_node_indices=*/fixture->scheduled_node_indices,
      /*.scheduled_node_count=*/1,
  };
}

class AmdgpuStorageLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &low_registry_.registry, IREE_SV("amdgpu.cdna3.core"));
    ASSERT_NE(descriptor_set_, nullptr);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t BuildLeaseTable(const loom_low_descriptor_t* descriptor,
                                loom_low_storage_lease_table_t* out_table) {
    InitializeScheduleFixture(descriptor_set_, descriptor, &fixture_);
    loom_low_storage_lease_provider_t provider = {};
    loom_amdgpu_storage_lease_provider(&provider);
    return loom_low_storage_lease_build(&fixture_.schedule, &provider, &arena_,
                                        out_table);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  ScheduleFixture fixture_;
};

TEST_F(AmdgpuStorageLeaseTest, LeasesBufferStoreVgprSources) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_DWORD);
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->storage_lease_count, 2u);

  loom_low_storage_lease_table_t table = {};
  IREE_ASSERT_OK(BuildLeaseTable(descriptor, &table));

  ASSERT_EQ(table.record_count, 2u);
  EXPECT_EQ(table.records[0].kind, LOOM_LOW_STORAGE_LEASE_SOURCE_READ);
  EXPECT_EQ(table.records[0].attachment,
            LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND);
  EXPECT_EQ(table.records[0].attachment_index, 0u);
  EXPECT_EQ(table.records[0].unit_count, 1u);
  EXPECT_EQ(table.records[0].release_class_id,
            LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  EXPECT_EQ(table.records[0].release_action_id,
            LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET);
  EXPECT_EQ(table.records[0].release_reason_id,
            LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE);

  EXPECT_EQ(table.records[1].kind, LOOM_LOW_STORAGE_LEASE_SOURCE_READ);
  EXPECT_EQ(table.records[1].attachment,
            LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND);
  EXPECT_EQ(table.records[1].attachment_index, 2u);
  EXPECT_EQ(table.records[1].unit_count, 1u);
  EXPECT_EQ(table.records[1].release_class_id,
            LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  EXPECT_EQ(table.records[1].release_reason_id,
            LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE);
}

TEST_F(AmdgpuStorageLeaseTest, LeasesGlobalLoadResult) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32);
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->storage_lease_count, 1u);

  loom_low_storage_lease_table_t table = {};
  IREE_ASSERT_OK(BuildLeaseTable(descriptor, &table));

  ASSERT_EQ(table.record_count, 1u);
  EXPECT_EQ(table.records[0].kind, LOOM_LOW_STORAGE_LEASE_RESULT_WRITE);
  EXPECT_EQ(table.records[0].attachment,
            LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT);
  EXPECT_EQ(table.records[0].attachment_index, 0u);
  EXPECT_EQ(table.records[0].unit_count, 1u);
  EXPECT_EQ(table.records[0].release_class_id,
            LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD);
  EXPECT_EQ(table.records[0].release_reason_id,
            LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE);
}

TEST_F(AmdgpuStorageLeaseTest, LeasesScalarLoadResult) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_S_BUFFER_LOAD_DWORD);
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->storage_lease_count, 1u);

  loom_low_storage_lease_table_t table = {};
  IREE_ASSERT_OK(BuildLeaseTable(descriptor, &table));

  ASSERT_EQ(table.record_count, 1u);
  EXPECT_EQ(table.records[0].kind, LOOM_LOW_STORAGE_LEASE_RESULT_WRITE);
  EXPECT_EQ(table.records[0].attachment,
            LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT);
  EXPECT_EQ(table.records[0].attachment_index, 0u);
  EXPECT_EQ(table.records[0].unit_count, 1u);
  EXPECT_EQ(table.records[0].release_class_id, LOOM_AMDGPU_WAIT_COUNTER_SMEM);
  EXPECT_EQ(table.records[0].release_reason_id,
            LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE);
}

TEST_F(AmdgpuStorageLeaseTest, DoesNotLeaseLdsStoreSourcesAsVmem) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32);
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->storage_lease_count, 0u);

  loom_low_storage_lease_table_t table = {};
  IREE_ASSERT_OK(BuildLeaseTable(descriptor, &table));
  EXPECT_EQ(table.record_count, 0u);
}

}  // namespace
}  // namespace loom
