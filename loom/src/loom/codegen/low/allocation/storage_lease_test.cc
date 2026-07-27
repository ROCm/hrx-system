// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease.h"

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationStorageLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    return module;
  }

  loom_value_id_t DefineValue(loom_module_t* module) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
    return value_id;
  }

  void AcquireValueDomain(loom_module_t* module,
                          const loom_value_id_t* value_ids,
                          iree_host_size_t value_count,
                          loom_local_value_domain_t* out_domain) {
    *out_domain = loom_local_value_domain_t{};
    out_domain->module = module;
    out_domain->flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED;
    loom_module_value_ordinal_scratch_acquire(module);
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      IREE_ASSERT_OK(loom_local_value_domain_register_value(
          out_domain, &arena_, value_ids[i], &value_ordinal));
      EXPECT_EQ((loom_value_ordinal_t)i, value_ordinal);
    }
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
};

loom_low_reg_class_t RegClass(uint16_t alias_set_id) {
  loom_low_reg_class_t reg_class = {};
  reg_class.alias_set_id = alias_set_id;
  return reg_class;
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_classes,
                                        iree_host_size_t reg_class_count) {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = reg_class_count;
  return descriptor_set;
}

loom_liveness_block_info_t LivenessBlock(uint32_t start_point,
                                         uint32_t end_point) {
  loom_liveness_block_info_t block = {};
  block.start_point = start_point;
  block.end_point = end_point;
  return block;
}

loom_liveness_analysis_t Liveness(const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count,
                                  const loom_value_id_t* value_ids,
                                  iree_host_size_t value_count) {
  loom_liveness_analysis_t liveness = {};
  liveness.blocks = blocks;
  liveness.block_count = block_count;
  liveness.value_ids = value_ids;
  liveness.value_count = value_count;
  return liveness;
}

loom_low_schedule_block_t ScheduleBlock(uint32_t scheduled_node_start,
                                        uint32_t scheduled_node_count) {
  loom_low_schedule_block_t block = {};
  block.scheduled_node_start = scheduled_node_start;
  block.scheduled_node_count = scheduled_node_count;
  return block;
}

loom_low_schedule_node_t ScheduleOperandNode(uint32_t block_index,
                                             uint32_t scheduled_ordinal,
                                             loom_value_ordinal_t operand) {
  loom_low_schedule_node_t node = {};
  node.block_index = block_index;
  node.scheduled_ordinal = scheduled_ordinal;
  node.operand_count = 1;
  node.value_ordinals.inline_value_ordinals[0] = operand;
  return node;
}

loom_low_schedule_table_t Schedule(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_liveness_analysis_t liveness, const loom_low_schedule_block_t* blocks,
    iree_host_size_t block_count, const loom_low_schedule_node_t* nodes,
    iree_host_size_t node_count, const uint32_t* scheduled_node_indices,
    iree_host_size_t scheduled_node_count) {
  loom_low_schedule_table_t schedule = {};
  schedule.module = module;
  schedule.function_op = function_op;
  schedule.value_ids = liveness.value_ids;
  schedule.value_count = (loom_value_ordinal_t)liveness.value_count;
  schedule.liveness = liveness;
  schedule.blocks = blocks;
  schedule.block_count = block_count;
  schedule.nodes = nodes;
  schedule.node_count = node_count;
  schedule.scheduled_node_indices = scheduled_node_indices;
  schedule.scheduled_node_count = scheduled_node_count;
  return schedule;
}

loom_low_storage_lease_record_t StorageLeaseRecord() {
  loom_low_storage_lease_record_t record = {};
  record.packet_index = 0;
  record.node_index = 0;
  record.block_index = 0;
  record.scheduled_ordinal = 0;
  record.kind = LOOM_LOW_STORAGE_LEASE_SOURCE_READ;
  record.attachment = LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND;
  record.attachment_index = 0;
  record.unit_offset = 1;
  record.unit_count = 2;
  record.release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS;
  record.release_class_id = 7;
  record.release_class_name = IREE_SV("test.progress");
  record.release_action_id = 9;
  record.release_action_name = IREE_SV("test.release-storage");
  record.release_reason_id = 11;
  record.release_reason_name = IREE_SV("test.storage-hazard");
  record.flags = LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE;
  return record;
}

loom_low_storage_lease_table_t StorageLeaseTable(
    const loom_low_schedule_table_t* schedule,
    const loom_low_storage_lease_record_t* records,
    iree_host_size_t record_count) {
  loom_low_storage_lease_table_t table = {};
  table.schedule = schedule;
  table.records = records;
  table.record_count = record_count;
  return table;
}

loom_low_allocation_assignment_t Assignment(loom_value_id_t value_id,
                                            uint16_t descriptor_reg_class_id,
                                            uint32_t start_point,
                                            uint32_t end_point,
                                            uint32_t location_base,
                                            uint32_t location_count) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = descriptor_reg_class_id;
  assignment.start_point = start_point;
  assignment.end_point = end_point;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  return assignment;
}

TEST_F(LowAllocationStorageLeaseTest, MaterializesAndReleasesConflictingLease) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));

  loom_module_t* module = AllocateModule();
  const loom_op_t* function_op =
      reinterpret_cast<const loom_op_t*>(static_cast<uintptr_t>(2));
  const loom_value_id_t value_ids[] = {
      DefineValue(module),
      DefineValue(module),
  };
  loom_local_value_domain_t value_domain = {};
  AcquireValueDomain(module, value_ids, IREE_ARRAYSIZE(value_ids),
                     &value_domain);

  const loom_liveness_block_info_t blocks[] = {
      LivenessBlock(/*start_point=*/0, /*end_point=*/3),
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks), value_domain.value_ids,
               value_domain.value_count);
  const loom_low_schedule_block_t schedule_blocks[] = {
      ScheduleBlock(/*scheduled_node_start=*/0,
                    /*scheduled_node_count=*/3),
  };
  const loom_low_schedule_node_t nodes[] = {
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/0,
                          /*operand=*/0),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/1,
                          /*operand=*/1),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/2,
                          /*operand=*/1),
  };
  const uint32_t scheduled_node_indices[] = {0, 1, 2};
  const loom_low_schedule_table_t schedule =
      Schedule(module, function_op, liveness, schedule_blocks,
               IREE_ARRAYSIZE(schedule_blocks), nodes, IREE_ARRAYSIZE(nodes),
               scheduled_node_indices, IREE_ARRAYSIZE(scheduled_node_indices));
  const loom_low_storage_lease_record_t records[] = {StorageLeaseRecord()};
  const loom_low_storage_lease_table_t lease_table =
      StorageLeaseTable(&schedule, records, IREE_ARRAYSIZE(records));

  loom_low_allocation_storage_lease_state_t state = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
      &lease_table, module, function_op, &value_domain, &liveness, &arena_,
      &state));

  const loom_low_allocation_assignment_t leased_assignment = Assignment(
      /*value_id=*/value_ids[0], /*descriptor_reg_class_id=*/0,
      /*start_point=*/0, /*end_point=*/1, /*location_base=*/10,
      /*location_count=*/4);
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_record_assignment(
      &state, &descriptor_set, &liveness, &leased_assignment,
      /*assignment_index=*/0, /*value_ordinal=*/0));
  ASSERT_EQ(state.instance_count, 1u);
  const loom_low_allocation_storage_lease_t* lease = &state.instances[0];
  EXPECT_EQ(lease->value_id, value_ids[0]);
  EXPECT_EQ(lease->start_point, 0u);
  EXPECT_EQ(lease->end_point, 3u);
  EXPECT_EQ(lease->location_base, 11u);
  EXPECT_EQ(lease->location_count, 2u);
  EXPECT_EQ(lease->release_action_index,
            LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE);

  const loom_low_allocation_assignment_t candidate = Assignment(
      /*value_id=*/value_ids[1], /*descriptor_reg_class_id=*/1,
      /*start_point=*/1, /*end_point=*/2, /*location_base=*/11,
      /*location_count=*/2);
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
      &state, &descriptor_set, &liveness, &candidate,
      /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
      &state, &descriptor_set, &liveness, &candidate,
      /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED));

  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_record_release_actions(
      &state, &descriptor_set, &liveness, &candidate,
      /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0));
  ASSERT_EQ(state.release_action_count, 1u);
  EXPECT_EQ(lease->release_action_index, 0u);
  EXPECT_EQ(lease->end_point, 1u);
  const loom_low_storage_release_action_t* action = &state.release_actions[0];
  EXPECT_EQ(action->insertion_packet_index, 1u);
  EXPECT_EQ(action->insertion_node_index, 1u);
  EXPECT_EQ(action->block_index, 0u);
  EXPECT_EQ(action->scheduled_ordinal, 1u);
  EXPECT_EQ(action->release_class_id, 7u);
  EXPECT_TRUE(iree_string_view_equal(action->release_class_name,
                                     IREE_SV("test.progress")));
  EXPECT_EQ(action->release_action_id, 9u);
  EXPECT_TRUE(iree_string_view_equal(action->release_action_name,
                                     IREE_SV("test.release-storage")));
  EXPECT_EQ(action->release_reason_id, 11u);
  EXPECT_TRUE(iree_string_view_equal(action->release_reason_name,
                                     IREE_SV("test.storage-hazard")));
  EXPECT_EQ(action->required_progress, 1u);
  EXPECT_EQ(action->lease_record_index, 0u);
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_finalize(&state));

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

TEST_F(LowAllocationStorageLeaseTest, RejectsLeaseOutsideAllocationLiveness) {
  loom_module_t* module = AllocateModule();
  const loom_op_t* function_op =
      reinterpret_cast<const loom_op_t*>(static_cast<uintptr_t>(2));
  const loom_value_id_t schedule_value_ids[] = {DefineValue(module)};
  const loom_value_id_t allocation_value_ids[] = {DefineValue(module)};
  loom_local_value_domain_t allocation_value_domain = {};
  AcquireValueDomain(module, allocation_value_ids,
                     IREE_ARRAYSIZE(allocation_value_ids),
                     &allocation_value_domain);

  const loom_liveness_block_info_t blocks[] = {
      LivenessBlock(/*start_point=*/0, /*end_point=*/2),
  };
  const loom_liveness_analysis_t schedule_liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks), schedule_value_ids,
               IREE_ARRAYSIZE(schedule_value_ids));
  const loom_liveness_analysis_t allocation_liveness = Liveness(
      blocks, IREE_ARRAYSIZE(blocks), allocation_value_domain.value_ids,
      allocation_value_domain.value_count);
  const loom_low_schedule_block_t schedule_blocks[] = {
      ScheduleBlock(/*scheduled_node_start=*/0,
                    /*scheduled_node_count=*/1),
  };
  const loom_low_schedule_node_t nodes[] = {
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/0,
                          /*operand=*/0),
  };
  const uint32_t scheduled_node_indices[] = {0};
  const loom_low_schedule_table_t schedule =
      Schedule(module, function_op, schedule_liveness, schedule_blocks,
               IREE_ARRAYSIZE(schedule_blocks), nodes, IREE_ARRAYSIZE(nodes),
               scheduled_node_indices, IREE_ARRAYSIZE(scheduled_node_indices));
  const loom_low_storage_lease_record_t records[] = {StorageLeaseRecord()};
  const loom_low_storage_lease_table_t lease_table =
      StorageLeaseTable(&schedule, records, IREE_ARRAYSIZE(records));

  loom_low_allocation_storage_lease_state_t state = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_allocation_storage_lease_state_initialize(
          &lease_table, module, function_op, &allocation_value_domain,
          &allocation_liveness, &arena_, &state));

  loom_local_value_domain_release(&allocation_value_domain);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
