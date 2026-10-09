// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/storage_lease_index.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"
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
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module, module->body, &arena_, out_domain));
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      IREE_ASSERT_OK(loom_local_value_domain_register_value(
          out_domain, value_ids[i], &value_ordinal));
      EXPECT_EQ((loom_value_ordinal_t)i, value_ordinal);
    }
  }

  // Blocks backing the test's module and allocation arenas.
  iree_arena_block_pool_t block_pool_;
  // Owner of allocation state and local value-domain storage.
  iree_arena_allocator_t arena_;
  // Context used to construct valid module-local values.
  loom_context_t context_;
};

loom_low_reg_class_t RegClass(uint16_t alias_set_id) {
  loom_low_reg_class_t reg_class = {.alias_set_id = alias_set_id};
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
  loom_liveness_block_info_t block = {.start_point = start_point,
                                      .end_point = end_point};
  return block;
}

loom_liveness_analysis_t Liveness(const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count,
                                  const loom_value_id_t* value_ids,
                                  iree_host_size_t value_count) {
  loom_liveness_analysis_t liveness = {.blocks = blocks,
                                       .block_count = block_count,
                                       .value_ids = value_ids,
                                       .value_count = value_count};
  return liveness;
}

loom_low_schedule_block_t ScheduleBlock(uint32_t scheduled_node_start,
                                        uint32_t scheduled_node_count) {
  loom_low_schedule_block_t block = {
      .scheduled_node_start = scheduled_node_start,
      .scheduled_node_count = scheduled_node_count};
  return block;
}

loom_low_schedule_node_t ScheduleOperandNode(uint32_t block_index,
                                             uint32_t scheduled_ordinal,
                                             loom_value_ordinal_t operand) {
  loom_low_schedule_node_t node = {.block_index = block_index,
                                   .scheduled_ordinal = scheduled_ordinal,
                                   .operand_count = 1};
  node.value_ordinals.inline_value_ordinals[0] = operand;
  return node;
}

loom_low_schedule_table_t Schedule(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_liveness_analysis_t liveness, const loom_low_schedule_block_t* blocks,
    iree_host_size_t block_count, const loom_low_schedule_node_t* nodes,
    iree_host_size_t node_count, const uint32_t* scheduled_node_indices,
    iree_host_size_t scheduled_node_count,
    const uint32_t* value_producer_nodes) {
  loom_low_schedule_table_t schedule = {
      .module = module,
      .function_op = function_op,
      .value_ids = liveness.value_ids,
      .value_count = (loom_value_ordinal_t)liveness.value_count,
      .value_producer_nodes = value_producer_nodes,
      .liveness = liveness,
      .blocks = blocks,
      .block_count = block_count,
      .nodes = nodes,
      .node_count = node_count,
      .scheduled_node_indices = scheduled_node_indices,
      .scheduled_node_count = scheduled_node_count};
  return schedule;
}

loom_low_storage_lease_record_t StorageLeaseRecord() {
  loom_low_storage_lease_record_t record = {
      .packet_index = 0,
      .node_index = 0,
      .block_index = 0,
      .scheduled_ordinal = 0,
      .kind = LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
      .attachment = LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND,
      .attachment_index = 0,
      .unit_offset = 1,
      .unit_count = 2,
      .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
      .release_class_id = 7,
      .release_class_name = IREE_SV("test.progress"),
      .release_action_id = 9,
      .release_action_name = IREE_SV("test.release-storage"),
      .release_reason_id = 11,
      .release_reason_name = IREE_SV("test.storage-hazard"),
      .flags = LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE};
  return record;
}

loom_low_storage_lease_table_t StorageLeaseTable(
    const loom_low_schedule_table_t* schedule,
    const loom_low_storage_lease_record_t* records,
    iree_host_size_t record_count) {
  loom_low_storage_lease_table_t table = {
      .schedule = schedule, .records = records, .record_count = record_count};
  return table;
}

loom_low_allocation_assignment_t Assignment(loom_value_id_t value_id,
                                            uint16_t descriptor_reg_class_id,
                                            uint32_t start_point,
                                            uint32_t end_point,
                                            uint32_t location_base,
                                            uint32_t location_count) {
  loom_low_allocation_assignment_t assignment = {
      .value_id = value_id,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .start_point = start_point,
      .end_point = end_point,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = location_base,
      .location_count = location_count};
  return assignment;
}

loom_low_placement_relation_t StorageRelation(
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal,
    uint32_t result_unit_offset, uint32_t source_unit_offset,
    uint32_t unit_count, loom_low_placement_cause_t cause,
    loom_low_placement_relation_flags_t flags =
        LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE) {
  loom_low_placement_relation_t relation = {};
  relation.result_ordinal = result_ordinal;
  relation.source_ordinal = source_ordinal;
  relation.result_unit_offset = result_unit_offset;
  relation.source_unit_offset = source_unit_offset;
  relation.unit_count = unit_count;
  relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
  relation.cause = cause;
  relation.flags = flags;
  relation.source_operand_index = LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE;
  return relation;
}

enum : iree_host_size_t {
  kStorageIdentityMaxValueCount = 16,
  kStorageIdentityMaxRelationCount = 24,
};

// Complete, data-only allocator fixture for one lease and a structural storage
// graph. Tests supply already-refined placement relations in their retained
// users-before-sources order, matching the production stage boundary.
struct StorageIdentityScenario {
  loom_module_t* module = nullptr;
  const loom_op_t* function_op = nullptr;
  loom_value_id_t value_ids[kStorageIdentityMaxValueCount] = {};
  loom_local_value_domain_t value_domain = {};
  loom_liveness_block_info_t blocks[1] = {};
  loom_liveness_analysis_t liveness = {};
  loom_low_schedule_block_t schedule_blocks[1] = {};
  loom_low_schedule_node_t schedule_nodes[1] = {};
  uint32_t scheduled_node_indices[1] = {};
  uint32_t value_producer_nodes[kStorageIdentityMaxValueCount] = {};
  loom_low_schedule_table_t schedule = {};
  loom_low_storage_lease_record_t lease_record = {};
  loom_low_storage_lease_table_t lease_table = {};
  loom_low_placement_relation_t relations[kStorageIdentityMaxRelationCount] =
      {};
  loom_low_placement_relation_range_t
      relation_ranges[kStorageIdentityMaxValueCount] = {};
  loom_value_ordinal_t storage_value_order[kStorageIdentityMaxValueCount] = {};
  loom_low_placement_table_t placement = {};
  loom_low_allocation_unit_liveness_value_t
      unit_values[kStorageIdentityMaxValueCount] = {};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  loom_low_allocation_storage_identity_t storage_identity = {};
  loom_low_allocation_storage_lease_state_t state = {};
};

class LowAllocationStorageIdentityTest : public LowAllocationStorageLeaseTest {
 protected:
  void InitializeScenario(const loom_low_descriptor_set_t* descriptor_set,
                          const uint32_t* unit_counts,
                          loom_value_ordinal_t value_count,
                          const loom_low_placement_relation_t* relations,
                          iree_host_size_t relation_count,
                          const loom_value_ordinal_t* storage_value_order,
                          loom_low_placement_storage_flags_t storage_flags,
                          StorageIdentityScenario* scenario) {
    ASSERT_LE(value_count, kStorageIdentityMaxValueCount);
    ASSERT_LE(relation_count, kStorageIdentityMaxRelationCount);
    *scenario = StorageIdentityScenario{};
    scenario->module = AllocateModule();
    scenario->function_op =
        reinterpret_cast<const loom_op_t*>(static_cast<uintptr_t>(2));
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      scenario->value_ids[i] = DefineValue(scenario->module);
      scenario->value_producer_nodes[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
    AcquireValueDomain(scenario->module, scenario->value_ids, value_count,
                       &scenario->value_domain);

    scenario->blocks[0] = LivenessBlock(/*start_point=*/0, /*end_point=*/10);
    scenario->liveness = Liveness(
        scenario->blocks, IREE_ARRAYSIZE(scenario->blocks),
        scenario->value_domain.value_ids, scenario->value_domain.value_count);
    scenario->schedule_blocks[0] = ScheduleBlock(/*scheduled_node_start=*/0,
                                                 /*scheduled_node_count=*/1);
    scenario->schedule_nodes[0] =
        ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/0,
                            /*operand=*/0);
    scenario->scheduled_node_indices[0] = 0;
    scenario->schedule = Schedule(
        scenario->module, scenario->function_op, scenario->liveness,
        scenario->schedule_blocks, IREE_ARRAYSIZE(scenario->schedule_blocks),
        scenario->schedule_nodes, IREE_ARRAYSIZE(scenario->schedule_nodes),
        scenario->scheduled_node_indices,
        IREE_ARRAYSIZE(scenario->scheduled_node_indices),
        scenario->value_producer_nodes);
    scenario->schedule.target.descriptor_set = descriptor_set;
    scenario->lease_record = StorageLeaseRecord();
    scenario->lease_record.unit_offset = 0;
    scenario->lease_record.unit_count = unit_counts[0];
    scenario->lease_table =
        StorageLeaseTable(&scenario->schedule, &scenario->lease_record, 1);

    for (iree_host_size_t i = 0; i < relation_count; ++i) {
      scenario->relations[i] = relations[i];
      loom_low_placement_relation_range_t* range =
          &scenario->relation_ranges[relations[i].result_ordinal];
      if (range->count == 0) {
        range->start = (uint32_t)i;
      } else {
        ASSERT_EQ(range->start + range->count, i);
      }
      ++range->count;
    }
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      scenario->storage_value_order[i] = storage_value_order[i];
    }
    scenario->placement.value_ids = scenario->value_domain.value_ids;
    scenario->placement.value_count = value_count;
    scenario->placement.relations = scenario->relations;
    scenario->placement.relation_count = relation_count;
    scenario->placement.ranges_by_result_ordinal = scenario->relation_ranges;
    scenario->placement.storage.flags = storage_flags;
    scenario->placement.storage_value_order = scenario->storage_value_order;
    scenario->placement.storage_value_order_count = value_count;

    iree_host_size_t point_count = 0;
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      ASSERT_LE(point_count, UINT32_MAX);
      scenario->unit_values[i].unit_point_start = (uint32_t)point_count;
      scenario->unit_values[i].acquisition_start_point = 0;
      point_count += unit_counts[i];
    }
    scenario->unit_liveness.values = scenario->unit_values;
    scenario->unit_liveness.point_count = point_count;
    IREE_ASSERT_OK(iree_arena_allocate_array(
        &arena_, point_count, sizeof(uint32_t),
        reinterpret_cast<void**>(&scenario->unit_liveness.end_points)));
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      for (uint32_t unit = 0; unit < unit_counts[i]; ++unit) {
        scenario->unit_liveness
            .end_points[scenario->unit_values[i].unit_point_start + unit] =
            i == 0 ? 1 : 2;
      }
    }
    IREE_ASSERT_OK(loom_low_allocation_storage_identity_initialize(
        &scenario->lease_table, &scenario->placement, &scenario->unit_liveness,
        &arena_, &scenario->storage_identity));
    IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
        &scenario->lease_table, scenario->module, scenario->function_op,
        &scenario->value_domain, &scenario->liveness,
        &scenario->storage_identity, &scenario->unit_liveness, &arena_,
        &scenario->state));
  }

  void DeinitializeScenario(StorageIdentityScenario* scenario) {
    loom_local_value_domain_release(&scenario->value_domain);
    loom_module_free(scenario->module);
  }

  uint32_t IdentityOrigin(const StorageIdentityScenario& scenario,
                          loom_value_ordinal_t value_ordinal,
                          uint32_t unit_offset) {
    return scenario.state
        .identity_origins[scenario.unit_values[value_ordinal].unit_point_start +
                          unit_offset];
  }

  void ExpectConflictOnScanAndIndex(
      StorageIdentityScenario* scenario,
      const loom_low_descriptor_set_t* descriptor_set,
      const loom_low_allocation_assignment_t& candidate,
      bool expected_conflict) {
    EXPECT_EQ(expected_conflict,
              loom_low_allocation_storage_lease_state_conflicts(
                  &scenario->state, descriptor_set, &scenario->liveness,
                  &candidate, /*ignored_value_ids=*/nullptr,
                  /*ignored_value_count=*/0,
                  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
    loom_low_allocation_storage_lease_unit_index_t* unit_index =
        scenario->state.unit_index;
    scenario->state.unit_index = nullptr;
    EXPECT_EQ(expected_conflict,
              loom_low_allocation_storage_lease_state_conflicts(
                  &scenario->state, descriptor_set, &scenario->liveness,
                  &candidate, /*ignored_value_ids=*/nullptr,
                  /*ignored_value_count=*/0,
                  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
    scenario->state.unit_index = unit_index;
  }
};

enum StorageIdentityValueOrdinal : loom_value_ordinal_t {
  kIdentitySource = 0,
  kIdentityTied = 1,
  kIdentityCopy = 2,
  kIdentityMove = 3,
  kIdentityLower = 4,
  kIdentityUpper = 5,
  kIdentityReconstructed = 6,
  kIdentityRepeated = 7,
  kIdentitySwapped = 8,
  kIdentityWritten = 9,
  kIdentityCaptured = 10,
  kIdentityMaterializedCopy = 11,
  kIdentitySeparated = 12,
  kIdentityValueCount = 13,
};

constexpr uint32_t kStorageIdentityUnitCounts[kIdentityValueCount] = {
    4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4,
};

// Placement retains users before sources. Reverse traversal therefore resolves
// each source root before assigning its users.
constexpr loom_value_ordinal_t kStorageIdentityOrder[kIdentityValueCount] = {
    kIdentityReconstructed, kIdentityRepeated, kIdentitySwapped,
    kIdentityWritten,       kIdentityCaptured, kIdentityMaterializedCopy,
    kIdentitySeparated,     kIdentityLower,    kIdentityUpper,
    kIdentityMove,          kIdentityCopy,     kIdentityTied,
    kIdentitySource,
};

iree_host_size_t BuildStorageIdentityRelations(
    loom_low_placement_relation_t* relations) {
  const loom_low_placement_relation_flags_t hard_alias =
      LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
      LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  const loom_low_placement_relation_flags_t preferred_alias =
      LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
      LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  iree_host_size_t count = 0;
  relations[count++] =
      StorageRelation(kIdentityTied, kIdentitySource, 0, 0, 4,
                      LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT, hard_alias);
  relations[count++] =
      StorageRelation(kIdentityCopy, kIdentityTied, 0, 0, 4,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityMove, kIdentityCopy, 0, 0, 4,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityLower, kIdentityMove, 0, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityUpper, kIdentityMove, 0, 2, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityReconstructed, kIdentityLower, 0, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityReconstructed, kIdentityUpper, 2, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityRepeated, kIdentityLower, 0, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentityRepeated, kIdentityLower, 2, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentitySwapped, kIdentityUpper, 0, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT, preferred_alias);
  relations[count++] =
      StorageRelation(kIdentitySwapped, kIdentityLower, 2, 0, 2,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT, preferred_alias);
  relations[count++] = StorageRelation(
      kIdentityWritten, kIdentityMove, 0, 0, 4,
      LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
      hard_alias | LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE);
  relations[count++] = StorageRelation(
      kIdentityCaptured, kIdentityMove, 0, 0, 4,
      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
      preferred_alias | LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART |
          LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART);
  relations[count++] = StorageRelation(
      kIdentityMaterializedCopy, kIdentityMove, 0, 0, 4,
      LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
      preferred_alias | LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART);
  relations[count++] =
      StorageRelation(kIdentitySeparated, kIdentityMove, 0, 0, 4,
                      LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY,
                      LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED);
  return count;
}

TEST_F(LowAllocationStorageIdentityTest,
       ResolvesFinalStructuralContentOrigins) {
  const loom_low_reg_class_t reg_classes[] = {RegClass(1)};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  loom_low_placement_relation_t relations[kStorageIdentityMaxRelationCount] =
      {};
  const iree_host_size_t relation_count =
      BuildStorageIdentityRelations(relations);
  StorageIdentityScenario scenario;
  InitializeScenario(&descriptor_set, kStorageIdentityUnitCounts,
                     kIdentityValueCount, relations, relation_count,
                     kStorageIdentityOrder,
                     LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                         LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES,
                     &scenario);

  ASSERT_NE(scenario.state.identity_origins, nullptr);
  for (uint32_t unit = 0; unit < 4; ++unit) {
    EXPECT_EQ(IdentityOrigin(scenario, kIdentitySource, unit), unit);
    EXPECT_EQ(IdentityOrigin(scenario, kIdentityTied, unit), unit);
    EXPECT_EQ(IdentityOrigin(scenario, kIdentityCopy, unit), unit);
    EXPECT_EQ(IdentityOrigin(scenario, kIdentityMove, unit), unit);
    EXPECT_EQ(IdentityOrigin(scenario, kIdentityReconstructed, unit), unit);
    EXPECT_EQ(IdentityOrigin(scenario, kIdentityMaterializedCopy, unit), unit);
  }
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityLower, 0), 0u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityLower, 1), 1u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityUpper, 0), 2u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityUpper, 1), 3u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityRepeated, 0), 0u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityRepeated, 1), 1u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityRepeated, 2), 0u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentityRepeated, 3), 1u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentitySwapped, 0), 2u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentitySwapped, 1), 3u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentitySwapped, 2), 0u);
  EXPECT_EQ(IdentityOrigin(scenario, kIdentitySwapped, 3), 1u);
  for (const loom_value_ordinal_t value :
       {kIdentityWritten, kIdentityCaptured, kIdentitySeparated}) {
    const uint32_t start = scenario.unit_values[value].unit_point_start;
    for (uint32_t unit = 0; unit < 4; ++unit) {
      EXPECT_EQ(IdentityOrigin(scenario, value, unit), start + unit);
    }
  }

  // Functions without leases never pay for identity state, even when their
  // placement graph contains aliases.
  const loom_low_storage_lease_table_t empty_lease_table = {};
  loom_low_allocation_storage_identity_t no_lease_identity = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_identity_initialize(
      &empty_lease_table, &scenario.placement, &scenario.unit_liveness, &arena_,
      &no_lease_identity));
  EXPECT_EQ(no_lease_identity.origins, nullptr);
  loom_low_allocation_storage_lease_state_t no_lease_state = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
      &empty_lease_table, scenario.module, scenario.function_op,
      &scenario.value_domain, &scenario.liveness, &no_lease_identity,
      &scenario.unit_liveness, &arena_, &no_lease_state));
  EXPECT_EQ(no_lease_state.identity_origins, nullptr);

  // Leases without structural aliases keep the original lease-only footprint.
  loom_low_placement_table_t no_alias_placement = scenario.placement;
  no_alias_placement.storage.flags = 0;
  no_alias_placement.storage_value_order = nullptr;
  no_alias_placement.storage_value_order_count = 0;
  loom_low_allocation_storage_identity_t no_alias_identity = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_identity_initialize(
      &scenario.lease_table, &no_alias_placement, &scenario.unit_liveness,
      &arena_, &no_alias_identity));
  EXPECT_EQ(no_alias_identity.origins, nullptr);
  loom_low_allocation_storage_lease_state_t no_alias_state = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
      &scenario.lease_table, scenario.module, scenario.function_op,
      &scenario.value_domain, &scenario.liveness, &no_alias_identity,
      &scenario.unit_liveness, &arena_, &no_alias_state));
  EXPECT_EQ(no_alias_state.identity_origins, nullptr);

  DeinitializeScenario(&scenario);
}

TEST_F(LowAllocationStorageIdentityTest,
       AppliesExactIdentityToScanIndexAndReleasePaths) {
  const loom_low_reg_class_t reg_classes[] = {RegClass(1), RegClass(1)};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  loom_low_placement_relation_t relations[kStorageIdentityMaxRelationCount] =
      {};
  const iree_host_size_t relation_count =
      BuildStorageIdentityRelations(relations);
  StorageIdentityScenario scenario;
  InitializeScenario(&descriptor_set, kStorageIdentityUnitCounts,
                     kIdentityValueCount, relations, relation_count,
                     kStorageIdentityOrder,
                     LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                         LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES,
                     &scenario);

  loom_low_allocation_assignment_t assignments[] = {Assignment(
      scenario.value_ids[kIdentitySource], /*descriptor_reg_class_id=*/0,
      /*start_point=*/0, /*end_point=*/1, /*location_base=*/16,
      /*location_count=*/4)};
  assignments[0].unit_count = 4;
  loom_low_allocation_storage_lease_state_record_assignment(
      &scenario.state, &descriptor_set, &scenario.liveness, assignments,
      /*assignment_index=*/0, kIdentitySource);
  ASSERT_EQ(scenario.state.instance_count, 1u);

  auto candidate = [&](loom_value_ordinal_t value_ordinal,
                       uint32_t location_base, uint32_t location_count) {
    loom_low_allocation_assignment_t result = Assignment(
        scenario.value_ids[value_ordinal], /*descriptor_reg_class_id=*/1,
        /*start_point=*/1, /*end_point=*/2, location_base, location_count);
    result.unit_count = kStorageIdentityUnitCounts[value_ordinal];
    result.unit_point_start =
        scenario.unit_values[value_ordinal].unit_point_start;
    return result;
  };

  for (const loom_value_ordinal_t value :
       {kIdentityTied, kIdentityCopy, kIdentityMove, kIdentityReconstructed,
        kIdentityMaterializedCopy}) {
    ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                                 candidate(value, 16, 4), false);
  }
  ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                               candidate(kIdentityLower, 16, 2), false);
  ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                               candidate(kIdentityUpper, 18, 2), false);

  for (const loom_value_ordinal_t value :
       {kIdentityRepeated, kIdentitySwapped, kIdentityWritten,
        kIdentityCaptured, kIdentitySeparated}) {
    ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                                 candidate(value, 16, 4), true);
  }
  // Equal roots alone are insufficient: each semantic unit must occupy the
  // exact leased physical unit represented by that root.
  ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                               candidate(kIdentityLower, 17, 2), true);
  ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                               candidate(kIdentityReconstructed, 20, 4), false);

  // The repeated aggregate has mismatched contents only in its upper half.
  // If those units are unused, its live lower half is still the leased content.
  const uint32_t repeated_start =
      scenario.unit_values[kIdentityRepeated].unit_point_start;
  scenario.unit_liveness.end_points[repeated_start + 2] = 1;
  scenario.unit_liveness.end_points[repeated_start + 3] = 1;
  ExpectConflictOnScanAndIndex(&scenario, &descriptor_set,
                               candidate(kIdentityRepeated, 16, 4), false);
  scenario.unit_liveness.end_points[repeated_start + 2] = 2;
  scenario.unit_liveness.end_points[repeated_start + 3] = 2;

  const loom_low_allocation_assignment_t reconstructed =
      candidate(kIdentityReconstructed, 16, 4);
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_record_release_actions(
      &scenario.state, &descriptor_set, &scenario.liveness, &reconstructed,
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  EXPECT_EQ(scenario.state.release_action_count, 0u);
  IREE_ASSERT_OK(
      loom_low_allocation_storage_lease_state_finalize(&scenario.state));
  DeinitializeScenario(&scenario);
}

TEST_F(LowAllocationStorageIdentityTest, RequiresExactExplicitRegisterViews) {
  loom_low_reg_class_t reg_classes[] = {RegClass(0), RegClass(0)};
  reg_classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                         LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  reg_classes[0].allocatable_count = 2;
  reg_classes[1].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                         LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  reg_classes[1].allocatable_count = 1;
  const uint16_t atomic_units[] = {0, 1, 0, 2, 3};
  const loom_low_physical_register_t physical_registers[] = {
      {
          .name_string_ref = 0,
          .atomic_unit_start = 0,
          .atomic_unit_count = 2,
          .reserved = 0,
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 2,
          .atomic_unit_count = 1,
          .reserved = 0,
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 3,
          .atomic_unit_count = 2,
          .reserved = 0,
      },
  };
  reg_classes[0].candidate_lookup.register_count = 3;
  reg_classes[1].candidate_lookup.ordinal_start = 3;
  reg_classes[1].candidate_lookup.register_base = 1;
  reg_classes[1].candidate_lookup.register_count = 1;
  reg_classes[1].physical_register_candidate_start = 2;
  const uint16_t candidate_ordinals[] = {0, UINT16_MAX, 1, 0};
  const uint16_t candidates[] = {0, 2, 1};
  const uint16_t allocation_ordinals[] = {0, 1, 0};
  loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  descriptor_set.physical_registers = physical_registers;
  descriptor_set.physical_register_count = IREE_ARRAYSIZE(physical_registers);
  descriptor_set.physical_register_candidate_ordinals = candidate_ordinals;
  descriptor_set.physical_register_candidate_ordinal_count =
      IREE_ARRAYSIZE(candidate_ordinals);
  descriptor_set.physical_register_candidate_ids = candidates;
  descriptor_set.physical_register_allocation_ordinals = allocation_ordinals;
  descriptor_set.physical_register_candidate_count = IREE_ARRAYSIZE(candidates);
  descriptor_set.physical_register_atomic_units = atomic_units;
  descriptor_set.physical_register_atomic_unit_count =
      IREE_ARRAYSIZE(atomic_units);

  constexpr uint32_t unit_counts[] = {1, 1, 1};
  constexpr loom_value_ordinal_t storage_order[] = {1, 2, 0};
  const loom_low_placement_relation_t relations[] = {StorageRelation(
      /*result_ordinal=*/1, /*source_ordinal=*/0,
      /*result_unit_offset=*/0, /*source_unit_offset=*/0, /*unit_count=*/1,
      LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY)};
  StorageIdentityScenario scenario;
  InitializeScenario(&descriptor_set, unit_counts, IREE_ARRAYSIZE(unit_counts),
                     relations, IREE_ARRAYSIZE(relations), storage_order,
                     LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES,
                     &scenario);

  loom_low_allocation_assignment_t assignments[] = {
      Assignment(scenario.value_ids[0], /*descriptor_reg_class_id=*/0,
                 /*start_point=*/0, /*end_point=*/1,
                 /*location_base=*/0, /*location_count=*/1)};
  assignments[0].unit_count = 1;
  loom_low_allocation_storage_lease_state_record_assignment(
      &scenario.state, &descriptor_set, &scenario.liveness, assignments,
      /*assignment_index=*/0, /*value_ordinal=*/0);

  auto candidate =
      Assignment(scenario.value_ids[1], /*descriptor_reg_class_id=*/0,
                 /*start_point=*/1, /*end_point=*/2,
                 /*location_base=*/0, /*location_count=*/1);
  candidate.unit_count = 1;
  candidate.unit_point_start = scenario.unit_values[1].unit_point_start;
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
      &scenario.state, &descriptor_set, &scenario.liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  // The narrow view aliases only one atomic part of the leased wide register.
  // Matching content roots cannot make that partial physical overlap exact.
  candidate.descriptor_reg_class_id = 1;
  candidate.location_base = 1;
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
      &scenario.state, &descriptor_set, &scenario.liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  // An unrelated SSA value in the exact wide view remains a real conflict.
  candidate.value_id = scenario.value_ids[2];
  candidate.unit_point_start = scenario.unit_values[2].unit_point_start;
  candidate.descriptor_reg_class_id = 0;
  candidate.location_base = 0;
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
      &scenario.state, &descriptor_set, &scenario.liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  // An empty candidate unit does not occupy its explicit register. Restoring
  // demand must conflict even though the leased source's SSA lifetime ended.
  scenario.unit_liveness.end_points[candidate.unit_point_start] = 1;
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
      &scenario.state, &descriptor_set, &scenario.liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  scenario.unit_liveness.end_points[candidate.unit_point_start] = 2;
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
      &scenario.state, &descriptor_set, &scenario.liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  // The other wide register occupies disjoint atomic storage.
  candidate.location_base = 2;
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
      &scenario.state, &descriptor_set, &scenario.liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  IREE_ASSERT_OK(
      loom_low_allocation_storage_lease_state_finalize(&scenario.state));
  DeinitializeScenario(&scenario);
}

TEST_F(LowAllocationStorageLeaseTest,
       OrdersOnlyCandidatesWithContinuousStorageLifetime) {
  const loom_low_reg_class_t reg_classes[] = {RegClass(/*alias_set_id=*/1)};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_liveness_segment_t storage_segments[] = {
      {/*start_point=*/10, /*end_point=*/20},
      {/*start_point=*/10, /*end_point=*/15},
      {/*start_point=*/16, /*end_point=*/20},
  };
  uint32_t unit_end_points[] = {20, 20};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
  unit_liveness.storage_segments.entries = storage_segments;
  uint32_t expiration_entry = 0;
  loom_low_allocation_storage_lease_state_t state = {};
  state.unit_liveness = &unit_liveness;
  state.availability_expiration_heap = &expiration_entry;

  auto candidate = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0,
      /*start_point=*/10, /*end_point=*/20, /*location_base=*/0,
      /*location_count=*/1);
  candidate.unit_count = 1;
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  for (const uint32_t end_point : {10, 15}) {
    unit_end_points[0] = end_point;
    EXPECT_FALSE(loom_low_allocation_storage_lease_state_can_order_candidate(
        &state, &descriptor_set, &candidate,
        LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  }
  unit_end_points[0] = 20;

  candidate.liveness_segments = {/*start=*/0, /*count=*/1};
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  candidate.liveness_segments = {/*start=*/1, /*count=*/1};
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  candidate.liveness_segments = {/*start=*/1, /*count=*/2};
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));

  candidate.liveness_segments = {/*start=*/0, /*count=*/1};
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED));

  // A location-only availability summary remains exact for a self-root value,
  // while a forwarded value needs candidate-specific identity comparison.
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {
      DefineValue(module),
      DefineValue(module),
  };
  loom_local_value_domain_t value_domain = {};
  AcquireValueDomain(module, value_ids, IREE_ARRAYSIZE(value_ids),
                     &value_domain);
  loom_low_allocation_unit_liveness_value_t unit_values[2] = {};
  unit_values[0].unit_point_start = 0;
  unit_values[1].unit_point_start = 1;
  const uint32_t identity_origins[] = {0, 0};
  state.value_domain = &value_domain;
  unit_liveness.values = unit_values;
  state.identity_origins = identity_origins;
  candidate.value_id = value_ids[0];
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  candidate.value_id = value_ids[1];
  candidate.unit_point_start = 1;
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_can_order_candidate(
      &state, &descriptor_set, &candidate,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

struct ReleasePoint {
  // Candidate lifetime start in the allocation liveness coordinate space.
  uint32_t program_point;
  // Scheduled packet at the point, or UINT32_MAX for a gap or empty extent.
  uint32_t packet_index;
  // Liveness block owning the packet when the point maps to a packet.
  uint32_t block_index;
  // Packet ordinal within its block when the point maps to a packet.
  uint32_t scheduled_ordinal;
  // Scheduled node defining the candidate, or NONE to use its lifetime start.
  uint32_t producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  // Program point of |producer_node|, or UINT32_MAX when it is absent.
  uint32_t producer_program_point = UINT32_MAX;
};

class LowAllocationStorageLeaseReleasePointTest
    : public LowAllocationStorageLeaseTest,
      public ::testing::WithParamInterface<ReleasePoint> {};

TEST_P(LowAllocationStorageLeaseReleasePointTest,
       MaterializesAndRefinesConflictingLeaseRelease) {
  const ReleasePoint point = GetParam();
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
      LivenessBlock(/*start_point=*/4, /*end_point=*/4),
      LivenessBlock(/*start_point=*/5, /*end_point=*/7),
      LivenessBlock(/*start_point=*/8, /*end_point=*/10),
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks), value_domain.value_ids,
               value_domain.value_count);
  const loom_low_schedule_block_t schedule_blocks[] = {
      ScheduleBlock(/*scheduled_node_start=*/0,
                    /*scheduled_node_count=*/3),
      ScheduleBlock(/*scheduled_node_start=*/3,
                    /*scheduled_node_count=*/0),
      ScheduleBlock(/*scheduled_node_start=*/3,
                    /*scheduled_node_count=*/2),
      ScheduleBlock(/*scheduled_node_start=*/5,
                    /*scheduled_node_count=*/2),
  };
  const loom_low_schedule_node_t nodes[] = {
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/0,
                          /*operand=*/0),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/1,
                          /*operand=*/1),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/2,
                          /*operand=*/1),
      ScheduleOperandNode(/*block_index=*/2, /*scheduled_ordinal=*/0,
                          /*operand=*/1),
      ScheduleOperandNode(/*block_index=*/2, /*scheduled_ordinal=*/1,
                          /*operand=*/1),
      ScheduleOperandNode(/*block_index=*/3, /*scheduled_ordinal=*/0,
                          /*operand=*/1),
      ScheduleOperandNode(/*block_index=*/3, /*scheduled_ordinal=*/1,
                          /*operand=*/1),
  };
  const uint32_t scheduled_node_indices[] = {0, 1, 2, 3, 4, 5, 6};
  const uint32_t value_producer_nodes[] = {
      LOOM_LOW_SCHEDULE_NODE_NONE,
      point.producer_node,
  };
  loom_low_schedule_table_t schedule =
      Schedule(module, function_op, liveness, schedule_blocks,
               IREE_ARRAYSIZE(schedule_blocks), nodes, IREE_ARRAYSIZE(nodes),
               scheduled_node_indices, IREE_ARRAYSIZE(scheduled_node_indices),
               value_producer_nodes);
  schedule.target.descriptor_set = &descriptor_set;
  loom_low_storage_lease_record_t records[] = {StorageLeaseRecord()};
  records[0].unit_count = 32;
  const loom_low_storage_lease_table_t lease_table =
      StorageLeaseTable(&schedule, records, IREE_ARRAYSIZE(records));
  const loom_low_allocation_storage_identity_t storage_identity = {};

  uint32_t unit_end_points[] = {point.program_point + 1};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points)};
  unit_liveness.storage_segments.entries = liveness.segments;
  loom_low_allocation_storage_lease_state_t state = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
      &lease_table, module, function_op, &value_domain, &liveness,
      &storage_identity, &unit_liveness, &arena_, &state));

  const loom_low_allocation_assignment_t leased_assignment = Assignment(
      /*value_id=*/value_ids[0], /*descriptor_reg_class_id=*/0,
      /*start_point=*/0, /*end_point=*/1, /*location_base=*/10,
      /*location_count=*/33);
  loom_low_allocation_storage_lease_state_record_assignment(
      &state, &descriptor_set, &liveness, &leased_assignment,
      /*assignment_index=*/0, /*value_ordinal=*/0);
  ASSERT_EQ(state.instance_count, 1u);
  ASSERT_NE(state.availability_expiration_heap, nullptr);
  EXPECT_EQ(state.availability_expiration_count, 1u);
  const loom_low_allocation_storage_lease_t* lease = &state.instances[0];
  EXPECT_EQ(lease->value_id, value_ids[0]);
  EXPECT_EQ(lease->start_point, 0u);
  EXPECT_EQ(lease->end_point, 10u);
  EXPECT_EQ(lease->location_base, 11u);
  EXPECT_EQ(lease->location_count, 32u);
  EXPECT_EQ(lease->release_action_index,
            LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE);

  auto candidate = Assignment(
      /*value_id=*/value_ids[1], /*descriptor_reg_class_id=*/1,
      /*start_point=*/point.program_point,
      /*end_point=*/point.program_point + 1, /*location_base=*/11,
      /*location_count=*/1);
  candidate.unit_count = 1;
  uint32_t available_base = UINT32_MAX;
  EXPECT_FALSE(
      loom_low_allocation_storage_lease_state_find_next_available_location(
          &state, &descriptor_set, &candidate,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN,
          /*minimum_base=*/11, /*maximum_base=*/11, &available_base));

  // Aggregate storage can be reserved before its earlier scalar definitions.
  // A release scheduled for the reservation still conflicts with those writes.
  if (point.program_point < 9) {
    const loom_low_allocation_assignment_t later_candidate =
        Assignment(value_ids[1], /*descriptor_reg_class_id=*/1,
                   /*start_point=*/9, /*end_point=*/10, /*location_base=*/11,
                   /*location_count=*/2);
    IREE_ASSERT_OK(
        loom_low_allocation_storage_lease_state_record_release_actions(
            &state, &descriptor_set, &liveness, &later_candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0));
    ASSERT_EQ(state.release_action_count, 1u);
    EXPECT_EQ(lease->end_point, 9u);
  }

  EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
      &state, &descriptor_set, &liveness, &candidate,
      /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  const bool maps_to_packet = point.packet_index != UINT32_MAX;
  EXPECT_EQ(!maps_to_packet,
            loom_low_allocation_storage_lease_state_conflicts(
                &state, &descriptor_set, &liveness, &candidate,
                /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
                LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED));

  if (maps_to_packet) {
    IREE_ASSERT_OK(
        loom_low_allocation_storage_lease_state_record_release_actions(
            &state, &descriptor_set, &liveness, &candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0));
    ASSERT_EQ(state.release_action_count, 1u);
    EXPECT_EQ(lease->release_action_index, 0u);
    EXPECT_EQ(lease->end_point, point.producer_program_point == UINT32_MAX
                                    ? point.program_point
                                    : point.producer_program_point);
    EXPECT_EQ(state.availability_expiration_count, 0u);
    EXPECT_TRUE(
        loom_low_allocation_storage_lease_state_find_next_available_location(
            &state, &descriptor_set, &candidate,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN,
            /*minimum_base=*/11, /*maximum_base=*/11, &available_base));
    EXPECT_EQ(available_base, 11u);
    const loom_low_storage_release_action_t* action = &state.release_actions[0];
    EXPECT_EQ(action->insertion_packet_index, point.packet_index);
    EXPECT_EQ(action->insertion_node_index, point.packet_index);
    EXPECT_EQ(action->block_index, point.block_index);
    EXPECT_EQ(action->scheduled_ordinal, point.scheduled_ordinal);
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
  }
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_finalize(&state));

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

INSTANTIATE_TEST_SUITE_P(
    BlockExtents, LowAllocationStorageLeaseReleasePointTest,
    ::testing::Values(ReleasePoint{1, 1, 0, 1}, ReleasePoint{2, 2, 0, 2},
                      ReleasePoint{3, UINT32_MAX, 0, 0},
                      ReleasePoint{4, UINT32_MAX, 0, 0},
                      ReleasePoint{5, 3, 2, 0}, ReleasePoint{6, 4, 2, 1},
                      ReleasePoint{7, UINT32_MAX, 0, 0},
                      ReleasePoint{8, 5, 3, 0}, ReleasePoint{9, 6, 3, 1},
                      // The scheduled producer controls release insertion even
                      // when its packet precedes the source liveness point.
                      ReleasePoint{9, 3, 2, 0, 3, 5}));

TEST_F(LowAllocationStorageLeaseTest,
       UsesExactProgressBoundAsPhysicalLeaseEndpoint) {
  const loom_low_reg_class_t reg_classes[] = {
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
      LivenessBlock(/*start_point=*/0, /*end_point=*/5),
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks), value_domain.value_ids,
               value_domain.value_count);
  const loom_low_schedule_block_t schedule_blocks[] = {
      ScheduleBlock(/*scheduled_node_start=*/0,
                    /*scheduled_node_count=*/5),
  };
  const loom_low_schedule_node_t nodes[] = {
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/0,
                          /*operand=*/0),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/1,
                          /*operand=*/0),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/2,
                          /*operand=*/0),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/3,
                          /*operand=*/0),
      ScheduleOperandNode(/*block_index=*/0, /*scheduled_ordinal=*/4,
                          /*operand=*/0),
  };
  const uint32_t scheduled_node_indices[] = {0, 1, 2, 3, 4};
  const uint32_t value_producer_nodes[] = {
      LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  loom_low_schedule_table_t schedule =
      Schedule(module, function_op, liveness, schedule_blocks,
               IREE_ARRAYSIZE(schedule_blocks), nodes, IREE_ARRAYSIZE(nodes),
               scheduled_node_indices, IREE_ARRAYSIZE(scheduled_node_indices),
               value_producer_nodes);
  schedule.target.descriptor_set = &descriptor_set;
  loom_low_storage_lease_record_t records[] = {StorageLeaseRecord()};
  records[0].release_before_scheduled_ordinal_plus_one = 4;
  const loom_low_storage_lease_table_t lease_table =
      StorageLeaseTable(&schedule, records, IREE_ARRAYSIZE(records));
  const loom_low_allocation_storage_identity_t storage_identity = {};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.storage_segments.entries = liveness.segments;

  loom_low_allocation_storage_lease_state_t state = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
      &lease_table, module, function_op, &value_domain, &liveness,
      &storage_identity, &unit_liveness, &arena_, &state));
  const loom_low_allocation_assignment_t leased_assignment = Assignment(
      /*value_id=*/value_ids[0], /*descriptor_reg_class_id=*/0,
      /*start_point=*/0, /*end_point=*/1, /*location_base=*/10,
      /*location_count=*/3);
  loom_low_allocation_storage_lease_state_record_assignment(
      &state, &descriptor_set, &liveness, &leased_assignment,
      /*assignment_index=*/0, /*value_ordinal=*/0);
  ASSERT_EQ(state.instance_count, 1u);
  EXPECT_EQ(state.instances[0].start_point, 0u);
  EXPECT_EQ(state.instances[0].end_point, 4u);

  // A result written by the packet before the bound begins at point 3 and
  // must still conflict with the lease.
  const loom_low_allocation_assignment_t before_bound = Assignment(
      /*value_id=*/value_ids[1], /*descriptor_reg_class_id=*/0,
      /*start_point=*/3, /*end_point=*/4, /*location_base=*/11,
      /*location_count=*/1);
  EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
      &state, &descriptor_set, &liveness, &before_bound,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
  // A result written by the bound packet begins at point 4 and may reuse the
  // released unit.
  const loom_low_allocation_assignment_t at_bound = Assignment(
      /*value_id=*/value_ids[1], /*descriptor_reg_class_id=*/0,
      /*start_point=*/4, /*end_point=*/5, /*location_base=*/11,
      /*location_count=*/1);
  EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
      &state, &descriptor_set, &liveness, &at_bound,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
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
  const uint32_t value_producer_nodes[] = {
      LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  const loom_low_schedule_table_t schedule =
      Schedule(module, function_op, schedule_liveness, schedule_blocks,
               IREE_ARRAYSIZE(schedule_blocks), nodes, IREE_ARRAYSIZE(nodes),
               scheduled_node_indices, IREE_ARRAYSIZE(scheduled_node_indices),
               value_producer_nodes);
  const loom_low_storage_lease_record_t records[] = {StorageLeaseRecord()};
  const loom_low_storage_lease_table_t lease_table =
      StorageLeaseTable(&schedule, records, IREE_ARRAYSIZE(records));
  const loom_low_allocation_storage_identity_t storage_identity = {};

  const loom_low_allocation_unit_liveness_t unit_liveness = {};
  loom_low_allocation_storage_lease_state_t state = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_allocation_storage_lease_state_initialize(
          &lease_table, module, function_op, &allocation_value_domain,
          &allocation_liveness, &storage_identity, &unit_liveness, &arena_,
          &state));

  loom_local_value_domain_release(&allocation_value_domain);
  loom_module_free(module);
}

TEST_F(LowAllocationStorageLeaseTest,
       PreservesLeasesAcrossStorageLifetimeHoles) {
  const loom_low_reg_class_t reg_classes[] = {RegClass(1), RegClass(1)};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_liveness_segment_t segments[] = {{0, 2}, {6, 10}, {1, 7}};
  loom_liveness_analysis_t liveness = {
      .segments = segments, .segment_count = IREE_ARRAYSIZE(segments)};
  uint32_t unit_end_points[] = {10};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points)};
  unit_liveness.storage_segments.entries = segments;

  for (const auto kind : {LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
                          LOOM_LOW_STORAGE_LEASE_RESULT_WRITE}) {
    for (const bool indexed : {false, true}) {
      SCOPED_TRACE(static_cast<int>(kind));
      SCOPED_TRACE(indexed);
      loom_low_storage_lease_record_t record = StorageLeaseRecord();
      record.kind = kind;
      record.attachment = kind == LOOM_LOW_STORAGE_LEASE_SOURCE_READ
                              ? LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND
                              : LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT;
      record.unit_offset = 0;
      const loom_low_storage_lease_table_t table =
          StorageLeaseTable(nullptr, &record, 1);
      loom_low_allocation_storage_lease_t lease = {
          .value_id = 0,
          .start_point = 2,
          .end_point = 6,
          .descriptor_reg_class_id = 0,
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 10,
          .location_count = 2};
      auto leased_assignment =
          Assignment(/*value_id=*/0, /*descriptor_reg_class_id=*/0,
                     /*start_point=*/0, /*end_point=*/1,
                     /*location_base=*/10, /*location_count=*/2);
      leased_assignment.unit_count = 2;
      uint8_t instance_written = 1;
      loom_low_allocation_storage_lease_state_t state = {};
      state.lease_table = &table;
      state.assignments = &leased_assignment;
      state.unit_liveness = &unit_liveness;
      state.instances = &lease;
      state.instance_written = &instance_written;
      state.instance_count = 1;
      loom_low_allocation_storage_lease_unit_index_t index = {};
      if (indexed) {
        IREE_ASSERT_OK(loom_low_allocation_storage_lease_unit_index_initialize(
            &index, &lease, /*lease_count=*/1, /*lease_unit_capacity=*/2,
            /*distinct_unit_capacity=*/2, &arena_));
        loom_low_allocation_storage_lease_unit_index_insert(
            &index, &descriptor_set, 0, 0);
        state.unit_index = &index;
      }

      // The aliasing register class overlaps the lease's second unit. Only
      // the candidate has a hole: the lease keeps its complete [2, 6) range.
      auto candidate = Assignment(/*value_id=*/1, /*descriptor_reg_class_id=*/1,
                                  /*start_point=*/0, /*end_point=*/10,
                                  /*location_base=*/11, /*location_count=*/1);
      candidate.liveness_segments = {0, 2};
      candidate.unit_count = 1;
      EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
          &state, &descriptor_set, &liveness, &candidate, nullptr, 0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
      IREE_ASSERT_OK(
          loom_low_allocation_storage_lease_state_record_release_actions(
              &state, &descriptor_set, &liveness, &candidate, nullptr, 0));
      EXPECT_EQ(state.release_action_count, 0u);
      EXPECT_EQ(lease.start_point, 2u);
      EXPECT_EQ(lease.end_point, 6u);

      // Incomplete storage segments retain the conservative interval. A real
      // overlapping segment also remains a conflict for either lease kind.
      candidate.liveness_segments = {};
      EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
          &state, &descriptor_set, &liveness, &candidate, nullptr, 0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
      candidate.liveness_segments = {2, 1};
      EXPECT_TRUE(loom_low_allocation_storage_lease_state_conflicts(
          &state, &descriptor_set, &liveness, &candidate, nullptr, 0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
      candidate.location_base = 12;
      EXPECT_FALSE(loom_low_allocation_storage_lease_state_conflicts(
          &state, &descriptor_set, &liveness, &candidate, nullptr, 0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN));
    }
  }
}

}  // namespace
}  // namespace loom
