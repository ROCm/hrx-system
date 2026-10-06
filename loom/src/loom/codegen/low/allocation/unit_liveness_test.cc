// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/util/cfg_graph.h"

namespace loom {
namespace {

class LowAllocationUnitLivenessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_arena_initialize(&block_pool_, &decision_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const auto* vtables = loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&decision_arena_);
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

  void RetainAndPropagateStorage(
      loom_low_allocation_unit_liveness_t* unit_liveness,
      const loom_liveness_analysis_t* liveness,
      const loom_low_placement_table_t* placement) {
    IREE_ASSERT_OK(loom_low_allocation_unit_liveness_retain_tied_storage(
        unit_liveness, liveness, placement, &arena_, &decision_arena_));
    loom_low_allocation_unit_liveness_propagate_storage_relations(unit_liveness,
                                                                  placement);
  }

  // Shared block pool for result, decision, and construction lifetimes.
  iree_arena_block_pool_t block_pool_;
  // Retains published point/segment arrays and prerequisite analyses.
  iree_arena_allocator_t arena_;
  // Owns query metadata used only while making allocation decisions.
  iree_arena_allocator_t decision_arena_;
  // Dialect context for resolved descriptor operations.
  loom_context_t context_;
};

loom_liveness_interval_t RegisterInterval(loom_value_id_t value_id,
                                          uint32_t start_point,
                                          uint32_t end_point,
                                          uint32_t unit_count) {
  loom_liveness_interval_t interval = {};
  interval.value_id = value_id;
  interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  interval.start_point = start_point;
  interval.end_point = end_point;
  interval.unit_count = unit_count;
  return interval;
}

loom_liveness_analysis_t Liveness(const loom_value_id_t* value_ids,
                                  iree_host_size_t value_count,
                                  const uint32_t* value_interval_indices,
                                  const loom_liveness_interval_t* intervals,
                                  iree_host_size_t interval_count,
                                  const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count) {
  loom_liveness_analysis_t liveness = {};
  liveness.value_ids = value_ids;
  liveness.value_count = value_count;
  liveness.value_interval_indices = value_interval_indices;
  liveness.intervals = intervals;
  liveness.interval_count = interval_count;
  liveness.blocks = blocks;
  liveness.block_count = block_count;
  return liveness;
}

class LowAllocationUnitLivenessExtentTest
    : public LowAllocationUnitLivenessTest,
      public ::testing::WithParamInterface<std::array<uint32_t, 2>> {};

TEST_P(LowAllocationUnitLivenessExtentTest, BoundsExtentBeforePointAllocation) {
  auto* module = AllocateModule();
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_type_id_t source_type = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module, loom_type_buffer(), &source_type));
  for (uint32_t i = 0; i < GetParam().size(); ++i) {
    // Distinct resource definitions have disjoint unused lifetimes. Their
    // total unit extent can exceed u32 while pressure remains representable.
    loom_op_t* resource = nullptr;
    IREE_ASSERT_OK(loom_low_resource_build(
        &builder, 0, LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER,
        LOOM_VALUE_ID_INVALID, i, source_type, 0, 0,
        loom_low_register_type(1, 0, GetParam()[i]), LOOM_LOCATION_UNKNOWN,
        &resource));
  }
  loom_local_value_domain_t domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module, module->body, &arena_, &domain));
  loom_liveness_analysis_t liveness = {};
  IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
      &domain, loom_liveness_order_empty(), &arena_, &liveness));
  ASSERT_EQ(liveness.interval_count, GetParam().size());
  for (uint32_t i = 0; i < GetParam().size(); ++i) {
    EXPECT_EQ(
        loom_liveness_interval_for_value_ordinal(&liveness, i)->unit_count,
        GetParam()[i]);
  }
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(loom_cfg_graph_build(module, module->body, &arena_, &graph));

  // Reject every backing allocation before reaching the OS, even if the
  // producer incorrectly attempts to allocate an unrepresentable point array.
  iree_host_size_t allocation_count = 0;
  const iree_allocator_t rejecting_allocator = {
      &allocation_count,
      [](void* self, iree_allocator_command_t, const void*,
         void**) -> iree_status_t {
        ++*static_cast<iree_host_size_t*>(self);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "point allocation rejected by test allocator");
      }};
  iree_arena_block_pool_t result_pool;
  iree_arena_block_pool_initialize(4096, rejecting_allocator, &result_pool);
  iree_arena_allocator_t result_arena;
  iree_arena_initialize(&result_pool, &result_arena);
  loom_low_reg_class_t register_class = {};
  register_class.alloc_unit_bits = 32;
  loom_low_descriptor_set_t descriptors = {};
  descriptors.stable_id = 1;
  descriptors.reg_classes = &register_class;
  descriptors.reg_class_count = 1;
  loom_low_resolved_target_t target = {};
  target.descriptor_set = &descriptors;
  loom_low_placement_table_t placement = {};
  loom_low_allocation_unit_liveness_t result = {};
  const uint64_t total = (uint64_t)GetParam()[0] + GetParam()[1];
  const bool fits = total <= UINT32_MAX;
  IREE_EXPECT_STATUS_IS(
      fits ? IREE_STATUS_RESOURCE_EXHAUSTED : IREE_STATUS_OUT_OF_RANGE,
      loom_low_allocation_unit_liveness_initialize(
          &target, &placement, &domain, &liveness, &graph, &result_arena,
          &decision_arena_, &result));
  // On a 32-bit host, a legal index extent may still overflow the byte size;
  // arena allocation rejects that before asking its backing allocator.
  EXPECT_EQ(allocation_count,
            fits && total <= IREE_HOST_SIZE_MAX / sizeof(uint32_t) ? 1u : 0u);
  EXPECT_EQ(result.start_points, nullptr);
  EXPECT_EQ(result.end_points, nullptr);
  iree_arena_deinitialize(&result_arena);
  iree_arena_block_pool_deinitialize(&result_pool);
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
}

INSTANTIATE_TEST_SUITE_P(
    UnitExtent, LowAllocationUnitLivenessExtentTest,
    ::testing::Values(std::array<uint32_t, 2>{1, 1},
                      std::array<uint32_t, 2>{UINT32_MAX - 2, 1},
                      std::array<uint32_t, 2>{UINT32_MAX - 1, 1},
                      std::array<uint32_t, 2>{UINT32_MAX, 1},
                      std::array<uint32_t, 2>{UINT32_MAX - 1, 2},
                      std::array<uint32_t, 2>{1u << 31, 1u << 31}));

TEST_F(LowAllocationUnitLivenessTest, RetainsImplicitReadsWithoutClobbering) {
  loom_low_reg_class_t classes[2] = {};
  for (auto& reg_class : classes) {
    reg_class.allocatable_count = 4;
    reg_class.alloc_unit_bits = 32;
  }
  // A finite inventory is physical without requiring the redundant flag.
  // Implicit writes name singleton state; reads may cover several units.
  classes[1].allocatable_count = 1;
  const loom_low_reg_class_alt_t alternatives[] = {
      {0, LOOM_LOW_REGISTER_PART_NONE, 0, 0},
      {1, LOOM_LOW_REGISTER_PART_NONE, 0, 0},
  };
  loom_low_operand_t operands[3] = {};
  operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  operands[0].reg_class_alt_count = 1;
  operands[0].unit_count = 1;
  for (uint16_t i = 0; i < 2; ++i) {
    operands[i + 1].role = LOOM_LOW_OPERAND_ROLE_IMPLICIT;
    operands[i + 1].source_value_index = LOOM_LOW_ID_NONE;
    operands[i + 1].reg_class_alt_start = i;
    operands[i + 1].reg_class_alt_count = 1;
    operands[i + 1].flags = LOOM_LOW_OPERAND_FLAG_IMPLICIT |
                            (i == 0 ? LOOM_LOW_OPERAND_FLAG_STATE_READ
                                    : LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    operands[i + 1].unit_count = i == 0 ? 3 : 1;
  }
  loom_low_descriptor_t descriptor = {};
  descriptor.operand_count = 3;
  descriptor.result_count = 1;
  loom_low_descriptor_set_t descriptors = {};
  descriptors.stable_id = 1;
  descriptors.reg_classes = classes;
  descriptors.reg_class_count = 2;
  descriptors.reg_class_alts = alternatives;
  descriptors.reg_class_alt_count = 2;
  descriptors.operands = operands;
  descriptors.operand_count = 3;
  descriptors.descriptors = &descriptor;
  descriptors.descriptor_count = 1;
  loom_low_resolved_target_t target = {};
  target.descriptor_set = &descriptors;
  auto* module = AllocateModule();
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* op = nullptr;
  const auto type = loom_low_register_type(1, 0, 1);
  IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
      &builder, &descriptors, &descriptor, 0, nullptr, 0, {}, &type, 1, nullptr,
      0, LOOM_LOCATION_UNKNOWN, &op));
  loom_local_value_domain_t domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module, module->body, &arena_, &domain));
  loom_liveness_analysis_t liveness = {};
  IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
      &domain, loom_liveness_order_empty(), &arena_, &liveness));
  loom_low_allocation_unit_liveness_t result = {};
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module, module->body, &arena_, &cfg_graph));
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      &target, nullptr, &domain, &liveness, &cfg_graph, &arena_,
      &decision_arena_, &result));
  ASSERT_NE(result.implicit_location_counts_by_reg_class, nullptr);
  EXPECT_EQ(result.implicit_location_counts_by_reg_class[0], 3u);
  EXPECT_EQ(result.implicit_location_counts_by_reg_class[1], 1u);
  EXPECT_EQ(result.clobbers.count, 1u);
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
}

TEST_F(LowAllocationUnitLivenessTest, ExcludesRequiredStorageComponents) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t values[] = {DefineValue(module), DefineValue(module),
                                    DefineValue(module)};
  loom_local_value_domain_t domain = {};
  AcquireValueDomain(module, values, IREE_ARRAYSIZE(values), &domain);
  const loom_value_ordinal_t roots[] = {1, 1, 2};
  loom_low_placement_table_t placement = {};
  placement.module = module;
  placement.tied_storage_origins_by_value_ordinal = roots;
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.tied_storage_placement = &placement;

  for (uint32_t ignored = 0; ignored < 2; ++ignored) {
    for (uint32_t value = 0; value < 3; ++value) {
      EXPECT_EQ(loom_low_allocation_unit_liveness_storage_is_ignored(
                    &unit_liveness, values[value], &values[ignored], 1),
                value < 2);
    }
  }
  EXPECT_FALSE(loom_low_allocation_unit_liveness_storage_is_ignored(
      &unit_liveness, values[0], nullptr, 0));
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
}

TEST_F(LowAllocationUnitLivenessTest, InitializesUnitStartsAndBoundaryUses) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {
      DefineValue(module),
      DefineValue(module),
  };
  loom_local_value_domain_t value_domain = {};
  AcquireValueDomain(module, value_ids, IREE_ARRAYSIZE(value_ids),
                     &value_domain);

  loom_region_t* body = module->body;
  loom_block_t* block = loom_region_entry_block(body);
  loom_low_resolved_target_t target = {};
  const uint32_t value_interval_indices[] = {0, 1};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(value_ids[0], /*start_point=*/2,
                       /*end_point=*/2, /*unit_count=*/2),
      RegisterInterval(value_ids[1], /*start_point=*/3,
                       /*end_point=*/3, /*unit_count=*/1),
  };
  const loom_value_id_t live_in_values[] = {value_ids[0]};
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/block,
          /*.start_point=*/5,
          /*.end_point=*/5,
          /*.live_in_values=*/live_in_values,
          /*.live_in_count=*/IREE_ARRAYSIZE(live_in_values),
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness = Liveness(
      value_domain.value_ids, value_domain.value_count, value_interval_indices,
      intervals, IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));

  loom_low_allocation_unit_liveness_t unit_liveness = {};
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(loom_cfg_graph_build(module, body, &arena_, &cfg_graph));
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      &target, nullptr, &value_domain, &liveness, &cfg_graph, &arena_,
      &decision_arena_, &unit_liveness));

  EXPECT_EQ(loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
                &unit_liveness, &liveness, /*value_ordinal=*/0),
            0u);
  EXPECT_EQ(loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
                &unit_liveness, &liveness, /*value_ordinal=*/1),
            2u);
  EXPECT_EQ(unit_liveness.values[0].acquisition_start_point, 2u);
  EXPECT_EQ(unit_liveness.values[1].acquisition_start_point, 3u);
  EXPECT_GT(decision_arena_.used_allocation_size, 0u);
  // Published points survive freeing all allocation-decision storage, not
  // merely returning its blocks to the reusable pool.
  iree_arena_reset(&decision_arena_);
  iree_arena_block_pool_trim(&block_pool_);
  ASSERT_EQ(unit_liveness.point_count, 3u);
  EXPECT_EQ(unit_liveness.start_points[0], 2u);
  EXPECT_EQ(unit_liveness.start_points[1], 2u);
  EXPECT_EQ(unit_liveness.start_points[2], 3u);
  EXPECT_EQ(unit_liveness.end_points[0], 6u);
  EXPECT_EQ(unit_liveness.end_points[1], 6u);
  EXPECT_EQ(unit_liveness.end_points[2], 4u);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

TEST_F(LowAllocationUnitLivenessTest, ExtendsTiedResultSourceUnits) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {
      DefineValue(module),
      DefineValue(module),
  };
  loom_local_value_domain_t value_domain = {};
  AcquireValueDomain(module, value_ids, IREE_ARRAYSIZE(value_ids),
                     &value_domain);

  loom_region_t* body = module->body;
  loom_block_t* block = loom_region_entry_block(body);
  loom_low_resolved_target_t target = {};
  const uint32_t value_interval_indices[] = {0, 1};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(value_ids[0], /*start_point=*/0,
                       /*end_point=*/0, /*unit_count=*/2),
      RegisterInterval(value_ids[1], /*start_point=*/7,
                       /*end_point=*/7, /*unit_count=*/2),
  };
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/block,
          /*.start_point=*/0,
          /*.end_point=*/0,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  loom_liveness_analysis_t liveness = Liveness(
      value_domain.value_ids, value_domain.value_count, value_interval_indices,
      intervals, IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));
  const loom_liveness_segment_t segments[] = {
      {/*.start_point=*/0, /*.end_point=*/1},
      {/*.start_point=*/7, /*.end_point=*/8},
  };
  const loom_liveness_segment_range_t value_segment_ranges[] = {
      {/*.start=*/0, /*.count=*/1},
      {/*.start=*/1, /*.count=*/1},
  };
  liveness.segments = segments;
  liveness.segment_count = IREE_ARRAYSIZE(segments);
  liveness.value_segment_ranges = value_segment_ranges;

  loom_low_allocation_unit_liveness_t unit_liveness = {};
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(loom_cfg_graph_build(module, body, &arena_, &cfg_graph));
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      &target, nullptr, &value_domain, &liveness, &cfg_graph, &arena_,
      &decision_arena_, &unit_liveness));
  ASSERT_EQ(unit_liveness.point_count, 4u);
  EXPECT_EQ(unit_liveness.end_points[0], 1u);
  EXPECT_EQ(unit_liveness.end_points[1], 1u);
  EXPECT_EQ(unit_liveness.end_points[2], 8u);
  EXPECT_EQ(unit_liveness.end_points[3], 8u);
  EXPECT_EQ(
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, /*value_ordinal=*/0)
          .count,
      1u);

  loom_low_placement_relation_t relations[] = {
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/1,
          /*.source_ordinal=*/0,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          {/*.location_mask=*/0},
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
              LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
          /*.priority=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  const loom_low_placement_relation_range_t ranges[] = {{0, 0}, {0, 1}};
  const loom_value_ordinal_t storage_order[] = {1, 0};
  const loom_value_ordinal_t tied_origins[] = {0, 0};
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;

  RetainAndPropagateStorage(&unit_liveness, &liveness, &placement);
  EXPECT_EQ(unit_liveness.start_points[2], 0u);
  EXPECT_EQ(unit_liveness.start_points[3], 0u);
  EXPECT_EQ(unit_liveness.end_points[0], 8u);
  EXPECT_EQ(unit_liveness.end_points[1], 8u);
  EXPECT_EQ(
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, /*value_ordinal=*/0)
          .count,
      0u);
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/1,
      /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/4));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/1,
          /*unit_offset=*/0, /*unit_count=*/2, /*program_point=*/8));

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

TEST_F(LowAllocationUnitLivenessTest, QueriesComponentStoragePerUnit) {
  const loom_value_id_t value_ids[] = {0};
  const uint32_t interval_indices[] = {0};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(0, 0, 8, 2),
  };
  loom_liveness_block_info_t block = {};
  block.end_point = 8;
  loom_liveness_analysis_t liveness =
      Liveness(value_ids, IREE_ARRAYSIZE(value_ids), interval_indices,
               intervals, IREE_ARRAYSIZE(intervals), &block, 1);
  const loom_liveness_segment_t segments[] = {{0, 8}};
  const loom_liveness_segment_range_t segment_ranges[] = {{0, 1}};
  liveness.segments = segments;
  liveness.segment_count = IREE_ARRAYSIZE(segments);
  liveness.value_segment_ranges = segment_ranges;

  loom_low_allocation_unit_liveness_value_t values[] = {{0, 0}};
  uint32_t starts[] = {0, 0};
  uint32_t ends[] = {8, 4};
  uint64_t incomplete_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = values;
  unit_liveness.start_points = starts;
  unit_liveness.end_points = ends;
  unit_liveness.point_count = IREE_ARRAYSIZE(ends);
  unit_liveness.values_with_incomplete_storage_segments = {
      IREE_ARRAYSIZE(value_ids), incomplete_words};
  unit_liveness.storage_segments.entries = segments;
  loom_low_placement_table_t placement = {};
  placement.value_count = IREE_ARRAYSIZE(value_ids);

  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/0,
      /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/6));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/0,
          /*unit_offset=*/1, /*unit_count=*/1, /*program_point=*/6));
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/0,
      /*unit_offset=*/0, /*unit_count=*/2, /*program_point=*/6));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/0,
          /*unit_offset=*/0, /*unit_count=*/2, /*program_point=*/8));
}

TEST_F(LowAllocationUnitLivenessTest, PropagatesTiedStorageAcrossOrdinalOrder) {
  const loom_value_id_t value_ids[] = {0, 1, 2, 3, 4, 5};
  const uint32_t interval_indices[] = {0, 1, 2, 3, 4, 5};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(0, 0, 1, 1),   RegisterInterval(1, 20, 31, 1),
      RegisterInterval(2, 0, 1, 1),   RegisterInterval(3, 2, 11, 1),
      RegisterInterval(4, 30, 36, 1), RegisterInterval(5, 35, 40, 1),
  };
  loom_liveness_block_info_t block = {};
  block.end_point = 40;
  const loom_liveness_analysis_t liveness =
      Liveness(value_ids, IREE_ARRAYSIZE(value_ids), interval_indices,
               intervals, IREE_ARRAYSIZE(intervals), &block, 1);

  loom_low_allocation_unit_liveness_value_t values[] = {
      {0, 0}, {1, 20}, {2, 0}, {3, 2}, {4, 30}, {5, 35}};
  uint32_t starts[] = {0, 20, 0, 2, 30, 35};
  uint32_t ends[] = {1, 31, 1, 11, 36, 40};
  uint64_t incomplete_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = values;
  unit_liveness.start_points = starts;
  unit_liveness.end_points = ends;
  unit_liveness.point_count = IREE_ARRAYSIZE(ends);
  unit_liveness.values_with_incomplete_storage_segments = {
      IREE_ARRAYSIZE(value_ids), incomplete_words};

  loom_low_placement_relation_t relations[3] = {};
  relations[0].result_ordinal = 1;
  relations[0].source_ordinal = 3;
  relations[1].result_ordinal = 4;
  relations[1].source_ordinal = 1;
  relations[2].result_ordinal = 5;
  relations[2].source_ordinal = 4;
  for (auto& relation : relations) {
    relation.unit_count = 1;
    relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relation.cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                     LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 1}, {1, 0}, {1, 0}, {1, 1}, {2, 1}};
  const loom_value_ordinal_t storage_order[] = {0, 2, 5, 4, 1, 3};
  const loom_value_ordinal_t tied_origins[] = {0, 3, 2, 3, 3, 3};
  loom_low_placement_table_t placement = {};
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = result_ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;

  RetainAndPropagateStorage(&unit_liveness, &liveness, &placement);
  EXPECT_EQ(ends[3], 40u);
  EXPECT_EQ(ends[1], 31u);
  EXPECT_EQ(ends[4], 36u);
  EXPECT_EQ(starts[1], 2u);
  EXPECT_EQ(starts[4], 2u);
  EXPECT_EQ(starts[5], 2u);
  // Physical unit starts flow forward, but acquisition only moves earlier
  // when a descendant needs its source sooner. A forward chain stays staggered.
  EXPECT_EQ(values[3].acquisition_start_point, 2u);
  EXPECT_EQ(values[1].acquisition_start_point, 20u);
  EXPECT_EQ(values[4].acquisition_start_point, 30u);
  EXPECT_EQ(values[5].acquisition_start_point, 35u);
  EXPECT_TRUE(iree_bitmap_test(
      unit_liveness.values_with_incomplete_storage_segments, 3));
  EXPECT_TRUE(iree_bitmap_test(
      unit_liveness.values_with_incomplete_storage_segments, 1));
  EXPECT_TRUE(iree_bitmap_test(
      unit_liveness.values_with_incomplete_storage_segments, 4));
}

TEST_F(LowAllocationUnitLivenessTest,
       RetainsEarlierDescendantAcquisitionAndSparseHoles) {
  const loom_value_id_t value_ids[] = {0, 1, 2, 3};
  const uint32_t interval_indices[] = {0, 1, 2, 3};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(0, 10, 17, 1),
      RegisterInterval(1, 11, 12, 1),
      RegisterInterval(2, 2, 14, 1),
      RegisterInterval(3, 17, 19, 1),
  };
  // The producer branches to an earlier-listed loop using its final identity
  // or to a later sibling that reads the original value.
  loom_liveness_block_info_t blocks[3] = {};
  blocks[0].end_point = 8;
  blocks[1].start_point = 10;
  blocks[1].end_point = 14;
  blocks[2].start_point = 16;
  blocks[2].end_point = 20;
  loom_liveness_analysis_t liveness = Liveness(
      value_ids, IREE_ARRAYSIZE(value_ids), interval_indices, intervals,
      IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));
  const loom_liveness_segment_t segments[] = {{10, 14}, {16, 17}, {11, 12},
                                              {2, 8},   {12, 14}, {17, 19}};
  const loom_liveness_segment_range_t segment_ranges[] = {
      {0, 2}, {2, 1}, {3, 2}, {5, 1}};
  liveness.segments = segments;
  liveness.segment_count = IREE_ARRAYSIZE(segments);
  liveness.value_segment_ranges = segment_ranges;

  loom_low_allocation_unit_liveness_value_t values[] = {
      {0, 10}, {1, 11}, {2, 2}, {3, 17}};
  uint32_t starts[] = {10, 11, 2, 17};
  uint32_t ends[] = {17, 12, 14, 19};
  uint64_t incomplete_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = values;
  unit_liveness.start_points = starts;
  unit_liveness.end_points = ends;
  unit_liveness.point_count = IREE_ARRAYSIZE(ends);
  unit_liveness.values_with_incomplete_storage_segments = {
      IREE_ARRAYSIZE(value_ids), incomplete_words};
  unit_liveness.storage_segments.entries = segments;

  loom_low_placement_relation_t relations[3] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(relations); ++i) {
    relations[i].result_ordinal = i + 1;
    relations[i].source_ordinal = i == 1 ? 1 : 0;
    relations[i].unit_count = 1;
    relations[i].kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relations[i].cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    relations[i].flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 1}, {1, 1}, {2, 1}};
  const loom_value_ordinal_t storage_order[] = {2, 3, 1, 0};
  const loom_value_ordinal_t tied_origins[] = {0, 0, 0, 0};
  loom_low_placement_table_t placement = {};
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = result_ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;

  RetainAndPropagateStorage(&unit_liveness, &liveness, &placement);
  EXPECT_EQ(values[0].acquisition_start_point, 2u);
  EXPECT_EQ(values[1].acquisition_start_point, 2u);
  EXPECT_EQ(values[2].acquisition_start_point, 2u);
  EXPECT_EQ(values[3].acquisition_start_point, 17u);
  EXPECT_EQ(starts[0], 2u);
  EXPECT_EQ(ends[0], 19u);
  EXPECT_EQ(ends[1], 12u);
  EXPECT_EQ(intervals[0].start_point, 10u);
  EXPECT_EQ(intervals[1].start_point, 11u);
  const auto reservation =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 0);
  ASSERT_EQ(reservation.count, 3u);
  const auto* retained =
      unit_liveness.storage_segments.entries + reservation.start;
  EXPECT_EQ(retained[0].start_point, 2u);
  EXPECT_EQ(retained[0].end_point, 8u);
  EXPECT_EQ(retained[1].start_point, 10u);
  EXPECT_EQ(retained[1].end_point, 14u);
  EXPECT_EQ(retained[2].start_point, 16u);
  EXPECT_EQ(retained[2].end_point, 19u);
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, 0, 0, 1, 3));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, 0, 0, 1, 9));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, 0, 0, 1, 15));

  // An incomplete component without sparse reservations uses its acquisition
  // point as the conservative hull start, including before the SSA origin.
  loom_low_allocation_unit_liveness_t linear_liveness = unit_liveness;
  linear_liveness.storage_segments.tied_sources = nullptr;
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &linear_liveness, &liveness, &placement, 0, 0, 1, 3));
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &linear_liveness, &liveness, &placement, 0, 0, 1, 9));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &linear_liveness, &liveness, &placement, 0, 0, 1, 1));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &linear_liveness, &liveness, &placement, 0, 0, 1, 19));

  // Only the range lookup is decision-owned. Published segments retain the
  // semantic prefix and tied-source reservations after that lookup is freed.
  EXPECT_GT(decision_arena_.used_allocation_size, 0u);
  iree_arena_reset(&decision_arena_);
  iree_arena_block_pool_trim(&block_pool_);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(segments); ++i) {
    EXPECT_EQ(unit_liveness.storage_segments.entries[i].start_point,
              segments[i].start_point);
    EXPECT_EQ(unit_liveness.storage_segments.entries[i].end_point,
              segments[i].end_point);
  }
  EXPECT_EQ(retained[0].start_point, 2u);
  EXPECT_EQ(retained[0].end_point, 8u);
  EXPECT_EQ(retained[1].start_point, 10u);
  EXPECT_EQ(retained[1].end_point, 14u);
  EXPECT_EQ(retained[2].start_point, 16u);
  EXPECT_EQ(retained[2].end_point, 19u);
}

TEST_F(LowAllocationUnitLivenessTest,
       RetainsTransitiveSparseTiedStorageFamily) {
  const loom_value_id_t value_ids[] = {0, 1, 2};
  const uint32_t interval_indices[] = {0, 1, 2};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(0, 0, 2, 1),
      RegisterInterval(1, 2, 7, 1),
      RegisterInterval(2, 6, 12, 1),
  };
  loom_liveness_block_info_t blocks[3] = {};
  blocks[0].end_point = 4;
  blocks[1].start_point = 5;
  blocks[1].end_point = 8;
  blocks[2].start_point = 9;
  blocks[2].end_point = 12;
  loom_liveness_analysis_t liveness = Liveness(
      value_ids, IREE_ARRAYSIZE(value_ids), interval_indices, intervals,
      IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));
  const loom_liveness_segment_t segments[] = {{0, 2}, {2, 7}, {6, 8}, {10, 12}};
  const loom_liveness_segment_range_t segment_ranges[] = {
      {0, 1}, {1, 1}, {2, 2}};
  liveness.segments = segments;
  liveness.segment_count = IREE_ARRAYSIZE(segments);
  liveness.value_segment_ranges = segment_ranges;

  loom_low_allocation_unit_liveness_value_t values[] = {{0, 0}, {1, 2}, {2, 6}};
  uint32_t starts[] = {0, 2, 6};
  uint32_t ends[] = {2, 7, 12};
  uint64_t incomplete_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = values;
  unit_liveness.start_points = starts;
  unit_liveness.end_points = ends;
  unit_liveness.point_count = IREE_ARRAYSIZE(ends);
  unit_liveness.values_with_incomplete_storage_segments = {
      IREE_ARRAYSIZE(value_ids), incomplete_words};
  unit_liveness.storage_segments.entries = segments;

  loom_low_placement_relation_t relations[2] = {};
  relations[0].result_ordinal = 1;
  relations[0].source_ordinal = 0;
  relations[1].result_ordinal = 2;
  relations[1].source_ordinal = 1;
  for (auto& relation : relations) {
    relation.unit_count = 1;
    relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relation.cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                     LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 1}, {1, 1}};
  const loom_value_ordinal_t storage_order[] = {2, 1, 0};
  const loom_value_ordinal_t tied_origins[] = {0, 0, 0};
  loom_low_placement_table_t placement = {};
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = result_ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;

  RetainAndPropagateStorage(&unit_liveness, &liveness, &placement);
  const loom_liveness_segment_range_t source =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 0);
  const loom_liveness_segment_range_t middle =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 1);
  ASSERT_EQ(source.count, 2u);
  EXPECT_EQ(source.start, middle.start);
  EXPECT_EQ(source.count, middle.count);
  const loom_liveness_segment_t* reservations =
      &unit_liveness.storage_segments.entries[source.start];
  EXPECT_EQ(reservations[0].start_point, 0u);
  EXPECT_EQ(reservations[0].end_point, 8u);
  EXPECT_EQ(reservations[1].start_point, 10u);
  EXPECT_EQ(reservations[1].end_point, 12u);
  EXPECT_EQ(ends[0], 12u);
  EXPECT_EQ(ends[1], 7u);
  // Every chain member reaches the retained component lifetime even when the
  // value ordinals run opposite the storage flow.
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/2,
      /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/7));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/2,
          /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/9));
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/0,
      /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/10));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/0,
          /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/12));
}

TEST_F(LowAllocationUnitLivenessTest,
       PropagatesConcatStartsThroughTiedResults) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {
      DefineValue(module),
      DefineValue(module),
      DefineValue(module),
      DefineValue(module),
  };
  loom_local_value_domain_t value_domain = {};
  AcquireValueDomain(module, value_ids, IREE_ARRAYSIZE(value_ids),
                     &value_domain);

  loom_region_t* body = module->body;
  loom_block_t* block = loom_region_entry_block(body);
  loom_low_resolved_target_t target = {};
  const uint32_t value_interval_indices[] = {0, 1, 2, 3};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(value_ids[0], /*start_point=*/0,
                       /*end_point=*/2, /*unit_count=*/2),
      RegisterInterval(value_ids[1], /*start_point=*/2,
                       /*end_point=*/9, /*unit_count=*/2),
      RegisterInterval(value_ids[2], /*start_point=*/5,
                       /*end_point=*/9, /*unit_count=*/2),
      RegisterInterval(value_ids[3], /*start_point=*/9,
                       /*end_point=*/12, /*unit_count=*/4),
  };
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/block,
          /*.start_point=*/0,
          /*.end_point=*/12,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness = Liveness(
      value_domain.value_ids, value_domain.value_count, value_interval_indices,
      intervals, IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));

  loom_low_allocation_unit_liveness_t unit_liveness = {};
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(loom_cfg_graph_build(module, body, &arena_, &cfg_graph));
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      &target, nullptr, &value_domain, &liveness, &cfg_graph, &arena_,
      &decision_arena_, &unit_liveness));

  loom_low_placement_relation_t relations[] = {
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/1,
          /*.source_ordinal=*/0,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          {/*.location_mask=*/0},
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
              LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
          /*.priority=*/0,
      },
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/3,
          /*.source_ordinal=*/1,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          {/*.location_mask=*/0},
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
              LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
          /*.priority=*/0,
      },
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/3,
          /*.source_ordinal=*/2,
          /*.result_unit_offset=*/2,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          {/*.location_mask=*/0},
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
              LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
          /*.priority=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  const loom_low_placement_relation_range_t ranges[] = {
      {0, 0}, {0, 1}, {1, 0}, {1, 2}};
  const loom_value_ordinal_t storage_order[] = {3, 1, 2, 0};
  const loom_value_ordinal_t tied_origins[] = {0, 0, 2, 3};
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;

  RetainAndPropagateStorage(&unit_liveness, &liveness, &placement);
  const uint32_t* concat_start_points =
      loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
          &unit_liveness, &liveness, /*value_ordinal=*/3);
  ASSERT_NE(concat_start_points, nullptr);
  EXPECT_EQ(concat_start_points[0], 0u);
  EXPECT_EQ(concat_start_points[1], 0u);
  EXPECT_EQ(concat_start_points[2], 5u);
  EXPECT_EQ(concat_start_points[3], 5u);
  EXPECT_EQ(unit_liveness.values[1].acquisition_start_point, 2u);
  EXPECT_EQ(unit_liveness.values[3].acquisition_start_point, 9u);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

TEST_F(LowAllocationUnitLivenessTest, RetainsSparseTiedStorageReservations) {
  const loom_value_id_t value_ids[] = {0, 1, 2, 3, 4, 5};
  const uint32_t interval_indices[] = {0, 1, 2, 3, 4, 5};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(0, 0, 13, 1),  RegisterInterval(1, 3, 5, 1),
      RegisterInterval(2, 8, 10, 1),  RegisterInterval(3, 13, 13, 1),
      RegisterInterval(4, 15, 17, 1), RegisterInterval(5, 17, 19, 1),
  };
  loom_liveness_block_info_t blocks[4] = {};
  blocks[0].end_point = 5;
  blocks[1].start_point = 6;
  blocks[1].end_point = 10;
  blocks[2].start_point = 11;
  blocks[2].end_point = 14;
  blocks[3].start_point = 15;
  blocks[3].end_point = 19;
  const loom_liveness_segment_t segments[] = {
      {0, 3}, {6, 8}, {11, 13}, {3, 5}, {8, 10}, {15, 17}, {17, 19},
  };
  const loom_liveness_segment_range_t ranges[] = {
      {0, 3}, {3, 1}, {4, 1}, {5, 0}, {5, 1}, {6, 1},
  };
  loom_liveness_analysis_t liveness = Liveness(
      value_ids, IREE_ARRAYSIZE(value_ids), interval_indices, intervals,
      IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));
  liveness.segments = segments;
  liveness.segment_count = IREE_ARRAYSIZE(segments);
  liveness.value_segment_ranges = ranges;

  // The second result has a storage use beyond its semantic end, and the
  // source at ordinal four has incomplete edge storage. The unused third result
  // still writes its physical destination at point 13.
  loom_low_allocation_unit_liveness_value_t values[] = {
      {0, 0}, {1, 3}, {2, 8}, {3, 13}, {4, 15}, {5, 17}};
  uint32_t starts[] = {0, 3, 8, 13, 15, 17};
  uint32_t ends[] = {13, 5, 11, 14, 17, 19};
  uint64_t incomplete[] = {(1u << 2) | (1u << 4)};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = values;
  unit_liveness.start_points = starts;
  unit_liveness.end_points = ends;
  unit_liveness.point_count = IREE_ARRAYSIZE(starts);
  unit_liveness.values_with_incomplete_storage_segments = {6, incomplete};
  unit_liveness.storage_segments.entries = segments;
  loom_low_placement_relation_t relations[4] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(relations); ++i) {
    relations[i].source_ordinal = i < 3 ? 0 : 4;
    relations[i].result_ordinal = i < 3 ? i + 1 : 5;
    relations[i].unit_count = 1;
    relations[i].kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relations[i].cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    relations[i].flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  loom_low_placement_table_t placement = {};
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 0}, {3, 1}};
  const loom_value_ordinal_t storage_order[] = {1, 2, 3, 5, 0, 4};
  const loom_value_ordinal_t tied_origins[] = {0, 0, 0, 0, 4, 4};
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = result_ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;
  RetainAndPropagateStorage(&unit_liveness, &liveness, &placement);

  const loom_liveness_segment_range_t source =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 0);
  ASSERT_EQ(source.count, 2u);
  const loom_liveness_segment_t* reservations =
      &unit_liveness.storage_segments.entries[source.start];
  EXPECT_EQ(reservations[0].start_point, 0u);
  EXPECT_EQ(reservations[0].end_point, 5u);
  EXPECT_EQ(reservations[1].start_point, 6u);
  EXPECT_EQ(reservations[1].end_point, 14u);
  EXPECT_EQ(ends[0], 14u);
  EXPECT_EQ(ends[4], 19u);
  EXPECT_EQ(
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 2)
          .count,
      0u);
  const loom_liveness_segment_range_t edge_source =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 4);
  ASSERT_EQ(edge_source.count, 1u);
  EXPECT_EQ(
      unit_liveness.storage_segments.entries[edge_source.start].start_point,
      15u);
  EXPECT_EQ(unit_liveness.storage_segments.entries[edge_source.start].end_point,
            19u);
  const loom_liveness_segment_range_t result =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, 1);
  EXPECT_EQ(result.start, ranges[1].start);
  EXPECT_EQ(result.count, ranges[1].count);
  // A fanout result observes storage retained by its sibling branch, while the
  // gap between component segments and an independent component stay free.
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/1,
      /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/9));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/1,
          /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/5));
  EXPECT_TRUE(loom_low_allocation_unit_liveness_storage_component_live_at_point(
      &unit_liveness, &liveness, &placement, /*value_ordinal=*/5,
      /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/18));
  EXPECT_FALSE(
      loom_low_allocation_unit_liveness_storage_component_live_at_point(
          &unit_liveness, &liveness, &placement, /*value_ordinal=*/5,
          /*unit_offset=*/0, /*unit_count=*/1, /*program_point=*/14));
  EXPECT_EQ(liveness.segments, segments);
  EXPECT_EQ(liveness.value_segment_ranges, ranges);
  EXPECT_EQ(liveness.segment_count, IREE_ARRAYSIZE(segments));
}

}  // namespace
}  // namespace loom
