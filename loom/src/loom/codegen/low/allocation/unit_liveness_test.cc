// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationUnitLivenessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
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
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      module, &target, nullptr, &value_domain, &liveness, &arena_,
      &unit_liveness));

  EXPECT_EQ(loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
                &unit_liveness, &liveness, /*value_ordinal=*/0),
            0u);
  EXPECT_EQ(loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
                &unit_liveness, &liveness, /*value_ordinal=*/1),
            2u);
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
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      module, &target, nullptr, &value_domain, &liveness, &arena_,
      &unit_liveness));
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

  const loom_low_placement_relation_t relations[] = {
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/1,
          /*.source_ordinal=*/0,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          /*.location_mask=*/0,
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
          /*.priority=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);

  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_propagate_storage_relations(
      &unit_liveness, &liveness, &placement));
  EXPECT_EQ(unit_liveness.start_points[2], 0u);
  EXPECT_EQ(unit_liveness.start_points[3], 0u);
  EXPECT_EQ(unit_liveness.end_points[0], 8u);
  EXPECT_EQ(unit_liveness.end_points[1], 8u);
  EXPECT_EQ(
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          &unit_liveness, &liveness, /*value_ordinal=*/0)
          .count,
      0u);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
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
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      module, &target, nullptr, &value_domain, &liveness, &arena_,
      &unit_liveness));

  const loom_low_placement_relation_t relations[] = {
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/1,
          /*.source_ordinal=*/0,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          /*.location_mask=*/0,
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
          /*.priority=*/0,
      },
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/3,
          /*.source_ordinal=*/1,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          /*.location_mask=*/0,
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
          /*.priority=*/0,
      },
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/3,
          /*.source_ordinal=*/2,
          /*.result_unit_offset=*/2,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          /*.location_mask=*/0,
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
          /*.priority=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);

  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_propagate_storage_relations(
      &unit_liveness, &liveness, &placement));
  const uint32_t* concat_start_points =
      loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
          &unit_liveness, &liveness, /*value_ordinal=*/3);
  ASSERT_NE(concat_start_points, nullptr);
  EXPECT_EQ(concat_start_points[0], 0u);
  EXPECT_EQ(concat_start_points[1], 0u);
  EXPECT_EQ(concat_start_points[2], 5u);
  EXPECT_EQ(concat_start_points[3], 5u);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
