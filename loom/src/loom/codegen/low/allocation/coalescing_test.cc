// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/coalescing.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationCoalescingTest : public ::testing::Test {
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

  // Backing blocks for module and allocation scratch.
  iree_arena_block_pool_t block_pool_;
  // Scratch storage for coalescing queries.
  iree_arena_allocator_t arena_;
  // Context shared by the fixture's modules.
  loom_context_t context_;
};

loom_liveness_value_class_t RegisterValueClass(uint64_t descriptor_set_id) {
  loom_liveness_value_class_t value_class = {
      .type_kind = LOOM_TYPE_REGISTER,
      .register_class_id = 0,
      .register_descriptor_set_stable_id = descriptor_set_id,
  };
  return value_class;
}

loom_liveness_interval_t Interval(loom_value_id_t value_id, uint32_t start,
                                  uint32_t end,
                                  loom_liveness_value_class_t value_class) {
  loom_liveness_interval_t interval = {
      .value_id = value_id,
      .start_point = start,
      .end_point = end,
      .value_class = value_class,
      .unit_count = 1,
  };
  return interval;
}

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint32_t start, uint32_t end,
    loom_liveness_value_class_t value_class, uint32_t location_base,
    uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {
      .value_id = value_id,
      .descriptor_reg_class_id = value_class.register_class_id,
      .start_point = start,
      .end_point = end,
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = location_base,
      .location_count = 1,
      .unit_point_start = unit_point_start,
  };
  return assignment;
}

TEST_F(LowAllocationCoalescingTest, ConcatMayPrecedeItsSourceAssignments) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {DefineValue(module), DefineValue(module),
                                       DefineValue(module)};
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
    loom_module_value_ordinal_scratch_set(module, value_ids[i], i);
  }

  const loom_liveness_value_class_t value_class = RegisterValueClass(17);
  loom_liveness_interval_t intervals[] = {
      Interval(value_ids[0], /*start=*/10, /*end=*/12, value_class),
      Interval(value_ids[1], /*start=*/11, /*end=*/12, value_class),
      Interval(value_ids[2], /*start=*/2, /*end=*/14, value_class),
  };
  intervals[2].unit_count = 2;
  const uint32_t interval_indices[] = {0, 1, 2};
  loom_liveness_analysis_t liveness = {
      .intervals = intervals,
      .interval_count = IREE_ARRAYSIZE(intervals),
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_interval_indices = interval_indices,
  };

  loom_low_placement_relation_t relations[2] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(relations); ++i) {
    relations[i].result_ordinal = 2;
    relations[i].source_ordinal = i;
    relations[i].result_unit_offset = i;
    relations[i].unit_count = 1;
    relations[i].kind = LOOM_LOW_PLACEMENT_RELATION_SUBRANGE;
    relations[i].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    relations[i].flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 0}, {0, 2}};
  const loom_low_placement_relation_range_t source_ranges[] = {
      {0, 1}, {1, 1}, {2, 0}};
  const uint32_t source_relations[] = {0, 1};
  loom_low_placement_table_t placement = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .relations = relations,
      .relation_count = IREE_ARRAYSIZE(relations),
      .ranges_by_result_ordinal = result_ranges,
      .relation_indices_by_source_ordinal = source_relations,
      .ranges_by_source_ordinal = source_ranges,
  };

  loom_low_allocation_assignment_t first_assignment =
      Assignment(value_ids[0], /*start=*/0, /*end=*/12, value_class,
                 /*location_base=*/0, /*unit_point_start=*/0);
  uint32_t assignment_indices[] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  loom_low_allocation_assignment_map_t assignment_map = {
      .module = module,
      .liveness = &liveness,
      .assignments = &first_assignment,
      .assignment_indices_by_value_ordinal = assignment_indices,
  };

  loom_low_reg_class_t reg_class = {
      .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
      .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
  };
  loom_low_descriptor_set_t descriptor_set = {
      .stable_id = 17,
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_low_resolved_target_t target = {
      .descriptor_set = &descriptor_set,
  };
  target.descriptor_set_key = IREE_SV("test");
  loom_low_allocation_target_constraints_t target_constraints = {
      .target = &target,
  };
  loom_low_allocation_search_context_t search_context = {
      .descriptor_set = &descriptor_set,
      .liveness = &liveness,
      .placement = &placement,
  };

  loom_low_allocation_coalescing_context_t context = {
      .arena = &arena_,
      .liveness = &liveness,
      .placement = &placement,
      .target_constraints = &target_constraints,
      .assignment_map = &assignment_map,
      .search_context = &search_context,
  };

  // Block layout can put a concat use before its defining block. Allocation
  // then reserves the result first, even when some operands were defined in
  // the entry block. The later sources can reuse that reservation.
  for (uint32_t assigned_count : {0u, 1u}) {
    SCOPED_TRACE(assigned_count);
    intervals[0].start_point = assigned_count ? 0 : 10;
    assignment_map.assignment_count = assigned_count;
    assignment_indices[0] = assigned_count ? 0 : UINT32_MAX;
    bool assigned = true;
    IREE_EXPECT_OK(loom_low_allocation_coalescing_assign_structural_interval(
        &context, &intervals[2], &assigned));
    EXPECT_FALSE(assigned);
    EXPECT_EQ(assignment_map.assignment_count, assigned_count);
    EXPECT_EQ(assignment_indices[2], UINT32_MAX);
  }

  for (loom_value_id_t value_id : value_ids) {
    loom_module_value_ordinal_scratch_clear(module, value_id);
  }
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

TEST_F(LowAllocationCoalescingTest,
       OptionalStructuralAliasMayPrecedeItsSourceAssignment) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {DefineValue(module),
                                       DefineValue(module)};
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
    loom_module_value_ordinal_scratch_set(module, value_ids[i], i);
  }

  const loom_liveness_value_class_t value_class = RegisterValueClass(17);
  loom_liveness_interval_t intervals[] = {
      Interval(value_ids[0], /*start=*/10, /*end=*/12, value_class),
      Interval(value_ids[1], /*start=*/2, /*end=*/14, value_class),
  };
  const uint32_t interval_indices[] = {0, 1};
  loom_liveness_analysis_t liveness = {
      .intervals = intervals,
      .interval_count = IREE_ARRAYSIZE(intervals),
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_interval_indices = interval_indices,
  };

  loom_low_placement_relation_t relation = {
      .result_ordinal = 1,
      .source_ordinal = 0,
      .unit_count = 1,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
               LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
  };
  const loom_low_placement_relation_range_t result_ranges[] = {{0, 0}, {0, 1}};
  const loom_low_placement_relation_range_t source_ranges[] = {{0, 1}, {1, 0}};
  const uint32_t source_relations[] = {0};
  loom_low_placement_table_t placement = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .relations = &relation,
      .relation_count = 1,
      .ranges_by_result_ordinal = result_ranges,
      .relation_indices_by_source_ordinal = source_relations,
      .ranges_by_source_ordinal = source_ranges,
  };

  uint32_t assignment_indices[] = {UINT32_MAX, UINT32_MAX};
  loom_low_allocation_assignment_map_t assignment_map = {
      .module = module,
      .liveness = &liveness,
      .assignment_indices_by_value_ordinal = assignment_indices,
  };

  loom_low_allocation_coalescing_context_t context = {
      .arena = &arena_,
      .liveness = &liveness,
      .placement = &placement,
      .assignment_map = &assignment_map,
  };

  for (loom_low_placement_cause_t cause :
       {LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY, LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE,
        LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE}) {
    SCOPED_TRACE(cause);
    const bool is_slice = cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE;
    intervals[0].unit_count = is_slice ? 2 : 1;
    relation.kind = is_slice ? LOOM_LOW_PLACEMENT_RELATION_SUBRANGE
                             : LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relation.cause = cause;
    relation.source_unit_offset = is_slice ? 1 : 0;
    bool assigned = true;
    IREE_EXPECT_OK(loom_low_allocation_coalescing_assign_structural_interval(
        &context, &intervals[1], &assigned));
    EXPECT_FALSE(assigned);
    EXPECT_EQ(assignment_map.assignment_count, 0u);
    EXPECT_EQ(assignment_indices[0], UINT32_MAX);
    EXPECT_EQ(assignment_indices[1], UINT32_MAX);
  }

  for (loom_value_id_t value_id : value_ids) {
    loom_module_value_ordinal_scratch_clear(module, value_id);
  }
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

TEST_F(LowAllocationCoalescingTest,
       AssignedConcatSourcesConstrainReservations) {
  // A source pair is already at units 3 and 4. The next source cannot use 5
  // until a temporary dies. Reserving the aggregate at 4 would put the new
  // source at 6, but materializing its first component would overwrite the
  // still-live second component. Repeat that component in the final slot.
  for (uint32_t width : {1u, 2u}) {
    for (uint32_t result_lifetime : {1u, 2u}) {
      SCOPED_TRACE(::testing::Message()
                   << "width=" << width << " lifetime=" << result_lifetime);
      loom_module_t* module = AllocateModule();
      loom_value_id_t value_ids[5];
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
        IREE_ASSERT_OK(loom_module_define_value(
            module, loom_low_register_type(17, 0, i == 3 ? 4 * width : width),
            &value_ids[i]));
      }
      loom_module_value_ordinal_scratch_acquire(module);
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
        loom_module_value_ordinal_scratch_set(module, value_ids[i], i);
      }
      const loom_liveness_value_class_t value_class = RegisterValueClass(17);
      loom_liveness_interval_t intervals[] = {
          Interval(value_ids[0], 1, 5, value_class),
          Interval(value_ids[1], 0, 8, value_class),
          Interval(value_ids[2], 3, 5, value_class),
          Interval(value_ids[3], 5, 5 + result_lifetime, value_class),
          Interval(value_ids[4], 0, 4, value_class),
      };
      const uint32_t interval_indices[] = {0, 1, 2, 3, 4};
      loom_liveness_analysis_t liveness = {
          .intervals = intervals,
          .interval_count = IREE_ARRAYSIZE(intervals),
          .value_ids = value_ids,
          .value_count = IREE_ARRAYSIZE(value_ids),
          .value_interval_indices = interval_indices,
      };
      loom_low_allocation_unit_liveness_value_t unit_values[5];
      uint32_t unit_start_points[16];
      uint32_t unit_end_points[16];
      uint32_t point_count = 0;
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
        intervals[i].unit_count = i == 3 ? 4 * width : width;
        unit_values[i] = {point_count, intervals[i].start_point};
        for (uint32_t unit = 0; unit < intervals[i].unit_count; ++unit) {
          unit_start_points[point_count] = intervals[i].start_point;
          unit_end_points[point_count++] = intervals[i].end_point;
        }
      }
      uint64_t incomplete_storage_words[] = {0};
      loom_low_allocation_unit_liveness_t unit_liveness = {
          .values = unit_values,
          .start_points = unit_start_points,
          .end_points = unit_end_points,
          .point_count = point_count,
          .values_with_incomplete_storage_segments = {5,
                                                      incomplete_storage_words},
      };
      loom_low_placement_relation_t relations[4] = {};
      const uint32_t source_ordinals[] = {0, 1, 2, 1};
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(relations); ++i) {
        relations[i].result_ordinal = 3;
        relations[i].source_ordinal = source_ordinals[i];
        relations[i].result_unit_offset = i * width;
        relations[i].unit_count = width;
        relations[i].kind = LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART;
        relations[i].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
        relations[i].flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
      }
      const loom_low_placement_relation_range_t result_ranges[] = {
          {0, 0}, {0, 0}, {0, 0}, {0, 4}, {4, 0}};
      const loom_low_placement_relation_range_t source_ranges[] = {
          {0, 1}, {1, 2}, {3, 1}, {4, 0}, {4, 0}};
      const uint32_t source_relations[] = {0, 1, 3, 2};
      loom_low_placement_operand_constraints_t operands[5] = {};
      loom_low_placement_table_t placement = {
          .value_ids = value_ids,
          .value_count = IREE_ARRAYSIZE(value_ids),
          .relations = relations,
          .relation_count = IREE_ARRAYSIZE(relations),
          .ranges_by_result_ordinal = result_ranges,
          .relation_indices_by_source_ordinal = source_relations,
          .ranges_by_source_ordinal = source_ranges,
          .operand_constraints_by_interval = operands,
      };
      loom_low_reg_class_t reg_class = {
          .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
          .alloc_unit_bits = 32,
          .allocatable_count = 16,
          .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
      };
      loom_low_descriptor_set_t descriptor_set = {
          .stable_id = 17,
          .reg_classes = &reg_class,
          .reg_class_count = 1,
      };
      loom_low_resolved_target_t target = {.descriptor_set = &descriptor_set};
      loom_low_allocation_resolved_reserved_range_t prefix = {
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_count = 3 * width,
      };
      uint32_t high_water[] = {6 * width};
      uint32_t constrained_high_water[] = {3 * width};
      loom_low_allocation_target_constraints_t constraints = {
          .module = module,
          .target = &target,
          .reserved_ranges = &prefix,
          .reserved_range_count = 1,
          .max_assigned_location_end_by_reg_class = high_water,
          .max_constrained_location_end_by_reg_class = constrained_high_water,
      };
      loom_low_allocation_assignment_t assigned[] = {
          Assignment(value_ids[0], 1, 5, value_class, 3 * width, 0),
          Assignment(value_ids[1], 0, 8, value_class, 4 * width, width),
          Assignment(value_ids[4], 0, 4, value_class, 5 * width, 7 * width),
      };
      for (auto& assignment : assigned) {
        assignment.unit_count = assignment.location_count = width;
      }
      const uint32_t assignment_indices[] = {0, 1, UINT32_MAX, UINT32_MAX, 2};
      loom_low_allocation_assignment_map_t assignment_map = {
          .module = module,
          .liveness = &liveness,
          .assignments = assigned,
          .assignment_count = IREE_ARRAYSIZE(assigned),
          .assignment_indices_by_value_ordinal = assignment_indices,
      };
      loom_low_allocation_active_set_t active_set = {};
      IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
          &descriptor_set, IREE_ARRAYSIZE(assigned), 9, 16, &arena_,
          &active_set));
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(assigned); ++i) {
        loom_low_allocation_active_set_insert(&active_set, &descriptor_set,
                                              assigned,
                                              IREE_ARRAYSIZE(assigned), i);
      }
      loom_low_allocation_storage_lease_state_t leases = {};
      loom_low_allocation_search_context_t search = {
          .module = module,
          .descriptor_set = &descriptor_set,
          .liveness = &liveness,
          .unit_liveness = &unit_liveness,
          .target_constraints = &constraints,
          .assignment_map = &assignment_map,
          .placement = &placement,
          .active_set = &active_set,
          .storage_leases = &leases,
      };
      loom_low_allocation_coalescing_context_t context = {
          .arena = &arena_,
          .liveness = &liveness,
          .placement = &placement,
          .target_constraints = &constraints,
          .assignment_map = &assignment_map,
          .search_context = &search,
          .append_assignment =
              [](void*, const loom_low_allocation_assignment_t*,
                 const loom_value_id_t*, uint16_t) -> iree_status_t {
            return iree_make_status(IREE_STATUS_ABORTED,
                                    "incompatible assembly must not reserve");
          },
      };
      bool did_assign = true;
      IREE_EXPECT_OK(
          loom_low_allocation_coalescing_assign_concat_source_interval(
              &context, &intervals[2], &did_assign));
      EXPECT_FALSE(did_assign);
      EXPECT_EQ(assignment_map.assignment_count, 3u);
      for (loom_value_id_t value_id : value_ids) {
        loom_module_value_ordinal_scratch_clear(module, value_id);
      }
      loom_module_value_ordinal_scratch_release(module);
      loom_module_free(module);
    }
  }
}

}  // namespace
}  // namespace loom
