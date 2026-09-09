// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/target_constraints.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/test/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/target_records.h"

namespace loom {
namespace {

typedef struct DiagnosticCapture {
  // Most recently emitted error definition, borrowed from the catalog.
  const loom_error_def_t* error;
  // Number of diagnostic emissions delivered to this capture.
  iree_host_size_t count;
} DiagnosticCapture;

static iree_status_t CaptureDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  DiagnosticCapture* capture = static_cast<DiagnosticCapture*>(user_data);
  capture->error = emission->error;
  ++capture->count;
  return iree_ok_status();
}

class LowAllocationTargetConstraintsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    const loom_target_bundle_t* target_bundle = loom_target_bundle_table_lookup(
        &loom_test_target_bundles, LOOM_TEST_TARGET_KIND_LOW_CORE);
    IREE_ASSERT(target_bundle != nullptr);
    loom_target_facts_builder_initialize(&loom_test_target_fact_type,
                                         target_bundle, &target_facts_);
    target_ = (loom_low_resolved_target_t){
        /*.target_facts=*/&target_facts_,
        /*.target_name=*/target_bundle->name,
        /*.descriptor_set_key=*/target_bundle->config->contract_set_key,
        /*.feature_bits=*/target_bundle->config->contract_feature_bits,
        /*.descriptor_set=*/loom_test_low_core_descriptor_set(),
    };
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint16_t RegisterClassId(iree_string_view_t name) const {
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    EXPECT_TRUE(loom_low_descriptor_set_lookup_register_class(
        target_.descriptor_set, name, &reg_class_id, nullptr));
    return reg_class_id;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_module_t module_ = {};
  loom_op_t function_op_ = {};
  // Complete synthetic target facts borrowed by |target_|.
  loom_target_facts_t target_facts_ = {};
  loom_low_resolved_target_t target_ = {};
};

TEST_F(LowAllocationTargetConstraintsTest, ClampsBudgetToDescriptorCapacity) {
  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.phys");
  budget.max_units = 64;

  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, &budget, /*budget_count=*/1,
      /*reserved_ranges=*/nullptr, /*reserved_range_count=*/0,
      /*emitter=*/iree_diagnostic_emitter_t{}, &arena_, &constraints));

  loom_low_allocation_class_capacity_t capacity = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_reg_class_capacity(
      &constraints, RegisterClassId(IREE_SV("test.phys")), &capacity));
  EXPECT_TRUE(capacity.is_bounded);
  EXPECT_TRUE(capacity.is_spillable);
  EXPECT_EQ(capacity.max_units, 32u);
  EXPECT_EQ(capacity.location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
}

TEST_F(LowAllocationTargetConstraintsTest, AppliesBudgetToUnboundedClass) {
  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.i32");
  budget.max_units = 7;

  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, &budget, /*budget_count=*/1,
      /*reserved_ranges=*/nullptr, /*reserved_range_count=*/0,
      /*emitter=*/iree_diagnostic_emitter_t{}, &arena_, &constraints));

  loom_low_allocation_class_capacity_t capacity = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_reg_class_capacity(
      &constraints, RegisterClassId(IREE_SV("test.i32")), &capacity));
  EXPECT_TRUE(capacity.is_bounded);
  EXPECT_EQ(capacity.max_units, 7u);
}

TEST_F(LowAllocationTargetConstraintsTest,
       MoveFailureRetainsDescriptorClassIdentity) {
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, nullptr, 0, nullptr, 0, {}, &arena_,
      &constraints));
  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_low_allocation_target_constraints_record_move_failure(
      &constraints, &function_op_, reg_class_id, 2, 3,
      IREE_SV("parallel-move-no-scratch-unit"));

  EXPECT_EQ(constraints.error_count, 1u);
  EXPECT_EQ(constraints.failure.op, &function_op_);
  EXPECT_EQ(constraints.failure.value_id, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(constraints.failure.descriptor_reg_class_id, reg_class_id);
  EXPECT_EQ(constraints.failure.value_class.type_kind, LOOM_TYPE_REGISTER);
  EXPECT_EQ(constraints.failure.value_class.register_class_id, reg_class_id);
  EXPECT_EQ(constraints.failure.value_class.register_descriptor_set_stable_id,
            target_.descriptor_set->stable_id);
  EXPECT_EQ(constraints.failure.budget_units, 2u);
  EXPECT_EQ(constraints.failure.required_unit_count, 3u);
}

TEST_F(LowAllocationTargetConstraintsTest, ReferenceClassCannotSpill) {
  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.i32"));
  loom_low_descriptor_set_t descriptor_set = *target_.descriptor_set;
  std::vector<loom_low_reg_class_t> reg_classes(
      descriptor_set.reg_classes,
      descriptor_set.reg_classes + descriptor_set.reg_class_count);
  reg_classes[reg_class_id].flags |= LOOM_LOW_REG_CLASS_FLAG_REFERENCE;
  descriptor_set.reg_classes = reg_classes.data();
  target_.descriptor_set = &descriptor_set;

  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, /*budgets=*/nullptr,
      /*budget_count=*/0, /*reserved_ranges=*/nullptr,
      /*reserved_range_count=*/0, /*emitter=*/iree_diagnostic_emitter_t{},
      &arena_, &constraints));

  loom_low_allocation_class_capacity_t capacity = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_reg_class_capacity(
      &constraints, reg_class_id, &capacity));
  EXPECT_FALSE(capacity.is_spillable);
}

TEST_F(LowAllocationTargetConstraintsTest,
       ValidatesAllocatableAndFixedLocationWindowsSeparately) {
  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.phys");
  budget.max_units = 16;

  DiagnosticCapture capture = {};
  const iree_diagnostic_emitter_t emitter = {
      /*.fn=*/CaptureDiagnostic,
      /*.user_data=*/&capture,
  };
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, &budget, /*budget_count=*/1,
      /*reserved_ranges=*/nullptr, /*reserved_range_count=*/0, emitter, &arena_,
      &constraints));

  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  bool valid = false;
  IREE_ASSERT_OK(
      loom_low_allocation_target_constraints_validate_register_location_capacity(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*location_base=*/15, /*location_count=*/1, IREE_SV("test"),
          &function_op_, &valid));
  EXPECT_TRUE(valid);

  IREE_ASSERT_OK(
      loom_low_allocation_target_constraints_validate_register_location_capacity(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*location_base=*/20, /*location_count=*/1, IREE_SV("test"),
          &function_op_, &valid));
  EXPECT_FALSE(valid);

  IREE_ASSERT_OK(
      loom_low_allocation_target_constraints_validate_register_location_capacity(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*location_base=*/32, /*location_count=*/8, IREE_SV("test"),
          &function_op_, &valid));
  EXPECT_TRUE(valid);

  IREE_ASSERT_OK(
      loom_low_allocation_target_constraints_validate_register_location_capacity(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*location_base=*/31, /*location_count=*/2, IREE_SV("test"),
          &function_op_, &valid));
  EXPECT_FALSE(valid);

  IREE_ASSERT_OK(
      loom_low_allocation_target_constraints_validate_register_location_capacity(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*location_base=*/40, /*location_count=*/1, IREE_SV("test"),
          &function_op_, &valid));
  EXPECT_FALSE(valid);
  EXPECT_EQ(capture.count, 3u);
  EXPECT_EQ(constraints.error_count, 3u);
}

TEST_F(LowAllocationTargetConstraintsTest,
       ReportsOverlappingReservedRangesAsDiagnostic) {
  loom_low_allocation_reserved_range_t reserved_ranges[2] = {};
  reserved_ranges[0].register_class = IREE_SV("test.phys");
  reserved_ranges[0].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  reserved_ranges[0].location_base = 4;
  reserved_ranges[0].location_count = 4;
  reserved_ranges[1].register_class = IREE_SV("test.phys");
  reserved_ranges[1].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  reserved_ranges[1].location_base = 7;
  reserved_ranges[1].location_count = 2;

  DiagnosticCapture capture = {};
  const iree_diagnostic_emitter_t emitter = {
      /*.fn=*/CaptureDiagnostic,
      /*.user_data=*/&capture,
  };
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, /*budgets=*/nullptr,
      /*budget_count=*/0, reserved_ranges, IREE_ARRAYSIZE(reserved_ranges),
      emitter, &arena_, &constraints));
  EXPECT_EQ(constraints.error_count, 1u);
  EXPECT_EQ(constraints.reserved_range_count, 1u);
  EXPECT_EQ(capture.count, 1u);
  EXPECT_EQ(capture.error, LOOM_ERR_BACKEND_031);
}

TEST_F(LowAllocationTargetConstraintsTest,
       SearchLimitIncludesAssignmentsAndReservedRanges) {
  loom_low_allocation_reserved_range_t reserved_range = {};
  reserved_range.register_class = IREE_SV("test.phys");
  reserved_range.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  reserved_range.location_base = 10;
  reserved_range.location_count = 2;

  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, /*budgets=*/nullptr,
      /*budget_count=*/0, &reserved_range, /*reserved_range_count=*/1,
      /*emitter=*/iree_diagnostic_emitter_t{}, &arena_, &constraints));

  const uint16_t phys_reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_low_allocation_target_constraints_record_location_extent(
      &constraints, phys_reg_class_id,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/3);

  EXPECT_EQ(
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &constraints, phys_reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER),
      12u);
}

TEST_F(LowAllocationTargetConstraintsTest,
       PhysicalExtentsExcludeAbiFixedLocationWindow) {
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, /*budgets=*/nullptr,
      /*budget_count=*/0, /*reserved_ranges=*/nullptr,
      /*reserved_range_count=*/0, /*emitter=*/iree_diagnostic_emitter_t{},
      &arena_, &constraints));

  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_low_allocation_target_constraints_record_location_extent(
      &constraints, reg_class_id,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/5,
      /*location_count=*/1);
  loom_low_allocation_target_constraints_record_location_extent(
      &constraints, reg_class_id,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/32,
      /*location_count=*/8);

  EXPECT_EQ(
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER),
      6u);
}

TEST_F(LowAllocationTargetConstraintsTest,
       PhysicalExtentsIncludeMoveScratchLocations) {
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, /*budgets=*/nullptr,
      /*budget_count=*/0, /*reserved_ranges=*/nullptr,
      /*reserved_range_count=*/0, /*emitter=*/iree_diagnostic_emitter_t{},
      &arena_, &constraints));

  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_low_allocation_target_constraints_record_location_extent(
      &constraints, reg_class_id,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/0,
      /*location_count=*/4);
  loom_low_allocation_target_constraints_record_location_extent(
      &constraints, reg_class_id,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/1);

  EXPECT_EQ(
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &constraints, reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER),
      5u);
}

TEST_F(LowAllocationTargetConstraintsTest,
       ReservedRangesConflictAcrossAliasedClasses) {
  loom_low_allocation_reserved_range_t reserved_range = {};
  reserved_range.register_class = IREE_SV("test.alias32");
  reserved_range.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  reserved_range.location_base = 0;
  reserved_range.location_count = 1;

  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      &module_, &function_op_, &target_, /*budgets=*/nullptr,
      /*budget_count=*/0, &reserved_range, /*reserved_range_count=*/1,
      /*emitter=*/iree_diagnostic_emitter_t{}, &arena_, &constraints));

  EXPECT_TRUE(loom_low_allocation_target_constraints_reserved_range_conflicts(
      &constraints, RegisterClassId(IREE_SV("test.alias64")),
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/0,
      /*location_count=*/1));
}

TEST_F(LowAllocationTargetConstraintsTest,
       IndexesFixedValuesAndLifetimeOverlap) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("fixed"), &block_pool_,
                                      nullptr, iree_allocator_system(),
                                      &module));
  const uint32_t ranges[][3] = {
      {12, 18, 0}, {2, 60, 1},  {8, 10, 0}, {2, 3, 0},
      {36, 40, 0}, {22, 25, 2}, {8, 12, 0}, {50, 52, 0},
  };
  constexpr uint32_t kFixedCount = IREE_ARRAYSIZE(ranges);
  constexpr uint32_t kValueCount = kFixedCount + 1;
  loom_value_id_t values[kValueCount];
  loom_liveness_interval_t intervals[kValueCount] = {};
  uint32_t interval_indices[kValueCount];
  uint32_t unit_starts[kValueCount];
  uint32_t unit_ends[kValueCount];
  loom_low_allocation_fixed_value_t fixed_values[kFixedCount] = {};
  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_liveness_value_class_t value_class = {};
  value_class.type_kind = LOOM_TYPE_REGISTER;
  value_class.register_descriptor_set_stable_id =
      target_.descriptor_set->stable_id;
  value_class.register_class_id = reg_class_id;
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < kValueCount; ++i) {
    IREE_ASSERT_OK(loom_module_define_value(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &values[i]));
    loom_module_value_ordinal_scratch_set(module, values[i], i);
    intervals[i].value_id = values[i];
    intervals[i].value_class = value_class;
    intervals[i].unit_count = 1;
    intervals[i].start_point = i < kFixedCount ? ranges[i][0] : 0;
    intervals[i].end_point = i < kFixedCount ? ranges[i][1] : 64;
    interval_indices[i] = i;
    unit_starts[i] = i;
    unit_ends[i] = intervals[i].end_point;
    if (i < kFixedCount) {
      fixed_values[i].value_id = values[i];
      fixed_values[i].location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
      fixed_values[i].location_base = ranges[i][2];
      fixed_values[i].location_count = 1;
    }
  }
  loom_local_value_domain_t domain = {};
  domain.module = module;
  domain.value_ids = values;
  domain.value_count = kValueCount;
  domain.flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED;
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = kValueCount;
  liveness.value_ids = values;
  liveness.value_count = kValueCount;
  liveness.value_interval_indices = interval_indices;
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.point_starts_by_value_ordinal = unit_starts;
  unit_liveness.end_points = unit_ends;
  unit_liveness.point_count = kValueCount;
  uint64_t incomplete_storage_words[] = {0};
  unit_liveness.values_with_incomplete_storage_segments = {
      kValueCount, incomplete_storage_words};
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op_, &target_, nullptr, 0, nullptr, 0, {}, &arena_,
      &constraints));
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_values(
      &constraints, &liveness, &domain, &unit_liveness, fixed_values,
      kFixedCount, &arena_));
  ASSERT_EQ(constraints.error_count, 0u);
  for (uint32_t i = 0; i < kFixedCount; ++i) {
    EXPECT_EQ(loom_low_allocation_target_constraints_fixed_value_for_value(
                  &constraints, values[i]),
              &constraints.fixed_values[i]);
  }
  EXPECT_EQ(loom_low_allocation_target_constraints_fixed_value_for_value(
                &constraints, values[kFixedCount]),
            nullptr);
  EXPECT_EQ(loom_low_allocation_target_constraints_fixed_value_for_value(
                &constraints, LOOM_VALUE_ID_INVALID),
            nullptr);

  loom_low_allocation_assignment_t candidate = {};
  candidate.value_id = values[kFixedCount];
  candidate.value_class = value_class;
  candidate.descriptor_reg_class_id = reg_class_id;
  candidate.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  candidate.location_count = 1;
  candidate.unit_count = 1;
  candidate.unit_point_start = kFixedCount;
  loom_low_placement_relation_range_t placement_ranges[kValueCount] = {};
  loom_low_placement_table_t placement = {};
  placement.value_ids = values;
  placement.value_count = kValueCount;
  placement.ranges_by_result_ordinal = placement_ranges;
  placement.ranges_by_source_ordinal = placement_ranges;
  // Disordered starts, equal starts, nested ranges, touching endpoints, and
  // ignored values all use the same half-open lifetime contract.
  for (uint32_t start = 0; start <= 64; ++start) {
    for (uint32_t length : {1u, 5u, 32u}) {
      candidate.start_point = start;
      candidate.end_point = start + length;
      unit_ends[kFixedCount] = candidate.end_point;
      for (uint32_t location = 0; location < 3; ++location) {
        candidate.location_base = location;
        for (uint16_t ignored_count : {0, 1}) {
          bool expected = false;
          for (uint32_t i = ignored_count; i < kFixedCount; ++i) {
            expected |= ranges[i][2] == location &&
                        ranges[i][0] < start + length && start < ranges[i][1];
          }
          EXPECT_EQ(
              loom_low_allocation_target_constraints_fixed_value_conflicts(
                  &constraints, &liveness, &unit_liveness, &placement,
                  &candidate, values, ignored_count),
              expected)
              << "start=" << start << " length=" << length
              << " location=" << location << " ignored=" << ignored_count;
        }
      }
    }
  }
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
  loom_context_deinitialize(&context);
}

}  // namespace
}  // namespace loom
