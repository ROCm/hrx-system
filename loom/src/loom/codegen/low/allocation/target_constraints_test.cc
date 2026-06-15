// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/target_constraints.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

loom_low_resolved_target_t TestTarget() {
  loom_low_resolved_target_t target = {};
  target.descriptor_set = loom_test_low_core_descriptor_set();
  target.descriptor_set_key = IREE_SV("test.low.core");
  return target;
}

typedef struct DiagnosticCapture {
  const loom_error_def_t* error;
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
    target_ = TestTarget();
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
  loom_low_allocation_assignment_t assignment = {};
  assignment.descriptor_reg_class_id = phys_reg_class_id;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = 4;
  assignment.location_count = 3;
  loom_low_allocation_target_constraints_record_assignment_location_end(
      &constraints, &assignment);

  EXPECT_EQ(
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &constraints, phys_reg_class_id,
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER),
      12u);
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

}  // namespace
}  // namespace loom
