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
#include "loom/target/registers.h"
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
       ArchitecturalReservationSurvivesNarrowerAllocationBudget) {
  // Linear and explicit physical classes both reserve architectural state
  // outside the candidate window available to ordinary values.
  const iree_string_view_t class_names[] = {IREE_SV("test.phys"),
                                            IREE_SV("test.explicit32")};
  for (iree_string_view_t class_name : class_names) {
    const uint16_t class_id = RegisterClassId(class_name);
    const auto* descriptor_set = target_.descriptor_set;
    const auto* reg_class = &descriptor_set->reg_classes[class_id];
    const uint32_t reserved_location =
        loom_low_reg_class_uses_explicit_physical_registers(reg_class)
            ? loom_low_descriptor_set_physical_register_candidate(
                  descriptor_set, class_id, /*ordinal=*/2)
            : 6;
    const loom_low_allocation_budget_t budget = {class_name, 2};
    const loom_low_allocation_reserved_range_t reservation = {
        class_name, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        reserved_location, 1};
    DiagnosticCapture capture = {};
    const iree_diagnostic_emitter_t emitter = {CaptureDiagnostic, &capture};
    loom_low_allocation_target_constraints_t constraints = {};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
        &module_, &function_op_, &target_, &budget, /*budget_count=*/1,
        &reservation, /*reserved_range_count=*/1, emitter, &arena_,
        &constraints));
    EXPECT_EQ(capture.count, 0u);
    ASSERT_EQ(constraints.error_count, 0u);
    ASSERT_EQ(constraints.reserved_range_count, 1u);
    EXPECT_EQ(constraints.reserved_ranges[0].location_base, reserved_location);

    loom_low_allocation_class_capacity_t capacity = {};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_reg_class_capacity(
        &constraints, class_id, &capacity));
    EXPECT_EQ(capacity.max_units, 2u);
    EXPECT_FALSE(
        loom_low_allocation_target_constraints_location_range_fits_capacity(
            descriptor_set, &capacity,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, reserved_location,
            /*location_count=*/1));
  }
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
  EXPECT_EQ(
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &constraints, RegisterClassId(IREE_SV("test.alias64")),
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER),
      1u);
}

TEST_F(LowAllocationTargetConstraintsTest,
       PropagatesFixedBindingsAcrossRetainedTiedOrigins) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("fixed"), &block_pool_,
                                      nullptr, iree_allocator_system(),
                                      &module));
  // Placement retains one origin for the required chain 1 -> 2 -> 0,
  // independently of local value registration order. Only its final value has
  // an explicit fixed binding.
  constexpr uint32_t kValueCount = 3;
  loom_value_id_t values[kValueCount];
  loom_liveness_interval_t intervals[kValueCount] = {};
  uint32_t interval_indices[] = {0, 1, 2};
  loom_low_allocation_unit_liveness_value_t unit_values[] = {
      {0, 2}, {1, 0}, {2, 1}};
  uint32_t unit_start_points[] = {2, 0, 1, 0};
  uint32_t unit_end_points[] = {3, 2, 3, 3};
  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_liveness_value_class_t value_class = {};
  value_class.type_kind = LOOM_TYPE_REGISTER;
  value_class.register_descriptor_set_stable_id =
      target_.descriptor_set->stable_id;
  value_class.register_class_id = reg_class_id;
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < kValueCount; ++i) {
    IREE_ASSERT_OK(loom_module_define_value(
        module,
        loom_low_register_type(target_.descriptor_set->stable_id, reg_class_id,
                               1),
        &values[i]));
    loom_module_value_ordinal_scratch_set(module, values[i], i);
    intervals[i].value_id = values[i];
    intervals[i].value_class = value_class;
    intervals[i].unit_count = 1;
    intervals[i].start_point = unit_start_points[i];
    intervals[i].end_point = unit_end_points[i];
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
  unit_liveness.values = unit_values;
  unit_liveness.start_points = unit_start_points;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
  uint64_t incomplete_storage_words[] = {0};
  unit_liveness.values_with_incomplete_storage_segments = {
      kValueCount, incomplete_storage_words};
  loom_low_placement_relation_t relations[2] = {};
  relations[0].result_ordinal = 0;
  relations[0].source_ordinal = 2;
  relations[1].result_ordinal = 2;
  relations[1].source_ordinal = 1;
  for (auto& relation : relations) {
    relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relation.cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
                     LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
    relation.unit_count = 1;
  }
  loom_low_placement_table_t placement = {};
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  const loom_value_ordinal_t tied_storage_origins[] = {1, 1, 1};
  placement.tied_storage_origins_by_value_ordinal = tied_storage_origins;
  const loom_low_allocation_fixed_value_t fixed = {
      values[0], LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 6, 1};
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op_, &target_, nullptr, 0, nullptr, 0, {}, &arena_,
      &constraints));
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_values(
      &constraints, &liveness, &domain, &unit_liveness, &placement, &fixed, 1,
      &arena_));
  ASSERT_EQ(constraints.error_count, 0u);
  EXPECT_EQ(constraints.preassigned_fixed_value_count, kValueCount);
  EXPECT_NE(constraints.fixed_index.subtree_tied_roots, nullptr);
  for (auto value : values) {
    const auto* binding =
        loom_low_allocation_target_constraints_preassigned_fixed_value_for_value(
            &constraints, value);
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->tied_root_ordinal, 1u);
    EXPECT_EQ(binding->assignment.location_base, 6u);
    EXPECT_FALSE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
        &constraints, &unit_liveness, &binding->assignment,
        /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
    // A temporary may overlap the component only after its owner is excluded.
    // Excluding any required alias releases the same complete reservation.
    loom_low_allocation_assignment_t temporary = binding->assignment;
    temporary.value_id = LOOM_VALUE_ID_INVALID;
    temporary.start_point = 0;
    temporary.end_point = 3;
    temporary.unit_point_start = 3;
    EXPECT_TRUE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
        &constraints, &unit_liveness, &temporary, nullptr, 0));
    EXPECT_FALSE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
        &constraints, &unit_liveness, &temporary, &value, 1));
  }
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
  loom_context_deinitialize(&context);
}

TEST_F(LowAllocationTargetConstraintsTest,
       FixedValuesRespectRetainedOperandWindow) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("fixed"), &block_pool_,
                                      nullptr, iree_allocator_system(),
                                      &module));
  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.phys"));
  loom_value_id_t value;
  IREE_ASSERT_OK(loom_module_define_value(
      module,
      loom_low_register_type(target_.descriptor_set->stable_id, reg_class_id,
                             1),
      &value));
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, value, 0);
  loom_local_value_domain_t domain = {};
  domain.module = module;
  domain.value_ids = &value;
  domain.value_count = 1;
  domain.flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED;
  loom_liveness_interval_t interval = {};
  interval.value_id = value;
  interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  interval.value_class.register_descriptor_set_stable_id =
      target_.descriptor_set->stable_id;
  interval.value_class.register_class_id = reg_class_id;
  interval.unit_count = 1;
  interval.end_point = 1;
  uint32_t zero = 0, one = 1;
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = &interval;
  liveness.interval_count = 1;
  liveness.value_ids = &value;
  liveness.value_count = 1;
  liveness.value_interval_indices = &zero;
  loom_low_allocation_unit_liveness_value_t unit_value = {};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = &unit_value;
  unit_liveness.start_points = &zero;
  unit_liveness.end_points = &one;
  unit_liveness.point_count = 1;
  uint64_t incomplete_storage_words[] = {0};
  unit_liveness.values_with_incomplete_storage_segments = {
      1, incomplete_storage_words};
  loom_low_placement_operand_constraints_t operand = {};
  operand.addressable_unit_count = 8;
  loom_low_placement_table_t placement = {};
  placement.operand_constraints_by_interval = &operand;
  // The class's ABI-fixed window at 32 is legal storage, but still cannot be
  // encoded by an operand restricted to the first eight registers.
  for (uint32_t location : {7, 8, 32}) {
    DiagnosticCapture capture = {};
    const iree_diagnostic_emitter_t emitter = {CaptureDiagnostic, &capture};
    loom_low_allocation_target_constraints_t constraints = {};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
        module, &function_op_, &target_, nullptr, 0, nullptr, 0, emitter,
        &arena_, &constraints));
    const loom_low_allocation_fixed_value_t fixed = {
        value, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, location, 1};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_values(
        &constraints, &liveness, &domain, &unit_liveness, &placement, &fixed, 1,
        &arena_));
    EXPECT_EQ(constraints.error_count, location < 8 ? 0u : 1u);
    EXPECT_EQ(constraints.fixed_value_count, location < 8 ? 1u : 0u);
    if (location >= 8) {
      EXPECT_EQ(capture.error, LOOM_ERR_BACKEND_022);
    }
  }
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
  loom_context_deinitialize(&context);
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
  loom_liveness_segment_t segments[kValueCount + 1] = {};
  loom_liveness_segment_range_t segment_ranges[kValueCount] = {};
  uint32_t interval_indices[kValueCount];
  loom_low_allocation_unit_liveness_value_t unit_values[kValueCount];
  uint32_t unit_starts[kValueCount];
  uint32_t unit_ends[kValueCount];
  loom_low_allocation_fixed_value_t fixed_values[kFixedCount] = {};
  const uint16_t reg_class_id = RegisterClassId(IREE_SV("test.i32"));
  loom_liveness_value_class_t value_class = {};
  value_class.type_kind = LOOM_TYPE_REGISTER;
  value_class.register_descriptor_set_stable_id =
      target_.descriptor_set->stable_id;
  value_class.register_class_id = reg_class_id;
  loom_module_value_ordinal_scratch_acquire(module);
  uint32_t segment_count = 0;
  for (uint32_t i = 0; i < kValueCount; ++i) {
    IREE_ASSERT_OK(loom_module_define_value(
        module,
        loom_low_register_type(target_.descriptor_set->stable_id, reg_class_id,
                               /*unit_count=*/1),
        &values[i]));
    loom_module_value_ordinal_scratch_set(module, values[i], i);
    intervals[i].value_id = values[i];
    intervals[i].value_class = value_class;
    intervals[i].unit_count = 1;
    intervals[i].start_point = i < kFixedCount ? ranges[i][0] : 0;
    intervals[i].end_point = i < kFixedCount ? ranges[i][1] : 64;
    segment_ranges[i].start = segment_count;
    if (i == 5) {
      segments[segment_count++] = {22, 23};
      segments[segment_count++] = {24, 25};
      segment_ranges[i].count = 2;
    } else {
      segments[segment_count++] = {intervals[i].start_point,
                                   intervals[i].end_point};
      segment_ranges[i].count = 1;
    }
    interval_indices[i] = i;
    unit_values[i] = {i, intervals[i].start_point};
    unit_starts[i] = intervals[i].start_point;
    unit_ends[i] = intervals[i].end_point;
    if (i < kFixedCount) {
      fixed_values[i].value_id = values[i];
      fixed_values[i].location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
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
  liveness.segments = segments;
  liveness.segment_count = segment_count;
  liveness.value_segment_ranges = segment_ranges;
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = unit_values;
  unit_liveness.start_points = unit_starts;
  unit_liveness.end_points = unit_ends;
  unit_liveness.point_count = kValueCount;
  uint64_t incomplete_storage_words[] = {0};
  unit_liveness.values_with_incomplete_storage_segments = {
      kValueCount, incomplete_storage_words};
  unit_liveness.storage_segments.entries = segments;
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op_, &target_, nullptr, 0, nullptr, 0, {}, &arena_,
      &constraints));
  const loom_low_placement_table_t placement = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_values(
      &constraints, &liveness, &domain, &unit_liveness, &placement,
      fixed_values, kFixedCount, &arena_));
  ASSERT_EQ(constraints.error_count, 0u);
  EXPECT_EQ(constraints.fixed_index.subtree_tied_roots, nullptr);
  EXPECT_EQ(
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &constraints, reg_class_id, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID),
      3u);
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
  candidate.descriptor_reg_class_id = reg_class_id;
  candidate.location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
  candidate.location_count = 1;
  candidate.unit_count = 1;
  candidate.unit_point_start = kFixedCount;
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
            if (ranges[i][2] != location) {
              continue;
            }
            for (uint32_t j = 0; j < segment_ranges[i].count; ++j) {
              const loom_liveness_segment_t& segment =
                  segments[segment_ranges[i].start + j];
              expected |= segment.start_point < start + length &&
                          start < segment.end_point;
            }
          }
          EXPECT_EQ(
              loom_low_allocation_target_constraints_fixed_storage_conflicts(
                  &constraints, &unit_liveness, &candidate, values,
                  ignored_count),
              expected)
              << "start=" << start << " length=" << length
              << " location=" << location << " ignored=" << ignored_count;
        }
      }
    }
  }
  constraints.fixed_index.generation = UINT32_MAX;
  candidate.start_point = 23;
  candidate.end_point = 24;
  candidate.location_base = 2;
  candidate.liveness_segments = {};
  unit_ends[kFixedCount] = candidate.end_point;
  EXPECT_FALSE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
      &constraints, &unit_liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));

  candidate.start_point = 0;
  candidate.end_point = 64;
  candidate.liveness_segments = segment_ranges[kFixedCount];
  unit_ends[kFixedCount] = candidate.end_point;
  segments[segment_ranges[kFixedCount].start] = {0, 1};
  EXPECT_FALSE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
      &constraints, &unit_liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
  segments[segment_ranges[kFixedCount].start] = {23, 24};
  EXPECT_FALSE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
      &constraints, &unit_liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
  segments[segment_ranges[kFixedCount].start] = {22, 23};
  EXPECT_TRUE(loom_low_allocation_target_constraints_fixed_storage_conflicts(
      &constraints, &unit_liveness, &candidate,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
  loom_local_value_domain_release(&domain);
  loom_module_free(module);
  loom_context_deinitialize(&context);
}

TEST_F(LowAllocationTargetConstraintsTest,
       IndexesExplicitWideAndNarrowPhysicalAliases) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("fixed_alias"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &module));

  const uint16_t wide_reg_class_id =
      RegisterClassId(IREE_SV("test.atomic.narrow"));
  const uint16_t narrow_reg_class_id =
      RegisterClassId(IREE_SV("test.explicit32"));
  constexpr uint32_t kValueCount = 2;
  loom_value_id_t values[kValueCount];
  loom_liveness_interval_t intervals[kValueCount] = {};
  uint32_t interval_indices[] = {0, 1};
  loom_low_allocation_unit_liveness_value_t unit_values[] = {{0, 0}, {1, 0}};
  uint32_t unit_starts[] = {0, 0};
  uint32_t unit_ends[] = {10, 10};
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < kValueCount; ++i) {
    const uint16_t reg_class_id =
        i == 0 ? wide_reg_class_id : narrow_reg_class_id;
    IREE_ASSERT_OK(loom_module_define_value(
        module,
        loom_low_register_type(target_.descriptor_set->stable_id, reg_class_id,
                               /*unit_count=*/1),
        &values[i]));
    loom_module_value_ordinal_scratch_set(module, values[i], i);
    intervals[i].value_id = values[i];
    intervals[i].start_point = 0;
    intervals[i].end_point = 10;
    intervals[i].unit_count = 1;
    intervals[i].value_class.type_kind = LOOM_TYPE_REGISTER;
    intervals[i].value_class.register_descriptor_set_stable_id =
        target_.descriptor_set->stable_id;
    intervals[i].value_class.register_class_id = reg_class_id;
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
  unit_liveness.values = unit_values;
  unit_liveness.start_points = unit_starts;
  unit_liveness.end_points = unit_ends;
  unit_liveness.point_count = kValueCount;
  uint64_t incomplete_storage_words[] = {0};
  unit_liveness.values_with_incomplete_storage_segments = {
      kValueCount, incomplete_storage_words};

  const uint32_t wide_register_id =
      loom_low_descriptor_set_physical_register_candidate(
          target_.descriptor_set, wide_reg_class_id, /*ordinal=*/0);
  const loom_low_allocation_fixed_value_t fixed_value = {
      values[0], LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      wide_register_id, 1};
  loom_low_allocation_target_constraints_t constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op_, &target_, nullptr, 0, nullptr, 0, {}, &arena_,
      &constraints));
  const loom_low_placement_table_t placement = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_values(
      &constraints, &liveness, &domain, &unit_liveness, &placement,
      &fixed_value, /*fixed_value_count=*/1, &arena_));
  ASSERT_EQ(constraints.error_count, 0u);

  loom_low_allocation_assignment_t candidate = {};
  candidate.value_id = values[1];
  candidate.descriptor_reg_class_id = narrow_reg_class_id;
  candidate.start_point = 0;
  candidate.end_point = 10;
  candidate.unit_count = 1;
  candidate.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  candidate.location_count = 1;
  candidate.unit_point_start = 1;
  for (uint16_t ordinal = 0; ordinal < 3; ++ordinal) {
    candidate.location_base =
        loom_low_descriptor_set_physical_register_candidate(
            target_.descriptor_set, narrow_reg_class_id, ordinal);
    EXPECT_EQ(loom_low_allocation_target_constraints_fixed_storage_conflicts(
                  &constraints, &unit_liveness, &candidate,
                  /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0),
              ordinal == 0 || ordinal == 2);
  }

  loom_local_value_domain_release(&domain);
  loom_module_free(module);
  loom_context_deinitialize(&context);
}

}  // namespace
}  // namespace loom
