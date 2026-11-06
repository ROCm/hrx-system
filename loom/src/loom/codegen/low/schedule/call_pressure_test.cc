// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/call_pressure.h"

#include "iree/testing/gtest.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/schedule/target_pressure.h"

namespace loom {
namespace {

class CallPressureTest : public ::testing::Test {
 protected:
  // Two register views with independently bounded storage unless aliased.
  loom_low_reg_class_t classes_[2] = {
      {.alloc_unit_bits = 32, .allocatable_count = 8},
      {.alloc_unit_bits = 64, .allocatable_count = 8},
  };
  // Descriptor-owned class and optional packing geometry.
  loom_low_descriptor_set_t descriptors_ = {
      .reg_classes = classes_,
      .reg_class_count = IREE_ARRAYSIZE(classes_),
  };
  // Live register units before consuming call arguments.
  uint64_t live_[2] = {};
  // Units whose final use is the call.
  uint64_t killed_[2] = {};
  // Post-call result-minus-argument changes, excluded at the clobber phase.
  int64_t delta_[2] = {};
  // Aggregate state when the two classes share a bank.
  loom_low_schedule_pressure_set_record_t sets_[2] = {};
  // Existing pressure view over the native records above.
  loom_low_schedule_pressure_state_t pressure_ = {
      .current_live_units_by_reg_class = live_,
      .pressure_sets = {.records = sets_},
      .candidate_delta_units_by_reg_class = delta_,
      .candidate_killed_units_by_reg_class = killed_,
  };
};

TEST_F(CallPressureTest, OnlySurvivorsConsumePreservedCapacity) {
  const loom_low_schedule_call_pressure_limit_t limit = {
      .preserved_unit_capacity = 4,
      .source_kind = LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS,
  };
  const struct {
    // Call operands that consume their last live units.
    uint64_t killed;
    // Units above the four-unit preserved capacity during the call.
    uint32_t debt;
  } cases[] = {{0, 3}, {1, 2}, {3, 0}, {7, 0}};
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.killed);
    live_[0] = 7;
    killed_[0] = test_case.killed;
    delta_[0] = 6 - static_cast<int64_t>(test_case.killed);
    loom_low_schedule_candidate_score_t score = {
        .projected_live_units = 13 - test_case.killed,
        .killed_live_units = test_case.killed,
        .produced_live_units = 6,
        .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
        .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
    };
    loom_low_schedule_call_pressure_score_limits(&descriptors_, &pressure_,
                                                 &limit, 1, &score);
    EXPECT_EQ(score.pressure_cliff_penalty, test_case.debt);
    EXPECT_EQ(score.flags,
              test_case.debt
                  ? LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT
                  : 0);
    EXPECT_EQ(score.pressure_cliff_units,
              test_case.debt ? 4u : LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE);
    EXPECT_EQ(live_[0], 7u);
    EXPECT_EQ(killed_[0], test_case.killed);
  }
}

TEST_F(CallPressureTest, AliasedWidthsUseOneSharedBank) {
  classes_[0].alias_set_id = classes_[1].alias_set_id = 1;
  live_[0] = 4;
  live_[1] = 3;
  killed_[0] = killed_[1] = 1;
  sets_[1].current_live_units = 7;
  sets_[1].candidate_killed_units = 2;
  const loom_low_schedule_call_pressure_limit_t limit = {
      .pressure_set_id = 1,
      .preserved_unit_capacity = 4,
      .source_kind = LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS,
  };
  loom_low_schedule_candidate_score_t score = {
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = 3,
  };
  loom_low_schedule_call_pressure_score_limits(&descriptors_, &pressure_,
                                               &limit, 1, &score);
  // Neither class alone exceeds four, but their five survivors share the bank.
  EXPECT_EQ(score.pressure_cliff_penalty, 1u);
  EXPECT_EQ(score.flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT);
  EXPECT_EQ(score.pressure_cliff_source_kind,
            LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS);
  EXPECT_EQ(score.pressure_cliff_source_id, 0u);
  EXPECT_EQ(score.pressure_cliff_units, 4u);
  EXPECT_EQ(score.units_until_pressure_cliff,
            LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE);
}

TEST_F(CallPressureTest, CallClobbersReduceSharedPackingCapacity) {
  loom_low_register_packing_resource_t resource = {
      .capacity = 8,
      .member_count = 2,
      .capacity_member_index = 0,
  };
  const loom_low_register_packing_resource_member_t members[] = {
      {.reg_class_id = 0, .register_unit_count = 2, .resource_unit_count = 1},
      {.reg_class_id = 1, .register_unit_count = 1, .resource_unit_count = 1},
  };
  descriptors_.register_packing_resources = &resource;
  descriptors_.register_packing_resource_count = 1;
  descriptors_.register_packing_resource_members = members;
  descriptors_.register_packing_resource_member_count = IREE_ARRAYSIZE(members);
  loom_low_schedule_build_state_t state = {};
  state.target.descriptor_set = &descriptors_;
  const loom_low_allocation_target_constraints_t constraints = {
      .target = &state.target,
  };
  state.allocation_constraints = &constraints;
  uint32_t class_limits[] = {8, 8};
  uint32_t packing_limits[] = {4};
  state.pressure_limits.by_reg_class = class_limits;
  state.pressure_limits.register_packing_resource_capacities = packing_limits;
  const loom_low_call_clobber_t clobber = {
      .register_class = 0,
      .location = 4,
      .count = 4,
  };
  loom_low_call_contract_t contract = {};
  loom_low_schedule_call_pressure_limit_t limits[3] = {};
  EXPECT_EQ(loom_low_schedule_call_pressure_build_limits(&state, &contract, {},
                                                         limits),
            0u);
  contract.clobbers = {&clobber, 1};
  uint64_t packed_units[] = {3};
  pressure_.current_register_packing_units = packed_units;
  live_[1] = 1;
  killed_[0] = 1;
  for (bool hard_capacity : {false, true}) {
    resource.flags =
        hard_capacity ? LOOM_LOW_REGISTER_PACKING_RESOURCE_FLAG_UNSPILLABLE : 0;
    const iree_host_size_t limit_count =
        loom_low_schedule_call_pressure_build_limits(&state, &contract, {},
                                                     limits);
    ASSERT_EQ(limit_count, 2u);
    EXPECT_EQ(limits[0].source_kind,
              LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS);
    EXPECT_EQ(limits[0].source_id, 0u);
    EXPECT_EQ(limits[0].preserved_unit_capacity, 4u);
    EXPECT_EQ(limits[1].source_kind,
              LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE);
    EXPECT_EQ(limits[1].source_id, 0u);
    EXPECT_EQ(limits[1].preserved_unit_capacity, 2u);
    EXPECT_EQ(limits[1].requires_hard_capacity, hard_capacity);
    for (uint64_t live_units : {3u, 4u}) {
      SCOPED_TRACE(::testing::Message()
                   << "live=" << live_units << ", hard=" << hard_capacity);
      live_[0] = live_units;
      loom_low_schedule_candidate_score_t score = {
          .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
          .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      };
      loom_low_schedule_call_pressure_score_limits(&descriptors_, &pressure_,
                                                   limits, limit_count, &score);
      // Killing one of three narrow units frees a pair; killing one of four
      // leaves both pairs occupied. The other class retains one whole unit.
      const uint32_t debt = live_units == 4 ? 1 : 0;
      EXPECT_EQ(score.pressure_cliff_penalty, debt);
      const uint16_t debt_flag =
          hard_capacity
              ? LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY
              : LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT;
      EXPECT_EQ(score.flags, debt ? debt_flag : 0);
      EXPECT_EQ(packed_units[0], 3u);
    }
  }
}

TEST_F(CallPressureTest, AccumulatesDebtWithoutReplacingEarlierCliff) {
  live_[0] = 7;
  live_[1] = 3;
  const loom_low_schedule_call_pressure_limit_t limits[] = {
      {.source_kind = LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS},
      {
          .source_id = 1,
          .source_kind = LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS,
      },
  };
  loom_low_schedule_candidate_score_t score = {
      .pressure_cliff_penalty = UINT32_MAX - 4,
      .pressure_cliff_units = 2,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .pressure_cliff_source_id = 1,
      .pressure_cliff_source_kind =
          LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS,
      .flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR,
  };
  loom_low_schedule_call_pressure_score_limits(
      &descriptors_, &pressure_, limits, IREE_ARRAYSIZE(limits), &score);
  EXPECT_EQ(score.pressure_cliff_penalty, UINT32_MAX);
  EXPECT_EQ(score.pressure_cliff_source_id, 1u);
  EXPECT_EQ(score.pressure_cliff_units, 2u);
  EXPECT_EQ(score.flags,
            LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR |
                LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT);
}

}  // namespace
}  // namespace loom
