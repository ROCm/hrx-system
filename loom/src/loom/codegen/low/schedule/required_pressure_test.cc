// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/required_pressure.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(RequiredPressureTest, ReplayKeepsSameDomainCapturesResident) {
  const loom_low_reg_class_t classes[] = {
      {.alias_set_id = 1},
      {.alias_set_id = 2},
      {.alias_set_id = 1},
      {
          .flags = LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
          .storage = {.explicit_registers = {.pressure_set_id = 1}},
      },
      {},
  };
  const uint16_t operand_indices[] = {0, 1, 2, 3};
  loom_low_operand_t operands[4] = {};
  const loom_low_descriptor_set_t descriptors = {
      .operands = operands,
      .operand_count = IREE_ARRAYSIZE(operands),
      .operand_indices = operand_indices,
      .operand_index_count = IREE_ARRAYSIZE(operand_indices),
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  const struct {
    // Recipe or storage boundary under test.
    const char* name;
    // Operand register classes; NONE denotes a non-register capture.
    uint16_t input_classes[2];
    // Number of captures in the recipe.
    uint16_t input_count;
    // Whether replay can release the result's physical domain.
    bool replayable;
  } cases[] = {
      {"constant", {}, 0, true},
      {"other domain", {1}, 1, true},
      {"independent class", {4}, 1, true},
      {"non-register capture", {LOOM_LOW_REG_CLASS_NONE}, 1, true},
      {"same class", {0}, 1, false},
      {"linear alias", {2}, 1, false},
      {"explicit physical alias", {3}, 1, false},
      {"second capture retains domain", {1, 2}, 2, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    for (bool rematerializable : {false, true}) {
      for (bool observable : {false, true}) {
        for (uint16_t result_count : {0, 1, 2}) {
          SCOPED_TRACE(::testing::Message()
                       << rematerializable << "," << observable << ","
                       << result_count);
          loom_low_descriptor_t descriptor = {
              .operand_count = (uint16_t)(result_count + test_case.input_count),
              .result_count = result_count,
          };
          for (uint16_t i = 0; i < descriptor.operand_count; ++i) {
            operands[i].role = i < result_count ? LOOM_LOW_OPERAND_ROLE_RESULT
                                                : LOOM_LOW_OPERAND_ROLE_OPERAND;
            operands[i].flags = i < result_count && rematerializable
                                    ? LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE
                                    : 0;
          }
          loom_low_schedule_value_record_t values[4] = {};
          loom_low_schedule_node_t node = {
              .descriptor = &descriptor,
              .traits = observable ? LOOM_TRAIT_OBSERVABLE_EFFECT : 0u,
              .operand_count = test_case.input_count,
              .result_count = result_count,
          };
          auto* ordinals = loom_low_schedule_node_value_ordinals(&node);
          for (uint16_t i = 0; i < descriptor.operand_count; ++i) {
            ordinals[i] = i;
            values[i].register_class_id =
                i < test_case.input_count ? test_case.input_classes[i] : 0;
            values[i].flags =
                i < test_case.input_count
                    ? 0
                    : LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
          }
          loom_low_schedule_build_state_t state = {
              .target = {.descriptor_set = &descriptors},
              .values = values,
          };
          loom_low_schedule_required_pressure_classify_node(&state, &node);
          for (uint16_t i = test_case.input_count; i < descriptor.operand_count;
               ++i) {
            EXPECT_EQ(
                iree_any_bit_set(
                    values[i].flags,
                    LOOM_LOW_SCHEDULE_VALUE_FLAG_REPLAYABLE_OUTSIDE_DOMAIN),
                test_case.replayable && rematerializable && !observable &&
                    result_count == 1);
          }
        }
      }
    }
  }
}

TEST(RequiredPressureTest, OwnershipAndWriteOverlapBelongToPhysicalDomains) {
  loom_low_schedule_completion_domain_t domains[] = {
      {.capacity = 4, .reg_class_id = 0},
      {.capacity = 4, .reg_class_id = 2},
  };
  uint16_t domain_ids[] = {0, 0, 1, UINT16_MAX};
  const loom_low_schedule_build_state_t state = {
      .pressure_limits =
          {
              .unspillable_completion_domains = domains,
              .unspillable_completion_domain_ids_by_reg_class = domain_ids,
              .unspillable_completion_domain_count = IREE_ARRAYSIZE(domains),
          },
  };
  const loom_low_schedule_value_record_t values[] = {
      {
          .unit_count = 2,
          .register_class_id = 0,
          .flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER |
                   LOOM_LOW_SCHEDULE_VALUE_FLAG_REPLAYABLE_OUTSIDE_DOMAIN,
      },
      {
          .unit_count = 2,
          .register_class_id = 1,
          .flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER,
      },
      {
          .unit_count = 4,
          .register_class_id = 2,
          .flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER,
      },
      {.unit_count = 1, .register_class_id = 0},
      {
          .unit_count = 1,
          .register_class_id = 3,
          .flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER,
      },
  };
  for (bool late_read : {false, true}) {
    SCOPED_TRACE(late_read);
    loom_low_schedule_required_pressure_t required[2] = {};
    loom_low_schedule_pressure_state_t pressure = {
        .required_pressure = required,
    };
    for (const auto& value : values) {
      loom_low_schedule_required_pressure_adjust_live(&state, &pressure, &value,
                                                      value.unit_count);
    }
    EXPECT_EQ(required[0].live_units, 4u);
    EXPECT_EQ(required[1].live_units, 4u);
    EXPECT_EQ(required[0].non_replayable_live_units, 2u);
    EXPECT_EQ(required[1].non_replayable_live_units, 4u);

    // The alias views exchange one unit, with transient overlap for late
    // reads. Emptying another bank cannot repay that shared bank's pressure.
    required[0].candidate_opens_lifetime = true;
    loom_low_schedule_required_pressure_adjust_candidate(&state, &pressure,
                                                         &values[0], 1, false);
    loom_low_schedule_required_pressure_adjust_candidate(
        &state, &pressure, &values[1], -1, late_read);
    loom_low_schedule_required_pressure_adjust_candidate(&state, &pressure,
                                                         &values[2], -4, false);
    loom_low_schedule_required_pressure_adjust_candidate(&state, &pressure,
                                                         &values[3], 1, false);
    loom_low_schedule_required_pressure_adjust_candidate(&state, &pressure,
                                                         &values[4], 1, false);
    EXPECT_EQ(required[0].candidate_delta_units, 0);
    EXPECT_EQ(required[0].candidate_write_delta_units, late_read ? 1 : 0);
    EXPECT_EQ(required[1].candidate_delta_units, -4);
    EXPECT_EQ(required[1].candidate_write_delta_units, -4);
    EXPECT_EQ(required[0].non_replayable_candidate_delta_units, -1);
    EXPECT_EQ(required[1].non_replayable_candidate_delta_units, -4);
    loom_low_schedule_candidate_score_t score = {};
    loom_low_schedule_required_pressure_score_candidate(&state, &pressure,
                                                        &score);
    EXPECT_EQ(
        iree_any_bit_set(
            score.flags,
            LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY),
        late_read);

    for (const auto& value : values) {
      loom_low_schedule_required_pressure_adjust_live(
          &state, &pressure, &value, -(int64_t)value.unit_count);
    }
    EXPECT_EQ(required[0].live_units, 0u);
    EXPECT_EQ(required[1].live_units, 0u);
    EXPECT_EQ(required[0].non_replayable_live_units, 0u);
    EXPECT_EQ(required[1].non_replayable_live_units, 0u);
  }
}

TEST(RequiredPressureTest, HardCapacityUsesTheLargestInstructionPhase) {
  loom_low_schedule_completion_domain_t domain = {.capacity = 4};
  const loom_low_schedule_build_state_t state = {
      .pressure_limits =
          {
              .unspillable_completion_domains = &domain,
              .unspillable_completion_domain_count = 1,
          },
  };
  const struct {
    // Required ownership before, during, and after a candidate.
    loom_low_schedule_required_pressure_t pressure;
    // Whether this candidate grows beyond the physical capacity.
    bool exceeds_capacity;
  } cases[] = {
      {{.live_units = 3, .candidate_delta_units = 1}, false},
      {{.live_units = 4, .candidate_delta_units = 1}, true},
      {{.live_units = 4, .candidate_write_delta_units = 1}, true},
      {{.live_units = 4, .candidate_early_added_units = 1}, true},
      {{
           .live_units = 3,
           .candidate_delta_units = 1,
           .candidate_write_delta_units = 1,
           .candidate_early_added_units = 1,
       },
       false},
      {{.live_units = 5}, false},
      {{.live_units = 5, .candidate_delta_units = -1}, false},
  };
  for (const auto& test_case : cases) {
    auto required = test_case.pressure;
    SCOPED_TRACE(::testing::Message()
                 << "live=" << required.live_units
                 << " delta=" << required.candidate_delta_units
                 << " write=" << required.candidate_write_delta_units
                 << " early=" << required.candidate_early_added_units);
    const loom_low_schedule_pressure_state_t pressure = {
        .required_pressure = &required,
    };
    loom_low_schedule_candidate_score_t score = {};
    loom_low_schedule_required_pressure_score_candidate(&state, &pressure,
                                                        &score);
    EXPECT_EQ(
        iree_any_bit_set(
            score.flags,
            LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY),
        test_case.exceeds_capacity);
  }
}

}  // namespace
}  // namespace loom
