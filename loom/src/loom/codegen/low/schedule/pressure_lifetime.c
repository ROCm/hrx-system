// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure_lifetime.h"

#include "iree/base/internal/math.h"
#include "loom/codegen/low/schedule/target_pressure.h"

void loom_low_schedule_pressure_lifetime_note_early_result(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    uint32_t unit_count) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE || unit_count == 0 ||
      pressure_state->candidate_lifetime.early_added_units == NULL) {
    return;
  }
  IREE_ASSERT(pressure_state->candidate_delta_touched_flags[reg_class_id]);
  pressure_state->candidate_lifetime.early_added_units[reg_class_id] =
      iree_math_saturating_add_u64(
          pressure_state->candidate_lifetime.early_added_units[reg_class_id],
          unit_count);
  if (pressure_state->pressure_sets.records == NULL) {
    return;
  }
  const uint16_t pressure_set_id =
      loom_low_schedule_reg_class_pressure_set_id(state, reg_class_id);
  if (pressure_set_id == 0) {
    return;
  }
  loom_low_schedule_pressure_set_record_t* record =
      &pressure_state->pressure_sets.records[pressure_set_id];
  IREE_ASSERT(iree_any_bit_set(
      record->flags, LOOM_LOW_SCHEDULE_PRESSURE_SET_FLAG_CANDIDATE_TOUCHED));
  record->candidate_lifetime.early_added_units = iree_math_saturating_add_u64(
      record->candidate_lifetime.early_added_units, unit_count);
}

uint64_t loom_low_schedule_pressure_lifetime_transient_growth(
    int64_t delta_units, uint64_t early_added_units,
    uint64_t late_released_units) {
  const int64_t write_delta = delta_units + (int64_t)late_released_units;
  return iree_max(early_added_units,
                  write_delta > 0 ? (uint64_t)write_delta : 0);
}
