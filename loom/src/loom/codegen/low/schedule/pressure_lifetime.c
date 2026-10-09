// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure_lifetime.h"

#include "iree/base/internal/math.h"
#include "loom/codegen/low/schedule/target_pressure.h"

static void loom_low_schedule_note_candidate_early_added_pressure(
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
  if (pressure_state->alias_sets.records == NULL) {
    return;
  }
  const uint16_t alias_set_id =
      state->target.descriptor_set->reg_classes[reg_class_id].alias_set_id;
  if (alias_set_id == 0) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  IREE_ASSERT(iree_any_bit_set(
      record->flags, LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED));
  record->candidate_lifetime.early_added_units = iree_math_saturating_add_u64(
      record->candidate_lifetime.early_added_units, unit_count);
}

static uint32_t loom_low_schedule_candidate_early_added_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, uint16_t result_index,
    uint32_t unit_count) {
  const loom_low_operand_t* result_operand = loom_low_descriptor_operand(
      state->target.descriptor_set, node->descriptor, result_index);
  if (!iree_any_bit_set(result_operand->flags,
                        LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER)) {
    return 0;
  }
  return iree_any_bit_set(result_operand->flags, LOOM_LOW_OPERAND_FLAG_TIED)
             ? 0
             : unit_count;
}

void loom_low_schedule_pressure_lifetime_note_early_result(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node, uint16_t result_index,
    uint32_t unit_count) {
  loom_low_schedule_note_candidate_early_added_pressure(
      state, pressure_state,
      state
          ->values[loom_low_schedule_node_const_result_ordinals(
              node)[result_index]]
          .register_class_id,
      loom_low_schedule_candidate_early_added_units(state, node, result_index,
                                                    unit_count));
}

uint64_t loom_low_schedule_pressure_lifetime_transient_growth(
    int64_t delta_units, uint64_t early_added_units,
    uint64_t late_released_units) {
  const int64_t write_delta = delta_units + (int64_t)late_released_units;
  return iree_max(early_added_units,
                  write_delta > 0 ? (uint64_t)write_delta : 0);
}
