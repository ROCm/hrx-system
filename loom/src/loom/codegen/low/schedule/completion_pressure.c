// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_pressure.h"

#include "loom/codegen/low/descriptor_traits.h"

void loom_low_schedule_completion_pressure_classify_node(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (node->descriptor == NULL || node->result_count != 1 ||
      iree_any_bit_set(node->traits, LOOM_TRAIT_OBSERVABLE_EFFECT) ||
      !loom_low_descriptor_result_can_rematerialize(
          state->target.descriptor_set, node->descriptor, 0)) {
    return;
  }
  loom_low_schedule_value_record_t* result =
      &state->values[loom_low_schedule_node_const_result_ordinals(node)[0]];
  if (!loom_low_schedule_value_has_constrained_register(result)) {
    return;
  }
  const uint16_t pressure_set = loom_low_reg_class_pressure_set_id(
      &state->target.descriptor_set->reg_classes[result->register_class_id]);
  const loom_value_ordinal_t* operands =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t i = 0; i < node->operand_count; ++i) {
    const uint16_t input_class = state->values[operands[i]].register_class_id;
    if (input_class == result->register_class_id ||
        (input_class != LOOM_LOW_REG_CLASS_NONE && pressure_set != 0 &&
         pressure_set ==
             loom_low_reg_class_pressure_set_id(
                 &state->target.descriptor_set->reg_classes[input_class]))) {
      return;
    }
  }
  result->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_REPLAYABLE_OUTSIDE_DOMAIN;
}

void loom_low_schedule_completion_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  for (uint16_t domain = 0;
       domain < state->pressure_limits.constrained_completion_domain_count;
       ++domain) {
    const loom_low_schedule_completion_pressure_t* record =
        &pressure_state->completion_pressure[domain];
    const int64_t growth =
        iree_max(record->candidate_delta_units,
                 iree_max(record->candidate_write_delta_units,
                          (int64_t)record->candidate_early_added_units));
    const uint32_t capacity =
        state->pressure_limits.constrained_completion_domains[domain].capacity;
    if (growth > 0 && record->live_units + (uint64_t)growth > capacity) {
      score->flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_CONSTRAINED_CAPACITY;
    }
  }
}
