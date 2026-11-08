// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical ownership and completion headroom in bounded register banks.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_PRESSURE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_PRESSURE_H_

#include "loom/codegen/low/schedule/pressure.h"

#ifdef __cplusplus
extern "C" {
#endif

// Storage remains charged to the pressure model's current owner across alias
// transfers. Constrained owners establish demand in their register domains;
// their borrowed aliases do not count the same units twice.
struct loom_low_schedule_completion_pressure_t {
  // Live register units from required lifetimes and singleton classes.
  uint64_t live_units;
  // Live units without a direct replay recipe outside this domain.
  uint64_t non_replayable_live_units;
  // Change in constrained ownership after the candidate completes.
  int64_t candidate_delta_units;
  // Change in ownership without a direct replay recipe outside this domain.
  int64_t non_replayable_candidate_delta_units;
  // Change in constrained ownership at the candidate's result-write boundary.
  int64_t candidate_write_delta_units;
  // Constrained result units overlapping all inputs before their reads.
  uint64_t candidate_early_added_units;
  // True when the candidate creates a used result in this domain.
  bool candidate_opens_lifetime;
};

// Retains direct rematerialization eligibility once during graph construction.
// A replayable packet needs no input storage from its result's physical bank;
// captures in aliased register classes still occupy that bank. Structural
// identities and multi-result packets retain their physical ownership, matching
// allocation repair's single-result descriptor recipe boundary.
void loom_low_schedule_completion_pressure_classify_node(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node);

// Returns the completion-pressure domain for a constrained value, or NULL
// when its register class has no bounded completion domain.
static inline loom_low_schedule_completion_pressure_t*
loom_low_schedule_completion_pressure_for_value(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_value_record_t* value) {
  if (!loom_low_schedule_value_has_constrained_register(value)) {
    return NULL;
  }
  const uint16_t domain = loom_low_schedule_constrained_completion_domain_id(
      state, value->register_class_id);
  return domain != UINT16_MAX ? &pressure_state->completion_pressure[domain]
                              : NULL;
}

// Transfers ownership at a live-value or alias boundary. The caller supplies
// the value's full owned delta, before subtracting transfers to other owners.
static inline void loom_low_schedule_completion_pressure_adjust_live(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_value_record_t* value, int64_t delta_units) {
  loom_low_schedule_completion_pressure_t* record =
      loom_low_schedule_completion_pressure_for_value(state, pressure_state,
                                                      value);
  if (record != NULL) {
    record->live_units += delta_units;
    if (!iree_any_bit_set(
            value->flags,
            LOOM_LOW_SCHEDULE_VALUE_FLAG_REPLAYABLE_OUTSIDE_DOMAIN)) {
      record->non_replayable_live_units += delta_units;
    }
  }
}

// Records a candidate ownership change and whether it occurs after writes.
static inline void loom_low_schedule_completion_pressure_adjust_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_value_record_t* value, int64_t delta_units,
    bool after_writes) {
  loom_low_schedule_completion_pressure_t* record =
      loom_low_schedule_completion_pressure_for_value(state, pressure_state,
                                                      value);
  if (record != NULL) {
    record->candidate_delta_units += delta_units;
    if (!iree_any_bit_set(
            value->flags,
            LOOM_LOW_SCHEDULE_VALUE_FLAG_REPLAYABLE_OUTSIDE_DOMAIN)) {
      record->non_replayable_candidate_delta_units += delta_units;
    }
    if (!after_writes) {
      record->candidate_write_delta_units += delta_units;
    }
  }
}

// Marks candidates whose constrained ownership grows beyond its physical bank.
void loom_low_schedule_completion_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_PRESSURE_H_
