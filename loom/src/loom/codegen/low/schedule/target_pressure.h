// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target pressure cliffs, shared-class limits, and derived resources.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_TARGET_PRESSURE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_TARGET_PRESSURE_H_

#include "loom/codegen/low/schedule/pressure.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_schedule_pressure_set_flag_bits_e {
  LOOM_LOW_SCHEDULE_PRESSURE_SET_FLAG_BLOCK_TOUCHED = 1u << 0,
  LOOM_LOW_SCHEDULE_PRESSURE_SET_FLAG_CANDIDATE_TOUCHED = 1u << 1,
};

enum loom_low_schedule_resource_pressure_flag_bits_e {
  LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED = 1u << 0,
};

// Builds dense bounded constrained completion domains after the scheduler has
// resolved per-class and shared pressure-set capacities.
iree_status_t loom_low_schedule_target_pressure_initialize_completion_domains(
    loom_low_schedule_build_state_t* state);

// Resolves function-local packing-resource capacities and leaves resources
// inactive when direct register pressure already carries the same signal.
iree_status_t loom_low_schedule_target_pressure_initialize_packing_resources(
    loom_low_schedule_build_state_t* state, const uint8_t* active_reg_classes);

// Returns the function-local capacity of |resource_id|, or UINT32_MAX when the
// resource is redundant for this function.
uint32_t loom_low_schedule_register_packing_resource_capacity(
    const loom_low_schedule_build_state_t* state, uint16_t resource_id);

// Resets the incrementally maintained current packing-resource footprints.
void loom_low_schedule_target_pressure_reset_current_packing_units(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state);

// Applies one register-class live-unit change to its packing resources.
void loom_low_schedule_target_pressure_adjust_current_packing_units(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    int64_t delta_units);

// Mutable pressure state for one shared register-class allocation domain.
struct loom_low_schedule_pressure_set_record_t {
  // Current live units in the shared pressure set.
  uint64_t current_live_units;
  // Live-unit delta projected for the candidate being scored.
  int64_t candidate_delta_units;
  // Candidate operand deaths in the shared pressure set.
  uint64_t candidate_killed_units;
  // Candidate instruction-phase overlap in the shared register namespace.
  struct {
    // Result storage created before ordinary input reads complete.
    uint64_t early_added_units;
    // Dying input units retained through result writes.
    uint64_t late_released_units;
  } candidate_lifetime;
  // Block-local headroom required by aligned contiguous values.
  uint32_t packing_reserve_units;
  // Mutable loom_low_schedule_pressure_set_flag_bits_e bits.
  uint8_t flags;
};

// Mutable high-water state for one derived target pressure resource.
struct loom_low_schedule_resource_pressure_record_t {
  // Current sum of rounded member-class high-water marks.
  uint64_t current_peak_units;
  // Additional resource units projected by the candidate being scored.
  uint64_t candidate_added_units;
  // Penalty accumulated above the authored source-order baseline.
  uint32_t pressure_cliff_penalty;
  // First target cliff not yet crossed by the scheduled high-water mark.
  uint16_t next_cliff_index;
  // Mutable loom_low_schedule_resource_pressure_flag_bits_e bits.
  uint8_t flags;
};

// Converts register units into the member's shared packing-resource units.
uint64_t loom_low_schedule_register_packing_contribution(
    uint64_t register_units,
    const loom_low_register_packing_resource_member_t* member);

// Projects available register units through a capacity-source member. Unlike
// live contributions this rounds down because an incomplete group cannot
// provide another resource unit.
uint32_t loom_low_schedule_register_packing_member_capacity(
    uint32_t register_unit_capacity,
    const loom_low_register_packing_resource_t* resource,
    const loom_low_register_packing_resource_member_t* member);

// Returns the retained packing-resource footprint of the node results.
uint64_t loom_low_schedule_node_register_packing_result_units(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t resource_id);

// Returns whether |node_index| carries storage from |reg_class_id| into a
// result class sharing the same indivisible register-packing resource.
bool loom_low_schedule_node_retains_aggregate_packing_from_class(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id);

// Resets live packing-resource completion state for |block_record|.
void loom_low_schedule_target_pressure_reset_packing_completions(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state);

// Adds or removes a live value from packing-resource completion state.
void loom_low_schedule_target_pressure_add_packing_completion_value(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal);
void loom_low_schedule_target_pressure_remove_packing_completion_value(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal);

// Advances stale completion anchors after a node has transferred or retired
// live packing-resource storage.
void loom_low_schedule_target_pressure_repair_packing_completions(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index);

// Returns the capacity of |completion_domain_id| when its live pressure is at
// or above that hard limit, or UINT32_MAX while it still has headroom.
uint32_t loom_low_schedule_target_pressure_full_constrained_completion_capacity(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint16_t completion_domain_id);

// Scores shared register-packing footprints across the candidate's lifetime
// phases. NODE_NONE omits node-specific completion priority; the native
// pressure records and resolved resource capacities still determine capacity
// and debt.
void loom_low_schedule_score_candidate_register_packing_resources(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score);

// Scores all target-authored pressure cliffs, limits, and derived resources
// against the candidate deltas already present in |pressure_state|.
void loom_low_schedule_target_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_TARGET_PRESSURE_H_
