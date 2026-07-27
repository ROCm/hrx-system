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

enum loom_low_schedule_alias_pressure_flag_bits_e {
  LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED = 1u << 0,
  LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED = 1u << 1,
};

enum loom_low_schedule_resource_pressure_flag_bits_e {
  LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED = 1u << 0,
};

// Mutable pressure state for one shared register-class namespace.
struct loom_low_schedule_alias_pressure_record_t {
  // Current live units in the shared alias namespace.
  uint64_t current_live_units;
  // Live-unit delta projected for the candidate being scored.
  int64_t candidate_delta_units;
  // Units created during the candidate's Early phase.
  uint64_t candidate_early_added_units;
  // Block-local headroom required by aligned contiguous values.
  uint32_t packing_reserve_units;
  // Mutable loom_low_schedule_alias_pressure_flag_bits_e bits.
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

// Scores all target-authored pressure cliffs, limits, and derived resources
// against the candidate deltas already present in |pressure_state|.
void loom_low_schedule_target_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_TARGET_PRESSURE_H_
