// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded nomination and comparison policy for ready schedule candidates.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_CANDIDATE_POLICY_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_CANDIDATE_POLICY_H_

#include "loom/codegen/low/schedule/pressure.h"
#include "loom/codegen/low/schedule/ready_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

// Result of selecting one node from a non-empty ready frontier.
typedef struct loom_low_schedule_candidate_selection_t {
  // Chosen node index.
  uint32_t chosen_node;
  // Highest-ranked rejected node, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t rejected_node;
  // Number of dependency-ready nodes when selection began.
  uint32_t ready_candidate_count;
  // Number of bounded nominees scored during selection.
  uint32_t scored_candidate_count;
  // Complete score for chosen_node when pressure scoring is active.
  loom_low_schedule_candidate_score_t chosen_score;
  // Complete score for rejected_node when present.
  loom_low_schedule_candidate_score_t rejected_score;
} loom_low_schedule_candidate_selection_t;

// Selects one node from a known non-empty |ready_policy| frontier.
void loom_low_schedule_candidate_policy_select(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t ready_candidate_count,
    loom_low_schedule_candidate_selection_t* out_selection);

// Records a requested structured decision row for |selection|.
void loom_low_schedule_candidate_policy_record_decision(
    loom_low_schedule_build_state_t* state, uint32_t block_index,
    uint32_t scheduled_ordinal,
    const loom_low_schedule_candidate_selection_t* selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_CANDIDATE_POLICY_H_
