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

// Reusable score storage for one scheduling run's bounded nomination policy.
typedef struct loom_low_schedule_candidate_policy_t {
  // Number of register-debt words in each exact score's row.
  iree_host_size_t register_debt_word_count;
  // Scratch-arena rows, one per exact score until the next selection.
  loom_low_schedule_register_debt_word_t* nominee_register_debts;
} loom_low_schedule_candidate_policy_t;

// Physical-pressure context used to rank a bounded set of ready candidates.
typedef enum loom_low_schedule_candidate_compare_mode_e {
  // No active physical-pressure recovery requirement.
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT = 0,
  // Prefer work that relieves active register pressure.
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF = 1,
  // Preserve a selected aggregate register-packing completion.
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION = 2,
} loom_low_schedule_candidate_compare_mode_t;

// Returns true when issue readiness should order two candidates advancing the
// same active unspillable completion. A storage handoff participating in
// aggregate packing recovery must reach the physical completion policy first;
// issuing its ready peer can fragment the bank the handoff is closing.
static inline bool loom_low_schedule_candidate_policy_uses_ready_progress_order(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  return state->options->strategy ==
             LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
         compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT &&
         (compare_mode !=
              LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION ||
          !iree_any_bit_set(
              lhs->flags | rhs->flags,
              LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE)) &&
         !iree_any_bit_set(state->options->flags,
                           LOOM_LOW_SCHEDULE_FLAG_HAS_LOWERED_SPILL_TRAFFIC) &&
         lhs->active_unspillable_completion_domain_id != UINT16_MAX &&
         lhs->active_unspillable_completion_domain_id ==
             rhs->active_unspillable_completion_domain_id &&
         lhs->effective_stall_cycles != rhs->effective_stall_cycles;
}

// Allocates bounded score storage when the strategy uses pressure scoring.
iree_status_t loom_low_schedule_candidate_policy_initialize(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_policy_t* out_policy);

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
    const loom_low_schedule_candidate_policy_t* policy,
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
