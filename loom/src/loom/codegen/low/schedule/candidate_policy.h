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

// Returns true when issue readiness should order constrained candidates after
// immediate constrained-capacity feasibility. A ready exact packing completion
// keeps its remaining headroom instead of waiting to release storage in another
// bank. Other storage handoffs participating in aggregate packing recovery
// reach the physical completion policy first; issuing their ready peers can
// fragment the bank those handoffs are closing.
static inline bool loom_low_schedule_candidate_policy_uses_ready_progress_order(
    const loom_low_schedule_options_t* options,
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  if (options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT ||
      iree_any_bit_set(options->flags,
                       LOOM_LOW_SCHEDULE_FLAG_HAS_LOWERED_SPILL_TRAFFIC) ||
      lhs->active_constrained_completion_domain_id == UINT16_MAX ||
      rhs->active_constrained_completion_domain_id == UINT16_MAX ||
      lhs->effective_stall_cycles == rhs->effective_stall_cycles) {
    return false;
  }
  if (lhs->active_constrained_completion_domain_id !=
      rhs->active_constrained_completion_domain_id) {
    return (lhs->effective_stall_cycles == 0 &&
            iree_any_bit_set(
                lhs->flags,
                LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXACT_PACKING_COMPLETION)) ||
           (rhs->effective_stall_cycles == 0 &&
            iree_any_bit_set(
                rhs->flags,
                LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXACT_PACKING_COMPLETION));
  }
  return compare_mode !=
             LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION ||
         !iree_any_bit_set(lhs->flags | rhs->flags,
                           LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE);
}

// Classifies pressure progress from the completed live-unit and storage facts
// in |score| and the producer-owned descriptor |frontier_progress|.
loom_low_schedule_pressure_progress_kind_t
loom_low_schedule_classify_candidate_pressure_progress(
    const loom_low_schedule_candidate_score_t* score,
    loom_low_schedule_pressure_progress_kind_t frontier_progress);

// Classifies new debt or bounded downstream demand relative to current
// persistent pressure. Existing debt alone does not make a candidate risky.
loom_low_schedule_pressure_risk_t
loom_low_schedule_classify_candidate_pressure_risk(
    const loom_low_schedule_candidate_score_t* score,
    uint32_t current_persistent_pressure_penalty);

// Selects the physical-pressure context shared by every nominee comparison.
// A packing completion takes precedence over general pressure relief.
loom_low_schedule_candidate_compare_mode_t
loom_low_schedule_choose_candidate_compare_mode(
    const loom_low_schedule_candidate_score_t* scores,
    iree_host_size_t score_count, uint32_t current_persistent_pressure_penalty);

// Tests a graph- or schedule-ranked nominee beyond the bounded source window.
// A growing nominee must come from the schedule view, remain free of pressure
// risk, and stay within the source nominees' critical-path band.
bool loom_low_schedule_candidate_policy_admits_ranked_nominee(
    const loom_low_schedule_candidate_score_t* score,
    uint32_t maximum_source_critical_path_cycles, bool is_graph_nominee);

// Tests the pressure eligibility of a recovery nominee after its latency-window
// checks. Default recovery cannot introduce debt. Pressure recovery requires
// progress unless it replaces deferred setup or escapes hard-capacity failure.
bool loom_low_schedule_candidate_policy_admits_recovery_nominee(
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* score,
    const loom_low_schedule_candidate_score_t* chosen_score);

// Returns true when |lhs| is preferred to |rhs| under the selected physical
// pressure context. Both scores borrow |register_debt_word_count| debt words
// from their owning policy. Comparison reads only retained score facts.
bool loom_low_schedule_candidate_score_less(
    const loom_low_schedule_options_t* options,
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs,
    iree_host_size_t register_debt_word_count);

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
