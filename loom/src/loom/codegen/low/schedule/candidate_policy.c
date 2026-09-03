// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/candidate_policy.h"

#include "loom/codegen/low/schedule/ready_frontier.h"

#define LOOM_LOW_SCHEDULE_READY_SOURCE_NOMINEE_COUNT 16
#define LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT 16
#define LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT 2
#define LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY 24

typedef enum loom_low_schedule_candidate_compare_mode_e {
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT = 0,
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF = 1,
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_MODE_COUNT = 2,
} loom_low_schedule_candidate_compare_mode_t;

enum loom_low_schedule_recovery_policy_flag_bits_e {
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REJECT_PRESSURE_DEBT = 1u << 0,
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REQUIRE_PRESSURE_PROGRESS = 1u << 1,
};

typedef struct loom_low_schedule_recovery_policy_t {
  // Ready views contributing additional recovery nominees.
  loom_low_schedule_ready_view_t ready_views[2];
  // Number of populated entries in ready_views.
  uint8_t ready_view_count;
  // loom_low_schedule_recovery_policy_flag_bits_e bits.
  uint8_t flags;
} loom_low_schedule_recovery_policy_t;

static const loom_low_schedule_recovery_policy_t kLoomLowScheduleRecoveryPolicies
    [LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_MODE_COUNT] = {
        [LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT] =
            {
                .ready_views = {LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE},
                .ready_view_count = 1,
                .flags =
                    LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REJECT_PRESSURE_DEBT,
            },
        [LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF] =
            {
                .ready_views =
                    {
                        LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE,
                        LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE,
                    },
                .ready_view_count = 2,
                .flags =
                    LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REQUIRE_PRESSURE_PROGRESS,
            },
};

static int loom_low_schedule_compare_candidate_pressure(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  if (lhs->projected_live_units != rhs->projected_live_units) {
    return lhs->projected_live_units < rhs->projected_live_units ? -1 : 1;
  }
  if (lhs->killed_live_units != rhs->killed_live_units) {
    return lhs->killed_live_units > rhs->killed_live_units ? -1 : 1;
  }
  if (lhs->produced_live_units != rhs->produced_live_units) {
    return lhs->produced_live_units < rhs->produced_live_units ? -1 : 1;
  }
  return 0;
}

static uint32_t loom_low_schedule_candidate_positive_pressure_growth(
    const loom_low_schedule_candidate_score_t* score) {
  if (score->produced_live_units <= score->killed_live_units) {
    return 0;
  }
  const uint64_t growth = score->produced_live_units - score->killed_live_units;
  return growth > UINT32_MAX ? UINT32_MAX : (uint32_t)growth;
}

static int loom_low_schedule_compare_candidate_pressure_efficiency(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const uint32_t lhs_growth =
      loom_low_schedule_candidate_positive_pressure_growth(lhs);
  const uint32_t rhs_growth =
      loom_low_schedule_candidate_positive_pressure_growth(rhs);
  const uint32_t lhs_demand =
      lhs->pressure_demand_units != 0 ? lhs->pressure_demand_units : 1;
  const uint32_t rhs_demand =
      rhs->pressure_demand_units != 0 ? rhs->pressure_demand_units : 1;
  const uint64_t lhs_cost = (uint64_t)lhs_growth * rhs_demand;
  const uint64_t rhs_cost = (uint64_t)rhs_growth * lhs_demand;
  if (lhs_cost == rhs_cost) {
    return 0;
  }
  return lhs_cost < rhs_cost ? -1 : 1;
}

static int loom_low_schedule_compare_candidate_live_values(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const int64_t lhs_delta = (int64_t)lhs->produced_live_value_count -
                            (int64_t)lhs->killed_live_value_count;
  const int64_t rhs_delta = (int64_t)rhs->produced_live_value_count -
                            (int64_t)rhs->killed_live_value_count;
  if (lhs_delta != rhs_delta) {
    return lhs_delta < rhs_delta ? -1 : 1;
  }
  if (lhs->killed_live_value_count != rhs->killed_live_value_count) {
    return lhs->killed_live_value_count > rhs->killed_live_value_count ? -1 : 1;
  }
  if (lhs->produced_live_value_count != rhs->produced_live_value_count) {
    return lhs->produced_live_value_count < rhs->produced_live_value_count ? -1
                                                                           : 1;
  }
  return 0;
}

static bool loom_low_schedule_candidate_shortens_producer_live_range(
    const loom_low_schedule_candidate_score_t* score) {
  return score->dependency_latency_cycles != 0 &&
         score->killed_live_units > score->produced_live_units;
}

static bool loom_low_schedule_candidate_compacts_live_values(
    const loom_low_schedule_candidate_score_t* score) {
  return score->storage_relation_count != 0 &&
         score->killed_live_value_count > score->produced_live_value_count;
}

static bool loom_low_schedule_candidate_has_better_pair_affinity(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  if (lhs->pair_affinity_score != rhs->pair_affinity_score) {
    return lhs->pair_affinity_score > rhs->pair_affinity_score;
  }
  return lhs->pair_placement_option_count > rhs->pair_placement_option_count;
}

static bool loom_low_schedule_candidate_pair_affinity_differs(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  return lhs->pair_affinity_score != rhs->pair_affinity_score ||
         (lhs->pair_affinity_score != 0 &&
          lhs->pair_placement_option_count != rhs->pair_placement_option_count);
}

static loom_low_schedule_candidate_compare_mode_t
loom_low_schedule_choose_candidate_compare_mode(
    const loom_low_schedule_candidate_score_t* scores,
    iree_host_size_t score_count,
    uint32_t current_persistent_pressure_penalty) {
  if (current_persistent_pressure_penalty != 0) {
    return LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF;
  }
  for (iree_host_size_t i = 0; i < score_count; ++i) {
    if (scores[i].pressure_risk != LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE) {
      return LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF;
    }
  }
  return LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT;
}

static bool loom_low_schedule_candidate_score_less(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const int pressure_order =
      loom_low_schedule_compare_candidate_pressure(lhs, rhs);
  const int pressure_efficiency_order =
      loom_low_schedule_compare_candidate_pressure_efficiency(lhs, rhs);
  const int live_value_order =
      loom_low_schedule_compare_candidate_live_values(lhs, rhs);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    if (compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF) {
      const bool lhs_makes_pressure_progress =
          lhs->pressure_progress_kind !=
          LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE;
      const bool rhs_makes_pressure_progress =
          rhs->pressure_progress_kind !=
          LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE;
      if (lhs_makes_pressure_progress != rhs_makes_pressure_progress) {
        return lhs_makes_pressure_progress;
      }
      if (lhs->pressure_cliff_penalty != rhs->pressure_cliff_penalty) {
        return lhs->pressure_cliff_penalty < rhs->pressure_cliff_penalty;
      }
      if (pressure_efficiency_order != 0) {
        return pressure_efficiency_order < 0;
      }
      if (pressure_order != 0) {
        return pressure_order < 0;
      }
      if (lhs_makes_pressure_progress &&
          lhs->critical_path_cycles != rhs->critical_path_cycles) {
        return lhs->critical_path_cycles > rhs->critical_path_cycles;
      }
    }
    if (lhs->effective_stall_cycles != rhs->effective_stall_cycles) {
      return lhs->effective_stall_cycles < rhs->effective_stall_cycles;
    }
    if (lhs->hazard_stall_cycles != rhs->hazard_stall_cycles) {
      return lhs->hazard_stall_cycles < rhs->hazard_stall_cycles;
    }
    if (lhs->resource_stall_cycles != rhs->resource_stall_cycles) {
      return lhs->resource_stall_cycles < rhs->resource_stall_cycles;
    }
    if (lhs->data_ready_stall_cycles != rhs->data_ready_stall_cycles) {
      return lhs->data_ready_stall_cycles < rhs->data_ready_stall_cycles;
    }
    if (compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT &&
        lhs->opened_completion_latency_cycles !=
            rhs->opened_completion_latency_cycles) {
      return lhs->opened_completion_latency_cycles >
             rhs->opened_completion_latency_cycles;
    }
    if (loom_low_schedule_candidate_pair_affinity_differs(lhs, rhs)) {
      return loom_low_schedule_candidate_has_better_pair_affinity(lhs, rhs);
    }
    if (live_value_order != 0 &&
        (loom_low_schedule_candidate_compacts_live_values(lhs) ||
         loom_low_schedule_candidate_compacts_live_values(rhs)) &&
        (pressure_order == 0 || pressure_order == live_value_order)) {
      return live_value_order < 0;
    }
    if (pressure_order != 0) {
      const bool lhs_shortens =
          loom_low_schedule_candidate_shortens_producer_live_range(lhs);
      const bool rhs_shortens =
          loom_low_schedule_candidate_shortens_producer_live_range(rhs);
      if (lhs_shortens && !rhs_shortens && pressure_order < 0) {
        return true;
      }
      if (rhs_shortens && !lhs_shortens && pressure_order > 0) {
        return false;
      }
      if (lhs_shortens && rhs_shortens) {
        return pressure_order < 0;
      }
    }
    if (lhs->critical_path_cycles != rhs->critical_path_cycles) {
      return lhs->critical_path_cycles > rhs->critical_path_cycles;
    }
  }
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING) {
    if (lhs->dependency_latency_cycles != rhs->dependency_latency_cycles) {
      return lhs->dependency_latency_cycles < rhs->dependency_latency_cycles;
    }
    if (lhs->latency_cycles != rhs->latency_cycles) {
      return lhs->latency_cycles > rhs->latency_cycles;
    }
  }
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      lhs->pressure_cliff_penalty != rhs->pressure_cliff_penalty) {
    return lhs->pressure_cliff_penalty < rhs->pressure_cliff_penalty;
  }
  if (loom_low_schedule_candidate_pair_affinity_differs(lhs, rhs)) {
    return loom_low_schedule_candidate_has_better_pair_affinity(lhs, rhs);
  }
  if (pressure_efficiency_order != 0) {
    return pressure_efficiency_order < 0;
  }
  if (pressure_order != 0) {
    return pressure_order < 0;
  }
  return lhs->source_ordinal < rhs->source_ordinal;
}

void loom_low_schedule_candidate_policy_record_decision(
    loom_low_schedule_build_state_t* state, uint32_t block_index,
    uint32_t scheduled_ordinal,
    const loom_low_schedule_candidate_selection_t* selection) {
  if (!state->candidate_decisions) {
    return;
  }
  if (selection->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return;
  }
  const loom_low_schedule_candidate_score_t* chosen_score =
      &selection->chosen_score;
  const loom_low_schedule_candidate_score_t* rejected_score =
      &selection->rejected_score;
  state->candidate_decisions[state->candidate_decision_count++] =
      (loom_low_schedule_candidate_decision_t){
          .block_index = block_index,
          .scheduled_ordinal = scheduled_ordinal,
          .ready_candidate_count = selection->ready_candidate_count,
          .scored_candidate_count = selection->scored_candidate_count,
          .chosen_node = selection->chosen_node,
          .rejected_node = selection->rejected_node,
          .chosen_dependency_latency_cycles =
              chosen_score->dependency_latency_cycles,
          .chosen_latency_cycles = chosen_score->latency_cycles,
          .chosen_pair_affinity_score = chosen_score->pair_affinity_score,
          .rejected_dependency_latency_cycles =
              rejected_score->dependency_latency_cycles,
          .rejected_latency_cycles = rejected_score->latency_cycles,
          .rejected_pair_affinity_score = rejected_score->pair_affinity_score,
          .chosen_projected_live_units = chosen_score->projected_live_units,
          .chosen_killed_live_units = chosen_score->killed_live_units,
          .chosen_produced_live_units = chosen_score->produced_live_units,
          .rejected_projected_live_units = rejected_score->projected_live_units,
          .rejected_killed_live_units = rejected_score->killed_live_units,
          .rejected_produced_live_units = rejected_score->produced_live_units,
          .chosen_data_ready_stall_cycles =
              chosen_score->data_ready_stall_cycles,
          .chosen_resource_stall_cycles = chosen_score->resource_stall_cycles,
          .chosen_hazard_stall_cycles = chosen_score->hazard_stall_cycles,
          .chosen_completion_wait_cycles = chosen_score->completion_wait_cycles,
          .chosen_effective_stall_cycles = chosen_score->effective_stall_cycles,
          .chosen_bottleneck_resource_id = chosen_score->bottleneck_resource_id,
          .chosen_pressure_cliff_penalty = chosen_score->pressure_cliff_penalty,
          .chosen_pressure_cliff_source =
              loom_low_schedule_pressure_source_name(
                  state, chosen_score->pressure_cliff_source_kind,
                  chosen_score->pressure_cliff_source_id),
          .chosen_pressure_cliff_units = chosen_score->pressure_cliff_units,
          .chosen_units_until_pressure_cliff =
              chosen_score->units_until_pressure_cliff,
          .rejected_data_ready_stall_cycles =
              rejected_score->data_ready_stall_cycles,
          .rejected_resource_stall_cycles =
              rejected_score->resource_stall_cycles,
          .rejected_hazard_stall_cycles = rejected_score->hazard_stall_cycles,
          .rejected_completion_wait_cycles =
              rejected_score->completion_wait_cycles,
          .rejected_effective_stall_cycles =
              rejected_score->effective_stall_cycles,
          .rejected_bottleneck_resource_id =
              rejected_score->bottleneck_resource_id,
          .rejected_pressure_cliff_penalty =
              rejected_score->pressure_cliff_penalty,
          .rejected_pressure_cliff_source =
              loom_low_schedule_pressure_source_name(
                  state, rejected_score->pressure_cliff_source_kind,
                  rejected_score->pressure_cliff_source_id),
          .rejected_pressure_cliff_units = rejected_score->pressure_cliff_units,
          .rejected_units_until_pressure_cliff =
              rejected_score->units_until_pressure_cliff,
      };
}

static bool loom_low_schedule_add_ready_nominee(uint32_t node_index,
                                                uint32_t* nominees,
                                                uint8_t* nominee_count) {
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE) return false;
  for (uint8_t i = 0; i < *nominee_count; ++i) {
    if (nominees[i] == node_index) return false;
  }
  IREE_ASSERT_LT(*nominee_count, LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY);
  nominees[(*nominee_count)++] = node_index;
  return true;
}

static void loom_low_schedule_add_ready_view_nominees(
    const loom_low_schedule_ready_policy_t* ready_policy,
    loom_low_schedule_ready_view_t view, uint8_t search_count,
    uint8_t nominee_limit, uint32_t* nominees, uint8_t* nominee_count) {
  uint32_t view_nominees[LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY];
  const uint8_t view_nominee_count = loom_low_schedule_ready_frontier_copy_best(
      &ready_policy->frontier, view, search_count, view_nominees);
  uint8_t added_nominee_count = 0;
  for (uint8_t i = 0; i < view_nominee_count; ++i) {
    if (loom_low_schedule_add_ready_nominee(view_nominees[i], nominees,
                                            nominee_count) &&
        ++added_nominee_count == nominee_limit) {
      break;
    }
  }
}

static uint8_t loom_low_schedule_collect_source_nominees(
    const loom_low_schedule_ready_policy_t* ready_policy,
    uint32_t* out_nominees) {
  uint8_t nominee_count = 0;
  loom_low_schedule_add_ready_view_nominees(
      ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE,
      LOOM_LOW_SCHEDULE_READY_SOURCE_NOMINEE_COUNT,
      LOOM_LOW_SCHEDULE_READY_SOURCE_NOMINEE_COUNT, out_nominees,
      &nominee_count);
  return nominee_count;
}

static void loom_low_schedule_collect_recovery_nominees(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const loom_low_schedule_recovery_policy_t* recovery_policy,
    uint32_t* nominees, uint8_t* nominee_count) {
  for (uint8_t i = 0; i < recovery_policy->ready_view_count; ++i) {
    loom_low_schedule_add_ready_view_nominees(
        ready_policy, recovery_policy->ready_views[i],
        LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT,
        LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT, nominees, nominee_count);
  }
  (void)loom_low_schedule_add_ready_nominee(
      loom_low_schedule_ready_policy_pair_nominee(state, ready_policy),
      nominees, nominee_count);
}

void loom_low_schedule_candidate_policy_select(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t ready_candidate_count,
    loom_low_schedule_candidate_selection_t* out_selection) {
  IREE_ASSERT_NE(ready_candidate_count, 0u);
  *out_selection = (loom_low_schedule_candidate_selection_t){
      .chosen_node = LOOM_LOW_SCHEDULE_NODE_NONE,
      .rejected_node = LOOM_LOW_SCHEDULE_NODE_NONE,
      .ready_candidate_count = ready_candidate_count,
  };
  if (!loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
    (void)loom_low_schedule_ready_frontier_copy_best(
        &ready_policy->frontier, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE,
        /*capacity=*/1, &out_selection->chosen_node);
    IREE_ASSERT_NE(out_selection->chosen_node, LOOM_LOW_SCHEDULE_NODE_NONE);
    return;
  }

  uint32_t nominees[LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY];
  loom_low_schedule_candidate_score_t
      nominee_scores[LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY];
  uint8_t nominee_count =
      loom_low_schedule_collect_source_nominees(ready_policy, nominees);
  IREE_ASSERT_NE(nominee_count, 0);
  for (uint8_t i = 0; i < nominee_count; ++i) {
    loom_low_schedule_pressure_score_candidate(state, pressure_state,
                                               ready_policy, indegrees,
                                               nominees[i], &nominee_scores[i]);
  }
  out_selection->scored_candidate_count = nominee_count;
  const loom_low_schedule_candidate_compare_mode_t compare_mode =
      loom_low_schedule_choose_candidate_compare_mode(
          nominee_scores, nominee_count,
          pressure_state->current_persistent_pressure_penalty);
  for (uint8_t i = 0; i < nominee_count; ++i) {
    const uint32_t node_index = nominees[i];
    if (out_selection->chosen_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        loom_low_schedule_candidate_score_less(state, compare_mode,
                                               &nominee_scores[i],
                                               &out_selection->chosen_score)) {
      if (out_selection->chosen_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
        out_selection->rejected_node = out_selection->chosen_node;
        out_selection->rejected_score = out_selection->chosen_score;
      }
      out_selection->chosen_node = node_index;
      out_selection->chosen_score = nominee_scores[i];
    } else if (out_selection->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
               loom_low_schedule_candidate_score_less(
                   state, compare_mode, &nominee_scores[i],
                   &out_selection->rejected_score)) {
      out_selection->rejected_node = node_index;
      out_selection->rejected_score = nominee_scores[i];
    }
  }

  const bool recover_pressure =
      compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF;
  // Keep zero-stall selection on the bounded source-order path unless its best
  // candidate would consume a longer-latency result while ready work remains
  // outside that window. The existing recovery views can then preserve the
  // latency window without scanning the complete ready set.
  const bool recover_latency_window =
      !recover_pressure &&
      out_selection->chosen_score.effective_stall_cycles == 0 &&
      out_selection->chosen_score.dependency_latency_cycles >
          out_selection->chosen_score.latency_cycles &&
      ready_candidate_count > nominee_count;
  if (!recover_pressure &&
      out_selection->chosen_score.effective_stall_cycles == 0 &&
      !recover_latency_window) {
    return;
  }
  const uint8_t source_nominee_count = nominee_count;
  const uint16_t minimum_recovery_latency_cycles =
      recover_latency_window
          ? out_selection->chosen_score.dependency_latency_cycles
          : 0;
  const loom_low_schedule_recovery_policy_t* recovery_policy =
      &kLoomLowScheduleRecoveryPolicies[compare_mode];
  loom_low_schedule_collect_recovery_nominees(
      state, ready_policy, recovery_policy, nominees, &nominee_count);
  for (uint8_t i = source_nominee_count; i < nominee_count; ++i) {
    loom_low_schedule_pressure_score_candidate(state, pressure_state,
                                               ready_policy, indegrees,
                                               nominees[i], &nominee_scores[i]);
    ++out_selection->scored_candidate_count;
    // Latency-window recovery must either replace the entire window that the
    // source nominee would consume or immediately expose an equally long
    // descriptor without growing its live frontier. Other cheap operations let
    // critical-path ordering walk setup chains without issuing enough
    // independent latency to amortize the resulting pressure.
    const bool replenishes_non_growing_latency_window =
        nominee_scores[i].unlocked_non_growing_descriptor_latency_cycles >=
        minimum_recovery_latency_cycles;
    if (nominee_scores[i].latency_cycles < minimum_recovery_latency_cycles &&
        !replenishes_non_growing_latency_window) {
      continue;
    }
    if (iree_any_bit_set(
            recovery_policy->flags,
            LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REJECT_PRESSURE_DEBT) &&
        nominee_scores[i].pressure_risk ==
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT) {
      continue;
    }
    if (iree_any_bit_set(
            recovery_policy->flags,
            LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REQUIRE_PRESSURE_PROGRESS) &&
        nominee_scores[i].pressure_progress_kind ==
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE) {
      continue;
    }
    if (loom_low_schedule_candidate_score_less(state, compare_mode,
                                               &nominee_scores[i],
                                               &out_selection->chosen_score)) {
      out_selection->rejected_node = out_selection->chosen_node;
      out_selection->rejected_score = out_selection->chosen_score;
      out_selection->chosen_node = nominees[i];
      out_selection->chosen_score = nominee_scores[i];
    } else if (out_selection->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
               loom_low_schedule_candidate_score_less(
                   state, compare_mode, &nominee_scores[i],
                   &out_selection->rejected_score)) {
      out_selection->rejected_node = nominees[i];
      out_selection->rejected_score = nominee_scores[i];
    }
  }
}
