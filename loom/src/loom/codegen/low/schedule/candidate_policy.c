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
#define LOOM_LOW_SCHEDULE_READY_CRITICAL_PATH_BAND_CYCLES 1

iree_status_t loom_low_schedule_candidate_policy_initialize(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_policy_t* out_policy) {
  *out_policy = (loom_low_schedule_candidate_policy_t){0};
  if (!loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
    return iree_ok_status();
  }
  out_policy->register_debt_word_count =
      (state->target.descriptor_set->reg_class_count + 63) / 64;
  return iree_arena_allocate_array(state->scratch_arena,
                                   LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY *
                                       out_policy->register_debt_word_count,
                                   sizeof(*out_policy->nominee_register_debts),
                                   (void**)&out_policy->nominee_register_debts);
}

static loom_low_schedule_register_debt_word_t*
loom_low_schedule_candidate_policy_reset_register_debt(
    const loom_low_schedule_candidate_policy_t* policy, uint8_t score_index) {
  if (policy->register_debt_word_count == 0) {
    return NULL;
  }
  loom_low_schedule_register_debt_word_t* row =
      &policy->nominee_register_debts[score_index *
                                      policy->register_debt_word_count];
  memset(row, 0, policy->register_debt_word_count * sizeof(*row));
  return row;
}

static bool loom_low_schedule_candidate_opens_other_register_debt(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_candidate_score_t* required,
    const loom_low_schedule_candidate_score_t* alternative) {
  const iree_host_size_t word_count =
      (state->target.descriptor_set->reg_class_count + 63) / 64;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    if ((alternative->register_debt[i].banks &
         ~required->register_debt[i].required_banks) != 0) {
      return true;
    }
  }
  return false;
}

enum loom_low_schedule_recovery_policy_flag_bits_e {
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REJECT_PRESSURE_DEBT = 1u << 0,
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REQUIRE_PRESSURE_PROGRESS = 1u << 1,
};

typedef struct loom_low_schedule_recovery_policy_t {
  // Ready views contributing additional recovery nominees.
  loom_low_schedule_ready_view_t ready_views[3];
  // Number of populated entries in ready_views.
  uint8_t ready_view_count;
  // loom_low_schedule_recovery_policy_flag_bits_e bits.
  uint8_t flags;
} loom_low_schedule_recovery_policy_t;

enum {
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_DEFAULT = 0,
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_PRESSURE = 1,
  LOOM_LOW_SCHEDULE_RECOVERY_POLICY_COUNT = 2,
};

static const loom_low_schedule_recovery_policy_t
    kLoomLowScheduleRecoveryPolicies[LOOM_LOW_SCHEDULE_RECOVERY_POLICY_COUNT] = {
        [LOOM_LOW_SCHEDULE_RECOVERY_POLICY_DEFAULT] =
            {
                .ready_views = {LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE},
                .ready_view_count = 1,
                .flags =
                    LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REJECT_PRESSURE_DEBT,
            },
        [LOOM_LOW_SCHEDULE_RECOVERY_POLICY_PRESSURE] =
            {
                .ready_views =
                    {
                        LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE,
                        LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE,
                        LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE,
                    },
                .ready_view_count = 3,
                .flags =
                    LOOM_LOW_SCHEDULE_RECOVERY_POLICY_FLAG_REQUIRE_PRESSURE_PROGRESS,
            },
};

// A pinned completion may need several setup values before any consumer can
// release its inputs. Advance that exact transaction while the setup fits;
// cone membership alone cannot justify opening another register debt.
static bool loom_low_schedule_candidate_advances_pinned_completion(
    const loom_low_schedule_candidate_score_t* score) {
  return score->active_unspillable_transaction_final_capacity != UINT32_MAX &&
         !iree_any_bit_set(
             score->flags,
             LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT |
                 LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY);
}

// Before spill materialization, allocation repair can retain a private value
// needed to reach an unspillable consumer. Advance that placement while its
// bounded downstream demand remains outside the near-cliff region. Once spill
// traffic exists the scorer withholds the retained-placement flag because
// mandatory helpers can still intervene.
static bool loom_low_schedule_candidate_advances_retained_unspillable_value(
    const loom_low_schedule_candidate_score_t* score) {
  const uint16_t retained_unspillable_flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_RETAINS_PRE_SPILL_PLACEMENT;
  return score->pressure_risk == LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE &&
         score->pressure_cliff_penalty == 0 &&
         iree_all_bits_set(score->flags, retained_unspillable_flags) &&
         !iree_any_bit_set(
             score->flags,
             LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_REMATERIALIZABLE_LEAF);
}

// Returns true when |score| establishes a new live storage value without
// reducing live units, exposing a storage step or descriptor, advancing a
// retained unspillable value while it still fits, advancing a feasible pinned
// completion, or releasing inputs along another completion.
// Scheduling such setup early only transfers or grows liveness in its
// destination register class and can hold scarce physical locations across
// unrelated work. Alias establishment and storage compaction produce no value
// or reduce live units, respectively, and remain immediately actionable.
static bool loom_low_schedule_candidate_defers_storage_setup(
    const loom_low_schedule_candidate_score_t* score) {
  if (!iree_any_bit_set(score->flags,
                        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_STORAGE_SETUP)) {
    return false;
  }
  const uint16_t actionable_flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  const bool advances_retained_unspillable_value =
      loom_low_schedule_candidate_advances_retained_unspillable_value(score);
  const bool advances_completion =
      (iree_any_bit_set(
           score->flags,
           LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_REMATERIALIZABLE_LEAF) &&
       loom_low_schedule_candidate_advances_pinned_completion(score)) ||
      (score->killed_live_units != 0 &&
       iree_any_bit_set(
           score->flags,
           LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION));
  return score->produced_live_value_count != 0 &&
         score->killed_live_units <= score->produced_live_units &&
         !iree_any_bit_set(score->flags, actionable_flags) &&
         !advances_retained_unspillable_value && !advances_completion;
}

// A pinned completion can require several input-free setup operations before
// its consumer is ready. Preserve that progress while their physical bank has
// headroom. Once setup grows its bank beyond capacity, cone membership alone is
// insufficient: memory or target-state ordering can still block its consumer
// behind earlier work.
static bool loom_low_schedule_candidate_defers_rematerializable_leaf(
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* score) {
  if (compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT ||
      loom_low_schedule_candidate_advances_pinned_completion(score)) {
    return false;
  }
  uint16_t actionable_flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXACT_PACKING_COMPLETION;
  // An input-free leaf can expose another setup without making its final
  // consumer ready. Completion-path membership does not justify opening that
  // live range when it incurs debt; retain its demand-driven placement.
  if (score->pressure_risk != LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT ||
      !iree_any_bit_set(score->flags,
                        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_STORAGE_SETUP)) {
    actionable_flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  }
  return score->produced_live_value_count != 0 &&
         score->killed_live_units == 0 &&
         iree_any_bit_set(
             score->flags,
             LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_REMATERIALIZABLE_LEAF) &&
         !iree_any_bit_set(score->flags, actionable_flags);
}

// A short, replayable register move can use an otherwise idle issue cycle, but
// it has no latency to hide while its sole consumer remains blocked. Prefer
// ready work that keeps the destination register lifetime closed.
static bool loom_low_schedule_candidate_defers_replayable_register_move(
    const loom_low_schedule_candidate_score_t* score) {
  const uint16_t actionable_flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  return score->produced_live_value_count != 0 &&
         iree_any_bit_set(
             score->flags,
             LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_SHORT_REPLAYABLE_REGISTER_MOVE) &&
         !iree_any_bit_set(score->flags, actionable_flags);
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline bool
loom_low_schedule_candidate_defers_materialization(
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* score) {
  return loom_low_schedule_candidate_defers_storage_setup(score) ||
         loom_low_schedule_candidate_defers_rematerializable_leaf(compare_mode,
                                                                  score);
}

static bool loom_low_schedule_candidate_unlocks_descriptor(
    const loom_low_schedule_candidate_score_t* score) {
  return iree_any_bit_set(score->flags,
                          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR);
}

static bool loom_low_schedule_candidate_exceeds_unspillable_capacity(
    const loom_low_schedule_candidate_score_t* score) {
  return iree_any_bit_set(
      score->flags,
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY);
}

static uint32_t loom_low_schedule_candidate_unspillable_completion_capacity(
    const loom_low_schedule_candidate_score_t* score) {
  return iree_min(score->active_unspillable_completion_capacity,
                  score->opened_unspillable_completion_capacity);
}

// Direct storage progress and setup that exposes a storage descriptor are
// equivalent at this decision point.
static bool loom_low_schedule_candidate_advances_storage(
    const loom_low_schedule_candidate_score_t* score) {
  const uint16_t descriptor_setup_flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_STORAGE_SETUP |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR;
  return iree_any_bit_set(score->flags,
                          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE) ||
         iree_all_bits_set(score->flags, descriptor_setup_flags);
}

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

// Returns true when the candidate can preserve a pending placement pair
// without waiting on a data or hazard prerequisite. Resource occupancy is not
// included: pair affinity describes an opportunity to replace two adjacent
// packets with one target packet, so charging the second component's ordinary
// standalone issue resource before considering the pair defeats the hint.
static bool loom_low_schedule_candidate_has_ready_pair_affinity(
    const loom_low_schedule_candidate_score_t* score) {
  return score->pair_affinity_score != 0 &&
         score->data_ready_stall_cycles == 0 && score->hazard_stall_cycles == 0;
}

static loom_low_schedule_candidate_compare_mode_t
loom_low_schedule_choose_candidate_compare_mode(
    const loom_low_schedule_candidate_score_t* scores,
    iree_host_size_t score_count,
    uint32_t current_persistent_pressure_penalty) {
  bool has_pressure_risk = current_persistent_pressure_penalty != 0;
  for (iree_host_size_t i = 0; i < score_count; ++i) {
    if (iree_any_bit_set(
            scores[i].flags,
            LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_NEEDS_COMPLETION_RECOVERY)) {
      return LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION;
    }
    if (scores[i].pressure_risk != LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE) {
      has_pressure_risk = true;
    }
  }
  return has_pressure_risk ? LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF
                           : LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT;
}

static bool loom_low_schedule_candidate_score_less(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const bool lhs_exceeds_unspillable_capacity =
      loom_low_schedule_candidate_exceeds_unspillable_capacity(lhs);
  const bool rhs_exceeds_unspillable_capacity =
      loom_low_schedule_candidate_exceeds_unspillable_capacity(rhs);
  if (lhs_exceeds_unspillable_capacity != rhs_exceeds_unspillable_capacity) {
    return !lhs_exceeds_unspillable_capacity;
  }
  const bool lhs_defers_materialization =
      loom_low_schedule_candidate_defers_materialization(compare_mode, lhs);
  const bool rhs_defers_materialization =
      loom_low_schedule_candidate_defers_materialization(compare_mode, rhs);
  // A setup whose consumer is blocked cannot relieve the pressure blocking
  // that consumer. Opening its destination first can occupy a singleton
  // register across earlier uses required to make the consumer ready.
  if (lhs_defers_materialization != rhs_defers_materialization) {
    return !lhs_defers_materialization;
  }
  if (iree_any_bit_set(
          lhs->flags | rhs->flags,
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_SHORT_REPLAYABLE_REGISTER_MOVE)) {
    const bool lhs_defers_register_move =
        loom_low_schedule_candidate_defers_replayable_register_move(lhs);
    const bool rhs_defers_register_move =
        loom_low_schedule_candidate_defers_replayable_register_move(rhs);
    if (lhs_defers_register_move != rhs_defers_register_move) {
      const loom_low_schedule_candidate_score_t* deferred =
          lhs_defers_register_move ? lhs : rhs;
      const loom_low_schedule_candidate_score_t* alternative =
          lhs_defers_register_move ? rhs : lhs;
      // Keep the move available to fill a real issue gap. When another
      // candidate can issue no later without opening more aggregate register
      // state, opening the move destination buys no latency and only lengthens
      // its lifetime.
      if (alternative->effective_stall_cycles <=
              deferred->effective_stall_cycles &&
          alternative->projected_live_units <= deferred->projected_live_units &&
          alternative->pressure_cliff_penalty <=
              deferred->pressure_cliff_penalty) {
        return !lhs_defers_register_move;
      }
    }
  }
  // A ready exact packing completion cannot wait for another bank's release:
  // that release may consume the last slot needed to finish the packing chain.
  // Other candidates sharing a full unspillable domain also compete by ready
  // progress, subject to aggregate handoff and spill-traffic ordering.
  if (loom_low_schedule_candidate_policy_uses_ready_progress_order(
          state, compare_mode, lhs, rhs)) {
    return lhs->effective_stall_cycles < rhs->effective_stall_cycles;
  }
  // Under pressure, an unspillable path with the same risk and no greater
  // physical cliff cost already owns its live footprint. Advance a neutral
  // storage handoff when it issues no later than the alternative so that
  // footprint can reach its consumer and drain. A zero-unit setup owns no
  // footprint yet, so it only wins when it fills an actual issue gap;
  // otherwise running setup early can overlap later storage lifetimes without
  // making room.
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT &&
      lhs->pressure_risk == rhs->pressure_risk &&
      iree_any_bit_set(
          lhs->flags,
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION) &&
      iree_any_bit_set(
          rhs->flags,
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION)) {
    const bool lhs_advances_storage =
        loom_low_schedule_candidate_advances_storage(lhs);
    const bool rhs_advances_storage =
        loom_low_schedule_candidate_advances_storage(rhs);
    if (lhs_advances_storage != rhs_advances_storage) {
      const loom_low_schedule_candidate_score_t* storage_candidate =
          lhs_advances_storage ? lhs : rhs;
      const loom_low_schedule_candidate_score_t* alternative =
          lhs_advances_storage ? rhs : lhs;
      const bool fills_issue_gap = storage_candidate->effective_stall_cycles <
                                   alternative->effective_stall_cycles;
      if (storage_candidate->produced_live_units ==
              storage_candidate->killed_live_units &&
          storage_candidate->pressure_cliff_penalty <=
              alternative->pressure_cliff_penalty &&
          storage_candidate->effective_stall_cycles <=
              alternative->effective_stall_cycles &&
          (storage_candidate->produced_live_units != 0 || fills_issue_gap)) {
        return lhs_advances_storage;
      }
    }
  }
  // Once hard-capacity feasibility ties, advance the tighter full bank's
  // pending completion. Waiting until both candidates exceed capacity lets
  // other chains consume the temporary storage that completion still needs.
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT) {
    if (lhs->active_unspillable_transaction_final_capacity !=
        rhs->active_unspillable_transaction_final_capacity) {
      return lhs->active_unspillable_transaction_final_capacity <
             rhs->active_unspillable_transaction_final_capacity;
    }
    const uint32_t lhs_completion_capacity =
        loom_low_schedule_candidate_unspillable_completion_capacity(lhs);
    const uint32_t rhs_completion_capacity =
        loom_low_schedule_candidate_unspillable_completion_capacity(rhs);
    if (lhs_completion_capacity != rhs_completion_capacity) {
      return lhs_completion_capacity < rhs_completion_capacity;
    }
  }
  const bool lhs_advances_constrained_completion = iree_any_bit_set(
      lhs->flags,
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION);
  const bool rhs_advances_constrained_completion = iree_any_bit_set(
      rhs->flags,
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT &&
      lhs_advances_constrained_completion !=
          rhs_advances_constrained_completion) {
    return lhs_advances_constrained_completion;
  }
  if (lhs_defers_materialization) {
    const bool lhs_unlocks_descriptor =
        loom_low_schedule_candidate_unlocks_descriptor(lhs);
    const bool rhs_unlocks_descriptor =
        loom_low_schedule_candidate_unlocks_descriptor(rhs);
    if (lhs_unlocks_descriptor != rhs_unlocks_descriptor) {
      return lhs_unlocks_descriptor;
    }
  }
  // Prefer repairable pressure over new required lifetimes within the same
  // banks. Opening debt in another bank is not relief: repeatedly postponing
  // a required reload can otherwise spill all of its consumers' setup there.
  // Cross-bank alternatives keep the ordinary pressure/cost comparison.
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT) {
    const bool lhs_opens_required_debt = iree_any_bit_set(
        lhs->flags,
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REQUIRED_REGISTER_DEBT);
    const bool rhs_opens_required_debt = iree_any_bit_set(
        rhs->flags,
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REQUIRED_REGISTER_DEBT);
    if (lhs_opens_required_debt != rhs_opens_required_debt) {
      const loom_low_schedule_candidate_score_t* required =
          lhs_opens_required_debt ? lhs : rhs;
      const loom_low_schedule_candidate_score_t* alternative =
          lhs_opens_required_debt ? rhs : lhs;
      if (!loom_low_schedule_candidate_opens_other_register_debt(
              state, required, alternative)) {
        return !lhs_opens_required_debt;
      }
    }
  }
  if (lhs_defers_materialization &&
      lhs->pressure_cliff_penalty != rhs->pressure_cliff_penalty) {
    return lhs->pressure_cliff_penalty < rhs->pressure_cliff_penalty;
  }
  const int pressure_order =
      loom_low_schedule_compare_candidate_pressure(lhs, rhs);
  const int pressure_efficiency_order =
      loom_low_schedule_compare_candidate_pressure_efficiency(lhs, rhs);
  const int live_value_order =
      loom_low_schedule_compare_candidate_live_values(lhs, rhs);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    if (compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT) {
      const bool lhs_makes_pressure_progress =
          lhs->pressure_progress_kind !=
          LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE;
      const bool rhs_makes_pressure_progress =
          rhs->pressure_progress_kind !=
          LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE;
      // Net live-unit reduction can exchange space in one register bank for
      // debt in another. Compare physical headroom before aggregate progress
      // when both candidates perform register work. Operations without any
      // register activity retain the preference for actionable progress.
      if ((lhs->killed_live_value_count != 0 ||
           lhs->produced_live_value_count != 0) &&
          (rhs->killed_live_value_count != 0 ||
           rhs->produced_live_value_count != 0) &&
          lhs->pressure_cliff_penalty != rhs->pressure_cliff_penalty) {
        return lhs->pressure_cliff_penalty < rhs->pressure_cliff_penalty;
      }
      // Keep equally viable constrained transactions in source order without
      // letting age override a physical register-bank cliff. Selected packing
      // and unspillable completions were compared by identity above.
      if (lhs_advances_constrained_completion &&
          rhs_advances_constrained_completion &&
          lhs->source_ordinal != rhs->source_ordinal) {
        return lhs->source_ordinal < rhs->source_ordinal;
      }
      // Pressure relief is not a license to leave an issue slot empty.
      // Spillable work that remains below every bounded pressure horizon can
      // run while recovery waits for its inputs. Unspillable activation stays
      // under the completion policy above: its downstream ownership cannot be
      // recovered after an optimistic setup choice. Preserve pair-placement
      // opportunities here because neither component's standalone stall
      // represents their combined issue cost.
      if (compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF &&
          lhs->pressure_risk == LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE &&
          rhs->pressure_risk == LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE &&
          lhs->pressure_cliff_penalty == rhs->pressure_cliff_penalty &&
          !iree_any_bit_set(
              lhs->flags | rhs->flags,
              LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION) &&
          !loom_low_schedule_candidate_pair_affinity_differs(lhs, rhs) &&
          lhs->effective_stall_cycles != rhs->effective_stall_cycles) {
        return lhs->effective_stall_cycles < rhs->effective_stall_cycles;
      }
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
    const bool lhs_has_ready_pair_affinity =
        loom_low_schedule_candidate_has_ready_pair_affinity(lhs);
    const bool rhs_has_ready_pair_affinity =
        loom_low_schedule_candidate_has_ready_pair_affinity(rhs);
    if (lhs_has_ready_pair_affinity != rhs_has_ready_pair_affinity) {
      return lhs_has_ready_pair_affinity;
    }
    if (lhs_has_ready_pair_affinity &&
        loom_low_schedule_candidate_pair_affinity_differs(lhs, rhs)) {
      return loom_low_schedule_candidate_has_better_pair_affinity(lhs, rhs);
    }
    if (lhs->effective_stall_cycles != rhs->effective_stall_cycles) {
      return lhs->effective_stall_cycles < rhs->effective_stall_cycles;
    }
    // Pressure-equivalent candidates that cannot yet retire storage have no
    // recovery benefit to justify opening a later live range. Preserve source
    // order once their actual issue delay also ties; the split between data
    // and resource stalls does not change when either candidate can issue.
    if (compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT &&
        lhs->pressure_progress_kind ==
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE &&
        rhs->pressure_progress_kind ==
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE &&
        pressure_efficiency_order == 0 && pressure_order == 0 &&
        lhs->source_ordinal != rhs->source_ordinal) {
      return lhs->source_ordinal < rhs->source_ordinal;
    }
    // When candidates reach the same issue cycle, keep work on the longer
    // dependency path ahead of the incidental split between data, hazard, and
    // resource stalls. Preserve that split for paths reaching unspillable
    // storage: opening another hard-capacity chain cannot be repaired by
    // spilling if its later lifetimes become infeasible.
    const bool has_unspillable_activation = iree_any_bit_set(
        lhs->flags | rhs->flags,
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION);
    if (!has_unspillable_activation &&
        lhs->critical_path_cycles != rhs->critical_path_cycles) {
      return lhs->critical_path_cycles > rhs->critical_path_cycles;
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
          .issue_cycle = state->nodes[selection->chosen_node].issue_cycle,
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
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return false;
  }
  for (uint8_t i = 0; i < *nominee_count; ++i) {
    if (nominees[i] == node_index) {
      return false;
    }
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
    uint8_t nominee_capacity, uint32_t* nominees, uint8_t* nominee_count) {
  for (uint8_t i = 0; i < recovery_policy->ready_view_count; ++i) {
    // Keep one slot for the pair nominee within the fixed scoring budget.
    const uint8_t nominee_limit =
        iree_min(LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT,
                 nominee_capacity - 1 - *nominee_count);
    if (nominee_limit == 0) {
      break;
    }
    loom_low_schedule_add_ready_view_nominees(
        ready_policy, recovery_policy->ready_views[i],
        LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT, nominee_limit, nominees,
        nominee_count);
  }
  (void)loom_low_schedule_add_ready_nominee(
      loom_low_schedule_ready_policy_pair_nominee(state, ready_policy),
      nominees, nominee_count);
}

void loom_low_schedule_candidate_policy_select(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_candidate_policy_t* policy,
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
  const uint8_t source_view_nominee_count =
      loom_low_schedule_collect_source_nominees(ready_policy, nominees);
  uint8_t nominee_count = source_view_nominee_count;
  uint32_t graph_nominee = LOOM_LOW_SCHEDULE_NODE_NONE;
  if (ready_candidate_count > nominee_count) {
    // Add the best graph-ranked candidate to the bounded source nominees so
    // equivalent dependency graphs do not select entirely from source order.
    loom_low_schedule_add_ready_view_nominees(
        ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE,
        /*search_count=*/1, /*nominee_limit=*/1, nominees, &nominee_count);
    if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
      const uint8_t before_graph_count = nominee_count;
      loom_low_schedule_add_ready_view_nominees(
          ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_GRAPH,
          /*search_count=*/1, /*nominee_limit=*/1, nominees, &nominee_count);
      if (nominee_count > before_graph_count) {
        graph_nominee = nominees[before_graph_count];
      }
    }
  }
  IREE_ASSERT_NE(nominee_count, 0);
  for (uint8_t i = 0; i < nominee_count; ++i) {
    loom_low_schedule_pressure_score_candidate(
        state, pressure_state, ready_policy, indegrees, nominees[i],
        loom_low_schedule_candidate_policy_reset_register_debt(policy, i),
        &nominee_scores[i]);
  }
  out_selection->scored_candidate_count = nominee_count;
  uint8_t rejected_nominee_count = 0;
  if (nominee_count > source_view_nominee_count) {
    // Graph ranking repairs source-window boundaries only within the current
    // critical-path band. A growing path can remain graph ranked at each
    // successive node, so it must also have enough modeled headroom for its
    // downstream demand. Pressure and latency recovery below admit risky
    // growing work when it makes concrete progress.
    uint32_t maximum_source_critical_path_cycles = 0;
    for (uint8_t i = 0; i < source_view_nominee_count; ++i) {
      maximum_source_critical_path_cycles =
          iree_max(maximum_source_critical_path_cycles,
                   nominee_scores[i].critical_path_cycles);
    }
    uint8_t admitted_count = source_view_nominee_count;
    for (uint8_t i = source_view_nominee_count; i < nominee_count; ++i) {
      const loom_low_schedule_candidate_score_t* nominee_score =
          &nominee_scores[i];
      const bool grows_live_frontier =
          nominee_score->produced_live_units > nominee_score->killed_live_units;
      const bool exceeds_critical_path_band =
          nominee_score->critical_path_cycles >
              maximum_source_critical_path_cycles &&
          nominee_score->critical_path_cycles -
                  maximum_source_critical_path_cycles >
              LOOM_LOW_SCHEDULE_READY_CRITICAL_PATH_BAND_CYCLES;
      if (grows_live_frontier && (nominees[i] == graph_nominee ||
                                  nominee_score->pressure_risk !=
                                      LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE ||
                                  exceeds_critical_path_band)) {
        // Keep the two possible rejected ordinary scores beyond recovery's
        // array budget. A recovery view may nominate the same node under a
        // different admission rule; its exact score has not changed.
        const uint8_t cached_index = LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY -
                                     1 - rejected_nominee_count++;
        nominees[cached_index] = nominees[i];
        nominee_scores[cached_index] = nominee_scores[i];
        continue;
      }
      nominees[admitted_count] = nominees[i];
      nominee_scores[admitted_count++] = nominee_scores[i];
    }
    nominee_count = admitted_count;
  }
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
      compare_mode != LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT;
  const bool recover_deferred_materialization =
      loom_low_schedule_candidate_defers_materialization(
          compare_mode, &out_selection->chosen_score) &&
      ready_candidate_count > nominee_count;
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
      !recover_latency_window && !recover_deferred_materialization) {
    return;
  }
  const uint8_t source_nominee_count = nominee_count;
  const loom_low_schedule_candidate_score_t latency_window_source_score =
      out_selection->chosen_score;
  const uint16_t minimum_recovery_latency_cycles =
      recover_latency_window
          ? out_selection->chosen_score.dependency_latency_cycles
          : 0;
  const loom_low_schedule_recovery_policy_t* recovery_policy =
      &kLoomLowScheduleRecoveryPolicies
          [compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT
               ? LOOM_LOW_SCHEDULE_RECOVERY_POLICY_DEFAULT
               : LOOM_LOW_SCHEDULE_RECOVERY_POLICY_PRESSURE];
  // Admission rejects candidates after scoring. Their work still consumes
  // the fixed scoring budget even though their array slots can be reused.
  const uint8_t recovery_nominee_capacity =
      LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY - rejected_nominee_count;
  loom_low_schedule_collect_recovery_nominees(
      state, ready_policy, recovery_policy, recovery_nominee_capacity, nominees,
      &nominee_count);
  for (uint8_t i = source_nominee_count; i < nominee_count; ++i) {
    uint8_t cached_index = recovery_nominee_capacity;
    while (cached_index < LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY &&
           nominees[cached_index] != nominees[i]) {
      ++cached_index;
    }
    if (cached_index < LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY) {
      nominee_scores[i] = nominee_scores[cached_index];
    } else {
      // Scores move when nominees are compacted or cached. Their debt rows
      // remain owned by the original scoring ordinal until selection ends.
      loom_low_schedule_pressure_score_candidate(
          state, pressure_state, ready_policy, indegrees, nominees[i],
          loom_low_schedule_candidate_policy_reset_register_debt(
              policy, out_selection->scored_candidate_count),
          &nominee_scores[i]);
      ++out_selection->scored_candidate_count;
    }
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
    if (recover_latency_window &&
        loom_low_schedule_compare_candidate_pressure(
            &nominee_scores[i], &latency_window_source_score) > 0) {
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
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE &&
        !loom_low_schedule_candidate_defers_materialization(
            compare_mode, &out_selection->chosen_score) &&
        !(loom_low_schedule_candidate_exceeds_unspillable_capacity(
              &out_selection->chosen_score) &&
          !loom_low_schedule_candidate_exceeds_unspillable_capacity(
              &nominee_scores[i]))) {
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
