// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/planning_statistics.h"

#include "iree/base/internal/math.h"

static void loom_low_planning_arena_statistics_merge(
    loom_low_planning_arena_statistics_t* target,
    const loom_low_planning_arena_statistics_t* source) {
  target->used_bytes_high_water =
      iree_max(target->used_bytes_high_water, source->used_bytes_high_water);
  target->owned_bytes_high_water =
      iree_max(target->owned_bytes_high_water, source->owned_bytes_high_water);
}

void loom_low_planning_statistics_accumulate(
    loom_low_planning_statistics_t* target,
    const loom_low_planning_statistics_t* source) {
#define LOOM_LOW_PLANNING_ACCUMULATE(field) \
  target->field = iree_math_saturating_add_u64(target->field, source->field)
  const bool target_has_residency_validation =
      target->residency.validation_count != 0;
  target->flags &= source->flags;
  LOOM_LOW_PLANNING_ACCUMULATE(frame_build_count);
  LOOM_LOW_PLANNING_ACCUMULATE(allocation_run_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.iteration_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.diagnostic_replay_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.spill_traffic_lowering_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.rematerialized_operand_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.live_range_split_operand_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.pair_replication_attempt_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.pair_replication_edit_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.pair_replication_rejection_count);
  LOOM_LOW_PLANNING_ACCUMULATE(repair.spill_materialization_batch_count);
  LOOM_LOW_PLANNING_ACCUMULATE(residency.contract_count);
  LOOM_LOW_PLANNING_ACCUMULATE(residency.candidate_count);
  target->residency.maximum_projected_required_tier =
      iree_max(target->residency.maximum_projected_required_tier,
               source->residency.maximum_projected_required_tier);
  if (source->residency.validation_count != 0) {
    target->residency.minimum_observed_allocated_tier =
        target_has_residency_validation
            ? iree_min(target->residency.minimum_observed_allocated_tier,
                       source->residency.minimum_observed_allocated_tier)
            : source->residency.minimum_observed_allocated_tier;
  }
  target->residency.maximum_observed_tier_shortfall =
      iree_max(target->residency.maximum_observed_tier_shortfall,
               source->residency.maximum_observed_tier_shortfall);
  LOOM_LOW_PLANNING_ACCUMULATE(residency.validation_count);
  LOOM_LOW_PLANNING_ACCUMULATE(residency.repair_attempt_count);
  LOOM_LOW_PLANNING_ACCUMULATE(residency.repair_count);
  LOOM_LOW_PLANNING_ACCUMULATE(residency.failure_count);
  loom_low_planning_arena_statistics_merge(&target->memory.frame_arena,
                                           &source->memory.frame_arena);
  loom_low_planning_arena_statistics_merge(&target->memory.repair_arena,
                                           &source->memory.repair_arena);
  loom_low_planning_arena_statistics_merge(&target->memory.scratch_arena,
                                           &source->memory.scratch_arena);
  LOOM_LOW_PLANNING_ACCUMULATE(memory.block_system_allocation_count);
  LOOM_LOW_PLANNING_ACCUMULATE(memory.block_system_allocation_bytes);
  LOOM_LOW_PLANNING_ACCUMULATE(memory.oversized_allocation_count);
  LOOM_LOW_PLANNING_ACCUMULATE(memory.oversized_allocation_bytes);
#undef LOOM_LOW_PLANNING_ACCUMULATE
}
