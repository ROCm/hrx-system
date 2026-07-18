// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scan-free work and memory statistics for target-low planning.

#ifndef LOOM_CODEGEN_LOW_PLANNING_STATISTICS_H_
#define LOOM_CODEGEN_LOW_PLANNING_STATISTICS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_planning_statistics_flag_bits_e {
  // Fresh block-pool system allocation counters are populated.
  LOOM_LOW_PLANNING_STATISTICS_FLAG_SYSTEM_ALLOCATIONS = 1u << 0,
};
typedef uint32_t loom_low_planning_statistics_flags_t;

// High-water allocation deltas for one arena lifetime.
typedef struct loom_low_planning_arena_statistics_t {
  // Maximum used bytes above the arena's planning-entry baseline.
  uint64_t used_bytes_high_water;
  // Maximum owned bytes above the arena's planning-entry baseline.
  uint64_t owned_bytes_high_water;
} loom_low_planning_arena_statistics_t;

// IR-changing work performed while converging on an emission frame.
typedef struct loom_low_planning_repair_statistics_t {
  // Number of complete repair iterations performed.
  uint64_t iteration_count;
  // Number of diagnostic frame builds replayed after a rejected silent build.
  uint64_t diagnostic_replay_count;
  // Number of spill-traffic lowering runs.
  uint64_t spill_traffic_lowering_count;
  // Number of operands rewritten by rematerialization.
  uint64_t rematerialized_operand_count;
  // Number of operands rewritten by live-range splitting.
  uint64_t live_range_split_operand_count;
  // Number of pair-replication plans attempted.
  uint64_t pair_replication_attempt_count;
  // Number of pair-replication edits proposed across all attempts.
  uint64_t pair_replication_edit_count;
  // Number of pair-replication plans rolled back.
  uint64_t pair_replication_rejection_count;
  // Number of spill-materialization batches applied.
  uint64_t spill_materialization_batch_count;
} loom_low_planning_repair_statistics_t;

// Allocation economics across one target-low planning invocation.
typedef struct loom_low_planning_memory_statistics_t {
  // Caller-owned arena retaining the accepted emission frame.
  loom_low_planning_arena_statistics_t frame_arena;
  // Arena retaining repair plans across planning iterations.
  loom_low_planning_arena_statistics_t repair_arena;
  // Arena reset between planning phases and iterations.
  loom_low_planning_arena_statistics_t scratch_arena;
  // Fresh fixed-size blocks allocated from the system.
  uint64_t block_system_allocation_count;
  // Total bytes of fresh fixed-size blocks allocated from the system.
  uint64_t block_system_allocation_bytes;
  // Fresh oversized arena allocations made from the system.
  uint64_t oversized_allocation_count;
  // Total bytes of fresh oversized arena allocations made from the system.
  uint64_t oversized_allocation_bytes;
} loom_low_planning_memory_statistics_t;

// Coarse target-low planning statistics collected without table rescans.
//
// Final schedule and allocation cardinalities remain in their existing compile
// report groups. This record only captures repeated work and arena economics
// that are otherwise invisible after the accepted frame replaces intermediate
// attempts.
typedef struct loom_low_planning_statistics_t {
  // Optional statistic categories populated in this record.
  loom_low_planning_statistics_flags_t flags;
  // Number of complete schedule-frame builds attempted.
  uint64_t frame_build_count;
  // Number of allocation runs reached after successful scheduling.
  uint64_t allocation_run_count;
  // IR-changing work performed while converging on the final frame.
  loom_low_planning_repair_statistics_t repair;
  // Arena and system allocation economics for the planning invocation.
  loom_low_planning_memory_statistics_t memory;
} loom_low_planning_statistics_t;

// Accumulates |source| into an already-populated |target|. Work and system
// allocation counts sum while per-arena high-water marks take their maximum.
// Optional categories remain available only when present in both records.
void loom_low_planning_statistics_accumulate(
    loom_low_planning_statistics_t* target,
    const loom_low_planning_statistics_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PLANNING_STATISTICS_H_
