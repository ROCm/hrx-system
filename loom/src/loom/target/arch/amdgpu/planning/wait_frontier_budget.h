// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static work admission for precise AMDGPU cross-block wait frontiers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_FRONTIER_BUDGET_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_FRONTIER_BUDGET_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ACCESS_COUNT UINT64_C(344)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRE_ADMISSION_VISITS \
  UINT64_C(131072)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COUNTER_BITS UINT64_C(2048)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRODUCER_SCANS UINT64_C(131072)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALIAS_CALLS UINT64_C(65536)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALIAS_BIT_VISITS UINT64_C(65536)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_TERM_MERGES UINT64_C(4194304)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_WORKLIST_POPS UINT64_C(131072)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_EDGE_EVALUATIONS UINT64_C(131072)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FORWARD_WORD_VISITS \
  UINT64_C(4194304)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FILTERED_ACCESS_VISITS \
  UINT64_C(2097152)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FILTERED_BIT_VISITS \
  UINT64_C(2097152)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COLLAPSE_ACCESS_VISITS \
  UINT64_C(262144)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COLLAPSE_BIT_VISITS \
  UINT64_C(262144)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_LOCAL_DRAIN_ACCESS_VISITS \
  UINT64_C(131072)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_LOCAL_DRAIN_BIT_VISITS \
  UINT64_C(131072)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_WORD_VISITS \
  UINT64_C(4096)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_COLLAPSE_VISITS \
  UINT64_C(512)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_COLLAPSE_BIT_VISITS \
  UINT64_C(512)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_DYNAMIC_DRAIN_BIT_VISITS \
  UINT64_C(262144)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_EFFECTIVE_TERMS UINT64_C(64)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALLOCATION_BYTES \
  UINT64_C(64 * 1024 * 1024)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MEMORY_SPACE_COUNT UINT64_C(6)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_XCNT_GROUP_COUNT UINT64_C(2)

// Profiling-derived additive admission score in millionths of a nanosecond.
// The sealed raw allowance is
//   (25000000 - 21641261.81780746) / 1.25 = 2686990.545754032 ns.
// This identity applies the safety factor exactly once; the modeled score must
// not be multiplied by 1.25 again. Positive costs are rounded upward and the
// raw allowance is rounded downward so integer admission is conservative.
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_SCORE_SCALE UINT64_C(1000000)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_SCORE_SCALED \
  UINT64_C(2686990545754)

// Query and producer-completion structure owned by wait-plan traversal.
typedef struct loom_amdgpu_wait_frontier_precise_runtime_bounds_t {
  bool arithmetic_valid;
  uint64_t barrier_query_count;
  uint64_t program_exit_query_count;
  uint64_t producer_completion_call_count;
  uint64_t producer_completion_guard_reject_count;
  uint64_t producer_completion_full_path_count;
  uint64_t producer_completion_full_path_read_space_count;
  uint64_t producer_completion_full_path_write_space_count;
  uint64_t direct_memory_query_count;
  uint64_t unclassified_memory_query_count;
  uint64_t direct_producer_completion_count;
  uint64_t unclassified_producer_completion_count;
} loom_amdgpu_wait_frontier_precise_runtime_bounds_t;

// Scalar tuple collected before any precise frontier allocation or publication.
// Every field is a count, never an address or live analysis object.
typedef struct loom_amdgpu_wait_frontier_precise_budget_input_t {
  uint64_t precise_access_count;
  uint64_t producer_counter_bit_count;
  uint64_t read_producer_counter_bit_count;
  uint64_t write_producer_counter_bit_count;
  uint64_t storage_lease_count;
  uint64_t precise_read_count;
  uint64_t precise_write_count;
  uint64_t read_effective_term_sum;
  uint64_t write_effective_term_sum;
  uint64_t precise_word_count;
  uint64_t block_count;
  uint64_t reachable_block_count;
  uint64_t unfiltered_forward_edge_count;
  uint64_t filtered_forward_edge_count;
  uint64_t backedge_count;
  uint64_t local_reset_event_count;
  uint64_t local_reset_counter_bit_count;
  uint64_t memory_space_count;
  uint64_t xcnt_group_count;
  uint64_t maximum_effective_term_count;
  uint64_t allocation_byte_count;
  uint64_t effect_use_count;
  uint64_t cfg_edge_count;
  uint64_t scheduled_node_count;
  uint64_t node_count;
  uint64_t dependency_count;
  uint64_t dependency_read_query_count;
  uint64_t dependency_write_query_count;
  uint64_t barrier_query_count;
  uint64_t program_exit_query_count;
  uint64_t producer_completion_call_count;
  uint64_t producer_completion_guard_reject_count;
  uint64_t producer_completion_full_path_count;
  uint64_t producer_completion_full_path_read_space_count;
  uint64_t producer_completion_full_path_write_space_count;
  uint64_t producer_complete_precise_access_visit_count;
  uint64_t producer_complete_storage_lease_visit_count;
  uint64_t direct_memory_query_count;
  uint64_t unclassified_memory_query_count;
  uint64_t direct_producer_completion_count;
  uint64_t unclassified_producer_completion_count;
  bool collector_heavy_path_possible;
} loom_amdgpu_wait_frontier_precise_budget_input_t;

// Derived work quantities checked independently against their profiling-derived
// caps. Keeping the dimensions explicit makes the admission decision auditable.
typedef struct loom_amdgpu_wait_frontier_precise_budget_usage_t {
  uint64_t precise_access_count;
  uint64_t producer_counter_bit_count;
  uint64_t producer_scan_visits;
  uint64_t eligible_alias_calls;
  uint64_t eligible_alias_bit_visits;
  uint64_t term_merge_visits;
  uint64_t worklist_pops;
  uint64_t edge_evaluations;
  uint64_t forward_word_visits;
  uint64_t filtered_access_visits;
  uint64_t filtered_producer_bit_visits;
  uint64_t collapse_access_visits;
  uint64_t collapse_producer_bit_visits;
  uint64_t local_drain_access_visits;
  uint64_t local_drain_bit_visits;
  uint64_t begin_block_word_visits;
  uint64_t begin_block_collapse_access_visits;
  uint64_t begin_block_collapse_producer_bit_visits;
  uint64_t dynamic_incoming_drain_bit_visits;
  uint64_t maximum_effective_term_count;
  uint64_t allocation_byte_count;
  uint64_t pre_admission_visits;
  uint64_t memory_query_calls;
  uint64_t memory_query_precise_access_visits;
  uint64_t memory_query_precise_bit_visits;
  uint64_t producer_complete_precise_access_visits;
  uint64_t producer_complete_storage_lease_visits;
  uint64_t structural_score_scaled;
} loom_amdgpu_wait_frontier_precise_budget_usage_t;

// Computes only V = 2E + 11B + 7Gcfg + 3S + N + D. This is used as a cheap
// fail-closed gate before validation scratch or collector work is allocated.
bool loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
    uint64_t effect_use_count, uint64_t block_count, uint64_t cfg_edge_count,
    uint64_t scheduled_node_count, uint64_t node_count,
    uint64_t dependency_count, uint64_t* out_visits);

// Derives all guarded work quantities with checked uint64 arithmetic. Returns
// false for overflow or an inconsistent raw tuple.
bool loom_amdgpu_wait_frontier_precise_budget_calculate(
    const loom_amdgpu_wait_frontier_precise_budget_input_t* input,
    loom_amdgpu_wait_frontier_precise_budget_usage_t* out_usage);

// Returns true only when every named usage dimension is within its cap.
bool loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(
    const loom_amdgpu_wait_frontier_precise_budget_usage_t* usage);

// Checked calculation followed by conjunctive admission.
bool loom_amdgpu_wait_frontier_precise_budget_is_admitted(
    const loom_amdgpu_wait_frontier_precise_budget_input_t* input);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_FRONTIER_BUDGET_H_
