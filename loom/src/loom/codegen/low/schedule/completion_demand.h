// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Incremental dependency demand for scarce-register completion priorities.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_DEMAND_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_DEMAND_H_

#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Six 64-way summary levels cover every 32-bit schedule-node index.
#define LOOM_LOW_SCHEDULE_COMPLETION_NOMINATION_LEVEL_CAPACITY 6u

// One pinned completion per hardware pressure domain. A completion remains
// selected until it is scheduled. Its same-block SSA ancestors are then also
// scheduled, so their demand bits need not be cleared before selecting another
// completion. Each node and incoming edge is visited at most once per domain.
// All storage belongs to the scheduling scratch arena and is fixed at creation.
typedef struct loom_low_schedule_completion_demand_t {
  // Incoming same-block SSA producer ranges, with a terminal sentinel.
  uint32_t* incoming_starts;
  // Producer nodes grouped by consumer in incoming_starts.
  uint32_t* producers;
  // Fixed traversal stack with one entry per schedule node.
  uint32_t* worklist;
  // Monotone demand bits, laid out as one node bitset per domain.
  uint64_t* demanded_bits;
  // Pinned completion node per domain, or NODE_NONE before the first request.
  uint32_t* roots;
  // Number of 64-bit words in each domain's node bitset.
  uint32_t words_per_domain;
  // Number of hardware pressure domains sharing the bitset layouts.
  uint16_t domain_count;
  // Fixed hierarchical nomination bitsets for ordered consumer selection.
  struct {
    // Consumer nominations and nonempty-word summaries, domain-major.
    uint64_t* bits;
    // Word offset of each summary level within one domain, leaves first.
    uint32_t
        level_starts[LOOM_LOW_SCHEDULE_COMPLETION_NOMINATION_LEVEL_CAPACITY];
    // Total nomination and summary words in each domain.
    uint32_t words_per_domain;
    // Number of populated level_starts entries.
    uint8_t level_count;
  } nominations;
} loom_low_schedule_completion_demand_t;

// Builds the reverse same-block SSA index and empty domain demand sets.
iree_status_t loom_low_schedule_completion_demand_initialize(
    const loom_low_schedule_dependency_index_t* index,
    const loom_low_schedule_node_t* nodes, uint16_t domain_count,
    iree_arena_allocator_t* arena,
    loom_low_schedule_completion_demand_t* out_demand);

// Nominates the sole remaining consumer of a live value in |domain|. Several
// values may nominate the same consumer without requiring reference counts:
// each nomination remains valid until that consumer runs. Insertion is bounded
// by the 32-bit node-index width, independently of the number of live values.
void loom_low_schedule_completion_demand_nominate(
    loom_low_schedule_completion_demand_t* demand, uint16_t domain,
    uint32_t consumer);

// Retires all nominations for a node when it is scheduled. Every scheduled
// node must be reported, including nodes that were not selected completions.
void loom_low_schedule_completion_demand_complete(
    loom_low_schedule_completion_demand_t* demand, uint32_t node);

// Pins the earliest nominated consumer, or returns NODE_NONE when none exists.
// An earlier selection remains pinned until its root is scheduled; nominations
// do not interrupt its dependency chain. Node scheduling is monotone for the
// lifetime of the demand table. No allocation occurs after initialization.
uint32_t loom_low_schedule_completion_demand_select(
    loom_low_schedule_completion_demand_t* demand,
    const loom_low_schedule_node_t* nodes, uint16_t domain);

// Returns whether an unscheduled node advances the domain's pinned completion.
// Bits for scheduled nodes remain set and are not meaningful to this query.
static inline bool loom_low_schedule_completion_demand_contains(
    const loom_low_schedule_completion_demand_t* demand, uint16_t domain,
    uint32_t node) {
  const uint64_t* bits = demand->demanded_bits +
                         (iree_host_size_t)domain * demand->words_per_domain;
  return (bits[node / 64] & (UINT64_C(1) << (node % 64))) != 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_DEMAND_H_
