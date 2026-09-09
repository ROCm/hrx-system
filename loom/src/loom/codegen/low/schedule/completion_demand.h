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
} loom_low_schedule_completion_demand_t;

// Builds the reverse same-block SSA index and empty domain demand sets.
iree_status_t loom_low_schedule_completion_demand_initialize(
    const loom_low_schedule_dependency_index_t* index,
    const loom_low_schedule_node_t* nodes, uint16_t domain_count,
    iree_arena_allocator_t* arena,
    loom_low_schedule_completion_demand_t* out_demand);

// Requests an unscheduled completion that releases live storage in |domain|.
// An earlier request remains pinned until its root is scheduled; new requests
// do not interrupt its dependency chain. Node scheduling is monotone for the
// lifetime of the demand table. No allocation occurs after initialization.
void loom_low_schedule_completion_demand_select(
    loom_low_schedule_completion_demand_t* demand,
    const loom_low_schedule_node_t* nodes, uint16_t domain, uint32_t root);

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
