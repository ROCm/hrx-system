// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable exact storage claims for resolved fixed-value assignments.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_FIXED_STORAGE_INDEX_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_FIXED_STORAGE_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_fixed_storage_record_t
    loom_low_allocation_fixed_storage_record_t;

// Storage-partitioned temporal index over immutable fixed assignments.
//
// Each record is one canonical atomic-storage and exact live-segment claim.
// Records are grouped by location kind and storage identity, with an implicit
// maximum-end interval tree inside each identity group. Query generations
// exclude the candidate and explicitly ignored fixed values without scanning
// the ignored set for every matching claim.
typedef struct loom_low_allocation_fixed_storage_index_t {
  // Exact claims ordered by storage identity then start point.
  loom_low_allocation_fixed_storage_record_t* records;
  // Tied root shared by every implicit record subtree, INVALID for mixed
  // subtrees, or NULL when no fixed assignment belongs to a shared component.
  loom_value_ordinal_t* subtree_tied_roots;
  // First storage record for each register-like location kind.
  uint32_t record_starts[2];
  // Number of storage records for each register-like location kind.
  uint32_t record_counts[2];
  // Last query generation excluding each resolved fixed value.
  uint32_t* excluded_generations;
  // Current nonzero query generation.
  uint32_t generation;
} loom_low_allocation_fixed_storage_index_t;

struct loom_low_allocation_target_constraints_t;
struct loom_low_allocation_assignment_t;
struct loom_low_allocation_unit_liveness_t;

// Builds the fixed-storage index owned by |constraints|. Retained storage
// comes from |arena|. Construction is linear in materialized exact claims plus
// bounded radix passes and returns temporary sorting blocks before completion.
iree_status_t loom_low_allocation_fixed_storage_index_initialize(
    struct loom_low_allocation_target_constraints_t* constraints,
    const struct loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena);

// Returns true when |candidate| violates its own fixed binding or conflicts
// with another fixed value or implicit physical write. Resolved whole-value
// tied components share reservations when their concrete storage matches;
// they never excuse clobbers.
bool loom_low_allocation_target_constraints_fixed_storage_conflicts(
    struct loom_low_allocation_target_constraints_t* constraints,
    const struct loom_low_allocation_unit_liveness_t* unit_liveness,
    const struct loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_FIXED_STORAGE_INDEX_H_
