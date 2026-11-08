// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical candidate preferences retained across scalar storage affinities.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PHYSICAL_DOMAINS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PHYSICAL_DOMAINS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable preferences shared by allocation attempts over the same input
// liveness, placement, target budgets, and reservations. Full scalar affinities
// carry their common physical domain and storage horizon. Atomic storage
// coverage relates classes whose named physical candidates differ. When broad
// demand exceeds general capacity, disjoint lifetimes reuse idle narrower
// storage while overlapping narrow demand retains the capacity it requires.
// These preferences never establish interference or permission to alias
// storage.
typedef struct loom_low_allocation_physical_domains_t {
  // Word offsets indexed by liveness interval index. UINT32_MAX denotes an
  // inert row; NULL denotes a function without physical scalar candidates.
  const uint32_t* offsets;
  // Primary penalty bits indexed by semantic candidate ordinal in the
  // interval's class.
  // Rows are word-aligned and have ceil(allocatable_count / 64) words.
  const uint64_t* penalty_words;
  // Candidates outside the common domain of a storage-affinity component.
  const uint64_t* affinity_words;
  // Reserved-narrower-storage bits. These also provide the second rank bit
  // when idle narrower storage is preferred.
  const uint64_t* reservation_words;
} loom_low_allocation_physical_domains_t;

typedef struct loom_low_allocation_physical_domain_row_t {
  // Primary candidate-penalty bits, or NULL when the row is inert.
  const uint64_t* penalty_words;
  // Candidates outside the component's common affinity domain, or NULL when
  // the row is inert.
  const uint64_t* affinity_words;
  // Reserved-narrower-storage bits, or NULL when the row is inert.
  const uint64_t* reservation_words;
} loom_low_allocation_physical_domain_row_t;

// Returns the retained rank for a semantic candidate ordinal. Rank zero is
// preferred; higher ranks violate more retained affinity or capacity
// preferences.
static inline uint32_t loom_low_allocation_physical_domain_row_candidate_rank(
    loom_low_allocation_physical_domain_row_t row, uint16_t candidate_ordinal) {
  if (!row.penalty_words) {
    return 0;
  }
  const uint64_t bit = UINT64_C(1) << (candidate_ordinal % 64);
  return (uint32_t)((row.penalty_words[candidate_ordinal / 64] & bit) != 0) +
         (uint32_t)((row.reservation_words[candidate_ordinal / 64] & bit) != 0);
}

// Returns true when the candidate occupies capacity reserved for a narrower
// physical domain. An inert row has no alternative candidate to preserve.
static inline bool
loom_low_allocation_physical_domain_row_candidate_is_reserved(
    loom_low_allocation_physical_domain_row_t row, uint16_t candidate_ordinal) {
  if (!row.penalty_words) {
    return false;
  }
  const uint64_t bit = UINT64_C(1) << (candidate_ordinal % 64);
  return (row.reservation_words[candidate_ordinal / 64] & bit) != 0;
}

// Returns true when the candidate lies outside the common physical domain of
// the interval's storage-affinity component.
static inline bool
loom_low_allocation_physical_domain_row_candidate_breaks_affinity(
    loom_low_allocation_physical_domain_row_t row, uint16_t candidate_ordinal) {
  if (!row.penalty_words) {
    return false;
  }
  const uint64_t bit = UINT64_C(1) << (candidate_ordinal % 64);
  return (row.affinity_words[candidate_ordinal / 64] & bit) != 0;
}

// Returns the retained candidate-penalty row for an interval. Bits address
// semantic candidate ordinals in the interval's register class, not
// physical-register IDs.
loom_low_allocation_physical_domain_row_t
loom_low_allocation_physical_domains_for_interval(
    const loom_low_allocation_physical_domains_t* domains,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_interval_t* interval);

// Builds preferences from final producer-owned placement, unit liveness, and
// target constraints. Whole-function budgets and reserved ranges remove
// unavailable physical candidates before overlapping-domain capacity is
// planned. Inputs are borrowed; all retained storage belongs to |arena|.
// Construction scratch is released before returning. Changed input facts
// require rebuilding the plan. Linear classes and aggregate views have no
// preference rows.
iree_status_t loom_low_allocation_physical_domains_build(
    const loom_low_allocation_target_constraints_t* target_constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_physical_domains_t* out_domains);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PHYSICAL_DOMAINS_H_
