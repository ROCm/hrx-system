// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral residency resource models and cliff evaluation.

#ifndef LOOM_TARGET_RESIDENCY_H_
#define LOOM_TARGET_RESIDENCY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// One target-provided residency cliff.
//
// A resource footprint at or above |cliff_units| has |tier_after| residency
// instead of |tier_before|. Cliffs for one resource form a strict descending
// tier chain ordered by increasing |cliff_units|.
typedef struct loom_target_residency_cliff_t {
  // Table-local direct or derived resource ID.
  uint16_t resource_id;
  // Resource units at which this cliff is crossed.
  uint32_t cliff_units;
  // Residency tier immediately before crossing the cliff.
  uint32_t tier_before;
  // Residency tier immediately after crossing the cliff.
  uint32_t tier_after;
} loom_target_residency_cliff_t;

// Contiguous residency-cliff rows for one direct resource.
typedef struct loom_target_residency_cliff_range_t {
  // First residency-cliff row for the resource.
  uint32_t start;
  // Number of residency-cliff rows for the resource.
  uint32_t count;
} loom_target_residency_cliff_range_t;

// Immutable direct residency resources.
//
// |cliffs| are grouped by resource ID and then increasing unit threshold.
// |cliff_ranges| is dense by direct resource ID.
typedef struct loom_target_residency_direct_resource_table_t {
  // Stable resource names indexed by direct resource ID.
  const iree_string_view_t* names;
  // Residency-cliff rows grouped by direct resource ID.
  const loom_target_residency_cliff_t* cliffs;
  // Number of entries in |cliffs|.
  iree_host_size_t cliff_count;
  // Dense cliff ranges indexed by direct resource ID.
  const loom_target_residency_cliff_range_t* cliff_ranges;
  // Number of direct resources and entries in |names| and |cliff_ranges|.
  uint16_t resource_count;
} loom_target_residency_direct_resource_table_t;

// One direct-resource contribution to a derived residency resource.
typedef struct loom_target_residency_derived_member_t {
  // Dense derived resource ID owning this member.
  uint16_t resource_id;
  // Dense contributing direct resource ID.
  uint16_t direct_resource_id;
  // Member units are rounded up to this granularity before aggregation.
  uint32_t contribution_granularity;
} loom_target_residency_derived_member_t;

// One target-defined residency resource derived from direct resources.
typedef struct loom_target_residency_derived_resource_t {
  // Stable target resource name used by diagnostics and reports.
  iree_string_view_t name;
  // Total resource units shared by resident target execution units.
  uint32_t pool_units;
  // Allocation granularity used by target resource accounting.
  uint32_t allocation_granularity;
  // First member row in the owning derived-resource table.
  uint16_t member_start;
  // Number of contiguous member rows owned by this resource.
  uint16_t member_count;
  // First cliff row in the owning derived-resource table.
  uint16_t cliff_start;
  // Number of contiguous cliff rows owned by this resource.
  uint16_t cliff_count;
} loom_target_residency_derived_resource_t;

// Contiguous reverse-index rows for one direct resource.
typedef struct loom_target_residency_derived_member_range_t {
  // First member-index row for the direct resource.
  uint16_t start;
  // Number of member-index rows for the direct resource.
  uint16_t count;
} loom_target_residency_derived_member_range_t;

// Immutable derived residency resources and direct-resource memberships.
//
// Members and cliffs are grouped by derived resource. The optional reverse
// index maps direct resources back to member rows for incremental hot paths.
// Both reverse-index pointers are either null or present together.
typedef struct loom_target_residency_derived_resource_table_t {
  // Derived residency-resource rows in stable target order.
  const loom_target_residency_derived_resource_t* resources;
  // Number of entries in |resources|.
  uint16_t resource_count;
  // Member rows grouped by derived resource.
  const loom_target_residency_derived_member_t* members;
  // Number of entries in |members|.
  uint16_t member_count;
  // Resource cliff rows grouped by derived resource.
  const loom_target_residency_cliff_t* cliffs;
  // Number of entries in |cliffs|.
  uint16_t cliff_count;
  // Optional member-row indices grouped by direct resource.
  const uint16_t* member_indices_by_direct_resource;
  // Optional dense reverse-index ranges by direct resource ID.
  const loom_target_residency_derived_member_range_t*
      member_ranges_by_direct_resource;
} loom_target_residency_derived_resource_table_t;

// Immutable target residency policy shared by planning and final accounting.
typedef struct loom_target_residency_model_t {
  // Best residency tier available before any resource cliff is crossed.
  uint32_t best_tier;
  // Direct target resources and their cliff chains.
  loom_target_residency_direct_resource_table_t direct_resources;
  // Resources derived from rounded direct-resource contributions.
  loom_target_residency_derived_resource_table_t derived_resources;
} loom_target_residency_model_t;

// Flags describing one cliff-chain evaluation.
typedef uint32_t loom_target_residency_cliff_evaluation_flags_t;
enum loom_target_residency_cliff_evaluation_flag_bits_e {
  // Reducing units can recover a better tier.
  LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_BETTER_TIER = 1u << 0,
  // Increasing units can cross a worse tier.
  LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_WORSE_TIER = 1u << 1,
};

// Evaluation of one ordered residency-cliff chain.
typedef struct loom_target_residency_cliff_evaluation_t {
  // Residency tier selected for the supplied units.
  uint32_t tier;
  // Immediate better tier recoverable by reducing units.
  uint32_t better_tier;
  // Units that must be removed to recover |better_tier|.
  uint64_t reduction_units_to_better_tier;
  // Immediate worse tier reached by increasing units.
  uint32_t worse_tier;
  // Absolute resource footprint at which |worse_tier| begins.
  uint32_t worse_cliff_units;
  // Additional units that cross into |worse_tier|.
  uint64_t additional_units_to_worse_tier;
  // Presence bits for optional better and worse tier fields.
  loom_target_residency_cliff_evaluation_flags_t flags;
} loom_target_residency_cliff_evaluation_t;

// Identifies the table owning a query result resource.
typedef enum loom_target_residency_resource_kind_e {
  // Resource units were supplied directly by the caller.
  LOOM_TARGET_RESIDENCY_RESOURCE_KIND_DIRECT = 0,
  // Resource units were derived from rounded direct contributions.
  LOOM_TARGET_RESIDENCY_RESOURCE_KIND_DERIVED = 1,
} loom_target_residency_resource_kind_t;

// Flags describing one resource in a whole-model query.
typedef uint32_t loom_target_residency_resource_evaluation_flags_t;
enum loom_target_residency_resource_evaluation_flag_bits_e {
  // This resource attains the model's current worst residency tier.
  LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_LIMITING = 1u << 0,
  // A future cliff can lower whole-model residency below its current tier.
  LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_HAS_NEXT_WORSE_TIER = 1u << 1,
};

// Evaluation of one direct or derived resource in a whole-model query.
typedef struct loom_target_residency_resource_evaluation_t {
  // Stable target resource name.
  iree_string_view_t name;
  // Table owning this resource.
  loom_target_residency_resource_kind_t kind;
  // Resource ID local to |kind|.
  uint16_t resource_id;
  // Directly supplied or derived resource units.
  uint64_t units;
  // Residency tier selected by this resource.
  uint32_t tier;
  // Next tier for this resource that would worsen whole-model residency.
  uint32_t next_worse_tier;
  // Absolute footprint at which |next_worse_tier| begins.
  uint32_t next_worse_cliff_units;
  // Additional units that would worsen whole-model residency.
  uint64_t additional_units_to_next_worse_tier;
  // Units this resource must shed to attain the query's next better tier.
  uint64_t reduction_units_to_next_better_tier;
  // Limiting and optional-field presence bits.
  loom_target_residency_resource_evaluation_flags_t flags;
} loom_target_residency_resource_evaluation_t;

// Whole-model residency evaluation and per-resource transition distances.
typedef struct loom_target_residency_query_t {
  // True when a non-null model was supplied and evaluated.
  bool model_available;
  // Best tier declared by the model.
  uint32_t best_tier;
  // Worst current tier across all direct and derived resources.
  uint32_t tier;
  // True when reducing limiting resources can improve |tier|.
  bool has_next_better_tier;
  // First higher whole-model tier reachable by reducing limiting resources.
  uint32_t next_better_tier;
  // Direct evaluations followed by derived evaluations.
  const loom_target_residency_resource_evaluation_t* resources;
  // Total number of entries in |resources|.
  iree_host_size_t resource_count;
  // Number of direct entries at the start of |resources|.
  iree_host_size_t direct_resource_count;
  // Number of resources attaining |tier| when |tier| is below |best_tier|.
  iree_host_size_t limiting_resource_count;
} loom_target_residency_query_t;

// Validated whole-model tier evaluator for repeated pressure queries.
typedef struct loom_target_residency_evaluator_t {
  // Immutable validated model, or NULL when residency is unavailable.
  const loom_target_residency_model_t* model;
  // Direct resource vector length required by |model|.
  iree_host_size_t direct_resource_count;
} loom_target_residency_evaluator_t;

// Returns true when |table| contains no direct resources and no cliffs.
static inline bool loom_target_residency_direct_resource_table_is_empty(
    const loom_target_residency_direct_resource_table_t* table) {
  return table->resource_count == 0 && table->cliff_count == 0;
}

// Returns the direct-resource cliff range for |resource_id|.
static inline loom_target_residency_cliff_range_t
loom_target_residency_direct_resource_cliff_range(
    const loom_target_residency_direct_resource_table_t* table,
    uint16_t resource_id) {
  return table->cliff_ranges[resource_id];
}

// Returns true when |table| contains no derived resources.
static inline bool loom_target_residency_derived_resource_table_is_empty(
    const loom_target_residency_derived_resource_table_t* table) {
  return table->resource_count == 0;
}

// Returns true when |model| contributes no direct or derived resources.
//
// A null model is unavailable, while a non-null empty model remains an
// available best-tier model. Callers needing that distinction use query.
static inline bool loom_target_residency_model_is_empty(
    const loom_target_residency_model_t* model) {
  return model == NULL ||
         (loom_target_residency_direct_resource_table_is_empty(
              &model->direct_resources) &&
          loom_target_residency_derived_resource_table_is_empty(
              &model->derived_resources));
}

// Rounds one derived-resource member contribution without wrapping.
uint64_t loom_target_residency_round_resource_units(uint64_t units,
                                                    uint32_t granularity);

// Returns the reverse-index range for |direct_resource_id|.
static inline loom_target_residency_derived_member_range_t
loom_target_residency_derived_resource_member_range(
    const loom_target_residency_derived_resource_table_t* table,
    uint16_t direct_resource_id) {
  return table->member_ranges_by_direct_resource[direct_resource_id];
}

// Evaluates one complete or suffix residency-cliff chain.
//
// |initial_tier| is the tier before the first supplied cliff. This lets an
// incremental planner evaluate only cliffs that remain actionable. The cliff
// chain must be strictly ordered, contiguous in tier, and descending.
void loom_target_residency_evaluate_cliffs(
    const loom_target_residency_cliff_t* cliffs, iree_host_size_t cliff_count,
    uint32_t initial_tier, uint64_t units,
    loom_target_residency_cliff_evaluation_t* out_evaluation);

// Validates |model| once and prepares a repeated whole-tier evaluator.
//
// A null model succeeds and creates an unavailable evaluator whose tier is
// always zero.
iree_status_t loom_target_residency_evaluator_initialize(
    const loom_target_residency_model_t* model,
    loom_target_residency_evaluator_t* out_evaluator);

// Evaluates one direct-resource vector with a prepared evaluator.
//
// This performs no allocation and does not revalidate immutable model tables.
// |direct_resource_units| must have exactly the count captured by initialize.
iree_status_t loom_target_residency_evaluator_evaluate_tier(
    const loom_target_residency_evaluator_t* evaluator,
    const uint64_t* direct_resource_units,
    iree_host_size_t direct_resource_unit_count, uint32_t* out_tier);

// Evaluates only the whole-model residency tier without allocating storage.
//
// |direct_resource_units| is dense by direct resource ID and must exactly
// match the model's direct resource count. A null model succeeds with tier zero
// when no direct resource values are supplied. This is the hot-path query for
// pressure sweeps; use |loom_target_residency_query| when transition distances
// and limiting-resource diagnostics are required.
iree_status_t loom_target_residency_evaluate_tier(
    const loom_target_residency_model_t* model,
    const uint64_t* direct_resource_units,
    iree_host_size_t direct_resource_unit_count, uint32_t* out_tier);

// Evaluates all resources and the resulting whole-model residency tier.
//
// |direct_resource_units| is dense by direct resource ID and must exactly
// match the model's direct resource count. Query storage is allocated from
// |arena| and remains valid until the arena is reset or deinitialized. A null
// model succeeds with |model_available| false and performs no allocation.
iree_status_t loom_target_residency_query(
    const loom_target_residency_model_t* model,
    const uint64_t* direct_resource_units,
    iree_host_size_t direct_resource_unit_count, iree_arena_allocator_t* arena,
    loom_target_residency_query_t* out_query);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_RESIDENCY_H_
