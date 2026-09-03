// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent costs derived from verified Low descriptor recipes.

#ifndef LOOM_CODEGEN_LOW_DESCRIPTOR_COST_H_
#define LOOM_CODEGEN_LOW_DESCRIPTOR_COST_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// One independently repeated descriptor stage in a realization recipe.
typedef struct loom_low_descriptor_recipe_entry_t {
  // Descriptor-set-local ordinal naming the instruction descriptor.
  uint32_t descriptor_ordinal;
  // Number of independent occurrences of this descriptor in the stage.
  uint32_t occurrence_count;
} loom_low_descriptor_recipe_entry_t;

static_assert(sizeof(loom_low_descriptor_recipe_entry_t) == 8,
              "loom_low_descriptor_recipe_entry_t must be 8 bytes");

// One dependency between recipe stages. Generated rows are sorted by target
// then source, and source_entry must precede target_entry.
typedef struct loom_low_descriptor_recipe_dependency_t {
  // Recipe entry that produces the dependency.
  uint16_t source_entry;
  // Recipe entry that consumes the dependency.
  uint16_t target_entry;
} loom_low_descriptor_recipe_dependency_t;

static_assert(sizeof(loom_low_descriptor_recipe_dependency_t) == 4,
              "loom_low_descriptor_recipe_dependency_t must be 4 bytes");

// Compact descriptor recipe supplied by a target realization candidate.
typedef struct loom_low_descriptor_recipe_t {
  // Descriptor stages in dependency-topological order.
  const loom_low_descriptor_recipe_entry_t* entries;
  // Number of descriptor stages in entries.
  uint16_t entry_count;
  // Sorted local dependency edges between descriptor stages.
  const loom_low_descriptor_recipe_dependency_t* dependencies;
  // Number of local dependency edges in dependencies.
  uint16_t dependency_count;
  // Additional durable register pressure created by the realization.
  const loom_low_pressure_delta_t* durable_pressure_deltas;
  // Number of rows in durable_pressure_deltas.
  uint16_t durable_pressure_delta_count;
} loom_low_descriptor_recipe_t;

// Aggregate descriptor resource occupancy for one target resource. Rows remain
// sorted by resource_id and omit resources unused by the recipe or schedule.
typedef struct loom_low_descriptor_resource_cost_t {
  // Target resource table identifier summarized by this record.
  uint16_t resource_id;
  // Borrowed stable resource name.
  iree_string_view_t resource_name;
  // Abstract resource kind used by generic diagnostics.
  loom_low_resource_kind_t resource_kind;
  // Generic resource flags from the descriptor table.
  loom_low_resource_flags_t resource_flags;
  // Resource units available per cycle in the descriptor model.
  uint16_t capacity_per_cycle;
  // Contention group identifier shared by related resources.
  uint16_t contention_group_id;
  // Number of issue-use rows accumulated for this resource.
  uint32_t use_count;
  // Sum of occupied cycles across all issue-use rows.
  uint64_t total_occupied_cycles;
  // Sum of cycles times units across all issue-use rows.
  uint64_t total_unit_cycles;
  // Ceiling of total_unit_cycles divided by capacity_per_cycle.
  uint64_t estimated_min_cycles;
  // Largest single issue-use units value observed for this resource.
  uint16_t peak_units_per_cycle;
} loom_low_descriptor_resource_cost_t;

// Aggregate durable register pressure for one descriptor recipe.
typedef struct loom_low_descriptor_pressure_cost_t {
  // Descriptor-set-local register class identifier.
  uint16_t reg_class_id;
  // Net durable allocation-unit delta for the register class.
  int64_t total_unit_delta;
} loom_low_descriptor_pressure_cost_t;

// Aggregate descriptor memory effects for one descriptor recipe.
typedef struct loom_low_descriptor_memory_cost_t {
  // Number of descriptor read-effect occurrences.
  uint64_t read_operation_count;
  // Number of descriptor write-effect occurrences.
  uint64_t write_operation_count;
  // Total known byte width of descriptor read effects.
  uint64_t read_byte_count;
  // Total known byte width of descriptor write effects.
  uint64_t write_byte_count;
  // Number of descriptor read effects without a known byte-aligned width.
  uint64_t read_unknown_width_count;
  // Number of descriptor write effects without a known byte-aligned width.
  uint64_t write_unknown_width_count;
} loom_low_descriptor_memory_cost_t;

// Structured cost evidence derived entirely from one active descriptor set.
typedef struct loom_low_descriptor_cost_t {
  // Borrowed descriptor set used to derive every row in this cost.
  const loom_low_descriptor_set_t* descriptor_set;
  // Least precise schedule model quality used by the recipe.
  loom_low_model_quality_t model_quality;
  // Total descriptor occurrences in the recipe.
  uint64_t instruction_count;
  // Maximum capacity-normalized cycles across resource rows.
  uint64_t maximum_resource_cycles;
  // Sum of capacity-normalized cycles across resource rows.
  uint64_t total_resource_cycles;
  // Longest dependency path through descriptor stages.
  uint64_t critical_path_cycles;
  // Memory operation and byte summaries.
  loom_low_descriptor_memory_cost_t memory;
  // Arena-owned resource cost rows sorted by resource_id.
  const loom_low_descriptor_resource_cost_t* resource_costs;
  // Number of rows in resource_costs.
  iree_host_size_t resource_cost_count;
  // Arena-owned durable pressure rows sorted by reg_class_id.
  const loom_low_descriptor_pressure_cost_t* pressure_costs;
  // Number of rows in pressure_costs.
  iree_host_size_t pressure_cost_count;
} loom_low_descriptor_cost_t;

// Candidate metadata used by the common deterministic cost comparator.
typedef struct loom_low_descriptor_cost_candidate_t {
  // Structured cost evidence for this candidate.
  const loom_low_descriptor_cost_t* cost;
  // Stable target-owned candidate identity used as the final tie breaker.
  uint64_t stable_key;
  // True when this is the target's canonical representation candidate.
  bool is_canonical;
} loom_low_descriptor_cost_candidate_t;

typedef enum loom_low_descriptor_cost_order_e {
  // Both candidates have identical ordering keys.
  LOOM_LOW_DESCRIPTOR_COST_ORDER_EQUIVALENT = 0,
  // The left candidate is preferred.
  LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT = 1,
  // The right candidate is preferred.
  LOOM_LOW_DESCRIPTOR_COST_ORDER_RIGHT = 2,
} loom_low_descriptor_cost_order_t;

// Initializes one dense resource accumulator row from a verified descriptor
// set. The caller may compact unused rows after accumulation.
void loom_low_descriptor_resource_cost_initialize(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t resource_id,
    loom_low_descriptor_resource_cost_t* out_cost);

// Accumulates repeated issue-use rows into a resource cost. This is the single
// resource accounting primitive used by both recipe planning and scheduling.
iree_status_t loom_low_descriptor_resource_cost_accumulate(
    const loom_low_issue_use_t* issue_use, uint32_t occurrence_count,
    loom_low_descriptor_resource_cost_t* inout_cost);

// Resolves and costs a compact descriptor recipe. Output arrays are owned by
// arena and remain sorted by their descriptor-set-local identifiers.
iree_status_t loom_low_descriptor_cost_compute(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_recipe_t* recipe, iree_arena_allocator_t* arena,
    loom_low_descriptor_cost_t* out_cost);

// Returns true when left has compatible exact-enough evidence, is no worse in
// every retained cost dimension, and is strictly better in at least one.
bool loom_low_descriptor_cost_dominates(
    const loom_low_descriptor_cost_t* left,
    const loom_low_descriptor_cost_t* right);

// Orders candidates by dominance, normalized resource work, dependency path,
// instruction count, canonical identity, and stable key. Unknown or
// incompatible evidence falls directly to canonical identity and stable key.
loom_low_descriptor_cost_order_t loom_low_descriptor_cost_compare(
    const loom_low_descriptor_cost_candidate_t* left,
    const loom_low_descriptor_cost_candidate_t* right);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_DESCRIPTOR_COST_H_
