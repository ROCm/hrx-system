// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse physical-representation selection for source SSA values.
//
// Targets contribute finite exact alternatives for individual values. Common
// lowering contributes equality relations between values whose physical
// representation must agree. The plan intersects alternatives and minimizes
// their aggregate target costs once all relations are known.

#ifndef LOOM_CODEGEN_LOW_REPRESENTATION_PLAN_H_
#define LOOM_CODEGEN_LOW_REPRESENTATION_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t loom_low_representation_id_t;

// Sentinel used when no physical representation has been selected.
#define LOOM_LOW_REPRESENTATION_ID_NONE \
  ((loom_low_representation_id_t)UINT16_MAX)

// Target cost for one exact representation alternative.
typedef struct loom_low_representation_cost_t {
  // Estimated runtime cost in target-defined comparable units.
  uint32_t runtime;
  // Estimated encoded program size used to break equal-runtime ties.
  uint32_t code_size;
} loom_low_representation_cost_t;

// One exact representation available at an operation boundary.
typedef struct loom_low_representation_candidate_t {
  // Physical representation identity in the active target policy's namespace.
  loom_low_representation_id_t representation;
  // Target cost incurred when this alternative is selected.
  loom_low_representation_cost_t cost;
} loom_low_representation_candidate_t;
static_assert(sizeof(loom_low_representation_candidate_t) == 12,
              "representation candidates must stay compact");

typedef struct loom_low_representation_node_t loom_low_representation_node_t;
typedef struct loom_low_representation_constraint_t
    loom_low_representation_constraint_t;

// Sparse representation-selection state retained for one source function.
typedef struct loom_low_representation_plan_t {
  // Arena owning all sparse plan storage.
  iree_arena_allocator_t* arena;
  // Number of values addressable by local value ordinal.
  loom_value_ordinal_t value_count;
  // Sparse-node ordinal by value ordinal, or UINT32_MAX when absent.
  uint32_t* node_ordinals;
  // Sparse participating-value nodes.
  loom_low_representation_node_t* nodes;
  // Number of initialized sparse nodes.
  uint32_t node_count;
  // Allocated sparse-node capacity.
  iree_host_size_t node_capacity;
  // Whether the plan has completed selection.
  bool solved;
} loom_low_representation_plan_t;

// Describes a component whose exact candidate intersection is empty.
typedef struct loom_low_representation_conflict_t {
  // Representative value ordinal from the conflicting component.
  loom_value_ordinal_t value_ordinal;
} loom_low_representation_conflict_t;

// Initializes an empty representation plan. Storage is allocated lazily when a
// value first participates.
void loom_low_representation_plan_initialize(
    loom_value_ordinal_t value_count, iree_arena_allocator_t* arena,
    loom_low_representation_plan_t* out_plan);

// Requires |left| and |right| to select the same physical representation.
iree_status_t loom_low_representation_plan_union(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t left,
    loom_value_ordinal_t right);

// Adds one operation-local exact representation domain and its target costs.
iree_status_t loom_low_representation_plan_constrain(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal,
    const loom_low_representation_candidate_t* candidates,
    iree_host_size_t candidate_count);

// Returns whether |value_ordinal|'s current component has received at least
// one exact candidate constraint. This may be queried during construction.
bool loom_low_representation_plan_component_is_constrained(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal);

// Selects the least-cost exact representation for every constrained component.
// Returns false and populates |out_conflict| when a component has no common
// representation. Unconstrained components intentionally remain unselected.
bool loom_low_representation_plan_solve(
    loom_low_representation_plan_t* plan,
    loom_low_representation_conflict_t* out_conflict);

// Returns the selected representation for |value_ordinal|, or false when the
// value did not participate or its component had no constraints.
bool loom_low_representation_plan_lookup(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal,
    loom_low_representation_id_t* out_representation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REPRESENTATION_PLAN_H_
