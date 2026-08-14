// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local storage placement relations for target-low allocation.
//
// Placement is the target-independent producer of storage-affinity facts. It
// records the IR-level relationships that make two low values profitable or
// mandatory to place in the same concrete storage without deciding where that
// storage is. Allocation consumes this table when assigning locations; move
// sequencing consumes the allocation result after unavoidable copies remain.

#ifndef LOOM_CODEGEN_LOW_PLACEMENT_H_
#define LOOM_CODEGEN_LOW_PLACEMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/placement_pair.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_placement_cause_bits_e {
  // Unknown or uninitialized placement cause.
  LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN = 0,
  // Semantically tied result requiring source/result storage identity.
  LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT = 1,
  // low.copy source/result storage affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY = 2,
  // low.slice source/result subrange affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE = 3,
  // low.concat source/result contiguous packing affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT = 4,
  // low.br edge payload source/block-argument affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH = 5,
  // low.scf.for initial iter_arg/body-argument affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_FOR = 6,
  // low.scf.yield payload/result or backedge affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD = 7,
  // Scheduled target-packet pair location affinity.
  LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY = 8,
} loom_low_placement_cause_bits_t;
typedef uint8_t loom_low_placement_cause_t;

enum loom_low_placement_relation_flag_bits_e {
  // The relation is required for the selected target operation semantics.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD = 1u << 0,
  // The relation removes a move when allocation can satisfy it.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED = 1u << 1,
  // The relation can justify overlapping target-visible storage.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE = 1u << 2,
};

// Bitset of loom_low_placement_relation_flag_bits_e values.
typedef uint16_t loom_low_placement_relation_flags_t;

// One concrete pair opportunity retained from the final schedule.
typedef struct loom_low_placement_pair_use_t {
  // First visible scheduled operation.
  const loom_op_t* first_op;
  // Second visible scheduled operation.
  const loom_op_t* second_op;
  // Index + 1 into the containing list placement_recipes table.
  uint16_t placement_recipe_index;
  // Relative benefit of satisfying this pair opportunity.
  uint16_t priority;
} loom_low_placement_pair_use_t;

// Concrete pair opportunities retained from one schedule.
typedef struct loom_low_placement_pair_use_list_t {
  // Borrowed pair-use rows.
  const loom_low_placement_pair_use_t* values;
  // Number of entries in values.
  iree_host_size_t count;
  // Borrowed target-provided placement recipes referenced by values.
  const loom_low_placement_pair_recipe_t* placement_recipes;
  // Number of entries in placement_recipes.
  iree_host_size_t placement_recipe_count;
} loom_low_placement_pair_use_list_t;

// Resolves one recipe value reference against a concrete scheduled pair.
loom_value_id_t loom_low_placement_pair_value_id(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_value_ref_t* ref);

// Returns true when assigning |separated_ref| a fresh SSA value removes every
// structural identity contradiction from one placement-recipe alternative.
// Physical-location feasibility remains an allocation decision.
bool loom_low_placement_pair_alternative_can_separate_ref(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_relation_t* relations,
    uint16_t relation_count,
    const loom_low_placement_pair_value_ref_t* separated_ref);

// Counts placement-recipe alternatives that are structurally possible for a
// concrete scheduled pair. This only considers SSA identity contradictions;
// physical-location legality remains an allocation decision.
uint16_t loom_low_placement_pair_possible_alternative_count(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe);

static inline loom_low_placement_pair_use_list_t
loom_low_placement_pair_use_list_empty(void) {
  return (loom_low_placement_pair_use_list_t){0};
}

// One directional placement relation keyed by result and source value ordinals.
typedef struct loom_low_placement_relation_t {
  // Operation that introduced this relation.
  const loom_op_t* op;
  // Result or destination value ordinal receiving preferred storage.
  loom_value_ordinal_t result_ordinal;
  // Source value ordinal providing preferred storage.
  loom_value_ordinal_t source_ordinal;
  // Unit offset inside the result assignment.
  uint32_t result_unit_offset;
  // Unit offset inside the source assignment.
  uint32_t source_unit_offset;
  // Number of units covered by this relation.
  uint32_t unit_count;
  // Low location bits compared by DIFFERENT_MASKED_LOCATION.
  uint32_t location_mask;
  // Structural relation shape.
  loom_low_placement_relation_kind_t kind;
  // IR feature that created the relation.
  loom_low_placement_cause_t cause;
  // Hard/soft relation behavior.
  loom_low_placement_relation_flags_t flags;
  // Relative benefit of satisfying this relation.
  uint16_t priority;
} loom_low_placement_relation_t;

// Contiguous relation range for one result value ordinal.
typedef struct loom_low_placement_relation_range_t {
  // First relation index for the value ordinal.
  uint32_t start;
  // Number of relation records for the value ordinal.
  uint32_t count;
} loom_low_placement_relation_range_t;

// Placement analysis table for one target-low function body.
typedef struct loom_low_placement_table_t {
  // Module containing the analyzed low function.
  const loom_module_t* module;
  // Region analyzed.
  const loom_region_t* region;
  // Local value IDs indexed by placement value ordinal.
  const loom_value_id_t* value_ids;
  // Number of local value IDs.
  loom_value_ordinal_t value_count;
  // Placement relations grouped by result value ordinal.
  const loom_low_placement_relation_t* relations;
  // Number of relation records.
  iree_host_size_t relation_count;
  // Number of relations constraining concrete location choice.
  iree_host_size_t location_relation_count;
  // Number of low.copy/slice/concat operations that may require packet moves.
  uint32_t packet_move_group_count;
  // Total units covered by low.copy/slice/concat relations.
  iree_host_size_t packet_move_unit_count;
  // Number of low.br operations that may require edge copies.
  uint32_t edge_copy_group_count;
  // Total units covered by low.br relations.
  iree_host_size_t branch_unit_count;
  // Relation ranges into |relations| indexed by result value ordinal.
  const loom_low_placement_relation_range_t* ranges_by_result_ordinal;
  // Relation indices grouped by source value ordinal. Each entry indexes
  // |relations|.
  const uint32_t* relation_indices_by_source_ordinal;
  // Relation ranges into |relation_indices_by_source_ordinal| indexed by source
  // value ordinal.
  const loom_low_placement_relation_range_t* ranges_by_source_ordinal;
} loom_low_placement_table_t;

// Returns true when |relation| can justify overlapping target-visible storage.
bool loom_low_placement_relation_can_alias(
    const loom_low_placement_relation_t* relation);

// Returns true when |cause| is a control-flow edge payload relation.
bool loom_low_placement_cause_is_edge(loom_low_placement_cause_t cause);

// Builds a function-local placement relation table over an acquired value
// domain and its liveness analysis.
iree_status_t loom_low_placement_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    loom_low_placement_pair_use_list_t pair_uses, iree_arena_allocator_t* arena,
    loom_low_placement_table_t* out_table);

// Returns the relation range for |result_ordinal|. The ordinal must belong to
// this placement table.
loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t result_ordinal);

// Returns the relation range for |source_ordinal|. The returned range indexes
// |relation_indices_by_source_ordinal|, whose records then index |relations|.
// The ordinal must belong to this placement table.
loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_source_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t source_ordinal);

// Returns the local value ID for |value_ordinal|.
loom_value_id_t loom_low_placement_value_id(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PLACEMENT_H_
