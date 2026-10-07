// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local storage placement relations for target-low allocation.
//
// Placement records required sharing, optional copy coalescing, disjoint
// storage, and target instruction-pair preferences. Allocation constructs the
// table using validated fixed locations, then consumes it when assigning the
// remaining storage. Move sequencing materializes the selected transfers.

#ifndef LOOM_CODEGEN_LOW_PLACEMENT_H_
#define LOOM_CODEGEN_LOW_PLACEMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement_capture.h"
#include "loom/codegen/low/placement_recipe.h"
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
  // low.copy source/result sharing or disjoint-placement preference.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY = 2,
  // low.move source/result sharing or disjoint-placement preference.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE = 3,
  // low.slice source/result subrange affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE = 4,
  // low.concat source/result contiguous packing affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT = 5,
  // low.br edge payload source/block-argument affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH = 6,
  // Structured-loop initial iter_arg/entry-argument affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY = 7,
  // low.scf.yield payload/result or backedge affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD = 8,
  // low.scf.condition payload/body-argument or result affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION = 9,
  // Scheduled target-packet pair location affinity.
  LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY = 10,
  // Target descriptor constraint coupling physical-register candidate
  // ordinals without aliasing their storage.
  LOOM_LOW_PLACEMENT_CAUSE_DESCRIPTOR_CONSTRAINT = 11,
} loom_low_placement_cause_bits_t;
typedef uint8_t loom_low_placement_cause_t;

enum loom_low_placement_relation_flag_bits_e {
  // The relation is required for the selected target operation semantics.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD = 1u << 0,
  // An optional placement objective, such as removing a move, freeing fixed
  // storage, or enabling instruction pairing. It does not establish legality.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED = 1u << 1,
  // The relation can justify overlapping target-visible storage.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE = 1u << 2,
  // The required tie writes new contents instead of forwarding an identity.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE = 1u << 3,
  // The edge forwards the destination's existing SSA bits through transparent
  // structural copies. Its source can share storage with the still-observable
  // destination. Destructive tied results do not preserve bit identity.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE = 1u << 4,
  // The concat part's edge reads use a saved aggregate component instead of
  // rereading the original source after a required storage write.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART = 1u << 5,
  // This concat part must be materialized at the concat's program point.
  // Other parts are transported directly by their decomposed edge reads.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART = 1u << 6,
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

// Resolves a recipe reference against its ordered operation bindings.
static inline loom_value_id_t loom_low_placement_operation_value_id(
    const loom_op_t* const* operations,
    const loom_low_placement_value_ref_t* ref) {
  const loom_op_t* op = operations[ref->operation_index];
  return ref->kind == LOOM_LOW_PLACEMENT_VALUE_OPERAND
             ? loom_op_const_operands(op)[ref->index]
             : loom_op_const_results(op)[ref->index];
}

// Resolves one recipe value reference against a concrete scheduled pair.
loom_value_id_t loom_low_placement_pair_value_id(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_value_ref_t* ref);

// Returns true when assigning |separated_ref| a fresh SSA value removes every
// structural identity contradiction from one placement-recipe alternative.
// Physical-location feasibility remains an allocation decision.
bool loom_low_placement_pair_alternative_can_separate_ref(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_preference_t* const* preferences,
    uint16_t preference_count,
    const loom_low_placement_value_ref_t* separated_ref);

// Counts placement-recipe alternatives that are structurally possible for a
// concrete scheduled pair. This only considers SSA identity contradictions;
// physical-location legality remains an allocation decision.
uint16_t loom_low_placement_pair_possible_alternative_count(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe);

// Returns the first structurally feasible alternative, or the first declared
// alternative when every one needs physical separation during allocation.
const loom_low_placement_preference_t* const*
loom_low_placement_select_pair_alternative(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe);

static inline loom_low_placement_pair_use_list_t
loom_low_placement_pair_use_list_empty(void) {
  return (loom_low_placement_pair_use_list_t){0};
}

// Sentinel used when a relation source is not an operand of the observed op.
#define LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE UINT16_MAX

// One directional placement relation between SSA storage providers.
// Construction may select a bit-equivalent source for a decomposed concat
// before indexing; the operation and result still identify the authored
// transfer.
typedef struct loom_low_placement_relation_t {
  // Operation that introduced this relation.
  const loom_op_t* op;
  // Authored result or destination value ordinal receiving preferred storage.
  loom_value_ordinal_t result_ordinal;
  // Selected source value ordinal providing preferred storage. This is the
  // authored operand unless construction selects an available equivalent copy.
  loom_value_ordinal_t source_ordinal;
  // Unit offset inside the result assignment.
  uint32_t result_unit_offset;
  // Unit offset inside the source assignment.
  uint32_t source_unit_offset;
  // Number of units covered by this relation.
  uint32_t unit_count;
  union {
    // Low location bits compared by DIFFERENT_MASKED_LOCATION.
    uint32_t location_mask;
    // Accepted program point of a WRITES_STORAGE relation's write.
    uint32_t write_point;
  };
  // Structural relation shape.
  loom_low_placement_relation_kind_t kind;
  // IR feature that created the relation.
  loom_low_placement_cause_t cause;
  // Hard/soft relation behavior.
  loom_low_placement_relation_flags_t flags;
  // Relative benefit of satisfying this relation.
  uint16_t priority;
  // Flat operand index of |source_ordinal| on |op|, or
  // LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE when the source is not an operand.
  uint16_t source_operand_index;
} loom_low_placement_relation_t;

#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(loom_low_placement_relation_t) == 40,
              "placement relations must remain compact");
#else
static_assert(sizeof(loom_low_placement_relation_t) == 36,
              "placement relations must remain compact");
#endif  // UINTPTR_MAX == UINT64_MAX

// Composes two retained relations over their overlapping intermediate units.
// Returns false for different intermediate values or disjoint unit ranges.
// The result retains the second relation's placement semantics and operation.
bool loom_low_placement_relation_compose(
    const loom_low_placement_relation_t* source_to_intermediate,
    const loom_low_placement_relation_t* intermediate_to_result,
    loom_low_placement_relation_t* out_relation);

// Composes a required tied-result alias with a concat source relation.
// Returns false unless the source tie can justify overlapping storage.
bool loom_low_placement_relation_compose_tied_concat_source(
    const loom_low_placement_relation_t* tied_relation,
    const loom_low_placement_relation_t* concat_relation,
    loom_low_placement_relation_t* out_relation);

// Contiguous relation range for one endpoint index key.
typedef struct loom_low_placement_relation_range_t {
  // First relation index for the endpoint key.
  uint32_t start;
  // Number of relation records for the endpoint key.
  uint32_t count;
} loom_low_placement_relation_range_t;

// Operand requirements combined across every use of one live interval.
typedef struct loom_low_placement_operand_constraints_t {
  // Smallest LOW_SUBSET addressable extent, or zero for no low-window bound.
  uint16_t addressable_unit_count;
  // Strongest required base alignment exponent in allocation units.
  uint8_t unit_alignment_log2;
  // An operand uses a target-selected address window. Renumbering preserves
  // its concrete location so it cannot change address-state transitions.
  bool has_target_address_state;
} loom_low_placement_operand_constraints_t;

enum loom_low_placement_storage_flag_bits_e {
  // Copy, move, slice, or concat relations can optionally share storage.
  LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES = 1u << 0,
  // Nonwriting required aliases share a content version with their source.
  LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES = 1u << 1,
  // Concat components need packet-versus-edge transport planning.
  LOOM_LOW_PLACEMENT_STORAGE_FLAG_CONCAT = 1u << 2,
};
typedef uint8_t loom_low_placement_storage_flags_t;

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
  // Placement relations grouped by result index key. Hard register-ordinal
  // relations use the result's tied storage origin; all other relations use
  // their authored result ordinal. An immediate tied source or defining copy
  // or move, when present, is first in its result's range; other relations
  // retain collection order.
  // Allocation refines optional alias permissions before assigning locations.
  loom_low_placement_relation_t* relations;
  // Number of relation records.
  iree_host_size_t relation_count;
  // Indices into |relations| for whole-value control-flow edge payloads, in
  // liveness operation order.
  const uint32_t* edge_relation_indices;
  // Number of entries in |edge_relation_indices|.
  uint32_t edge_relation_count;
  // Storage facts retained while collecting relations, before alias refinement.
  struct {
    // Mandatory writing relation indices in nondecreasing write-point order.
    const uint32_t* write_relation_indices;
    // Number of entries in write_relation_indices.
    uint32_t write_relation_count;
    // Alias families present in the collected structural storage relations.
    loom_low_placement_storage_flags_t flags;
  } storage;
  // Number of relations constraining concrete location choice.
  iree_host_size_t location_relation_count;
  // Number of hard relations constraining concrete location choice.
  uint32_t hard_location_relation_count;
  // Number of low.copy/move/slice/concat operations that may require packet
  // moves.
  uint32_t packet_move_group_count;
  // Total units covered by low.copy/move/slice/concat relations.
  iree_host_size_t packet_move_unit_count;
  // Number of low.br operations that may require edge copies.
  uint32_t edge_copy_group_count;
  // Total units covered by low.br relations.
  iree_host_size_t branch_unit_count;
  // Maximum raw move units contributed by any one packet or branch operation.
  iree_host_size_t max_move_group_unit_count;
  // Relation ranges into |relations| indexed by result ordinal or, for hard
  // register-ordinal constraints, the result's tied storage origin.
  const loom_low_placement_relation_range_t* ranges_by_result_ordinal;
  // Relation indices grouped by source ordinal or, for hard register-ordinal
  // constraints, the source's tied storage origin. Each entry indexes
  // |relations| without changing the selected endpoints.
  const uint32_t* relation_indices_by_source_ordinal;
  // Relation ranges into |relation_indices_by_source_ordinal| indexed by the
  // same source ordinal or storage-origin key.
  const loom_low_placement_relation_range_t* ranges_by_source_ordinal;
  // Local value ordinals in users-before-sources order for structural SSA
  // storage relations. Tied results and aliasable low copy/move/slice/concat
  // relations form an acyclic graph independent of block layout; allocation
  // analyses use this retained order for transitive storage facts.
  const loom_value_ordinal_t* storage_value_order;
  // Number of entries in |storage_value_order|. Zero when the function has no
  // structural SSA storage relations; otherwise equal to |value_count|.
  loom_value_ordinal_t storage_value_order_count;
  // Origin value in each exact tied-storage component, indexed by local value
  // ordinal. Unconnected values name themselves. NULL when the function has no
  // tied storage.
  const loom_value_ordinal_t* tied_storage_origins_by_value_ordinal;
  // Operand requirements indexed by liveness interval. Exact tied-storage
  // components share their strongest requirements. NULL when unconstrained;
  // optional slice/concat aliases are checked at their concrete derived base.
  const loom_low_placement_operand_constraints_t*
      operand_constraints_by_interval;
  // Exceptional concat sources, sorted by their placement relation index.
  // NULL when every edge can read the original component storage.
  const loom_low_placement_capture_t* captures;
  // Number of initialized exceptional concat source rows.
  uint32_t capture_count;
} loom_low_placement_table_t;

// One allocation-local instruction preference bound during placement analysis.
typedef struct loom_low_placement_preference_use_t {
  // Borrowed target recipe.
  const loom_low_placement_preference_t* preference;
  // First entry in the containing index's bindings array.
  uint32_t binding_start;
  // Instruction unit weight or retained 16-bit scheduled-pair priority.
  uint16_t priority;
  // Retained periodic-cost facts for allocation-local score reuse.
  struct {
    // Low candidate bits needed by the predicates, including offset carries.
    // Structural predicates are invariant after storage assignment; only the
    // masked predicates contribute location dependencies during renumbering.
    uint8_t location_bit_count;
    // One plus log2 of the cell bound, capped by finite register domains.
    // Zero disables assignment-time memoization for nonperiodic costs.
    uint8_t index_bit_count_plus_one;
  } memo;
} loom_low_placement_preference_use_t;

typedef struct loom_low_placement_preference_binding_t {
  // Actual value named by the recipe, even when another value is tied to it.
  loom_value_ordinal_t value_ordinal;
  // First slot in a scheduled-pair use with the same mandatory storage
  // origin. Deferred instruction uses retain their own slot index.
  uint32_t representative;
} loom_low_placement_preference_binding_t;

// Working index for allocation decisions, not part of the returned placement
// or allocation table. Its owner releases it after final register numbering.
typedef struct loom_low_placement_preference_index_t {
  // Bound uses in collection order.
  const loom_low_placement_preference_use_t* uses;
  // Bound real values, grouped by use.
  const loom_low_placement_preference_binding_t* bindings;
  // Scheduled-pair use indexes, grouped by mandatory storage origin and
  // deduplicated within each origin. Instruction uses are numbered only after
  // storage assignment and do not participate in assignment queries.
  const uint32_t* use_indices;
  // Offsets into use_indices, with value_count + 1 entries; NULL when inert.
  const uint32_t* offsets_by_origin;
  // Number of bound uses.
  uint32_t use_count;
  // Leading instruction uses deferred until final register numbering.
  uint32_t instruction_use_count;
  // Number of bound values.
  uint32_t binding_count;
  // Maximum uses incident to any one origin, for reusable query storage.
  uint32_t max_incident_use_count;
  // Maximum total binding slots of uses incident to any one origin.
  uint32_t max_incident_binding_count;
  // Maximum periodic-use cell bound, for one reusable assignment-attempt memo.
  uint32_t max_memo_entry_count;
} loom_low_placement_preference_index_t;

// Returns true when |relation| can justify overlapping target-visible storage.
bool loom_low_placement_relation_can_alias(
    const loom_low_placement_relation_t* relation);

// Returns true when |cause| is a control-flow edge payload relation.
bool loom_low_placement_cause_is_edge(loom_low_placement_cause_t cause);

// Returns the relation range keyed by |result_ordinal|. Hard register-ordinal
// constraints are keyed by the endpoint's tied storage origin; all other
// relations are keyed by their authored SSA endpoint. The ordinal must belong
// to this placement table.
loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t result_ordinal);

// Returns the defining low.copy or low.move relation for |value_ordinal|, or
// NULL for values not produced by a retained transfer. This is a constant-time
// lookup of the producer-reserved first relation, not a copy-chain walk. The
// relation's current flags determine whether it permits coalescing.
const loom_low_placement_relation_t*
loom_low_placement_defining_transfer_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal);

// Returns the immediate tied-storage source for tied result |value_ordinal|.
// This is a constant-time lookup of the producer-reserved first relation and
// does not flatten across destructive writes. The value must be a tied result.
loom_value_ordinal_t loom_low_placement_tied_source_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal);

// Returns the relation range keyed by |source_ordinal|, using the same storage-
// origin projection as result ranges. The returned range indexes
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
