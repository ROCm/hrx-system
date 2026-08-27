// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Completed allocation tables and read-only table queries.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_TABLE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_TABLE_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_storage_lease_unit_index_t
    loom_low_allocation_storage_lease_unit_index_t;

typedef enum loom_low_allocation_remark_kind_e {
  // Unknown or uninitialized remark kind.
  LOOM_LOW_ALLOCATION_REMARK_UNKNOWN = 0,
  // An interval could not fit in the configured register budget.
  LOOM_LOW_ALLOCATION_REMARK_SPILL = 1,
} loom_low_allocation_remark_kind_t;

typedef enum loom_low_allocation_copy_kind_e {
  // Unknown or uninitialized copy decision kind.
  LOOM_LOW_ALLOCATION_COPY_UNKNOWN = 0,
  // The copy source and result share one assigned location.
  LOOM_LOW_ALLOCATION_COPY_COALESCED = 1,
  // The copy must remain a target move/copy after allocation.
  LOOM_LOW_ALLOCATION_COPY_MATERIALIZED = 2,
} loom_low_allocation_copy_kind_t;

typedef enum loom_low_allocation_failure_blocking_kind_e {
  // No specific blocking constraint was recorded.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_UNKNOWN = 0,
  // The failing interval itself is wider than the register-class budget.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET = 1,
  // A live assignment occupies a candidate location and cannot be evicted.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT = 2,
  // A fixed value, reserved range, or storage lease blocks a candidate
  // location.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT = 3,
  // The allocator scanned the candidate locations without finding a legal
  // placement or a more specific blocking constraint.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION = 4,
} loom_low_allocation_failure_blocking_kind_t;

typedef enum loom_low_allocation_value_scratch_flag_bits_e {
  // The lease acquired the module value-ordinal scratch map and must release
  // it.
  LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED = 1u << 0,
} loom_low_allocation_value_scratch_flag_bits_t;

// Bitset of loom_low_allocation_value_scratch_flag_bits_t values.
typedef uint16_t loom_low_allocation_value_scratch_flags_t;

// Allocation remark for agent/tool feedback.
typedef struct loom_low_allocation_remark_t {
  // Remark category.
  loom_low_allocation_remark_kind_t kind;
  // Assignment index associated with this remark.
  uint32_t assignment_index;
  // Budget that was exceeded, or UINT32_MAX when no fixed budget was active.
  uint32_t budget_units;
  // Units requested by the assignment.
  uint32_t required_units;
} loom_low_allocation_remark_t;

// Terminal hard-allocation failure for one interval.
typedef struct loom_low_allocation_failure_t {
  // Stable failure code emitted by the structured diagnostic.
  iree_string_view_t failure_code;
  // SSA value whose interval could not be assigned.
  loom_value_id_t value_id;
  // Pressure/allocation class for |value_id|.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Program point where the failed interval starts.
  uint32_t start_point;
  // One-past-last storage program point required by the failed interval.
  uint32_t end_point;
  // Allocation units required by the failed interval.
  uint32_t required_unit_count;
  // Maximum allocation units available, or UINT32_MAX when unbounded.
  uint32_t budget_units;
  // Maximum boundary-live units observed for this pressure class.
  uint32_t peak_live_units;
  // Candidate location kind used while diagnosing the failure.
  loom_low_allocation_location_kind_t location_kind;
  // Candidate base physical register or target ID used for conflict reporting,
  // or UINT32_MAX when no concrete candidate was inspected.
  uint32_t location_base;
  // Candidate location width used for conflict reporting, or zero when no
  // concrete candidate was inspected.
  uint32_t location_count;
  // Structured category describing the first blocking constraint found.
  loom_low_allocation_failure_blocking_kind_t blocking_kind;
  // Assignment index for an active-assignment conflict, or UINT32_MAX.
  uint32_t conflict_assignment_index;
  // SSA value occupying the conflicting assignment, or LOOM_VALUE_ID_INVALID.
  loom_value_id_t conflict_value_id;
  // Program point where the conflicting assignment starts, or UINT32_MAX.
  uint32_t conflict_start_point;
  // One-past-last storage program point for the conflicting assignment, or
  // UINT32_MAX.
  uint32_t conflict_end_point;
  // Conflicting assignment location kind.
  loom_low_allocation_location_kind_t conflict_location_kind;
  // Conflicting assignment base physical register or target ID, or UINT32_MAX.
  uint32_t conflict_location_base;
  // Conflicting assignment location width, or zero when unavailable.
  uint32_t conflict_location_count;
} loom_low_allocation_failure_t;

// Returns true when |failure| describes a terminal hard-allocation failure.
static inline bool loom_low_allocation_failure_is_present(
    const loom_low_allocation_failure_t* failure) {
  return failure != NULL && !iree_string_view_is_empty(failure->failure_code);
}

// Spill materialization plan for one spilled assignment.
typedef struct loom_low_allocation_spill_plan_t {
  // SSA value represented by the spilled assignment.
  loom_value_id_t value_id;
  // Spilled assignment index in |assignments|.
  uint32_t assignment_index;
  // Spill slot ordinal assigned to this interval.
  uint32_t slot_index;
  // Storage space used by the spill slot.
  loom_low_spill_slot_space_t slot_space;
  // Slot size in bytes.
  uint32_t byte_size;
  // Required slot alignment in bytes.
  uint32_t byte_alignment;
  // Predicted stores needed by the current synthetic spill plan. Actual
  // materialization may differ after earlier spill rewrites in the same frame.
  uint32_t store_count;
  // Predicted operand-use reloads in the current synthetic spill plan. Actual
  // materialization may differ after earlier spill rewrites in the same frame.
  uint32_t reload_count;
} loom_low_allocation_spill_plan_t;

// Copy/coalescing decision for one low.copy op.
typedef struct loom_low_allocation_copy_decision_t {
  // Source SSA value consumed by the low.copy op.
  loom_value_id_t source_value_id;
  // Result SSA value produced by the low.copy op.
  loom_value_id_t result_value_id;
  // Assignment index for |source_value_id|.
  uint32_t source_assignment_index;
  // Assignment index for |result_value_id|.
  uint32_t result_assignment_index;
  // Whether allocation coalesced or must materialize the copy.
  loom_low_allocation_copy_kind_t kind;
} loom_low_allocation_copy_decision_t;

// Copy required to materialize one low.br edge payload into a destination block
// argument location.
typedef struct loom_low_allocation_edge_copy_t {
  // Operand index in the owning low.br payload.
  uint16_t payload_index;
  // SSA value providing this copied segment.
  loom_value_id_t source_value_id;
  // Destination block argument receiving this copied segment.
  loom_value_id_t destination_value_id;
  // Assignment index for |source_value_id|.
  uint32_t source_assignment_index;
  // Assignment index for |destination_value_id|.
  uint32_t destination_assignment_index;
  // Unit offset inside the source assignment.
  uint32_t source_unit_offset;
  // Unit offset inside the destination assignment.
  uint32_t destination_unit_offset;
  // Number of assignment units copied by this segment.
  uint32_t unit_count;
} loom_low_allocation_edge_copy_t;

// Contiguous edge-copy group for one low.br terminator.
typedef struct loom_low_allocation_edge_copy_group_t {
  // low.br terminator that owns this outgoing edge.
  const loom_op_t* terminator_op;
  // Source-order ordinal of |terminator_op| in its low function body.
  uint32_t source_ordinal;
  // First edge-copy record for |terminator_op|.
  iree_host_size_t copy_start;
  // Number of edge-copy records for |terminator_op|.
  iree_host_size_t copy_count;
  // Final sequential physical moves emitted before |terminator_op|.
  loom_low_move_group_t move_group;
} loom_low_allocation_edge_copy_group_t;

// Final move group for one materialized packet-local parallel move operation.
typedef struct loom_low_allocation_packet_move_group_t {
  // Source-order ordinal of the owning low.copy, low.move, low.slice, or
  // low.concat.
  uint32_t source_ordinal;
  // Structural placement cause that produced the move group.
  loom_low_placement_cause_t cause;
  // Final sequential physical moves emitted by the owning operation.
  loom_low_move_group_t move_group;
} loom_low_allocation_packet_move_group_t;

// Assignment-backed storage lease over target-visible physical units.
//
// Each record corresponds to one entry in |storage_leases.records|. The lease
// record names the scheduled packet attachment and target release class; this
// allocation-side record names the concrete assignment subrange that must
// remain unavailable for incompatible reuse while the lease is active.
typedef struct loom_low_allocation_storage_lease_t {
  // Index into |storage_leases.records|.
  uint32_t lease_record_index;
  // Assignment carrying the leased value.
  uint32_t assignment_index;
  // SSA value whose assignment owns the leased units.
  loom_value_id_t value_id;
  // Program point where the lease becomes active.
  uint32_t start_point;
  // Program point where allocation may reuse the leased units.
  uint32_t end_point;
  // Release action covering this lease, or STORAGE_RELEASE_ACTION_INDEX_NONE.
  uint32_t release_action_index;
  // Descriptor-set-local register class owning the leased units.
  uint16_t descriptor_reg_class_id;
  // Target-visible storage kind for the leased units.
  loom_low_allocation_location_kind_t location_kind;
  // First physical register or target ID leased.
  uint32_t location_base;
  // Number of contiguous physical units leased.
  uint32_t location_count;
} loom_low_allocation_storage_lease_t;

// Allocation table for one target-low function body. All arrays are
// arena-owned by the caller-provided arena passed to
// loom_low_allocate_function.
typedef struct loom_low_allocation_table_t {
  // Module containing the allocated low function.
  loom_module_t* module;
  // Target-low function operation allocated by this table.
  const loom_op_t* function_op;
  // Resolved target context selected by |function_op|.
  loom_low_resolved_target_t target;
  // Liveness analysis that produced the allocated intervals.
  loom_liveness_analysis_t liveness;
  // Placement relations consumed while assigning intervals.
  loom_low_placement_table_t placement;
  // Resolved fixed SSA value locations consumed by this allocation.
  const loom_low_allocation_resolved_fixed_value_t* fixed_values;
  // Number of records in |fixed_values|.
  iree_host_size_t fixed_value_count;
  // Allocation mode requested on the low function, or 0 for the default.
  uint8_t allocation_mode;
  // Number of error diagnostics emitted while attempting allocation.
  uint32_t error_count;
  // Per-interval assignments in allocation order.
  const loom_low_allocation_assignment_t* assignments;
  // Number of records in |assignments|.
  iree_host_size_t assignment_count;
  // Dense physical extents retained from assignment and move planning.
  struct {
    // Maximum one-past-last assigned or move-scratch location indexed by
    // descriptor register class ID.
    const uint32_t* ends_by_reg_class;
    // Number of entries in |ends_by_reg_class|.
    iree_host_size_t count;
  } physical_extents;
  // Assignment indices by liveness local value ordinal. Entries without an
  // assignment contain UINT32_MAX.
  const uint32_t* assignment_indices_by_value_ordinal;
  // First live storage program point for each assigned unit.
  const uint32_t* unit_start_points;
  // One-past-last live program point for each assigned unit.
  const uint32_t* unit_end_points;
  // Number of records in |unit_start_points| and |unit_end_points|.
  iree_host_size_t unit_point_count;
  // Spill materialization plans in assignment order.
  const loom_low_allocation_spill_plan_t* spill_plans;
  // Number of records in |spill_plans|.
  iree_host_size_t spill_plan_count;
  // Allocation remarks in assignment order.
  const loom_low_allocation_remark_t* remarks;
  // Number of records in |remarks|.
  iree_host_size_t remark_count;
  // Terminal hard-allocation failure when |error_count| is non-zero.
  loom_low_allocation_failure_t failure;
  // Copy/coalescing decisions in source order.
  const loom_low_allocation_copy_decision_t* copy_decisions;
  // Number of records in |copy_decisions|.
  iree_host_size_t copy_decision_count;
  // Edge-copy records grouped by low.br terminator source order.
  const loom_low_allocation_edge_copy_t* edge_copies;
  // Number of records in |edge_copies|.
  iree_host_size_t edge_copy_count;
  // Per-low.br groups indexing |edge_copies|.
  const loom_low_allocation_edge_copy_group_t* edge_copy_groups;
  // Number of records in |edge_copy_groups|.
  iree_host_size_t edge_copy_group_count;
  // Packet-local move groups in source order.
  const loom_low_allocation_packet_move_group_t* packet_move_groups;
  // Number of records in |packet_move_groups|.
  iree_host_size_t packet_move_group_count;
  // Final sequential physical move rows shared by all move groups.
  const loom_low_move_t* moves;
  // Indices into |moves| for the first row writing each cycle-scratch location.
  const iree_host_size_t* scratch_move_indices;
  // Number of final move rows across packet-local move groups.
  iree_host_size_t packet_move_count;
  // Target storage-lease facts consumed by this allocation.
  loom_low_storage_lease_table_t storage_leases;
  // Assignment-backed storage-lease records in storage-lease table order.
  const loom_low_allocation_storage_lease_t* storage_lease_instances;
  // Number of records in |storage_lease_instances|.
  iree_host_size_t storage_lease_instance_count;
  // Physical-unit index over |storage_lease_instances| retained from
  // allocation, or NULL when no register-like leases were materialized.
  const loom_low_allocation_storage_lease_unit_index_t*
      storage_lease_unit_index;
  // Allocator-requested storage release actions in allocation order.
  const loom_low_storage_release_action_t* storage_release_actions;
  // Number of records in |storage_release_actions|.
  iree_host_size_t storage_release_action_count;
  // Number of assignments whose location kind is SPILL_SLOT.
  iree_host_size_t spill_count;
  // Number of low.copy ops coalesced into one location.
  iree_host_size_t coalesced_copy_count;
  // Number of low.copy ops that must remain materialized.
  iree_host_size_t materialized_copy_count;
  // Resolved whole-function target-owned location ranges.
  const loom_low_allocation_resolved_reserved_range_t* reserved_ranges;
  // Number of records in |reserved_ranges|.
  iree_host_size_t reserved_range_count;
  // Function CFG used to construct this allocation. Its block adjacency
  // remains valid while spill materialization rewrites branch payloads without
  // changing topology; edge terminator pointers remain snapshot facts.
  loom_cfg_graph_t cfg_graph;
} loom_low_allocation_table_t;

// Active allocation-owned lease over the module value-ordinal scratch map.
typedef struct loom_low_allocation_value_scratch_t {
  // Module whose scratch map is borrowed by this lease.
  loom_module_t* module;
  // Allocation table whose liveness value order defines the active map.
  const loom_low_allocation_table_t* table;
  // Value IDs indexed by allocation-local ordinal.
  const loom_value_id_t* value_ids;
  // Number of records in |value_ids|.
  iree_host_size_t value_count;
  // Lease lifecycle flags.
  loom_low_allocation_value_scratch_flags_t flags;
} loom_low_allocation_value_scratch_t;

// Acquires |table|'s local value map in module ordinal scratch.
iree_status_t loom_low_allocation_acquire_value_scratch(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_value_scratch_t* out_scratch);

// Releases an allocation value scratch lease.
void loom_low_allocation_release_value_scratch(
    loom_low_allocation_value_scratch_t* scratch);

// Returns the assignment for |value_ordinal|, or NULL when that local value has
// no assignment. |out_assignment_index| is optional.
const loom_low_allocation_assignment_t*
loom_low_allocation_assignment_for_value_ordinal(
    const loom_low_allocation_table_t* table,
    loom_value_ordinal_t value_ordinal, uint32_t* out_assignment_index);

// Maps |value_id| through the active module value-ordinal scratch map to its
// assignment, or NULL when the active local domain has no assignment for the
// value. |out_assignment_index| is optional.
const loom_low_allocation_assignment_t*
loom_low_allocation_try_map_active_value_assignment(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index);

// Maps |value_id| through the active module value-ordinal scratch map to its
// assignment. The value must have an assignment; missing entries are compiler
// bugs.
const loom_low_allocation_assignment_t*
loom_low_allocation_map_active_value_assignment(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index);

// Finds the edge-copy group for the source-order node, or NULL when the node
// has no edge-copy payload.
const loom_low_allocation_edge_copy_group_t*
loom_low_allocation_find_edge_copy_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal);

// Finds the materialized packet-local move group for the source-order node, or
// NULL when the node emits no final moves.
const loom_low_allocation_packet_move_group_t*
loom_low_allocation_find_packet_move_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal);

// Resolves the descriptor-set register class spelling for |assignment|.
iree_status_t loom_low_allocation_assignment_register_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t* out_register_class_name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_TABLE_H_
