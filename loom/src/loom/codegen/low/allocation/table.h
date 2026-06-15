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
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  // Predicted stores needed by the current synthetic spill plan.
  uint32_t store_count;
  // Predicted operand-use reloads in the current synthetic spill plan.
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

// Scratch unit reserved for sequencing cyclic edge-copy moves.
typedef struct loom_low_allocation_edge_copy_temporary_t {
  // Storage class of the cyclic move set.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Target-visible scratch location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register, target ID, or spill slot ordinal used as scratch.
  uint32_t location;
} loom_low_allocation_edge_copy_temporary_t;

// Scratch unit reserved for sequencing one cyclic packet-local move set.
typedef struct loom_low_allocation_packet_move_temporary_t {
  // Storage class of the cyclic move set.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Target-visible scratch location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register, target ID, or spill slot ordinal used as scratch.
  uint32_t location;
} loom_low_allocation_packet_move_temporary_t;

// Contiguous edge-copy group for one low.br terminator.
typedef struct loom_low_allocation_edge_copy_group_t {
  // low.br terminator that owns this outgoing edge.
  const loom_op_t* terminator_op;
  // Source-order ordinal of |terminator_op| in its low function body.
  uint32_t source_ordinal;
  // Program point where the edge copies execute before |terminator_op|.
  uint32_t program_point;
  // First edge-copy record for |terminator_op|.
  uint32_t copy_start;
  // Number of edge-copy records for |terminator_op|.
  uint32_t copy_count;
  // First temporary record reserved for |terminator_op|.
  uint32_t temporary_start;
  // Number of temporary records reserved for |terminator_op|.
  uint32_t temporary_count;
} loom_low_allocation_edge_copy_group_t;

// Scratch-unit group for one packet-local parallel move operation.
typedef struct loom_low_allocation_packet_move_temporary_group_t {
  // low.copy, low.slice, or low.concat operation that owns the move set.
  const loom_op_t* op;
  // Source-order ordinal of |op| in its low function body.
  uint32_t source_ordinal;
  // Program point where the packet-local moves execute.
  uint32_t program_point;
  // First scratch record for |op|.
  uint32_t temporary_start;
  // Number of scratch records reserved for |op|.
  uint32_t temporary_count;
} loom_low_allocation_packet_move_temporary_group_t;

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
  // Allocation mode requested on the low function, or 0 for the default.
  uint8_t allocation_mode;
  // Number of error diagnostics emitted while attempting allocation.
  uint32_t error_count;
  // Per-interval assignments in allocation order.
  const loom_low_allocation_assignment_t* assignments;
  // Number of records in |assignments|.
  iree_host_size_t assignment_count;
  // Assignment indices by liveness local value ordinal. Entries without an
  // assignment contain UINT32_MAX.
  const uint32_t* assignment_indices_by_value_ordinal;
  // One-past-last live program point for each assigned unit.
  const uint32_t* unit_end_points;
  // Number of records in |unit_end_points|.
  iree_host_size_t unit_end_point_count;
  // Spill materialization plans in assignment order.
  const loom_low_allocation_spill_plan_t* spill_plans;
  // Number of records in |spill_plans|.
  iree_host_size_t spill_plan_count;
  // Allocation remarks in assignment order.
  const loom_low_allocation_remark_t* remarks;
  // Number of records in |remarks|.
  iree_host_size_t remark_count;
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
  // Scratch units reserved for cyclic edge-copy groups.
  const loom_low_allocation_edge_copy_temporary_t* edge_copy_temporaries;
  // Number of records in |edge_copy_temporaries|.
  iree_host_size_t edge_copy_temporary_count;
  // Per-packet groups indexing |packet_move_temporaries|.
  const loom_low_allocation_packet_move_temporary_group_t*
      packet_move_temporary_groups;
  // Number of records in |packet_move_temporary_groups|.
  iree_host_size_t packet_move_temporary_group_count;
  // Scratch units reserved for cyclic packet-local move groups.
  const loom_low_allocation_packet_move_temporary_t* packet_move_temporaries;
  // Number of records in |packet_move_temporaries|.
  iree_host_size_t packet_move_temporary_count;
  // Target storage-lease facts consumed by this allocation.
  loom_low_storage_lease_table_t storage_leases;
  // Assignment-backed storage-lease records in storage-lease table order.
  const loom_low_allocation_storage_lease_t* storage_lease_instances;
  // Number of records in |storage_lease_instances|.
  iree_host_size_t storage_lease_instance_count;
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

// Finds the packet-local move temporary group for the source-order node, or
// NULL when the node has no cyclic packet-local move set.
const loom_low_allocation_packet_move_temporary_group_t*
loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
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
