// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Spill plan construction for allocation assignments.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_PLAN_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Predicted memory traffic for materializing one spilled value.
typedef struct loom_low_allocation_spill_plan_traffic_t {
  // Predicted stores needed by the current synthetic spill plan.
  uint32_t store_count;
  // Predicted bytes stored by the current synthetic spill plan.
  uint64_t store_bytes;
  // Predicted operand-use reloads in the current synthetic spill plan.
  uint32_t reload_count;
  // Predicted bytes reloaded by the current synthetic spill plan.
  uint64_t reload_bytes;
} loom_low_allocation_spill_plan_traffic_t;

// Reusable traffic-query scratch for one immutable function CFG. Contents are
// empty between queries; storage belongs to the allocation attempt's arena.
typedef struct loom_low_allocation_spill_plan_traffic_workspace_t {
  // Slice-use totals and touched-list links, indexed by region block index.
  struct loom_low_allocation_spill_plan_slice_group_t* slice_groups_by_block;
} loom_low_allocation_spill_plan_traffic_workspace_t;

// Reference to an original SSA value or a value produced by an earlier spill
// materialization action. Exactly one field is valid.
typedef struct loom_low_spill_materialization_value_ref_t {
  // Original SSA value, or LOOM_VALUE_ID_INVALID for an action result.
  loom_value_id_t value_id;
  // Producing action index, or UINT32_MAX for an original SSA value.
  uint32_t action_index;
} loom_low_spill_materialization_value_ref_t;

typedef enum loom_low_spill_materialization_target_kind_e {
  // Operand on an operation retained from the source function.
  LOOM_LOW_SPILL_MATERIALIZATION_TARGET_OPERATION_OPERAND = 0,
  // Payload for an original destination block argument on the current branch.
  LOOM_LOW_SPILL_MATERIALIZATION_TARGET_BRANCH_ARGUMENT = 1,
  // Logical source consumed by an earlier materialization action.
  LOOM_LOW_SPILL_MATERIALIZATION_TARGET_ACTION_SOURCE = 2,
} loom_low_spill_materialization_target_kind_t;

// Mutable logical source targeted by a spill reload. Branch targets name the
// destination argument instead of a positional operand so earlier argument
// removal cannot invalidate the target. Action sources are resolved before the
// consuming traffic action is emitted.
typedef struct loom_low_spill_materialization_target_t {
  // Target kind selecting the active union member.
  uint8_t kind;
  // Operand index for operation targets.
  uint16_t operand_index;
  // Original destination block argument for branch targets.
  loom_value_id_t block_argument_value_id;
  // Owner of the targeted operand occurrence.
  union {
    // Source operation owning an original operand.
    loom_op_t* operation;
    // Predecessor block whose current terminator owns a branch payload.
    loom_block_t* predecessor_block;
    // Earlier action consuming the logical source.
    uint32_t action_index;
  } owner;
} loom_low_spill_materialization_target_t;

typedef enum loom_low_spill_materialization_action_kind_e {
  // Store a spilled value at its definition or function entry.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE = 0,
  // Store a current predecessor payload before its branch.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE = 1,
  // Reload a complete spilled value for one operand occurrence.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD = 2,
  // Reload one addressable slice and erase the original slice operation.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD = 3,
  // Reload a complete spilled value shared by a block-local slice group.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD = 4,
  // Remove one current predecessor payload from its branch.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT = 5,
  // Remove a spilled non-entry block argument from its block.
  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT = 6,
} loom_low_spill_materialization_action_kind_t;

// One ordered spill materialization action. The action sequence models the
// semantic operand and CFG effects that application performs, so later actions
// target stable occurrences instead of rediscovering uses from rewritten IR.
typedef struct loom_low_spill_materialization_action_t {
  // Action kind selecting the active union member.
  loom_low_spill_materialization_action_kind_t kind;
  // Index of the spill plan whose storage this action reads or writes.
  uint32_t spill_plan_index;
  union {
    struct {
      // Current payload stored on the predecessor edge.
      loom_low_spill_materialization_value_ref_t source;
      // Predecessor block receiving the store before its current terminator.
      loom_block_t* predecessor_block;
    } edge_store;
    struct {
      // Operand occurrence replaced by the reload result.
      loom_low_spill_materialization_target_t target;
    } operand_reload;
    struct {
      // Slice operation replaced and erased by this reload.
      loom_op_t* slice_op;
      // Byte offset within the spill slot.
      int64_t byte_offset;
    } narrow_slice_reload;
    struct {
      // Earliest slice operation anchoring the shared reload.
      loom_op_t* first_slice_op;
      // First slice operand target in the plan's flat target table.
      uint32_t target_start;
      // Number of slice operand targets sharing the reload.
      uint32_t target_count;
    } shared_slice_reload;
    struct {
      // Predecessor block whose current branch loses one payload.
      loom_block_t* predecessor_block;
      // Original destination argument identifying the current payload index.
      loom_value_id_t block_argument_value_id;
    } remove_branch_argument;
    struct {
      // Original block argument removed from its current signature position.
      loom_value_id_t value_id;
    } remove_block_argument;
  } value;
} loom_low_spill_materialization_action_t;

enum {
  // The instruction executes after its original liveness point instead of
  // before it. Action counts are bounded below this bit during plan creation.
  LOOM_LOW_SPILL_MATERIALIZATION_INSTRUCTION_AFTER_BIT = UINT32_C(1) << 31,
};

// One traffic-producing action projected into accepted program-point and final
// execution order. The action retains its immutable IR mutation anchor; target
// recipes may expand it into native instructions while preserving this order.
typedef struct loom_low_spill_materialization_instruction_t {
  // Liveness operation row whose boundary locates the generated instruction.
  uint32_t operation_index;
  // Action index with INSTRUCTION_AFTER_BIT selecting the execution side.
  uint32_t action_index_and_after;
} loom_low_spill_materialization_instruction_t;

// Returns the transformation-order action referenced by |instruction|.
static inline uint32_t loom_low_spill_materialization_instruction_action_index(
    const loom_low_spill_materialization_instruction_t* instruction) {
  return instruction->action_index_and_after &
         ~LOOM_LOW_SPILL_MATERIALIZATION_INSTRUCTION_AFTER_BIT;
}

// Returns true when |instruction| executes after its anchor point.
static inline bool loom_low_spill_materialization_instruction_is_after(
    const loom_low_spill_materialization_instruction_t* instruction) {
  return iree_any_bit_set(instruction->action_index_and_after,
                          LOOM_LOW_SPILL_MATERIALIZATION_INSTRUCTION_AFTER_BIT);
}

// Exact memory traffic for one selected spill plan.
typedef struct loom_low_spill_materialization_record_t {
  // True when the spilled value is a block argument.
  bool value_is_block_argument;
  // Number of stores materialized for this spill.
  uint32_t store_count;
  // Number of bytes stored for this spill.
  uint64_t store_bytes;
  // Number of reloads materialized for this spill.
  uint32_t reload_count;
  // Number of bytes reloaded for this spill.
  uint64_t reload_bytes;
} loom_low_spill_materialization_record_t;

// Returns true when |record| retains traffic that requires spill storage.
static inline bool loom_low_spill_materialization_record_has_traffic(
    const loom_low_spill_materialization_record_t* record) {
  return record->store_count != 0 || record->reload_count != 0;
}

// Immutable ordered realization of one selected spill-plan batch.
typedef struct loom_low_spill_materialization_plan_t {
  // Actions in transformation order.
  const loom_low_spill_materialization_action_t* actions;
  // Number of records in |actions|.
  iree_host_size_t action_count;
  // Traffic-producing actions in accepted point and per-anchor execution order.
  const loom_low_spill_materialization_instruction_t* instructions;
  // Number of records in |instructions|.
  iree_host_size_t instruction_count;
  // Shared-slice operand targets referenced by action spans.
  const loom_low_spill_materialization_target_t* targets;
  // Number of records in |targets|.
  iree_host_size_t target_count;
  // Exact action and traffic records indexed by input spill plan.
  const loom_low_spill_materialization_record_t* records;
  // Number of records in |records|.
  iree_host_size_t record_count;
  // Earliest traffic-producing plan sharing each spill reservation, or NULL
  // when every traffic-producing plan has an independent reservation.
  // Records without traffic map to themselves and require no reservation.
  const uint32_t* slot_representatives;
} loom_low_spill_materialization_plan_t;

typedef enum loom_low_spill_materialization_traffic_kind_e {
  LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_STORE = 0,
  LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_RELOAD = 1,
} loom_low_spill_materialization_traffic_kind_t;

// Target-independent access shape for one materialization action. This is a
// transient query result; the ordered action remains the retained source of
// structural mutation and value identity.
typedef struct loom_low_spill_materialization_traffic_t {
  // Store or reload traffic for the reported action.
  loom_low_spill_materialization_traffic_kind_t kind;
  // Complete register value transferred by the action.
  loom_type_t value_type;
  // Byte offset within the action's spill slot.
  int64_t byte_offset;
} loom_low_spill_materialization_traffic_t;

enum {
  // Equal-byte full reloads are only worthwhile after removing enough reload
  // packets to offset the longer live range of the reloaded tuple.
  LOOM_LOW_ALLOCATION_DENSE_SLICE_RELOAD_MIN_SLICE_COUNT = 8u,
};

// Computes the byte size and alignment required for spilling |assignment|.
iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment);

// Returns true and the complete result byte size when |use_op| is a slice of
// a spilled register range whose units can be addressed individually in bytes.
// The assignment and operation belong to verified Low IR.
bool loom_low_allocation_spill_plan_slice_reload_byte_size(
    const loom_module_t* module,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t spill_byte_size, const loom_op_t* use_op,
    uint32_t* out_reload_byte_size);

// Returns true when a block-local slice group should share one full reload.
bool loom_low_allocation_spill_plan_use_full_slice_reload(
    uint32_t slice_count, uint64_t narrow_reload_bytes,
    uint32_t spill_byte_size);

// Resolves the traffic shape for one retained action in constant time. Target
// planners use this instead of interpreting mutable IR or reconstructing action
// result types. Returns false for structural-only actions.
bool loom_low_spill_materialization_plan_query_traffic(
    const loom_module_t* module,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_spill_materialization_plan_t* plan,
    iree_host_size_t action_index,
    loom_low_spill_materialization_traffic_t* out_traffic);

// Builds the exact ordered actions for the selected spill batch. The source IR
// and allocation snapshot remain immutable. All retained arrays are owned by
// |arena|; branch and generated-use scratch is released before return.
iree_status_t loom_low_spill_materialization_plan_build(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count, iree_arena_allocator_t* arena,
    loom_low_spill_materialization_plan_t* out_plan);

// Allocates traffic-query scratch once for |body|. Call when spill selection
// first becomes necessary, using the owning allocation attempt's scratch arena.
iree_status_t loom_low_allocation_spill_plan_traffic_workspace_initialize(
    const loom_region_t* body, iree_arena_allocator_t* arena,
    loom_low_allocation_spill_plan_traffic_workspace_t* out_workspace);

// Computes the predicted memory traffic for spilling |assignment| using
// |workspace| initialized for |cfg_graph|'s region. Queries are sequential.
// Block arguments must belong to |cfg_graph|'s region; nested-region arguments
// are excluded by spill-register requirement classification before planning.
iree_status_t loom_low_allocation_spill_plan_traffic(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits,
    loom_low_allocation_spill_plan_traffic_workspace_t* workspace,
    loom_low_allocation_spill_plan_traffic_t* out_traffic);

// Appends the spill materialization plan for |assignment| using its previously
// computed |traffic|. The traffic fact must describe the same assignment and
// allocation-unit width passed here.
iree_status_t loom_low_allocation_spill_plan_append(
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, uint16_t alloc_unit_bits,
    loom_low_spill_slot_space_t spill_slot_space,
    const loom_low_allocation_spill_plan_traffic_t* traffic,
    loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t* inout_spill_plan_count);

// Appends a spill remark for |assignment_index|.
void loom_low_allocation_spill_remark_record(
    loom_low_allocation_remark_t* remarks, iree_host_size_t* inout_remark_count,
    uint32_t assignment_index, uint32_t budget_units, uint32_t required_units);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_PLAN_H_
