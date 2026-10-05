// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-guided value rematerialization and allocation-pressure repair.
//
// This layer mutates IR only when a pure descriptor packet explicitly opts its
// result in to rematerialization. Allocation and scheduling retain the evidence
// that selects a candidate; this utility owns cloning the producer ownership
// chain near each user and retaining private producer placement across schedule
// rebuilds. Every packet in a tied chain must opt in. Repeated operands in one
// user share the same cloned chain.

#ifndef LOOM_CODEGEN_LOW_REMATERIALIZATION_H_
#define LOOM_CODEGEN_LOW_REMATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_rematerialization_trigger_e {
  // Unknown or uninitialized rematerialization trigger.
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_UNKNOWN = 0,
  // A terminal allocation failure was repaired by rematerialization.
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE = 1,
  // A predicted spill plan was avoided by rematerialization.
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_SPILL_PLAN = 2,
} loom_low_allocation_rematerialization_trigger_t;

typedef struct loom_low_value_rematerialization_result_t {
  // SSA value whose defining packet was rematerialized or placement retained.
  loom_value_id_t value_id;
  // Number of descriptor packet clones inserted across all distinct users.
  uint32_t cloned_packet_count;
  // Number of operand uses rewritten to cloned packet results.
  uint32_t rewritten_operand_count;
  // Number of original private results whose placement was newly retained.
  uint32_t retained_placement_count;
} loom_low_value_rematerialization_result_t;

typedef struct loom_low_allocation_rematerialization_result_t {
  // Descriptor-guided value rematerialization performed by the repair.
  loom_low_value_rematerialization_result_t value;
  // Original register class in the allocation's target descriptor set.
  // Retained by value across IR rewrites and analysis snapshot retirement.
  uint16_t descriptor_reg_class_id;
} loom_low_allocation_rematerialization_result_t;

typedef struct loom_low_rematerialization_batch_result_t {
  // Total descriptor packets cloned across the repaired values.
  uint32_t cloned_packet_count;
  // Total operand uses rewritten across the repaired values.
  uint32_t rewritten_operand_count;
  // Total original private results whose placement was newly retained.
  uint32_t retained_placement_count;
} loom_low_rematerialization_batch_result_t;

// Retained producer facts for one scheduling/allocation repair lifecycle.
// Initialize with the repair arena and an empty bitmap before the first
// attempt. The arena and state survive analysis rebuilds until the repair loop
// finishes. Marked results have private per-user placement, either from cloning
// or from retaining an already-adjacent definition after scheduling separated
// it from its user. Inserting other operand definitions does not make them
// candidates again. Cloning a consumer invalidates placement for its inputs;
// spill materialization invalidates all placement. Value IDs remain stable
// across the supported repair mutations.
typedef struct loom_low_rematerialization_state_t {
  // Arena retaining placement membership across repair attempts.
  iree_arena_allocator_t* arena;
  // Module value IDs with per-user placement since the last invalidation.
  iree_bitmap_t per_user_values;
  // Optional caller-owned module-value bitmap whose register requirements
  // each clone inherits from its source. Growth uses the repair arena.
  iree_bitmap_t* required_register_values;
} loom_low_rematerialization_state_t;

// Invalidates per-user placement after spill traffic changes live ranges.
// Retains allocated membership storage for subsequent repair attempts.
void loom_low_rematerialization_invalidate_placement(
    loom_low_rematerialization_state_t* state);

// Rematerializes a descriptor-backed SSA value once near each distinct user.
// A tied result recreates its complete rematerializable ownership chain so each
// clone consumes private storage. Multiple operands of one user share one
// rematerialized value.
// Consumes verified IR: the definition dominates its existing operand uses.
// Recipes with consumed external inputs are ineligible, so every remaining
// input and external type/attribute capture remains available at each clone.
// When |schedule| is the completed order that failed allocation, an original
// data-dependent result already adjacent to its sole user can retain that
// placement instead of cloning if scheduling separated the pair. The caller
// must consume retained placement by rebuilding the schedule before allocating
// again. NULL permits only IR-rewriting repairs. The snapshot is queried only
// before mutation; callers rewriting a preplanned batch pass NULL.
//
// Returns OK with a zero result when |value_id| is not a safe rematerialization
// candidate. A preplanned batch may retain immutable snapshot facts while
// rewriting; callers rebuild analyses before making further planning decisions.
iree_status_t loom_low_rematerialize_value_uses(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, const struct loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result);

// Repairs cross-block rematerializations in over-budget unspillable classes
// as one consumer-first batch. When no batch applies, repairs the terminal
// failure frontier: private placements accumulate while the snapshot remains
// unchanged, stopping before a clone once placement warrants a new schedule.
// Otherwise one genuine rewrite ends use of that snapshot. Each cloned source
// emits its own optional decision.
// |schedule| is the completed order that failed allocation when the caller
// will rebuild scheduling with retained placement. Source-order allocation
// callers pass NULL and only consume IR-rewriting repairs.
//
// Returns OK with a zero result when the failure is not a rematerialization
// candidate. User IR failures remain allocation diagnostics; status failures
// are reserved for allocation failures retaining repair state or rewriting IR.
iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const struct loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result);

// Attempts to repair one predicted spill plan by rematerializing its value
// instead of materializing storage traffic.
//
// Returns OK with a zero result when no spill-plan value is a rematerialization
// candidate. When a value is rewritten, callers must rebuild allocation before
// consulting the old allocation table again.
iree_status_t loom_low_allocation_rematerialize_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result);

// Emits a structured remark describing a successful rematerialization result.
iree_status_t loom_low_allocation_rematerialization_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_rematerialization_trigger_t trigger,
    const loom_low_allocation_rematerialization_result_t* result,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REMATERIALIZATION_H_
