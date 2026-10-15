// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Value rematerialization and allocation-pressure repair.
//
// Target packets explicitly opt their results in to rematerialization; pure
// structural materializations are also eligible. Allocation and scheduling
// retain the evidence that selects a candidate. This utility owns cloning the
// declared producer recipe near each user and retaining private producer
// placement across schedule rebuilds. Tied ownership predecessors are
// mandatory; descriptor-declared replay sources enrich a packet's recipe when
// available. Repeated operands in one user share the same cloned recipe.

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

typedef struct loom_low_value_rematerialization_result_t {
  // SSA value whose defining packet was rematerialized or placement retained.
  loom_value_id_t value_id;
  // Number of producer clones inserted across all distinct users.
  uint32_t cloned_packet_count;
  // Number of operand uses rewritten to cloned packet results.
  uint32_t rewritten_operand_count;
  // Number of original private results whose placement was newly retained.
  uint32_t retained_placement_count;
} loom_low_value_rematerialization_result_t;

typedef struct loom_low_rematerialization_batch_result_t {
  // Number of values repaired by cloning or retained placement.
  uint32_t repaired_value_count;
  // Total producer operations cloned across the repaired values.
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
// spill materialization invalidates placement for values whose users it
// changes. Value IDs remain stable across the supported repair mutations.
typedef struct loom_low_rematerialization_state_t {
  // Arena retaining placement membership across repair attempts.
  iree_arena_allocator_t* arena;
  // Module value IDs with per-user placement since the last invalidation.
  iree_bitmap_t per_user_values;
  // Optional caller-owned module-value bitmap whose register requirements
  // each clone inherits from its source. Growth uses the repair arena.
  iree_bitmap_t* required_register_values;
} loom_low_rematerialization_state_t;

// Rematerializes an eligible SSA value once near each distinct user.
// A result recreates its complete declared producer recipe so each tied clone
// consumes private storage and optional replay sources remain adjacent.
// Multiple operands of one user share one rematerialized value.
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

// Repairs pressure from unspillable classes and register-required spill
// traffic. Selects cross-block demand in over-budget unspillable classes first,
// then the live failure frontier if that demand cannot be repaired. Each set
// forms one consumer-first batch. Private placements retain the completed
// schedule until the first clone invalidates it. Subsequent clones capture
// fresh operand uses without consulting the allocation or schedule snapshot.
// One optional diagnostic summarizes the complete batch. |schedule| is the
// completed order that failed allocation when the caller will rebuild
// scheduling with retained placement. Source-order allocation callers pass NULL
// and only consume IR-rewriting repairs.
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

// Repairs predicted spills as one consumer-first rematerialization batch.
// Candidate selection consumes the immutable allocation snapshot. Application
// retains only value IDs and ordering facts, then queries fresh uses for each
// rewrite so producers see uses introduced by earlier consumer clones.
// Callers rebuild allocation once after the complete batch. One optional
// diagnostic summarizes the batch; its trigger class is the first spill plan.
//
// Returns OK with a zero result when no spill-plan value can be rematerialized.
iree_status_t loom_low_allocation_rematerialize_spill_plans(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REMATERIALIZATION_H_
