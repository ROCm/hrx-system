// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Value rematerialization and allocation-pressure repair.
//
// Target packets explicitly opt their results in to rematerialization. Shared
// structural composition can recreate a closed graph over operand-free target
// leaves. A narrow view retains its
// wider input as a capture instead of rebuilding that input for each use.
// Other native recipes recreate tied ownership predecessors and available
// descriptor-declared sources, retaining their remaining inputs as external
// captures. Recipes preserve consuming order. Each distinct user receives a
// private producer graph, shared by repeated operands of that user.
//
// Allocation and scheduling supply function-local value domains and ordering
// facts. Rematerialization computes closure once per original value in a repair
// batch, then applies consumer-first repairs using fresh operand uses. Private
// producer placement survives schedule rebuilds until another rewrite changes
// its uses.

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

// Repairs a scheduler's architectural-state failure by recreating the selected
// value privately for each user. The failed schedule supplies its retained
// function-local value domain and CFG. Recipe planning consumes verified SSA,
// preserving ownership order and rejecting consumed external inputs. Callers
// rebuild the schedule after rewriting. Returns OK with a zero result when the
// selected value has no safe rematerialization recipe.
iree_status_t loom_low_schedule_rematerialize_failure(
    loom_module_t* module, const struct loom_low_schedule_table_t* schedule,
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
// Every recipe producer and use retains its repeated execution domain.
// Callers rebuild allocation once after the complete batch. One optional
// diagnostic summarizes the batch; its trigger class is the first spill plan.
//
// Returns OK with a zero result when no spill-plan value can be rematerialized.
iree_status_t loom_low_allocation_rematerialize_spill_plans(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result);

// Reconstructs closed constant compositions at the selected reload sites.
// Unlike speculative pressure repair, this replaces storage traffic that would
// execute at those same users, including users inside loops. Each private
// result requires a register, preventing reconstruction/spill cycles. The
// caller supplies state->required_register_values and retains it across
// rebuilds. Returns the remaining storage plans, borrowing |table| when no
// reconstruction applies or allocated from |arena| otherwise. Other allocation
// facts remain an immutable snapshot. Consume the returned plans before
// rebuilding allocation.
iree_status_t loom_low_allocation_reconstruct_spills(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    const loom_low_allocation_spill_plan_t** out_spill_plans,
    iree_host_size_t* out_spill_plan_count,
    loom_low_rematerialization_batch_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REMATERIALIZATION_H_
