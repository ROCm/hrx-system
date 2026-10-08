// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-pressure repair by fixed-value live-range detachment.
//
// Fixed values model ABI or target locations that must occupy a particular
// physical slot while live. When such a value stays live long enough to force a
// spill, this utility can insert one low.copy or ownership-preserving low.move
// after the fixed source is materialized and rewrite later users to the
// transfer result. Entry sources are copied after the complete live-in/resource
// preamble; other sources are copied immediately after their definition.
// Preamble uses retain the original input. The fixed source's live range ends
// at the transfer; the result has a preference for disjoint storage but no
// fixed-location requirement. Rebuilding allocation determines whether the
// split avoids spills.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Aggregate edits from one fixed-value detachment batch.
typedef struct loom_low_allocation_live_range_split_result_t {
  // Number of low.copy or low.move transfer packets inserted.
  uint32_t transfer_packet_count;
  // Total operand uses rewritten across all detached values.
  uint64_t rewritten_operand_count;
} loom_low_allocation_live_range_split_result_t;

// One committed detached-copy edit used to repair placement-sensitive pairs.
typedef struct loom_low_allocation_pair_replication_edit_t {
  // Detached low.copy inserted for this source value.
  loom_op_t* copy_op;
  // Original value replicated by |copy_op|.
  loom_value_id_t source_value_id;
  // Detached copy result used by rewritten pair operands.
  loom_value_id_t replica_value_id;
} loom_low_allocation_pair_replication_edit_t;

// Transactional result from placement-sensitive pair source replication.
typedef struct loom_low_allocation_pair_replication_result_t {
  // Arena-owned committed edit records.
  loom_low_allocation_pair_replication_edit_t* edits;
  // Number of records in |edits|.
  iree_host_size_t edit_count;
  // Satisfied pair-recipe packet savings before replication.
  uint64_t baseline_satisfied_packet_savings;
} loom_low_allocation_pair_replication_result_t;

// Detaches each distinct fixed retained-read blocker recorded by allocation.
// Immutable fixed IDs and classes remain valid edit inputs while uses are
// rewritten; liveness and assignments are not queried after edits.
// If no retained blocker can be detached, attempts one ordinary overlapping
// fixed value instead. Each detached value emits its decision to |emitter|.
//
// Returns OK with a zero result when no fixed value is a safe split candidate.
// Callers rebuild allocation once after the complete batch. Temporary
// rewriter storage retires before return; new IR belongs to the module.
iree_status_t loom_low_allocation_split_fixed_value_spill_plans(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* out_result);

// Replicates shared operands when concrete placement-pair recipes predict that
// one detached copy can recover more native packets than it costs.
//
// Only |pair_uses| and allocation tables are inspected; this does not walk the
// function IR. Candidate replicas are inserted as one transaction. Callers
// rebuild scheduling and allocation, evaluate actual packet savings and
// resource use, and retain or roll back the edits. While replicas are present,
// the original schedule and allocation are only a comparison baseline.
// Rolling back restores their validity when no other IR edits intervened.
iree_status_t loom_low_allocation_replicate_pair_sources(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_placement_pair_use_list_t pair_uses, iree_arena_allocator_t* arena,
    loom_low_allocation_pair_replication_result_t* out_result);

// Sums native packet savings for pair recipes satisfied by |table|.
iree_status_t loom_low_allocation_satisfied_pair_packet_savings(
    const loom_low_allocation_table_t* table,
    loom_low_placement_pair_use_list_t pair_uses, uint64_t* out_packet_savings);

// Restores all operands rewritten by |result| and erases its detached copies.
// Preserves every original operation and its order, allowing the caller to
// retain the pre-replication schedule and allocation instead of rebuilding
// them.
iree_status_t loom_low_allocation_rollback_pair_replication(
    loom_module_t* module,
    const loom_low_allocation_pair_replication_result_t* result,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_
