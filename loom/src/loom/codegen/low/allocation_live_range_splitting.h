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
// immediately after the fixed source is materialized and rewrite later users
// to the transfer result. The fixed physical location then dies at the transfer
// while the ordinary virtual value remains allocatable by normal rules.

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

typedef enum loom_low_allocation_live_range_split_trigger_e {
  // Unknown or uninitialized split trigger.
  LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_UNKNOWN = 0,
  // A predicted spill plan was avoided by detaching a fixed value.
  LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN = 1,
} loom_low_allocation_live_range_split_trigger_t;

typedef struct loom_low_allocation_live_range_split_result_t {
  // Fixed SSA value whose physical live range was detached.
  loom_value_id_t source_value_id;
  // Transfer result carrying the ordinary virtual live range.
  loom_value_id_t split_value_id;
  // Original allocation assignment associated with |source_value_id|, or
  // UINT32_MAX when the fixed value had no completed assignment.
  uint32_t source_assignment_index;
  // Number of low.copy or low.move transfer packets inserted.
  uint32_t transfer_packet_count;
  // Number of operand uses rewritten to |split_value_id|.
  uint32_t rewritten_operand_count;
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

// Attempts to repair one predicted spill plan by detaching an overlapping
// fixed value from its fixed physical location.
//
// Returns OK with a zero result when no fixed value is a safe split candidate.
// When a value is rewritten, callers must rebuild allocation before consulting
// the old allocation table again.
iree_status_t loom_low_allocation_split_fixed_value_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* out_result);

// Replicates shared operands when concrete placement-pair recipes prove that
// one detached copy can recover more native packets than it costs.
//
// Only |pair_uses| and allocation tables are inspected; this does not walk the
// function IR. All profitable replicas are committed as one transaction so
// callers can rebuild scheduling and allocation once. A non-empty result must
// either be retained or passed to
// loom_low_allocation_rollback_pair_replication before consulting the old
// schedule or allocation tables again.
iree_status_t loom_low_allocation_replicate_pair_sources(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_placement_pair_use_list_t pair_uses, iree_arena_allocator_t* arena,
    loom_low_allocation_pair_replication_result_t* out_result);

// Sums native packet savings for pair recipes satisfied by |table|.
iree_status_t loom_low_allocation_satisfied_pair_packet_savings(
    const loom_low_allocation_table_t* table,
    loom_low_placement_pair_use_list_t pair_uses, uint64_t* out_packet_savings);

// Restores all operands rewritten by |result| and erases its detached copies.
iree_status_t loom_low_allocation_rollback_pair_replication(
    loom_module_t* module,
    const loom_low_allocation_pair_replication_result_t* result,
    iree_arena_allocator_t* arena);

// Emits a structured remark describing a successful live-range split result.
iree_status_t loom_low_allocation_live_range_split_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_live_range_split_trigger_t trigger,
    const loom_low_allocation_live_range_split_result_t* result,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_
