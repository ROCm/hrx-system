// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-pressure repair by descriptor-guided rematerialization.
//
// This layer is intentionally allocation-informed but not part of the
// allocator: allocation reports the hard pressure failure, and this utility
// mutates IR only when the failed or conflicting value is a pure descriptor
// packet whose result explicitly opts in to rematerialization.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_REMATERIALIZATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_REMATERIALIZATION_H_

#include "iree/base/api.h"
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
  // A recorded structured-placement alternative repaired a residency cliff.
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_RESIDENCY_CLIFF = 3,
} loom_low_allocation_rematerialization_trigger_t;

typedef struct loom_low_allocation_rematerialization_result_t {
  // SSA value whose defining packet was rematerialized.
  loom_value_id_t value_id;
  // Original allocation assignment associated with |value_id|, or UINT32_MAX.
  uint32_t assignment_index;
  // Number of descriptor packet clones inserted near operand users.
  uint32_t cloned_packet_count;
  // Number of operand uses rewritten to cloned packet results.
  uint32_t rewritten_operand_count;
} loom_low_allocation_rematerialization_result_t;

// Attempts to repair a terminal hard allocation failure by rematerializing a
// descriptor-backed value whose live range creates unspillable pressure.
//
// Returns OK with a zero result when the failure is not a rematerialization
// candidate. User IR failures remain allocation diagnostics; status failures
// are reserved for compiler infrastructure invariants while cloning or
// rewriting already-validated descriptor packets.
iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result);

// Attempts to repair one predicted spill plan by rematerializing its value
// instead of materializing storage traffic.
//
// Returns OK with a zero result when no spill-plan value is a rematerialization
// candidate. When a value is rewritten, callers must rebuild allocation before
// consulting the old allocation table again.
iree_status_t loom_low_allocation_rematerialize_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result);

// Rematerializes |value_id| only at the supplied recorded placement uses.
//
// When |materialization_ops| is nonempty, it is a dependency-ordered bounded
// producer slice authorized by source motion legality. The slice is cloned
// once before the earliest recorded use in each exact use block, with
// |materialization_inputs| retained as captures. Otherwise the defining
// descriptor is rematerialized independently at each use. Stale uses already
// rewritten by an earlier repair are ignored. Returns a zero result when the
// value or surviving uses cannot be safely rematerialized.
iree_status_t loom_low_allocation_rematerialize_candidate_uses(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_value_id_t value_id, loom_op_t* const* materialization_ops,
    uint32_t materialization_op_count,
    const loom_value_id_t* materialization_inputs,
    uint32_t materialization_input_count, const loom_use_t* uses,
    uint32_t use_count, iree_arena_allocator_t* arena,
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

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_REMATERIALIZATION_H_
