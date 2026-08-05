// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-guided value rematerialization and allocation-pressure repair.
//
// This layer mutates IR only when a pure descriptor packet explicitly opts its
// result in to rematerialization. Allocation and scheduling retain the evidence
// that selects a candidate; this utility owns cloning the producer near each
// use and removing the original long-lived value.

#ifndef LOOM_CODEGEN_LOW_REMATERIALIZATION_H_
#define LOOM_CODEGEN_LOW_REMATERIALIZATION_H_

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
} loom_low_allocation_rematerialization_trigger_t;

typedef struct loom_low_value_rematerialization_result_t {
  // SSA value whose defining packet was rematerialized.
  loom_value_id_t value_id;
  // Number of descriptor packet clones inserted near operand users.
  uint32_t cloned_packet_count;
  // Number of operand uses rewritten to cloned packet results.
  uint32_t rewritten_operand_count;
} loom_low_value_rematerialization_result_t;

typedef struct loom_low_allocation_rematerialization_result_t {
  // Descriptor-guided value rematerialization performed by the repair.
  loom_low_value_rematerialization_result_t value;
  // Original allocation assignment associated with |value|, or UINT32_MAX.
  uint32_t assignment_index;
} loom_low_allocation_rematerialization_result_t;

// Rematerializes a descriptor-backed SSA value near all of its uses.
//
// Returns OK with a zero result when |value_id| is not a safe rematerialization
// candidate. When rewritten, callers must discard analyses of the old IR and
// rebuild them before continuing.
iree_status_t loom_low_rematerialize_value_uses(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result);

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
