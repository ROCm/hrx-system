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
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_rematerialization_result_t {
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_REMATERIALIZATION_H_
