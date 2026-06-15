// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent allocation construction for target-low functions.
//
// Loom low functions remain SSA after allocation. Physical registers, target
// local IDs, and spill slots are table facts over live intervals, while
// copies, split intervals, spills, and reloads are explicit IR when a later
// pass decides to materialize them.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/diagnostics.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling allocation table construction.
typedef struct loom_low_allocation_options_t {
  // Optional operation order used for live intervals.
  loom_liveness_order_t liveness_order;
  // Descriptor registry available to the allocator.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional runtime/device target overlay applied when compatible with the
  // function's target record.
  loom_target_selection_t target_selection;
  // Explicit per-class register budgets.
  const loom_low_allocation_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // Fixed locations for precolored SSA values.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
  // Whole-function target-owned location ranges.
  const loom_low_allocation_reserved_range_t* reserved_ranges;
  // Number of entries in |reserved_ranges|.
  iree_host_size_t reserved_range_count;
  // Optional target storage leases built over the same scheduled low function
  // represented by |liveness_order|.
  loom_low_storage_lease_table_t storage_leases;
  // Structured diagnostic emitter for allocation failures and feedback.
  iree_diagnostic_emitter_t emitter;
  // Optional structured allocation feedback to emit.
  loom_low_allocation_diagnostic_flags_t diagnostic_flags;
} loom_low_allocation_options_t;

// Allocates one target-low function body and writes an arena-owned table.
// This first allocator is deliberately simple and deterministic: it performs
// per-class linear-scan-style first-fit assignment over liveness intervals,
// uses target allocatable unit counts or explicit budgets as hard limits, and
// reports spills as table remarks without mutating IR.
iree_status_t loom_low_allocate_function(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_H_
