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
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/allocation/diagnostics.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/codegen/low/storage_transport.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_schedule_table_t;

// Options controlling allocation table construction.
typedef struct loom_low_allocation_options_t {
  // Optional retained result relations requested by downstream consumers.
  loom_low_allocation_flags_t flags;
  // Optional final schedule. Allocation uses its retained operation order for
  // liveness and its node ordinals for schedule-sensitive coalescing.
  const struct loom_low_schedule_table_t* schedule;
  // Explicit per-class register budgets.
  const loom_low_allocation_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // Fixed locations for precolored SSA values.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
  // Borrowed invocation locations indexed by formal-argument ordinal, including
  // unused arguments. The entry arguments prefix the local value domain. A
  // missing suffix has no entry transport; these are not lifetime constraints.
  const loom_low_allocation_abi_location_t* entry_locations;
  // Number of entries in |entry_locations|, at most the formal argument count.
  iree_host_size_t entry_location_count;
  // Borrowed outgoing locations indexed by function result. These constrain
  // each return boundary, not SSA lifetimes.
  const loom_low_allocation_abi_location_t* exit_locations;
  // Number of entries in |exit_locations|, at most the function result count.
  iree_host_size_t exit_location_count;
  // Physical call effects resolved from retained target convention bindings.
  loom_low_call_contract_provider_t call_contracts;
  // Proven synchronous boundary storage for this immutable function snapshot.
  const loom_low_storage_transport_t* storage_transport;
  // Spaces the consumer can access synchronously within final move groups.
  // Empty for targets requiring separately scheduled spill expansion.
  loom_low_storage_space_set_t move_storage_spaces;
  // Whole-function target-owned location ranges.
  const loom_low_allocation_reserved_range_t* reserved_ranges;
  // Number of entries in |reserved_ranges|.
  iree_host_size_t reserved_range_count;
  // Optional target storage leases built over |schedule|.
  loom_low_storage_lease_table_t storage_leases;
  // Structured diagnostic emitter for invalid input constraints. Planning
  // failures are retained in the table for allocation_diagnostics_emit.
  iree_diagnostic_emitter_t emitter;
  // Function-local residency view. Direct resources are dense by descriptor
  // register-class ID for the resolved low target.
  loom_target_residency_view_t residency;
  // Immutable instruction preferences for the resolved descriptor set.
  loom_low_placement_instruction_preferences_t instruction_preferences;
  // Borrowed bitmap indexed by module value ID. Set values require register
  // storage throughout allocation.
  iree_bitmap_t required_register_values;
} loom_low_allocation_options_t;

// Allocates one modeled target-low function body and writes an arena-owned
// table. |model| must remain live and its function semantically immutable until
// this function returns. The allocator performs deterministic per-class
// interval assignment and records failures/spills as table facts without
// mutating IR. The caller publishes terminal planning diagnostics by calling
// loom_low_allocation_diagnostics_emit on the accepted table. The caller must
// first admit an absent or virtual function allocation mode; assigned and fixed
// modes require a retained allocation table instead of synthesis.
iree_status_t loom_low_allocate_function(
    const loom_low_function_model_t* model,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_H_
