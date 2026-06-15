// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent emission frame construction for low functions.
//
// The emission frame is the shared producer contract for low emitters: it runs
// the scheduler and allocator over one prepared target-low function, verifies
// that their tables describe the same target function, and returns the
// arena-owned frame that packet emitters consume. It assumes ordinary pass
// pipelines have already prepared the low IR. This layer does not run
// optimization passes, emit bytes, text, JSON, or target artifacts; each target
// emitter owns those artifact decisions.

#ifndef LOOM_CODEGEN_LOW_FRAME_H_
#define LOOM_CODEGEN_LOW_FRAME_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling emission frame construction for one low function.
typedef struct loom_low_emission_frame_options_t {
  // Descriptor registry available to scheduling and allocation.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional runtime/device target overlay applied when compatible with the
  // function's target record.
  loom_target_selection_t target_selection;
  // Optional source-derived memory summaries for the scheduled low function.
  loom_low_memory_access_table_t memory_access_table;
  // Optional target-provided register-pressure cliff table.
  loom_low_schedule_pressure_cliff_list_t schedule_pressure_cliffs;
  // Optional target-provided descriptor pair-affinity table.
  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities;
  // Candidate selection strategy used by the scheduler.
  loom_low_schedule_strategy_t schedule_strategy;
  // Optional structured scheduler feedback to emit.
  loom_low_schedule_diagnostic_flags_t schedule_diagnostic_flags;
  // Explicit per-class register budgets passed to allocation.
  const loom_low_allocation_budget_t* allocation_budgets;
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed locations for precolored SSA values passed to allocation.
  const loom_low_allocation_fixed_value_t* allocation_fixed_values;
  // Number of entries in |allocation_fixed_values|.
  iree_host_size_t allocation_fixed_value_count;
  // Whole-function target-owned location ranges passed to allocation.
  const loom_low_allocation_reserved_range_t* allocation_reserved_ranges;
  // Number of entries in |allocation_reserved_ranges|.
  iree_host_size_t allocation_reserved_range_count;
  // Optional target storage-lease provider queried over the final schedule
  // before allocation.
  const loom_low_storage_lease_provider_t* storage_lease_provider;
  // Optional structured allocation feedback to emit.
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags;
  // Structured diagnostic emitter shared by scheduling and allocation.
  iree_diagnostic_emitter_t emitter;
} loom_low_emission_frame_options_t;

// Emission-ready production frame for one prepared target-low function. The
// frame and its nested tables borrow from the caller-provided module and arena.
typedef struct loom_low_emission_frame_t {
  // Module containing the prepared low function.
  const loom_module_t* module;
  // Prepared target-low function operation.
  const loom_op_t* function_op;
  // Resolved target context shared by the nested schedule/allocation tables.
  loom_low_resolved_target_t target;
  // Schedule table for the prepared function.
  loom_low_schedule_table_t schedule;
  // Allocation table for the prepared function.
  loom_low_allocation_table_t allocation;
} loom_low_emission_frame_t;

// Target callback that rewrites structural low.spill/low.reload traffic into
// target packets before the next emission-frame scheduling/allocation round.
typedef iree_status_t (*loom_low_emission_frame_lower_spill_traffic_fn_t)(
    void* user_data, loom_module_t* module, loom_op_t* low_func_op,
    iree_arena_allocator_t* arena);

// Summary from target address-state materialization.
typedef struct loom_low_emission_frame_materialize_address_state_result_t {
  // True if the target inserted, removed, or updated packets.
  bool changed;
  // Number of user-facing diagnostics emitted while materializing address
  // state.
  uint32_t error_count;
} loom_low_emission_frame_materialize_address_state_result_t;

// Target callback that materializes packet address-selection state exposed by
// allocated operands with LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE. The
// callback may rewrite |low_func_op| based on the accepted frame, and any
// change causes a fresh schedule/allocation round before final validation.
typedef iree_status_t (*loom_low_emission_frame_materialize_address_state_fn_t)(
    void* user_data, loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result);

// Target callback that validates the accepted final spill-free frame after
// target-independent packet addressability has already been checked.
typedef iree_status_t (*loom_low_emission_frame_validate_fn_t)(
    void* user_data, const loom_low_emission_frame_t* frame,
    iree_arena_allocator_t* arena);

// Options controlling final spill-free emission frame construction.
typedef struct loom_low_emission_frame_spill_free_options_t {
  // Materialization contract used when final allocation exposes new spills.
  loom_low_allocation_materialization_options_t materialization_options;
  // Target-owned lowering callback for structural spill traffic.
  loom_low_emission_frame_lower_spill_traffic_fn_t lower_spill_traffic;
  // Caller-owned data passed to |lower_spill_traffic|.
  void* lower_spill_traffic_user_data;
  // Optional target-owned packet address-state materializer.
  loom_low_emission_frame_materialize_address_state_fn_t
      materialize_address_state;
  // Caller-owned data passed to |materialize_address_state|.
  void* materialize_address_state_user_data;
  // Optional final target validation callback.
  loom_low_emission_frame_validate_fn_t validate_frame;
  // Caller-owned data passed to |validate_frame|.
  void* validate_frame_user_data;
} loom_low_emission_frame_spill_free_options_t;

// Schedules, allocates, and validates one target-low function for target
// emitters. |arena| must outlive |out_frame|.
iree_status_t loom_low_emission_frame_build(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame);

// Builds an emission frame and greedily materializes target-lowerable spill
// traffic until the final frame contains no spill assignments or spill plans.
// The loop materializes one spill plan per iteration because each rewrite can
// change liveness and invalidate later spill plans in the current frame.
// Materialization, final addressability, and final-frame convergence
// diagnostics follow the normal target-entry convention: if an error diagnostic
// is emitted, the function returns OK with |out_frame| left empty and the
// caller must check its diagnostic emitter before consuming the frame.
iree_status_t loom_low_emission_frame_build_spill_free(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_FRAME_H_
