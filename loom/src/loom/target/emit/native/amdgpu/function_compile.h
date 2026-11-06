// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native function compilation from one prepared target-low function.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_COMPILE_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_COMPILE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/arch/amdgpu/abi/call_abi.h"
#include "loom/target/emit/native/amdgpu/function_plan.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inputs fixed before compiling one target-low function.
typedef struct loom_amdgpu_function_compile_options_t {
  // Descriptor registry available to scheduling and allocation.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Already resolved representation and target contract for the function.
  const loom_low_resolved_target_t* resolved_target;
  // Analysis-derived memory summaries for the function.
  const loom_low_memory_access_map_t* memory_accesses;
  // Producer-proven execution counts for natural Low loops.
  const loom_low_loop_execution_map_t* loop_execution;
  // Fixed locations for precolored SSA values.
  const loom_low_allocation_fixed_value_t* allocation_fixed_values;
  // Number of entries in |allocation_fixed_values|.
  iree_host_size_t allocation_fixed_value_count;
  // Sparse values supplied in physical coordinates at function entry.
  const loom_low_allocation_entry_value_t* allocation_entry_values;
  // Number of entries in |allocation_entry_values|.
  iree_host_size_t allocation_entry_value_count;
  // Incoming locations indexed by formal argument.
  const loom_low_allocation_abi_location_t* allocation_entry_locations;
  // Number of entries in |allocation_entry_locations|.
  iree_host_size_t allocation_entry_location_count;
  // Outgoing locations indexed by function result.
  const loom_low_allocation_abi_location_t* allocation_exit_locations;
  // Number of entries in |allocation_exit_locations|.
  iree_host_size_t allocation_exit_location_count;
  // Callee convention facts consumed before allocation.
  loom_low_call_contract_provider_t call_contracts;
  // Callee machine-state contracts consumed during packet planning.
  loom_amdgpu_call_state_provider_t call_states;
  // Segment bytes occupied by callees below this function's local storage.
  // Fixed for authored spill lowering, allocation repair, and final placement.
  loom_amdgpu_storage_layout_segment_sizes_t storage_segment_prefix;
  // Whether this function participates as a callable helper.
  bool callable;
  // Whole-function target-owned location ranges unavailable to allocation.
  const loom_low_allocation_reserved_range_t* allocation_reserved_ranges;
  // Number of entries in |allocation_reserved_ranges|.
  iree_host_size_t allocation_reserved_range_count;
  // Structured diagnostic emitter for scheduling and allocation failures.
  iree_diagnostic_emitter_t emitter;
  // Optional compile report receiving spill and planning facts.
  loom_target_compile_report_t* report;
} loom_amdgpu_function_compile_options_t;

// Complete native compilation product for one target-low function.
typedef struct loom_amdgpu_function_compilation_t {
  // Spill-free scheduled and allocated Low function.
  loom_low_emission_frame_t frame;
  // Function-local native packet and storage decisions.
  loom_amdgpu_function_plan_t plan;
  // Coarse planning statistics retained for entry-level reporting.
  loom_low_planning_statistics_t planning_statistics;
} loom_amdgpu_function_compilation_t;

// Compiles one prepared target-low function through native packet planning.
//
// User-dependent failures emit diagnostics and leave |out_compiled| false.
// |arena| must outlive |out_compilation|.
iree_status_t loom_amdgpu_function_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_amdgpu_function_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_amdgpu_function_compilation_t* out_compilation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_COMPILE_H_
