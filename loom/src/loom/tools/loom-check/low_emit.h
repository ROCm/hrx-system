// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared helpers for target-owned loom-check emit providers over target-low
// functions.

#ifndef LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_
#define LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/frame.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_check_diagnostic_collector_t
    loom_check_diagnostic_collector_t;

enum {
  LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS = 8,
  LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES = 16,
};

// Textual fixed-location request parsed from RUN options before the selected
// low function has been resolved.
typedef struct loom_check_low_emit_fixed_value_spec_t {
  // SSA value name without the leading '%'.
  iree_string_view_t value_name;
  // Target-visible fixed location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register or target ID.
  uint32_t location_base;
  // Number of contiguous units fixed at |location_base|.
  uint32_t location_count;
} loom_check_low_emit_fixed_value_spec_t;

// Parses a shared low emit scheduling strategy value.
iree_status_t loom_check_low_emit_parse_schedule_strategy(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_strategy_t* out_strategy);

// Parses one <register-class>=<units> allocation budget token.
iree_status_t loom_check_low_emit_parse_allocation_budget(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count);

// Parses one fixed=%value:<location-kind>:<base>:<count> allocation token.
// Location kinds use the same stable spellings as low allocation JSON:
// physical_register and target_id.
iree_status_t loom_check_low_emit_parse_fixed_value_spec(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_capacity, iree_host_size_t* fixed_spec_count);

// Parses either a fixed=... token or a <register-class>=<units> budget token.
iree_status_t loom_check_low_emit_parse_allocation_option(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count,
    loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_capacity, iree_host_size_t* fixed_spec_count);

// Finds a module-local target-low function definition by symbol name.
iree_status_t loom_check_low_emit_find_low_function_def(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_diagnostic_emitter_t emitter, loom_op_t** out_low_function);

// Resolves parsed fixed-location specs against the selected low function body.
// The returned fixed value array is allocated from |arena|.
iree_status_t loom_check_low_emit_resolve_fixed_value_specs(
    loom_module_t* module, loom_op_t* low_function,
    const loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_count, iree_diagnostic_emitter_t emitter,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count, bool* out_resolved,
    iree_arena_allocator_t* arena);

// Packetizes the selected low function through the registry linked into the
// emit provider request. |out_frame| stores table pointers allocated
// from request->case_arena.
iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t function_symbol_name,
    loom_low_schedule_strategy_t schedule_strategy,
    const loom_low_allocation_budget_t* allocation_budgets,
    iree_host_size_t allocation_budget_count,
    const loom_check_low_emit_fixed_value_spec_t* allocation_fixed_specs,
    iree_host_size_t allocation_fixed_spec_count,
    loom_low_schedule_pair_affinity_list_t schedule_pair_affinities,
    const loom_low_storage_lease_provider_t* storage_lease_provider,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    loom_low_emission_frame_t* out_frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_
