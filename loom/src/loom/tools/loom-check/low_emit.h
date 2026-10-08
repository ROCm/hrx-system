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

// Fixed-location requests for one RUN option string. Initialization reserves
// exactly one slot per fixed= token; parsing that same string fills the list.
typedef struct loom_check_low_emit_fixed_value_spec_list_t {
  // Case-arena-owned storage for the option string's fixed-location requests.
  loom_check_low_emit_fixed_value_spec_t* specs;
  // Number of successfully parsed requests in |specs|.
  iree_host_size_t count;
} loom_check_low_emit_fixed_value_spec_list_t;

// Outcome of resolving one authored value selector in a low function body.
typedef enum loom_check_low_emit_value_resolution_kind_e {
  LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_NOT_FOUND = 0,
  LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_RESOLVED,
  LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_AMBIGUOUS,
} loom_check_low_emit_value_resolution_kind_t;

// Resolution of one authored value selector in a low function body.
typedef struct loom_check_low_emit_value_resolution_t {
  // Whether the selector was absent, unique, or ambiguous.
  loom_check_low_emit_value_resolution_kind_t kind;
  // Unique value or first value with the selected name.
  loom_value_id_t value_id;
  // Second value with the selected name when |kind| is ambiguous.
  loom_value_id_t ambiguous_value_id;
} loom_check_low_emit_value_resolution_t;

// Reserves the fixed-location requests in |options| before its parser runs.
iree_status_t loom_check_low_emit_fixed_value_spec_list_initialize(
    iree_string_view_t options, iree_arena_allocator_t* arena,
    loom_check_low_emit_fixed_value_spec_list_t* out_list);

// Parses a shared low emit scheduling strategy value.
iree_status_t loom_check_low_emit_parse_schedule_strategy(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_strategy_t* out_strategy);

// Parses a shared low emit schedule-diagnostics value.
iree_status_t loom_check_low_emit_parse_schedule_diagnostics(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_diagnostic_flags_t* out_flags);

// Parses a shared low emit allocation-diagnostics value.
iree_status_t loom_check_low_emit_parse_allocation_diagnostics(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_allocation_diagnostic_flags_t* out_flags);

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
    loom_check_low_emit_fixed_value_spec_list_t* fixed_specs);

// Parses either a fixed=... token or a <register-class>=<units> budget token.
iree_status_t loom_check_low_emit_parse_allocation_option(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count,
    loom_check_low_emit_fixed_value_spec_list_t* fixed_specs);

// Finds a module-local target-low function definition by symbol name.
iree_status_t loom_check_low_emit_find_low_function_def(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_diagnostic_emitter_t emitter, loom_op_t** out_low_function);

// Resolves an SSA name or numeric value ID within |low_function|. The function
// must be a target-low definition with a body.
void loom_check_low_emit_resolve_function_value(
    const loom_module_t* module, const loom_op_t* low_function,
    iree_string_view_t value_name,
    loom_check_low_emit_value_resolution_t* out_resolution);

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
// from request->case_arena. The request supplies the descriptor registry and
// diagnostic emitter; authored fixed specs supply the fixed allocation values.
// All other frame options, including reserved ranges, pass through unchanged.
iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module,
    iree_string_view_t function_symbol_name,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_check_low_emit_fixed_value_spec_t* allocation_fixed_specs,
    iree_host_size_t allocation_fixed_spec_count,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    loom_low_emission_frame_t* out_frame, bool* out_accepted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_
