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

// Realizes required physical operand ties before direct Low emission. This
// applies shipping Low preparation without optimization or target rewriting.
// The function must have been verified against |descriptor_set|. Independent
// results may be replaced, so downstream consumers observe the realized form.
iree_status_t loom_check_low_emit_materialize_operand_ties(
    loom_module_t* module, loom_op_t* function,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena);

// Packetizes the selected low function through the registry linked into the
// emit provider request. |out_frame| stores table pointers allocated
// from request->case_arena. The request supplies the descriptor registry and
// diagnostic emitter. All other frame options, including ABI locations,
// reserved ranges, and fixed values, pass through unchanged.
iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module,
    iree_string_view_t function_symbol_name,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    loom_low_emission_frame_t* out_frame, bool* out_accepted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_
