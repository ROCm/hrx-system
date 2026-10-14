// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Diagnostic collection and annotation matching shared by loom-check modes.

#ifndef LOOM_TOOLS_LOOM_CHECK_DIAGNOSTICS_H_
#define LOOM_TOOLS_LOOM_CHECK_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/error/emitter.h"
#include "loom/error/parameter.h"
#include "loom/error/source.h"
#include "loom/ir/module.h"
#include "loom/testing/test_diagnostic.h"
#include "loom/testing/test_file.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// A diagnostic collected by loom-check for matching and reporting.
typedef loom_test_diagnostic_t loom_check_collected_diagnostic_t;

// Accumulates diagnostics emitted while executing one loom-check case.
typedef struct loom_check_diagnostic_collector_t {
  // Collected diagnostic entries, arena-allocated.
  loom_check_collected_diagnostic_t* diagnostics;

  // Number of populated entries in diagnostics.
  iree_host_size_t count;

  // Number of error-severity diagnostics collected so far.
  iree_host_size_t error_count;

  // Allocated entry capacity of diagnostics.
  iree_host_size_t capacity;

  // Arena used for diagnostic entries and rendered message storage.
  iree_arena_allocator_t* arena;

  // Host allocator used by temporary string builders.
  iree_allocator_t host_allocator;

  // Main source identity whose lines can match annotations and receive edits.
  iree_string_view_t filename;

  // Current parsed module for full type rendering, or NULL during parse
  // recovery.
  const loom_module_t* module;

  // Printer context used to render type-bearing diagnostic parameters.
  loom_low_descriptor_text_print_context_t type_print_context;

  // File-level result receiving structured diagnostic JSON captures.
  loom_check_result_t* result;
} loom_check_diagnostic_collector_t;

// Materializes structured emission requests into collected diagnostics.
typedef struct loom_check_diagnostic_emitter_capture_t {
  // Diagnostic collector shared by the executing check case.
  loom_check_diagnostic_collector_t* diagnostic_collector;

  // Module containing any operation referenced by emitted diagnostics.
  const loom_module_t* module;

  // Source resolver for source-backed operation locations in this case.
  loom_source_resolver_t source_resolver;

  // Subsystem identity to store in materialized diagnostics.
  loom_emitter_t emitter;

  // Number of diagnostics materialized through this capture.
  iree_host_size_t emission_count;
} loom_check_diagnostic_emitter_capture_t;

// Returns whether the collector contains an error, ignoring notes and warnings.
bool loom_check_diagnostic_collector_has_error(
    const loom_check_diagnostic_collector_t* collector);

// Appends one fully materialized diagnostic and updates severity counts.
iree_status_t loom_check_diagnostic_collector_append(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_collected_diagnostic_t* diagnostic);

// Diagnostic sink callback. Renders, stores, and JSON-captures one diagnostic.
iree_status_t loom_check_diagnostic_collector_sink(
    void* user_data, const loom_diagnostic_t* diagnostic);

// Diagnostic emitter callback. Pass a
// loom_check_diagnostic_emitter_capture_t* as user_data.
iree_status_t loom_check_diagnostic_emitter_capture_emit(
    void* user_data, const loom_diagnostic_emission_t* emission);

// Emits a structured diagnostic anchored to the first source line in
// |test_case| that contains Loom IR. This is for request-level failures that
// cannot be attached to an existing operation.
iree_status_t loom_check_diagnostic_collector_emit_case_source(
    loom_check_diagnostic_collector_t* collector,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_emitter_t emitter, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count);

// Matches collected diagnostics against annotations, sets result->raw_outcome,
// and builds failure detail/update edits when annotations do not match.
iree_status_t loom_check_diagnostic_collector_finish(
    loom_check_diagnostic_collector_t* collector,
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_DIAGNOSTICS_H_
