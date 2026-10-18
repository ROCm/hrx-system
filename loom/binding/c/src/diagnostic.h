// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_DIAGNOSTIC_STORAGE_H_
#define LOOMC_DIAGNOSTIC_STORAGE_H_

#include "loom/error/diagnostic.h"
#include "loom/error/source.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-aware type rendering state retained while diagnostics are captured.
typedef struct loomc_diagnostic_type_printer_t {
  // Module owning type identities referenced by diagnostic parameters.
  const loom_module_t* module;
  // Optional canonical target-aware printer options.
  const loom_text_print_options_t* text_print_options;
} loomc_diagnostic_type_printer_t;

// Returns a formatter backed by |printer|, or the context-free formatter when
// |printer| is NULL.
LOOMC_API_PRIVATE loom_type_formatter_t loomc_diagnostic_type_printer_formatter(
    const loomc_diagnostic_type_printer_t* printer);

// Synchronous adapter from native Loom diagnostics to one public result.
typedef struct loomc_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
  // Optional source owning parser or decoder input bytes.
  const loomc_source_t* source;
  // Optional module resolving emission locations and type identities.
  const loom_module_t* module;
  // Optional exact source resolver for module-backed locations.
  loom_source_resolver_t source_resolver;
  // Native emitter identity assigned to diagnostic emissions.
  loom_emitter_t emitter;
  // Optional target-aware type printer options borrowed during capture.
  const loom_text_print_options_t* text_print_options;
} loomc_diagnostic_capture_t;

// Initializes one synchronous native-to-public diagnostic adapter.
LOOMC_API_PRIVATE void loomc_diagnostic_capture_initialize(
    loomc_result_t* result, const loomc_source_t* source,
    const loom_module_t* module, loom_source_resolver_t source_resolver,
    loom_emitter_t emitter, const loom_text_print_options_t* text_print_options,
    loomc_diagnostic_capture_t* out_capture);

// Captures one native diagnostic through a loom_diagnostic_sink_t.
LOOMC_API_PRIVATE iree_status_t
loomc_diagnostic_capture(void* user_data, const loom_diagnostic_t* diagnostic);

// Captures one location-unresolved diagnostic emission.
LOOMC_API_PRIVATE iree_status_t loomc_diagnostic_capture_emission(
    void* user_data, const loom_diagnostic_emission_t* emission);

// Installs a borrowed sink that observes native Loom diagnostics while result
// is mutable. Passing NULL clears the sink before result escapes the operation.
LOOMC_API_PRIVATE void loomc_result_set_loom_diagnostic_sink(
    loomc_result_t* result, const loom_diagnostic_sink_t* sink);

// Returns the optional native Loom diagnostic sink installed on result.
LOOMC_API_PRIVATE const loom_diagnostic_sink_t*
loomc_result_loom_diagnostic_sink(const loomc_result_t* result);

// Adds a rendered Loom diagnostic and its related locations, retaining their
// source identities and optional text in result. Reuses |source| only when it
// owns the identified contents. Native related locations obey the bounded
// LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS contract.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_loom_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    const loom_diagnostic_t* diagnostic,
    const loom_text_print_options_t* text_print_options);

// Resolves an emission against |module| or its explicit module override and
// adds the diagnostic to result, which owns all resolved source identities.
// Related operations without a module override use the active |module|.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_loom_diagnostic_emission(
    loomc_result_t* result, const loom_module_t* module,
    loom_source_resolver_t source_resolver, loom_emitter_t emitter,
    const loom_diagnostic_emission_t* emission,
    const loom_text_print_options_t* text_print_options);

// Verifies a Loom module and adds verifier diagnostics to result. A zero error
// limit selects the binding's default diagnostic limit.
LOOMC_API_PRIVATE loomc_status_t loomc_result_verify_loom_module(
    const loomc_context_t* context, const loom_module_t* module,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loomc_result_t* result);

// Adds a rendered status as a result diagnostic without consuming status.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status);

// Returns true when status should be represented as an operation diagnostic.
LOOMC_API_PRIVATE bool loomc_status_is_result_diagnostic(loomc_status_t status);

// Adds a status diagnostic and marks the result failed.
LOOMC_API_PRIVATE loomc_status_t loomc_result_fail_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status);

// Adds a status diagnostic, marks the result failed, and frees status.
LOOMC_API_PRIVATE loomc_status_t loomc_result_fail_status_diagnostic_consume(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_DIAGNOSTIC_STORAGE_H_
