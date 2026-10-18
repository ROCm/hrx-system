// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_EMIT_STORAGE_H_
#define LOOMC_EMIT_STORAGE_H_

#include "diagnostic.h"
#include "loom/error/diagnostic.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/report.h"
#include "loom/util/json.h"
#include "loomc/emit.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fully resolved target emission options for one compiler transaction.
typedef struct loomc_emit_resolved_options_t {
  // Artifact format requested by the caller.
  loomc_string_view_t artifact_format;

  // Artifact identifier requested by the caller.
  loomc_string_view_t identifier;

  // Artifact classes requested by the caller.
  loomc_emit_artifact_flags_t artifact_flags;

  // Artifact manifest mode requested by the caller.
  loomc_artifact_manifest_mode_t artifact_manifest_mode;

  // Artifact manifest identifier requested by the caller.
  loomc_string_view_t artifact_manifest_identifier;

  // Compile report mode requested by the caller.
  loomc_compile_report_mode_t compile_report_mode;

  // Compile report serialization format requested by the caller.
  loomc_compile_report_format_t compile_report_format;

  // Compile report identifier requested by the caller.
  loomc_string_view_t compile_report_identifier;

  // Full extension chain passed to target-specific emitters.
  const void* option_chain;
} loomc_emit_resolved_options_t;

// One emission lifecycle spanning target compilation and final byte emission.
typedef struct loomc_emit_transaction_t {
  // Resolved caller options.
  loomc_emit_resolved_options_t options;

  // Result receiving diagnostics and artifacts.
  loomc_result_t* result;

  // Target emitter selected exactly once for the transaction.
  const loom_target_emitter_t* emitter;

  // Optional report spanning request resolution, compilation, and emission.
  loom_target_compile_report_t compile_report;

  // True when |compile_report| owns storage and must be deinitialized.
  bool compile_report_initialized;

  // Canonical diagnostic JSON retained only for detailed JSON reports.
  loom_json_value_list_t compile_report_diagnostics;

  // Borrowed result sink writing into |compile_report_diagnostics|.
  loom_diagnostic_sink_t compile_report_diagnostic_sink;

  // Target-aware printer options borrowed by the report diagnostic sink.
  const loom_text_print_options_t* diagnostic_text_print_options;

  // True when diagnostic JSON capture was installed on |result|.
  bool compile_report_diagnostics_initialized;
} loomc_emit_transaction_t;

// Resolves public options into an unbound emission transaction.
LOOMC_API_PRIVATE loomc_status_t loomc_emit_transaction_initialize(
    const loomc_emit_options_t* options, loomc_result_t* result,
    loomc_emit_transaction_t* out_transaction);

// Returns the exact public artifact format constraint, or an empty view.
LOOMC_API_PRIVATE loomc_string_view_t loomc_emit_transaction_artifact_format(
    const loomc_emit_transaction_t* transaction);

// Sets the target printer environment used by detailed diagnostic reports.
LOOMC_API_PRIVATE void loomc_emit_transaction_set_diagnostic_environment(
    loomc_emit_transaction_t* transaction,
    const loomc_target_environment_t* target_environment);

// Selects an emitter from the target environment using resolved options.
LOOMC_API_PRIVATE loomc_status_t loomc_emit_transaction_select_emitter(
    loomc_emit_transaction_t* transaction,
    const loom_target_environment_t* target_environment);

// Binds the emitter already selected by the core compile request.
LOOMC_API_PRIVATE void loomc_emit_transaction_bind_emitter(
    loomc_emit_transaction_t* transaction,
    const loom_target_emitter_t* emitter);

// Returns the optional report passed through compiler stages and emission.
LOOMC_API_PRIVATE loom_target_compile_report_t*
loomc_emit_transaction_compile_report(loomc_emit_transaction_t* transaction);

// Records the terminal compiler status when emission did not own it.
LOOMC_API_PRIVATE void loomc_emit_transaction_record_status(
    loomc_emit_transaction_t* transaction, iree_status_code_t status_code);

// Verifies and emits one prepared module into the transaction result.
LOOMC_API_PRIVATE loomc_status_t loomc_emit_transaction_emit(
    loomc_emit_transaction_t* transaction,
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module);

// Appends any requested compile report artifact to the transaction result.
LOOMC_API_PRIVATE loomc_status_t
loomc_emit_transaction_finish(loomc_emit_transaction_t* transaction);

// Releases transaction-owned report storage.
LOOMC_API_PRIVATE void loomc_emit_transaction_deinitialize(
    loomc_emit_transaction_t* transaction);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_EMIT_STORAGE_H_
