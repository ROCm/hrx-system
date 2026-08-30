// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared candidate compilation options for Loom execution tools.

#ifndef LOOM_TOOLING_EXECUTION_COMPILE_OPTIONS_H_
#define LOOM_TOOLING_EXECUTION_COMPILE_OPTIONS_H_

#include "iree/base/api.h"
#include "loom/ir/function_version.h"
#include "loom/target/pipeline_options.h"
#include "loom/target/reporting/artifact_manifest.h"
#include "loom/target/reporting/report.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_compile_report_capture_t
    loom_run_compile_report_capture_t;

typedef enum loom_run_candidate_artifact_flag_bits_e {
  LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE = 0u,
  // Requests target-owned textual executable listings when the selected backend
  // can produce them without changing the loaded executable.
  LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING = 1u << 0,
} loom_run_candidate_artifact_flag_bits_t;

typedef uint32_t loom_run_candidate_artifact_flags_t;

typedef struct loom_run_candidate_artifact_manifest_options_t {
  // Selected manifest detail mode.
  loom_target_artifact_manifest_mode_t mode;

  // Manifest sidecar output identifier.
  iree_string_view_t identifier;

  // Emitted artifact name recorded inside the manifest.
  iree_string_view_t artifact_name;
} loom_run_candidate_artifact_manifest_options_t;

typedef struct loom_run_candidate_compile_options_t {
  // Optional compiler-owned function versions participating in artifact
  // emission. The list and its version objects are borrowed for the call.
  const loom_function_version_list_t* function_versions;
  // Target pipeline options used to prepare the module for artifact emission.
  // Artifact providers use this to publish runtime support required by
  // compiler-generated target-low code.
  loom_target_pipeline_options_t target_pipeline_options;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before stopping. Zero uses a conservative
  // default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional one-shot report capture receiving emitted diagnostics for report
  // output adapters.
  loom_run_compile_report_capture_t* report_capture;
  // Optional debug artifacts requested from the selected backend.
  loom_run_candidate_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_run_candidate_artifact_manifest_options_t artifact_manifest;
} loom_run_candidate_compile_options_t;

// Initializes compile options with stderr diagnostics and a small error cap.
void loom_run_candidate_compile_options_initialize(
    loom_run_candidate_compile_options_t* out_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_COMPILE_OPTIONS_H_
