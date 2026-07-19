// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral analysis of final compiler artifact bytes.

#ifndef LOOM_TOOLING_EXECUTION_ARTIFACT_ANALYZER_H_
#define LOOM_TOOLING_EXECUTION_ARTIFACT_ANALYZER_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/target/reporting/report.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable final-artifact state borrowed for one synchronous analysis call.
typedef struct loom_run_artifact_analysis_request_t {
  // Target-native format of |artifact_data|.
  loom_target_artifact_format_t artifact_format;

  // Final target-native bytes produced by the selected artifact provider.
  iree_const_byte_span_t artifact_data;

  // Provider-owned target key used to emit |artifact_data|.
  iree_string_view_t target_key;

  // Target-neutral bundle resolved for the artifact, when available.
  const loom_target_bundle_t* target_bundle;

  // Sink receiving analyzer diagnostics. A NULL callback may count findings
  // internally but cannot publish rendered diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;

  // Maximum diagnostics the analyzer may publish before rejecting analysis.
  // Zero requests the analyzer's conservative default.
  uint32_t max_errors;

  // Optional candidate-owned compile report receiving structured analysis.
  loom_target_compile_report_t* report;
} loom_run_artifact_analysis_request_t;

// Synchronously analyzes |request|. The callback must not retain any request
// field after returning.
typedef iree_status_t (*loom_run_artifact_analyze_fn_t)(
    void* user_data, const loom_run_artifact_analysis_request_t* request);

// Borrowed final-artifact analyzer selected by a compiler embedding.
typedef struct loom_run_artifact_analyzer_t {
  // Synchronous analysis callback.
  loom_run_artifact_analyze_fn_t fn;

  // Caller-owned payload forwarded to |fn|.
  void* user_data;
} loom_run_artifact_analyzer_t;

// Invokes |analyzer| synchronously over |request|.
iree_status_t loom_run_artifact_analyzer_analyze(
    const loom_run_artifact_analyzer_t* analyzer,
    const loom_run_artifact_analysis_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_ARTIFACT_ANALYZER_H_
