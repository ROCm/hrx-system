// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-low compilation and textual artifacts for loom-check providers.

#ifndef LOOM_TOOLS_LOOM_CHECK_SOURCE_LOW_H_
#define LOOM_TOOLS_LOOM_CHECK_SOURCE_LOW_H_

#include "loom/sanitizer/options.h"
#include "loom/tools/loom-check/execute.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_check_emit_source_low_output_e {
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_MODULE = 0,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW = 1,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE = 2,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE = 3,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_NONE = 4,
} loom_check_emit_source_low_output_t;

typedef enum loom_check_source_low_option_bits_e {
  LOOM_CHECK_SOURCE_LOW_OPTION_OUTPUT = 1u << 0,
  LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS = 1u << 1,
  LOOM_CHECK_SOURCE_LOW_OPTION_CONTROL_FLOW = 1u << 2,
  LOOM_CHECK_SOURCE_LOW_OPTION_SANITIZER = 1u << 3,
  LOOM_CHECK_SOURCE_LOW_OPTION_SANITIZER_REPORTING = 1u << 4,
  LOOM_CHECK_SOURCE_LOW_OPTION_TARGET = 1u << 5,
} loom_check_source_low_option_bits_t;
typedef uint32_t loom_check_source_low_options_t;

// Parsed source-low request. Strings borrow the RUN line for the case lifetime.
typedef struct loom_check_source_low_request_t {
  // Textual product to compare, or NONE to check diagnostics only.
  loom_check_emit_source_low_output_t output;
  // Explicitly supplied options, used to reject duplicates and incompatible
  // modes.
  loom_check_source_low_options_t options;
  // Requested source-to-low legality diagnostics.
  loomc_source_to_low_diagnostic_flags_t diagnostic_flags;
  // Control-flow shape for the selected lowering pipeline.
  loomc_target_control_flow_lowering_t control_flow_lowering;
  // Requested sanitizer instrumentation and reporting policy.
  loom_sanitizer_options_t sanitizer;
  // Optional source function to specialize. With TARGET but no name, select
  // the sole definition or the unique public entry among private helpers.
  iree_string_view_t function_name;
  // Complete family:selector target spelling borrowed from the RUN line.
  iree_string_view_t target;
} loom_check_source_low_request_t;

// Parses optional @function and source-low options. An explicit function
// requires target=family:selector; pipeline-text modes accept neither.
iree_status_t loom_check_source_low_parse(
    iree_string_view_t text, loom_check_source_low_request_t* request);

// Runs source lowering and compares IR using an optional caller-owned profile.
// Target providers can refine public feature facts before entering this path.
iree_status_t loom_check_source_low_execute(
    const loom_check_emit_provider_request_t* provider_request,
    const loom_check_source_low_request_t* request,
    loomc_target_profile_t* target_profile);

// Source-consuming emit provider for source-low and source-to-low targets.
extern const loom_check_emit_provider_t loom_check_source_low_emit_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_SOURCE_LOW_H_
