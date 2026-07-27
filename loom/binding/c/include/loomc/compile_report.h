// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_REPORT_H_
#define LOOMC_COMPILE_REPORT_H_

#include "loomc/base.h"

/// @file
/// Compile report emission controls.
///
/// Compile reports are optional machine-readable sidecars for target emission.
/// They describe the selected backend, target, terminal status, artifact size,
/// and target-provided compiler analysis facts available at the selected
/// verbosity.
///
/// Report generation is opt-in. Omitting this descriptor, or setting `mode` to
/// `LOOMC_COMPILE_REPORT_MODE_NONE`, keeps emission on the normal artifact
/// path and avoids detail-only report collection.

#ifdef __cplusplus
extern "C" {
#endif

/// Loom compile report JSON artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON "loom-compile-report-json"

/// Compile report detail mode.
typedef enum loomc_compile_report_mode_e {
  /// Does not request a compile report.
  LOOMC_COMPILE_REPORT_MODE_NONE = 0,

  /// Requests stable target, status, artifact, and summary compiler facts.
  LOOMC_COMPILE_REPORT_MODE_SUMMARY = 1,

  /// Requests summary facts plus detail rows when producers support them.
  LOOMC_COMPILE_REPORT_MODE_DETAILS = 2,
} loomc_compile_report_mode_t;

/// Compile report emission options.
///
/// Attach this descriptor through `loomc_emit_options_t::next`. The descriptor
/// controls JSON report production for the emitted target artifact. It does not
/// run compilation passes, force target analyses, or write filesystem paths.
typedef struct loomc_compile_report_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Selected report detail mode.
  loomc_compile_report_mode_t mode;

  /// Result artifact identifier for the compile report JSON. Empty derives
  /// from the emitted artifact identifier by appending `.compile-report.json`.
  loomc_string_view_t identifier;
} loomc_compile_report_options_t;

/// Parses a stable compile report mode spelling.
///
/// Accepts `""`, `"none"`, `"summary"`, `"details"`, `"json"`,
/// `"json-summary"`, or `"json-details"`.
LOOMC_API_EXPORT loomc_status_t loomc_compile_report_mode_parse(
    loomc_string_view_t value, loomc_compile_report_mode_t* out_mode);

/// Returns the stable JSON/CLI spelling for `mode`.
LOOMC_API_EXPORT loomc_string_view_t
loomc_compile_report_mode_name(loomc_compile_report_mode_t mode);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_REPORT_H_
