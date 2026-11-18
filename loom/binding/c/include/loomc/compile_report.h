// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_REPORT_H_
#define LOOMC_COMPILE_REPORT_H_

#include "loomc/base.h"

/// @file
/// Compile report controls.
///
/// Compile reports are optional structured or human-readable sidecars for
/// pass execution and target emission. They describe the backend, target,
/// terminal status, artifact size, and target-provided compiler analysis facts
/// available at the selected verbosity. Summary and detail reports include
/// configuration bindings applied by the module's last successful compilation.
/// These bindings survive release of the configuration module and compile
/// result, workspace trimming, and module cloning. IR serialization alone does
/// not carry invocation bindings.
///
/// Report generation is opt-in. Omitting this descriptor, or setting `mode` to
/// `LOOMC_COMPILE_REPORT_MODE_NONE`, avoids report collection and formatting.

#ifdef __cplusplus
extern "C" {
#endif

/// Loom compile report JSON artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON "loom-compile-report-json"

/// Loom compile report text artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT "loom-compile-report-text"

/// Compile report serialization format.
typedef enum loomc_compile_report_format_e {
  /// Emits structured diagnostic JSON for this compiler version.
  LOOMC_COMPILE_REPORT_FORMAT_JSON = 0,

  /// Emits bounded human-readable text.
  LOOMC_COMPILE_REPORT_FORMAT_TEXT = 1,
} loomc_compile_report_format_t;

/// Compile report detail mode.
typedef enum loomc_compile_report_mode_e {
  /// Does not request a compile report.
  LOOMC_COMPILE_REPORT_MODE_NONE = 0,

  /// Requests target, status, artifact, and summary compiler facts.
  LOOMC_COMPILE_REPORT_MODE_SUMMARY = 1,

  /// Requests summary facts plus detail rows when producers support them.
  LOOMC_COMPILE_REPORT_MODE_DETAILS = 2,
} loomc_compile_report_mode_t;

/// Requests a result-owned target compile report.
///
/// Attach through `loomc_pass_run_options_t::next` or
/// `loomc_compile_options_t::next` to report facts from an attempted pass
/// program. Rejection before the program starts produces no report. A diagnosed
/// pass failure retains completed observations in its failed result.
///
/// Attach through `loomc_emit_options_t::next` to report the target-artifact
/// transaction, including transactions rejected before emission. For combined
/// compilation and emission, attach to the nested emission options so one
/// report covers both stages.
///
/// Command-product construction accepts the same extension on its options
/// chain. Once planning starts, it reports resolved command order and storage
/// decisions in a result-owned artifact. Reports are diagnostic observations,
/// not a stable programming interface for reconstructing or executing a command
/// program.
///
/// Reports do not run extra passes, force target analyses, or write filesystem
/// paths. Result bytes retain no module or target owners. An infrastructure
/// failure returns no result and stops further report serialization. This
/// extension and native compile report options are mutually exclusive.
typedef struct loomc_compile_report_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Selected report detail mode.
  loomc_compile_report_mode_t mode;

  /// Report serialization format. Zero selects structured JSON.
  loomc_compile_report_format_t format;

  /// Result artifact identifier. Empty stays empty for pass execution and
  /// standalone compilation. During emission, an empty identifier derives from
  /// the emitted artifact identifier by appending `.compile-report.json` or
  /// `.compile-report.txt` according to `format`. It remains empty when
  /// compilation fails before selecting an emitter and no artifact identifier
  /// was provided.
  loomc_string_view_t identifier;
} loomc_compile_report_options_t;

/// Parses a stable compile report mode spelling.
///
/// Accepts `""`, `"none"`, `"summary"`, `"details"`, `"json"`,
/// `"json-summary"`, or `"json-details"`.
///
/// @param value Mode spelling to parse. The view need not be NUL-terminated.
/// @param out_mode Receives the parsed mode on success.
/// @return OK when `value` is recognized. Invalid argument is returned when
/// `out_mode` is `NULL` or the spelling is unsupported.
LOOMC_API_EXPORT loomc_status_t loomc_compile_report_mode_parse(
    loomc_string_view_t value, loomc_compile_report_mode_t* out_mode);

/// Returns the stable JSON/CLI spelling for `mode`.
///
/// @param mode Mode to name. Unrecognized values use the `"none"` spelling.
/// @return Borrowed static string view containing the stable spelling.
LOOMC_API_EXPORT loomc_string_view_t
loomc_compile_report_mode_name(loomc_compile_report_mode_t mode);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_REPORT_H_
