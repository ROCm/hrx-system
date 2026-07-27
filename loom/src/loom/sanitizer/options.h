// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sanitizer enablement options shared by compiler front doors and target
// pipeline construction.

#ifndef LOOM_SANITIZER_OPTIONS_H_
#define LOOM_SANITIZER_OPTIONS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_sanitizer_check_bit_e {
  // Enables memory access assertions.
  LOOM_SANITIZER_CHECK_ACCESS = 1u << 0,
  // Enables SSA value assertions.
  LOOM_SANITIZER_CHECK_VALUE = 1u << 1,
  // Enables operation-level assertions.
  LOOM_SANITIZER_CHECK_OPERATION = 1u << 2,
  // Enables data-race observations.
  LOOM_SANITIZER_CHECK_RACE = 1u << 3,
};
// Bitset of loom_sanitizer_check_bit_e values selecting sanitizer checks.
typedef uint64_t loom_sanitizer_checks_t;

enum loom_sanitizer_flag_bit_e {
  // No sanitizer flags are currently defined.
  LOOM_SANITIZER_FLAG_NONE = 0u,
};
// Bitset of loom_sanitizer_flag_bit_e values controlling sanitizer behavior.
typedef uint32_t loom_sanitizer_flags_t;

typedef enum loom_sanitizer_reporting_mode_e {
  // Use the target's default structured sanitizer report path.
  LOOM_SANITIZER_REPORTING_MODE_DEFAULT = 0,
  // Trap directly on sanitizer failures without emitting structured reports.
  LOOM_SANITIZER_REPORTING_MODE_TRAP = 1,
  // Emit structured sanitizer reports without using a direct trap.
  LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY = 2,
} loom_sanitizer_reporting_mode_t;

// Check bits understood by this compiler.
#define LOOM_SANITIZER_CHECKS_KNOWN                           \
  ((loom_sanitizer_checks_t)(LOOM_SANITIZER_CHECK_ACCESS |    \
                             LOOM_SANITIZER_CHECK_VALUE |     \
                             LOOM_SANITIZER_CHECK_OPERATION | \
                             LOOM_SANITIZER_CHECK_RACE))

// Flag bits understood by this compiler.
#define LOOM_SANITIZER_FLAGS_KNOWN ((loom_sanitizer_flags_t)0u)

typedef struct loom_sanitizer_options_t {
  // Sanitizer checks enabled by the caller.
  loom_sanitizer_checks_t checks;
  // Additional behavior flags enabled by the caller.
  loom_sanitizer_flags_t flags;
  // Target failure reporting behavior for lowered sanitizer assertions.
  loom_sanitizer_reporting_mode_t reporting_mode;
} loom_sanitizer_options_t;

// Returns true when sanitizer instrumentation is enabled.
static inline bool loom_sanitizer_options_is_enabled(
    const loom_sanitizer_options_t* options) {
  return options != NULL && options->checks != 0;
}

// Validates that all sanitizer option bits are understood by this compiler.
static inline iree_status_t loom_sanitizer_options_validate(
    const loom_sanitizer_options_t* options) {
  if (options == NULL) {
    return iree_ok_status();
  }
  if ((options->checks & ~LOOM_SANITIZER_CHECKS_KNOWN) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer options contain unknown check bits");
  }
  if ((options->flags & ~LOOM_SANITIZER_FLAGS_KNOWN) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer options contain unknown flag bits");
  }
  switch (options->reporting_mode) {
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT:
    case LOOM_SANITIZER_REPORTING_MODE_TRAP:
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "sanitizer options contain unknown reporting mode");
  }
  return iree_ok_status();
}

// Parses a sanitizer check set such as "access", "value|operation", "all",
// or "none". |diagnostic_name| is included in parse failures and should name
// the option being parsed, such as "--sanitizer" or "pass option 'checks'".
iree_status_t loom_sanitizer_checks_parse(iree_string_view_t value,
                                          iree_string_view_t diagnostic_name,
                                          loom_sanitizer_checks_t* out_checks);

// Parses a sanitizer option check set with no behavior flags. |checks == 0|
// returns disabled sanitizer options.
iree_status_t loom_sanitizer_options_parse_checks(
    iree_string_view_t value, iree_string_view_t diagnostic_name,
    loom_sanitizer_options_t* out_options);

// Parses a sanitizer reporting mode such as "default", "trap", or
// "report-only".
iree_status_t loom_sanitizer_reporting_mode_parse(
    iree_string_view_t value, iree_string_view_t diagnostic_name,
    loom_sanitizer_reporting_mode_t* out_mode);

// Formats a sanitizer check set in canonical pass-pipeline spelling.
iree_status_t loom_sanitizer_checks_format(loom_sanitizer_checks_t checks,
                                           iree_string_view_t* out_value);

// Formats a sanitizer reporting mode in canonical pass-pipeline spelling.
iree_status_t loom_sanitizer_reporting_mode_format(
    loom_sanitizer_reporting_mode_t mode, iree_string_view_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_OPTIONS_H_
