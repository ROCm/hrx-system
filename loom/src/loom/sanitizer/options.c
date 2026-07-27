// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/options.h"

static iree_string_view_t loom_sanitizer_diagnostic_name(
    iree_string_view_t diagnostic_name) {
  return iree_string_view_is_empty(diagnostic_name)
             ? IREE_SV("sanitizer checks")
             : diagnostic_name;
}

static iree_status_t loom_sanitizer_parse_check_token(
    iree_string_view_t token, iree_string_view_t diagnostic_name,
    loom_sanitizer_checks_t* inout_checks) {
  if (iree_string_view_is_empty(token)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s has an empty token",
                            (int)diagnostic_name.size, diagnostic_name.data);
  }
  if (iree_string_view_equal(token, IREE_SV("none"))) {
    *inout_checks = 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("all"))) {
    *inout_checks = LOOM_SANITIZER_CHECKS_KNOWN;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("access"))) {
    *inout_checks |= LOOM_SANITIZER_CHECK_ACCESS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("asan"))) {
    *inout_checks |= LOOM_SANITIZER_CHECK_ACCESS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("value"))) {
    *inout_checks |= LOOM_SANITIZER_CHECK_VALUE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("operation"))) {
    *inout_checks |= LOOM_SANITIZER_CHECK_OPERATION;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("ubsan"))) {
    *inout_checks |=
        LOOM_SANITIZER_CHECK_VALUE | LOOM_SANITIZER_CHECK_OPERATION;
    return iree_ok_status();
  }
  if (iree_string_view_equal(token, IREE_SV("race")) ||
      iree_string_view_equal(token, IREE_SV("tsan"))) {
    *inout_checks |= LOOM_SANITIZER_CHECK_RACE;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s has unknown token '%.*s'",
                          (int)diagnostic_name.size, diagnostic_name.data,
                          (int)token.size, token.data);
}

iree_status_t loom_sanitizer_checks_parse(iree_string_view_t value,
                                          iree_string_view_t diagnostic_name,
                                          loom_sanitizer_checks_t* out_checks) {
  diagnostic_name = loom_sanitizer_diagnostic_name(diagnostic_name);
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s must be non-empty", (int)diagnostic_name.size,
                            diagnostic_name.data);
  }

  loom_sanitizer_checks_t checks = 0;
  iree_string_view_t remaining = value;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    intptr_t separator_position =
        iree_string_view_split(remaining, '|', &token, &remaining);
    token = iree_string_view_trim(token);
    if ((iree_string_view_equal(token, IREE_SV("none")) ||
         iree_string_view_equal(token, IREE_SV("all"))) &&
        (checks != 0 || separator_position >= 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s token '%.*s' cannot be combined",
                              (int)diagnostic_name.size, diagnostic_name.data,
                              (int)token.size, token.data);
    }
    IREE_RETURN_IF_ERROR(
        loom_sanitizer_parse_check_token(token, diagnostic_name, &checks));
    if (separator_position >= 0 &&
        iree_string_view_is_empty(iree_string_view_trim(remaining))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s has a trailing separator",
                              (int)diagnostic_name.size, diagnostic_name.data);
    }
    if (separator_position < 0) break;
  }

  const loom_sanitizer_options_t options = {
      .checks = checks,
  };
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_validate(&options));
  *out_checks = checks;
  return iree_ok_status();
}

iree_status_t loom_sanitizer_options_parse_checks(
    iree_string_view_t value, iree_string_view_t diagnostic_name,
    loom_sanitizer_options_t* out_options) {
  loom_sanitizer_checks_t checks = 0;
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_checks_parse(value, diagnostic_name, &checks));
  *out_options = (loom_sanitizer_options_t){
      .checks = checks,
  };
  return iree_ok_status();
}

iree_status_t loom_sanitizer_reporting_mode_parse(
    iree_string_view_t value, iree_string_view_t diagnostic_name,
    loom_sanitizer_reporting_mode_t* out_mode) {
  diagnostic_name = loom_sanitizer_diagnostic_name(diagnostic_name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("default"))) {
    *out_mode = LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("trap"))) {
    *out_mode = LOOM_SANITIZER_REPORTING_MODE_TRAP;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("report-only"))) {
    *out_mode = LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s expected 'default', 'trap', or "
                          "'report-only', got '%.*s'",
                          (int)diagnostic_name.size, diagnostic_name.data,
                          (int)value.size, value.data);
}

iree_status_t loom_sanitizer_checks_format(loom_sanitizer_checks_t checks,
                                           iree_string_view_t* out_value) {
  switch (checks) {
    case 0:
      *out_value = IREE_SV("none");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS:
      *out_value = IREE_SV("access");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_VALUE:
      *out_value = IREE_SV("value");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_OPERATION:
      *out_value = IREE_SV("operation");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE:
      *out_value = IREE_SV("access|value");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_OPERATION:
      *out_value = IREE_SV("access|operation");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_VALUE | LOOM_SANITIZER_CHECK_OPERATION:
      *out_value = IREE_SV("value|operation");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("access|race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_VALUE | LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("value|race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_OPERATION | LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("operation|race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE |
        LOOM_SANITIZER_CHECK_OPERATION:
      *out_value = IREE_SV("access|value|operation");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE |
        LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("access|value|race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_OPERATION |
        LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("access|operation|race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_VALUE | LOOM_SANITIZER_CHECK_OPERATION |
        LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("value|operation|race");
      return iree_ok_status();
    case LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE |
        LOOM_SANITIZER_CHECK_OPERATION | LOOM_SANITIZER_CHECK_RACE:
      *out_value = IREE_SV("access|value|operation|race");
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown sanitizer check bits");
  }
}

iree_status_t loom_sanitizer_reporting_mode_format(
    loom_sanitizer_reporting_mode_t mode, iree_string_view_t* out_value) {
  switch (mode) {
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT:
      *out_value = IREE_SV("default");
      return iree_ok_status();
    case LOOM_SANITIZER_REPORTING_MODE_TRAP:
      *out_value = IREE_SV("trap");
      return iree_ok_status();
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY:
      *out_value = IREE_SV("report-only");
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown sanitizer reporting mode");
  }
}
