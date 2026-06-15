// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/compile_report.h"

#include "loomc/status.h"

loomc_status_t loomc_compile_report_mode_parse(
    loomc_string_view_t value, loomc_compile_report_mode_t* out_mode) {
  if (out_mode == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_mode must not be NULL");
  }
  if (loomc_string_view_is_empty(value) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("none"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_NONE;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(value, loomc_make_cstring_view("summary")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json-summary"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_SUMMARY;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(value, loomc_make_cstring_view("details")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json-details"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    return loomc_ok_status();
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "unsupported compile report mode");
}

loomc_string_view_t loomc_compile_report_mode_name(
    loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
      return loomc_make_cstring_view("summary");
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return loomc_make_cstring_view("details");
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    default:
      return loomc_make_cstring_view("none");
  }
}
