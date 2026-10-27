// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format.h"

void loom_target_compile_report_format_options_initialize(
    loom_target_compile_report_format_options_t* out_options) {
  *out_options = (loom_target_compile_report_format_options_t){
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
}

iree_string_view_t loom_target_compile_report_format_mode_name(
    loom_target_compile_report_format_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY:
      return IREE_SV("summary");
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS:
      return IREE_SV("details");
    default:
      return IREE_SV("unknown");
  }
}

loom_target_compile_report_detail_flags_t
loom_target_compile_report_requested_detail_flags(
    loom_target_compile_report_format_mode_t mode) {
  const loom_target_compile_report_detail_flags_t summary_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_RESIDENCY_CONSTRAINTS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  switch (mode) {
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE:
      return LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE;
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY:
      return summary_flags;
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS:
      return summary_flags | LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS |
             LOOM_TARGET_COMPILE_REPORT_DETAIL_PIPELINE_PLAN_ROWS;
    default:
      return LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE;
  }
}

iree_status_t loom_target_compile_report_format_mode_parse(
    iree_string_view_t value,
    loom_target_compile_report_format_mode_t* out_mode) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported compile report mode '%.*s'; expected "
                          "'none', 'summary', or 'details'",
                          (int)value.size, value.data);
}
