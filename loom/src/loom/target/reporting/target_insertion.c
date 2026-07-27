// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/target_insertion.h"

#include "loom/target/reporting/report.h"
#include "loom/target/reporting/row_list.h"

iree_string_view_t loom_target_compile_report_target_insertion_kind_name(
    loom_target_compile_report_target_insertion_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_STATE:
      return IREE_SV("state");
    case LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_WAIT:
      return IREE_SV("wait");
    case LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_DELAY:
      return IREE_SV("delay");
    case LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_OTHER:
      return IREE_SV("other");
    case LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_NONE:
    default:
      return IREE_SV("none");
  }
}

void loom_target_compile_report_accumulate_target_insertion_summary(
    loom_target_compile_report_target_insertion_summary_t* target,
    const loom_target_compile_report_target_insertion_summary_t* source) {
  target->static_packet_count += source->static_packet_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
}

iree_status_t loom_target_compile_report_record_target_insertion_row(
    struct loom_target_compile_report_t* report,
    const loom_target_compile_report_target_insertion_row_t* row) {
  loom_target_compile_report_target_insertion_summary_t summary = {
      .static_packet_count = row->static_packet_count,
  };
  if (iree_any_bit_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT)) {
    summary.exact_dynamic_packet_count = row->static_packet_count;
    summary.dynamic_packet_count = row->dynamic_packet_count;
  } else {
    summary.unknown_dynamic_packet_count = row->static_packet_count;
  }
  loom_target_compile_report_accumulate_target_insertion_summary(
      &report->target_insertion_summary, &summary);
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS;
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->target_insertion_rows, sizeof(*row), report->allocator, row);
}
