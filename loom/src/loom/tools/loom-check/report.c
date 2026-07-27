// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/report.h"

#include <string.h>

iree_status_t loom_check_file_report_initialize(
    const loom_check_file_t* file, iree_arena_allocator_t* arena,
    loom_check_file_report_t* out_report) {
  memset(out_report, 0, sizeof(*out_report));

  out_report->case_count = file->case_count;
  if (file->case_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, file->case_count, sizeof(iree_host_size_t),
        (void**)&out_report->case_annotation_offsets));
  }

  iree_host_size_t annotation_count = 0;
  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    out_report->case_annotation_offsets[i] = annotation_count;
    annotation_count += file->cases[i].annotation_count;
  }
  out_report->annotation_count = annotation_count;

  iree_host_size_t word_count = iree_bitmap_calculate_words(annotation_count);
  if (word_count > 0) {
    uint64_t* words = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(uint64_t), (void**)&words));
    memset(words, 0, word_count * sizeof(uint64_t));
    out_report->matched_annotations = (iree_bitmap_t){
        .bit_count = annotation_count,
        .words = words,
    };
  }

  return iree_ok_status();
}

void loom_check_file_report_reset_matches(loom_check_file_report_t* report) {
  iree_bitmap_reset_all(report->matched_annotations);
}

static iree_status_t loom_check_file_report_annotation_ordinal(
    const loom_check_file_report_t* report, iree_host_size_t case_index,
    iree_host_size_t annotation_index, iree_host_size_t* out_ordinal) {
  if (case_index >= report->case_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "case index %zu out of bounds for %zu cases",
                            case_index, report->case_count);
  }

  iree_host_size_t case_begin = report->case_annotation_offsets[case_index];
  iree_host_size_t case_end =
      case_index + 1 < report->case_count
          ? report->case_annotation_offsets[case_index + 1]
          : report->annotation_count;
  iree_host_size_t case_annotation_count = case_end - case_begin;
  if (annotation_index >= case_annotation_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "annotation index %zu out of bounds for %zu annotations in case %zu",
        annotation_index, case_annotation_count, case_index);
  }

  *out_ordinal = case_begin + annotation_index;
  return iree_ok_status();
}

iree_status_t loom_check_file_report_mark_annotation_matched(
    loom_check_file_report_t* report, iree_host_size_t case_index,
    iree_host_size_t annotation_index) {
  iree_host_size_t ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_ordinal(
      report, case_index, annotation_index, &ordinal));
  iree_bitmap_set(report->matched_annotations, ordinal);
  return iree_ok_status();
}

iree_status_t loom_check_file_report_annotation_matched(
    const loom_check_file_report_t* report, iree_host_size_t case_index,
    iree_host_size_t annotation_index, bool* out_matched) {
  iree_host_size_t ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_ordinal(
      report, case_index, annotation_index, &ordinal));
  *out_matched = iree_bitmap_test(report->matched_annotations, ordinal);
  return iree_ok_status();
}
