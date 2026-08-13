// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/memory_summary.h"

#include <string.h>

#include "loom/target/reporting/row_list.h"

static bool loom_target_compile_report_memory_interval_has_envelope(
    const loom_target_compile_report_memory_interval_t* interval) {
  const loom_target_compile_report_memory_interval_flags_t range_flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE;
  return iree_all_bits_set(interval->flags, range_flags) &&
         interval->end_max_bytes >= interval->begin_min_bytes;
}

static uint64_t loom_target_compile_report_memory_interval_span(
    int64_t begin_bytes, int64_t end_bytes) {
  return end_bytes >= begin_bytes ? (uint64_t)end_bytes - (uint64_t)begin_bytes
                                  : 0;
}

typedef struct loom_target_compile_report_static_memory_interval_t {
  int64_t begin_bytes;
  int64_t end_bytes;
} loom_target_compile_report_static_memory_interval_t;

typedef enum loom_target_compile_report_memory_interval_unique_kind_e {
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_NONE = 0,
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_STATIC = 1,
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_SYMBOLIC = 2,
} loom_target_compile_report_memory_interval_unique_kind_t;

typedef struct loom_target_compile_report_memory_interval_unique_delta_t {
  loom_target_compile_report_memory_interval_unique_kind_t kind;
  uint64_t unique_byte_delta;
} loom_target_compile_report_memory_interval_unique_delta_t;

static bool loom_target_compile_report_memory_interval_is_exact_static(
    const loom_target_compile_report_memory_interval_t* interval,
    loom_target_compile_report_static_memory_interval_t* out_static_interval) {
  if (!loom_target_compile_report_memory_interval_has_envelope(interval) ||
      interval->begin_min_bytes != interval->begin_max_bytes ||
      interval->end_min_bytes != interval->end_max_bytes) {
    return false;
  }
  *out_static_interval = (loom_target_compile_report_static_memory_interval_t){
      .begin_bytes = interval->begin_min_bytes,
      .end_bytes = interval->end_max_bytes,
  };
  return true;
}

static bool loom_target_compile_report_memory_interval_has_exact_symbolic(
    const loom_target_compile_report_memory_interval_t* interval) {
  const loom_target_compile_report_memory_interval_flags_t required_flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR;
  return iree_all_bits_set(interval->flags, required_flags) &&
         interval->exact_length_bytes != 0;
}

static bool loom_target_compile_report_source_low_memory_row_has_root(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return !iree_string_view_is_empty(row->source_root_name) ||
         row->source_root_argument_index != UINT16_MAX;
}

static bool loom_target_compile_report_source_low_memory_rows_match_interval(
    const loom_target_compile_report_source_low_memory_row_t* lhs,
    const loom_target_compile_report_source_low_memory_row_t* rhs,
    bool match_operation_kind) {
  if (!loom_target_compile_report_source_low_memory_row_has_root(lhs) ||
      !loom_target_compile_report_source_low_memory_row_has_root(rhs)) {
    return false;
  }
  if (!iree_string_view_equal(lhs->function_name, rhs->function_name) ||
      !iree_string_view_equal(lhs->source_root_name, rhs->source_root_name) ||
      lhs->source_root_argument_index != rhs->source_root_argument_index ||
      !iree_string_view_equal(lhs->memory_space, rhs->memory_space)) {
    return false;
  }
  return !match_operation_kind ||
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind);
}

static bool loom_target_compile_report_static_memory_intervals_overlap(
    loom_target_compile_report_static_memory_interval_t lhs,
    loom_target_compile_report_static_memory_interval_t rhs,
    loom_target_compile_report_static_memory_interval_t* out_overlap) {
  const int64_t begin_bytes = iree_max(lhs.begin_bytes, rhs.begin_bytes);
  const int64_t end_bytes = iree_min(lhs.end_bytes, rhs.end_bytes);
  if (begin_bytes >= end_bytes) {
    return false;
  }
  *out_overlap = (loom_target_compile_report_static_memory_interval_t){
      .begin_bytes = begin_bytes,
      .end_bytes = end_bytes,
  };
  return true;
}

static bool loom_target_compile_report_memory_interval_envelopes_are_disjoint(
    const loom_target_compile_report_memory_interval_t* lhs,
    const loom_target_compile_report_memory_interval_t* rhs) {
  if (!loom_target_compile_report_memory_interval_has_envelope(lhs) ||
      !loom_target_compile_report_memory_interval_has_envelope(rhs)) {
    return false;
  }
  return lhs->end_max_bytes <= rhs->begin_min_bytes ||
         rhs->end_max_bytes <= lhs->begin_min_bytes;
}

static void loom_target_compile_report_insert_sorted_static_memory_interval(
    loom_target_compile_report_static_memory_interval_t* intervals,
    iree_host_size_t* inout_count,
    loom_target_compile_report_static_memory_interval_t interval) {
  iree_host_size_t index = *inout_count;
  while (index > 0 &&
         (intervals[index - 1].begin_bytes > interval.begin_bytes ||
          (intervals[index - 1].begin_bytes == interval.begin_bytes &&
           intervals[index - 1].end_bytes > interval.end_bytes))) {
    intervals[index] = intervals[index - 1];
    --index;
  }
  intervals[index] = interval;
  ++*inout_count;
}

static uint64_t loom_target_compile_report_static_memory_interval_union_bytes(
    const loom_target_compile_report_static_memory_interval_t* intervals,
    iree_host_size_t interval_count) {
  if (interval_count == 0) {
    return 0;
  }
  int64_t begin_bytes = intervals[0].begin_bytes;
  int64_t end_bytes = intervals[0].end_bytes;
  uint64_t byte_count = 0;
  for (iree_host_size_t i = 1; i < interval_count; ++i) {
    if (intervals[i].begin_bytes <= end_bytes) {
      end_bytes = iree_max(end_bytes, intervals[i].end_bytes);
      continue;
    }
    byte_count +=
        loom_target_compile_report_memory_interval_span(begin_bytes, end_bytes);
    begin_bytes = intervals[i].begin_bytes;
    end_bytes = intervals[i].end_bytes;
  }
  return byte_count + loom_target_compile_report_memory_interval_span(
                          begin_bytes, end_bytes);
}

static iree_status_t
loom_target_compile_report_calculate_source_low_unique_interval_delta(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row,
    bool match_operation_kind,
    loom_target_compile_report_memory_interval_unique_delta_t* out_delta) {
  *out_delta = (loom_target_compile_report_memory_interval_unique_delta_t){0};
  if (iree_allocator_is_null(report->allocator) ||
      !loom_target_compile_report_source_low_memory_row_has_root(row)) {
    return iree_ok_status();
  }
  loom_target_compile_report_static_memory_interval_t row_interval = {0};
  const bool row_is_exact_static =
      loom_target_compile_report_memory_interval_is_exact_static(
          &row->source_interval, &row_interval);
  const bool row_is_exact_symbolic =
      !row_is_exact_static &&
      loom_target_compile_report_memory_interval_has_exact_symbolic(
          &row->source_interval);
  if (!row_is_exact_static && !row_is_exact_symbolic) {
    return iree_ok_status();
  }
  out_delta->kind =
      row_is_exact_static
          ? LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_STATIC
          : LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_SYMBOLIC;
  const uint64_t row_byte_count =
      row_is_exact_static
          ? loom_target_compile_report_memory_interval_span(
                row_interval.begin_bytes, row_interval.end_bytes)
          : row->source_interval.exact_length_bytes;
  if (report->source_low_memory_rows.count == 0) {
    out_delta->unique_byte_delta = row_byte_count;
    return iree_ok_status();
  }

  loom_target_compile_report_static_memory_interval_t* overlaps = NULL;
  if (row_is_exact_static) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        report->allocator, report->source_low_memory_rows.count,
        sizeof(*overlaps), (void**)&overlaps));
  }

  iree_host_size_t overlap_count = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_row_t* rows =
        (const loom_target_compile_report_source_low_memory_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_source_low_memory_row_t* existing =
          &rows[i];
      if (!loom_target_compile_report_source_low_memory_rows_match_interval(
              existing, row, match_operation_kind)) {
        continue;
      }
      if (row_is_exact_static) {
        loom_target_compile_report_static_memory_interval_t existing_interval =
            {0};
        loom_target_compile_report_static_memory_interval_t overlap = {0};
        if (loom_target_compile_report_memory_interval_is_exact_static(
                &existing->source_interval, &existing_interval)) {
          if (loom_target_compile_report_static_memory_intervals_overlap(
                  row_interval, existing_interval, &overlap)) {
            loom_target_compile_report_insert_sorted_static_memory_interval(
                overlaps, &overlap_count, overlap);
          }
          continue;
        }
        if (loom_target_compile_report_memory_interval_has_exact_symbolic(
                &existing->source_interval) &&
            !loom_target_compile_report_memory_interval_envelopes_are_disjoint(
                &row->source_interval, &existing->source_interval)) {
          iree_allocator_free(report->allocator, overlaps);
          *out_delta =
              (loom_target_compile_report_memory_interval_unique_delta_t){0};
          return iree_ok_status();
        }
        continue;
      }

      loom_target_compile_report_static_memory_interval_t existing_interval = {
          0};
      if (loom_target_compile_report_memory_interval_is_exact_static(
              &existing->source_interval, &existing_interval)) {
        if (!loom_target_compile_report_memory_interval_envelopes_are_disjoint(
                &row->source_interval, &existing->source_interval)) {
          *out_delta =
              (loom_target_compile_report_memory_interval_unique_delta_t){0};
          return iree_ok_status();
        }
        continue;
      }
      if (!loom_target_compile_report_memory_interval_has_exact_symbolic(
              &existing->source_interval)) {
        continue;
      }
      if (row->source_interval.begin_expr_id ==
              existing->source_interval.begin_expr_id &&
          row->source_interval.end_expr_id ==
              existing->source_interval.end_expr_id) {
        out_delta->unique_byte_delta = 0;
        return iree_ok_status();
      }
      if (existing->source_interval.end_expr_id ==
              row->source_interval.begin_expr_id ||
          row->source_interval.end_expr_id ==
              existing->source_interval.begin_expr_id ||
          loom_target_compile_report_memory_interval_envelopes_are_disjoint(
              &row->source_interval, &existing->source_interval)) {
        continue;
      }
      *out_delta =
          (loom_target_compile_report_memory_interval_unique_delta_t){0};
      return iree_ok_status();
    }
  }

  if (row_is_exact_static) {
    const uint64_t covered_byte_count =
        loom_target_compile_report_static_memory_interval_union_bytes(
            overlaps, overlap_count);
    iree_allocator_free(report->allocator, overlaps);
    out_delta->unique_byte_delta = row_byte_count > covered_byte_count
                                       ? row_byte_count - covered_byte_count
                                       : 0;
  } else {
    out_delta->unique_byte_delta = row_byte_count;
  }
  return iree_ok_status();
}

static void loom_target_compile_report_merge_memory_interval_envelope_bounds(
    loom_target_compile_report_memory_interval_summary_t* target,
    int64_t begin_min_bytes, int64_t end_max_bytes) {
  if (target->packet_count == 0) {
    target->envelope_begin_min_bytes = begin_min_bytes;
    target->envelope_end_max_bytes = end_max_bytes;
  } else {
    target->envelope_begin_min_bytes =
        iree_min(target->envelope_begin_min_bytes, begin_min_bytes);
    target->envelope_end_max_bytes =
        iree_max(target->envelope_end_max_bytes, end_max_bytes);
  }
  target->envelope_byte_count = loom_target_compile_report_memory_interval_span(
      target->envelope_begin_min_bytes, target->envelope_end_max_bytes);
}

static void loom_target_compile_report_accumulate_memory_interval_summary(
    loom_target_compile_report_memory_interval_summary_t* target,
    const loom_target_compile_report_memory_interval_t* interval,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta) {
  if (!loom_target_compile_report_memory_interval_has_envelope(interval)) {
    return;
  }
  loom_target_compile_report_merge_memory_interval_envelope_bounds(
      target, interval->begin_min_bytes, interval->end_max_bytes);
  ++target->packet_count;
  switch (unique_delta.kind) {
    case LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_STATIC:
      ++target->exact_static_packet_count;
      target->unique_byte_count += unique_delta.unique_byte_delta;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_SYMBOLIC:
      ++target->exact_symbolic_packet_count;
      target->unique_byte_count += unique_delta.unique_byte_delta;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_NONE:
    default:
      break;
  }
}

static void loom_target_compile_report_merge_memory_interval_envelope_summary(
    loom_target_compile_report_memory_interval_summary_t* target,
    const loom_target_compile_report_memory_interval_summary_t* source) {
  if (source->packet_count == 0) {
    return;
  }
  loom_target_compile_report_merge_memory_interval_envelope_bounds(
      target, source->envelope_begin_min_bytes, source->envelope_end_max_bytes);
  target->packet_count += source->packet_count;
}

static void loom_target_compile_report_forget_memory_interval_unique_accounting(
    loom_target_compile_report_memory_interval_summary_t* summary) {
  summary->exact_static_packet_count = 0;
  summary->exact_symbolic_packet_count = 0;
  summary->unique_byte_count = 0;
}

void loom_target_compile_report_accumulate_bank_service_summaries(
    loom_target_compile_report_bank_service_summary_t* target,
    const loom_target_compile_report_bank_service_summary_t* source) {
  target->modeled_packet_count += source->modeled_packet_count;
  target->exact_packet_count += source->exact_packet_count;
  target->unknown_packet_count += source->unknown_packet_count;
  target->conflict_free_packet_count += source->conflict_free_packet_count;
  target->conflicted_packet_count += source->conflicted_packet_count;
  target->required_round_count += source->required_round_count;
  target->uncontended_round_count += source->uncontended_round_count;
  target->extra_round_count += source->extra_round_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
  target->dynamic_required_round_count += source->dynamic_required_round_count;
  target->dynamic_uncontended_round_count +=
      source->dynamic_uncontended_round_count;
  target->dynamic_extra_round_count += source->dynamic_extra_round_count;
  target->maximum_request_multiplicity =
      iree_max(target->maximum_request_multiplicity,
               source->maximum_request_multiplicity);
}

void loom_target_compile_report_accumulate_subgroup_access_summaries(
    loom_target_compile_report_subgroup_access_summary_t* target,
    const loom_target_compile_report_subgroup_access_summary_t* source) {
  target->modeled_packet_count += source->modeled_packet_count;
  target->exact_packet_count += source->exact_packet_count;
  target->unknown_packet_count += source->unknown_packet_count;
  target->dense_packet_count += source->dense_packet_count;
  target->gapped_packet_count += source->gapped_packet_count;
  target->overlapping_packet_count += source->overlapping_packet_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
  target->dynamic_dense_packet_count += source->dynamic_dense_packet_count;
  target->dynamic_gapped_packet_count += source->dynamic_gapped_packet_count;
  target->dynamic_overlapping_packet_count +=
      source->dynamic_overlapping_packet_count;
}

void loom_target_compile_report_accumulate_source_low_memory_summaries(
    loom_target_compile_report_source_low_memory_summary_t* target,
    const loom_target_compile_report_source_low_memory_summary_t* source) {
  target->packet_count += source->packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->load_packet_count += source->load_packet_count;
  target->store_packet_count += source->store_packet_count;
  target->scalar_packet_count += source->scalar_packet_count;
  target->vector_packet_count += source->vector_packet_count;
  target->source_lane_count += source->source_lane_count;
  target->source_byte_count += source->source_byte_count;
  target->read_byte_count += source->read_byte_count;
  target->write_byte_count += source->write_byte_count;
  target->issued_read_byte_count += source->issued_read_byte_count;
  target->issued_write_byte_count += source->issued_write_byte_count;
  target->issued_read_unknown_width_count +=
      source->issued_read_unknown_width_count;
  target->issued_write_unknown_width_count +=
      source->issued_write_unknown_width_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
  target->dynamic_source_byte_count += source->dynamic_source_byte_count;
  target->dynamic_read_byte_count += source->dynamic_read_byte_count;
  target->dynamic_write_byte_count += source->dynamic_write_byte_count;
  target->dynamic_issued_read_byte_count +=
      source->dynamic_issued_read_byte_count;
  target->dynamic_issued_write_byte_count +=
      source->dynamic_issued_write_byte_count;
  target->dynamic_issued_read_unknown_width_count +=
      source->dynamic_issued_read_unknown_width_count;
  target->dynamic_issued_write_unknown_width_count +=
      source->dynamic_issued_write_unknown_width_count;
  target->contiguous_vector_packet_count +=
      source->contiguous_vector_packet_count;
  target->strided_vector_packet_count += source->strided_vector_packet_count;
  target->unknown_stride_vector_packet_count +=
      source->unknown_stride_vector_packet_count;
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->interval_envelope, &source->interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->read_interval_envelope, &source->read_interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->write_interval_envelope, &source->write_interval_envelope);
}

static void loom_target_compile_report_merge_source_low_memory_summary(
    loom_target_compile_report_source_low_memory_summary_t* target,
    const loom_target_compile_report_source_low_memory_summary_t* source) {
  target->packet_count += source->packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->load_packet_count += source->load_packet_count;
  target->store_packet_count += source->store_packet_count;
  target->scalar_packet_count += source->scalar_packet_count;
  target->vector_packet_count += source->vector_packet_count;
  target->source_lane_count += source->source_lane_count;
  target->source_byte_count += source->source_byte_count;
  target->read_byte_count += source->read_byte_count;
  target->write_byte_count += source->write_byte_count;
  target->issued_read_byte_count += source->issued_read_byte_count;
  target->issued_write_byte_count += source->issued_write_byte_count;
  target->issued_read_unknown_width_count +=
      source->issued_read_unknown_width_count;
  target->issued_write_unknown_width_count +=
      source->issued_write_unknown_width_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
  target->dynamic_source_byte_count += source->dynamic_source_byte_count;
  target->dynamic_read_byte_count += source->dynamic_read_byte_count;
  target->dynamic_write_byte_count += source->dynamic_write_byte_count;
  target->dynamic_issued_read_byte_count +=
      source->dynamic_issued_read_byte_count;
  target->dynamic_issued_write_byte_count +=
      source->dynamic_issued_write_byte_count;
  target->dynamic_issued_read_unknown_width_count +=
      source->dynamic_issued_read_unknown_width_count;
  target->dynamic_issued_write_unknown_width_count +=
      source->dynamic_issued_write_unknown_width_count;
  target->contiguous_vector_packet_count +=
      source->contiguous_vector_packet_count;
  target->strided_vector_packet_count += source->strided_vector_packet_count;
  target->unknown_stride_vector_packet_count +=
      source->unknown_stride_vector_packet_count;
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->interval_envelope, &source->interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->read_interval_envelope, &source->read_interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->write_interval_envelope, &source->write_interval_envelope);
  loom_target_compile_report_forget_memory_interval_unique_accounting(
      &target->interval_envelope);
  loom_target_compile_report_forget_memory_interval_unique_accounting(
      &target->read_interval_envelope);
  loom_target_compile_report_forget_memory_interval_unique_accounting(
      &target->write_interval_envelope);
}

static void loom_target_compile_report_merge_source_low_memory_root_summary(
    loom_target_compile_report_source_low_memory_root_summary_t* target,
    const loom_target_compile_report_source_low_memory_root_summary_t* source) {
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static void loom_target_compile_report_merge_source_low_memory_argument_summary(
    loom_target_compile_report_source_low_memory_argument_summary_t* target,
    const loom_target_compile_report_source_low_memory_argument_summary_t*
        source) {
  if (!iree_string_view_is_empty(target->source_root_name) &&
      (iree_string_view_is_empty(source->source_root_name) ||
       !iree_string_view_equal(target->source_root_name,
                               source->source_root_name))) {
    target->source_root_name = iree_string_view_empty();
  }
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static void
loom_target_compile_report_merge_source_low_memory_argument_packet_summary(
    loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        target,
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        source) {
  if (!iree_string_view_is_empty(target->source_root_name) &&
      (iree_string_view_is_empty(source->source_root_name) ||
       !iree_string_view_equal(target->source_root_name,
                               source->source_root_name))) {
    target->source_root_name = iree_string_view_empty();
  }
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static void loom_target_compile_report_merge_source_low_memory_strategy_summary(
    loom_target_compile_report_source_low_memory_strategy_summary_t* target,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        source) {
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static loom_target_compile_report_source_low_memory_root_summary_t*
loom_target_compile_report_find_source_low_memory_root_summary(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t source_root_name, uint16_t source_root_argument_index,
    iree_string_view_t memory_space) {
  const bool has_root_identity = !iree_string_view_is_empty(source_root_name) ||
                                 source_root_argument_index != UINT16_MAX;
  if (!has_root_identity) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_root_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_root_summary_t* summaries =
        (loom_target_compile_report_source_low_memory_root_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_root_summary_t* summary =
          &summaries[i];
      if (iree_string_view_equal(summary->function_name, function_name) &&
          iree_string_view_equal(summary->source_root_name, source_root_name) &&
          summary->source_root_argument_index == source_root_argument_index &&
          iree_string_view_equal(summary->memory_space, memory_space)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_root_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_root_summary_t* row) {
  loom_target_compile_report_source_low_memory_root_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_root_summary(
          report, row->function_name, row->source_root_name,
          row->source_root_argument_index, row->memory_space);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_root_summary(summary,
                                                                    row);
    return iree_ok_status();
  } else if (iree_string_view_is_empty(row->source_root_name) &&
             row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_root_summaries, sizeof(*row),
      report->allocator, row);
}

static loom_target_compile_report_source_low_memory_argument_summary_t*
loom_target_compile_report_find_source_low_memory_argument_summary(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    uint16_t source_root_argument_index, iree_string_view_t memory_space) {
  if (source_root_argument_index == UINT16_MAX) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_argument_summary_t* summaries =
        (loom_target_compile_report_source_low_memory_argument_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_argument_summary_t* summary =
          &summaries[i];
      if (iree_string_view_equal(summary->function_name, function_name) &&
          summary->source_root_argument_index == source_root_argument_index &&
          iree_string_view_equal(summary->memory_space, memory_space)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_argument_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_argument_summary_t*
        row) {
  loom_target_compile_report_source_low_memory_argument_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_argument_summary(
          report, row->function_name, row->source_root_argument_index,
          row->memory_space);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_argument_summary(summary,
                                                                        row);
    return iree_ok_status();
  } else if (row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_argument_summaries, sizeof(*row),
      report->allocator, row);
}

static loom_target_compile_report_source_low_memory_argument_packet_summary_t
loom_target_compile_report_source_low_memory_argument_packet_summary_from_row(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return (
      loom_target_compile_report_source_low_memory_argument_packet_summary_t){
      .function_name = row->function_name,
      .source_root_name = row->source_root_name,
      .source_root_argument_index = row->source_root_argument_index,
      .memory_space = row->memory_space,
      .operation_kind = row->operation_kind,
      .packet_key = row->packet_key,
      .strategy_key = row->strategy_key,
      .fallback_reason = row->fallback_reason,
      .storage_element_format = row->storage_element_format,
      .storage_scale_format = row->storage_scale_format,
      .storage_secondary_scale_format = row->storage_secondary_scale_format,
      .storage_payload_packing = row->storage_payload_packing,
      .storage_scale_topology = row->storage_scale_topology,
      .storage_affine_policy = row->storage_affine_policy,
      .storage_rounding_policy = row->storage_rounding_policy,
      .storage_codebook_policy = row->storage_codebook_policy,
      .storage_sparsity_policy = row->storage_sparsity_policy,
  };
}

static bool
loom_target_compile_report_source_low_memory_argument_packet_summaries_match(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        lhs,
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        rhs) {
  return iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         lhs->source_root_argument_index == rhs->source_root_argument_index &&
         iree_string_view_equal(lhs->memory_space, rhs->memory_space) &&
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind) &&
         iree_string_view_equal(lhs->packet_key, rhs->packet_key) &&
         iree_string_view_equal(lhs->strategy_key, rhs->strategy_key) &&
         iree_string_view_equal(lhs->fallback_reason, rhs->fallback_reason) &&
         iree_string_view_equal(lhs->storage_element_format,
                                rhs->storage_element_format) &&
         iree_string_view_equal(lhs->storage_scale_format,
                                rhs->storage_scale_format) &&
         iree_string_view_equal(lhs->storage_secondary_scale_format,
                                rhs->storage_secondary_scale_format) &&
         iree_string_view_equal(lhs->storage_payload_packing,
                                rhs->storage_payload_packing) &&
         iree_string_view_equal(lhs->storage_scale_topology,
                                rhs->storage_scale_topology) &&
         iree_string_view_equal(lhs->storage_affine_policy,
                                rhs->storage_affine_policy) &&
         iree_string_view_equal(lhs->storage_rounding_policy,
                                rhs->storage_rounding_policy) &&
         iree_string_view_equal(lhs->storage_codebook_policy,
                                rhs->storage_codebook_policy) &&
         iree_string_view_equal(lhs->storage_sparsity_policy,
                                rhs->storage_sparsity_policy);
}

static loom_target_compile_report_source_low_memory_argument_packet_summary_t*
loom_target_compile_report_find_source_low_memory_argument_packet_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        row) {
  if (row->source_root_argument_index == UINT16_MAX) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_packet_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summaries =
            (loom_target_compile_report_source_low_memory_argument_packet_summary_t*)
                loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_argument_packet_summary_t*
          summary = &summaries[i];
      if (loom_target_compile_report_source_low_memory_argument_packet_summaries_match(
              summary, row)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_argument_packet_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        row) {
  loom_target_compile_report_source_low_memory_argument_packet_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_argument_packet_summary(
          report, row);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_argument_packet_summary(
        summary, row);
    return iree_ok_status();
  } else if (row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_argument_packet_summaries, sizeof(*row),
      report->allocator, row);
}

static loom_target_compile_report_source_low_memory_strategy_summary_t
loom_target_compile_report_source_low_memory_strategy_summary_from_row(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return (loom_target_compile_report_source_low_memory_strategy_summary_t){
      .function_name = row->function_name,
      .memory_space = row->memory_space,
      .operation_kind = row->operation_kind,
      .packet_key = row->packet_key,
      .strategy_key = row->strategy_key,
      .fallback_reason = row->fallback_reason,
      .storage_element_format = row->storage_element_format,
      .storage_scale_format = row->storage_scale_format,
      .storage_secondary_scale_format = row->storage_secondary_scale_format,
      .storage_payload_packing = row->storage_payload_packing,
      .storage_scale_topology = row->storage_scale_topology,
      .storage_affine_policy = row->storage_affine_policy,
      .storage_rounding_policy = row->storage_rounding_policy,
      .storage_codebook_policy = row->storage_codebook_policy,
      .storage_sparsity_policy = row->storage_sparsity_policy,
  };
}

static bool
loom_target_compile_report_source_low_memory_strategy_summaries_match(
    const loom_target_compile_report_source_low_memory_strategy_summary_t* lhs,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        rhs) {
  return iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         iree_string_view_equal(lhs->memory_space, rhs->memory_space) &&
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind) &&
         iree_string_view_equal(lhs->packet_key, rhs->packet_key) &&
         iree_string_view_equal(lhs->strategy_key, rhs->strategy_key) &&
         iree_string_view_equal(lhs->fallback_reason, rhs->fallback_reason) &&
         iree_string_view_equal(lhs->storage_element_format,
                                rhs->storage_element_format) &&
         iree_string_view_equal(lhs->storage_scale_format,
                                rhs->storage_scale_format) &&
         iree_string_view_equal(lhs->storage_secondary_scale_format,
                                rhs->storage_secondary_scale_format) &&
         iree_string_view_equal(lhs->storage_payload_packing,
                                rhs->storage_payload_packing) &&
         iree_string_view_equal(lhs->storage_scale_topology,
                                rhs->storage_scale_topology) &&
         iree_string_view_equal(lhs->storage_affine_policy,
                                rhs->storage_affine_policy) &&
         iree_string_view_equal(lhs->storage_rounding_policy,
                                rhs->storage_rounding_policy) &&
         iree_string_view_equal(lhs->storage_codebook_policy,
                                rhs->storage_codebook_policy) &&
         iree_string_view_equal(lhs->storage_sparsity_policy,
                                rhs->storage_sparsity_policy);
}

static loom_target_compile_report_source_low_memory_strategy_summary_t*
loom_target_compile_report_find_source_low_memory_strategy_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        row) {
  if (iree_string_view_is_empty(row->strategy_key)) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_strategy_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_strategy_summary_t* summaries =
        (loom_target_compile_report_source_low_memory_strategy_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_strategy_summary_t* summary =
          &summaries[i];
      if (loom_target_compile_report_source_low_memory_strategy_summaries_match(
              summary, row)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_strategy_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        row) {
  loom_target_compile_report_source_low_memory_strategy_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_strategy_summary(report,
                                                                         row);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_strategy_summary(summary,
                                                                        row);
    return iree_ok_status();
  } else if (iree_string_view_is_empty(row->strategy_key)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_strategy_summaries, sizeof(*row),
      report->allocator, row);
}

static bool loom_target_compile_report_source_low_bank_service_summaries_match(
    const loom_target_compile_report_source_low_bank_service_summary_t* lhs,
    const loom_target_compile_report_source_low_bank_service_summary_t* rhs) {
  return iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         iree_string_view_equal(lhs->source_op_name, rhs->source_op_name) &&
         lhs->source_op_kind == rhs->source_op_kind &&
         iree_string_view_equal(lhs->source_root_name, rhs->source_root_name) &&
         lhs->source_root_argument_index == rhs->source_root_argument_index &&
         iree_string_view_equal(lhs->memory_space, rhs->memory_space) &&
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind) &&
         iree_string_view_equal(lhs->packet_key, rhs->packet_key) &&
         iree_string_view_equal(lhs->strategy_key, rhs->strategy_key) &&
         iree_string_view_equal(lhs->model_key, rhs->model_key) &&
         iree_string_view_equal(lhs->model_revision, rhs->model_revision) &&
         iree_string_view_equal(lhs->model_evidence, rhs->model_evidence) &&
         iree_string_view_equal(lhs->request_policy, rhs->request_policy) &&
         lhs->wave_size == rhs->wave_size &&
         lhs->bank_count == rhs->bank_count &&
         lhs->bank_word_byte_count == rhs->bank_word_byte_count &&
         lhs->packet_word_count == rhs->packet_word_count;
}

static loom_target_compile_report_source_low_bank_service_summary_t*
loom_target_compile_report_find_source_low_bank_service_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_bank_service_summary_t* row) {
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_bank_service_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_bank_service_summary_t* summaries =
        (loom_target_compile_report_source_low_bank_service_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_bank_service_summary_t* summary =
          &summaries[i];
      if (loom_target_compile_report_source_low_bank_service_summaries_match(
              summary, row)) {
        return summary;
      }
    }
  }
  return NULL;
}

static void loom_target_compile_report_merge_unknown_bank_service_reason(
    loom_target_compile_report_source_low_bank_service_summary_t* target,
    iree_string_view_t source_reason, bool source_has_mixed_reasons,
    uint64_t source_unknown_packet_count) {
  if (source_unknown_packet_count == 0) {
    return;
  }
  if (target->summary.unknown_packet_count == 0) {
    target->unknown_reason = source_reason;
    target->has_mixed_unknown_reasons = source_has_mixed_reasons;
    return;
  }
  if (target->has_mixed_unknown_reasons || source_has_mixed_reasons ||
      !iree_string_view_equal(target->unknown_reason, source_reason)) {
    target->unknown_reason = iree_string_view_empty();
    target->has_mixed_unknown_reasons = true;
  }
}

static void loom_target_compile_report_accumulate_bank_service_summary(
    loom_target_compile_report_bank_service_summary_t* summary,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  const loom_target_compile_report_bank_service_t* bank_service =
      &row->bank_service;
  if (iree_string_view_is_empty(bank_service->model_key)) {
    return;
  }

  ++summary->modeled_packet_count;
  if (!iree_string_view_equal(bank_service->proof, IREE_SV("exact"))) {
    ++summary->unknown_packet_count;
    ++summary->unknown_dynamic_packet_count;
    return;
  }

  ++summary->exact_packet_count;
  if (iree_string_view_equal(bank_service->classification,
                             IREE_SV("conflict-free"))) {
    ++summary->conflict_free_packet_count;
  } else if (iree_string_view_equal(bank_service->classification,
                                    IREE_SV("conflicted"))) {
    ++summary->conflicted_packet_count;
  }
  summary->required_round_count += bank_service->required_rounds;
  summary->uncontended_round_count += bank_service->uncontended_rounds;
  summary->extra_round_count += bank_service->extra_rounds;
  summary->maximum_request_multiplicity =
      iree_max(summary->maximum_request_multiplicity,
               bank_service->maximum_request_multiplicity);

  if (row->execution_count_plus_one ==
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN) {
    ++summary->unknown_dynamic_packet_count;
    return;
  }

  const uint64_t execution_count = row->execution_count_plus_one - 1;
  uint64_t dynamic_required_round_count = 0;
  uint64_t dynamic_uncontended_round_count = 0;
  uint64_t dynamic_extra_round_count = 0;
  const bool dynamic_counts_ok =
      iree_checked_mul_u64(bank_service->required_rounds, execution_count,
                           &dynamic_required_round_count) &&
      iree_checked_mul_u64(bank_service->uncontended_rounds, execution_count,
                           &dynamic_uncontended_round_count) &&
      iree_checked_mul_u64(bank_service->extra_rounds, execution_count,
                           &dynamic_extra_round_count);
  uint64_t new_dynamic_packet_count = summary->dynamic_packet_count;
  uint64_t new_dynamic_required_round_count =
      summary->dynamic_required_round_count;
  uint64_t new_dynamic_uncontended_round_count =
      summary->dynamic_uncontended_round_count;
  uint64_t new_dynamic_extra_round_count = summary->dynamic_extra_round_count;
  const bool dynamic_accumulation_ok =
      dynamic_counts_ok &&
      iree_checked_add_u64(new_dynamic_packet_count, execution_count,
                           &new_dynamic_packet_count) &&
      iree_checked_add_u64(new_dynamic_required_round_count,
                           dynamic_required_round_count,
                           &new_dynamic_required_round_count) &&
      iree_checked_add_u64(new_dynamic_uncontended_round_count,
                           dynamic_uncontended_round_count,
                           &new_dynamic_uncontended_round_count) &&
      iree_checked_add_u64(new_dynamic_extra_round_count,
                           dynamic_extra_round_count,
                           &new_dynamic_extra_round_count);
  if (!dynamic_accumulation_ok) {
    ++summary->unknown_dynamic_packet_count;
    return;
  }

  ++summary->exact_dynamic_packet_count;
  summary->dynamic_packet_count = new_dynamic_packet_count;
  summary->dynamic_required_round_count = new_dynamic_required_round_count;
  summary->dynamic_uncontended_round_count =
      new_dynamic_uncontended_round_count;
  summary->dynamic_extra_round_count = new_dynamic_extra_round_count;
}

static loom_target_compile_report_source_low_bank_service_summary_t
loom_target_compile_report_source_low_bank_service_summary_from_row(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  const loom_target_compile_report_bank_service_t* bank_service =
      &row->bank_service;
  return (loom_target_compile_report_source_low_bank_service_summary_t){
      .function_name = row->function_name,
      .source_op_name = row->source_op_name,
      .source_op_kind = row->source_op_kind,
      .source_root_name = row->source_root_name,
      .source_root_argument_index = row->source_root_argument_index,
      .memory_space = row->memory_space,
      .operation_kind = row->operation_kind,
      .packet_key = row->packet_key,
      .strategy_key = row->strategy_key,
      .model_key = bank_service->model_key,
      .model_revision = bank_service->model_revision,
      .model_evidence = bank_service->model_evidence,
      .request_policy = bank_service->request_policy,
      .wave_size = bank_service->wave_size,
      .bank_count = bank_service->bank_count,
      .bank_word_byte_count = bank_service->bank_word_byte_count,
      .packet_word_count = bank_service->packet_word_count,
  };
}

static iree_status_t
loom_target_compile_report_record_source_low_bank_service_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_bank_service_summary_t* row) {
  loom_target_compile_report_source_low_bank_service_summary_t* summary =
      loom_target_compile_report_find_source_low_bank_service_summary(report,
                                                                      row);
  if (summary != NULL) {
    loom_target_compile_report_merge_unknown_bank_service_reason(
        summary, row->unknown_reason, row->has_mixed_unknown_reasons,
        row->summary.unknown_packet_count);
    loom_target_compile_report_accumulate_bank_service_summaries(
        &summary->summary, &row->summary);
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_bank_service_summaries, sizeof(*row),
      report->allocator, row);
}

static iree_status_t
loom_target_compile_report_record_source_low_bank_service_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  if (iree_string_view_is_empty(row->bank_service.model_key)) {
    return iree_ok_status();
  }
  loom_target_compile_report_source_low_bank_service_summary_t key =
      loom_target_compile_report_source_low_bank_service_summary_from_row(row);
  loom_target_compile_report_source_low_bank_service_summary_t* summary =
      loom_target_compile_report_find_source_low_bank_service_summary(report,
                                                                      &key);
  if (summary == NULL) {
    loom_target_compile_report_merge_unknown_bank_service_reason(
        &key, row->bank_service.unknown_reason,
        /*source_has_mixed_reasons=*/false,
        iree_string_view_equal(row->bank_service.proof, IREE_SV("exact")) ? 0
                                                                          : 1);
    loom_target_compile_report_accumulate_bank_service_summary(&key.summary,
                                                               row);
    return loom_target_compile_report_row_list_append(
        &report->source_low_bank_service_summaries, sizeof(key),
        report->allocator, &key);
  }
  loom_target_compile_report_merge_unknown_bank_service_reason(
      summary, row->bank_service.unknown_reason,
      /*source_has_mixed_reasons=*/false,
      iree_string_view_equal(row->bank_service.proof, IREE_SV("exact")) ? 0
                                                                        : 1);
  loom_target_compile_report_accumulate_bank_service_summary(&summary->summary,
                                                             row);
  return iree_ok_status();
}

static bool loom_target_compile_report_subgroup_accesses_match(
    const loom_target_compile_report_subgroup_access_t* lhs,
    const loom_target_compile_report_subgroup_access_t* rhs) {
  if (!iree_string_view_equal(lhs->proof, rhs->proof) ||
      !iree_string_view_equal(lhs->lane_address_proof,
                              rhs->lane_address_proof) ||
      !iree_string_view_equal(lhs->active_lane_proof, rhs->active_lane_proof) ||
      !iree_string_view_equal(lhs->lane_mapping, rhs->lane_mapping) ||
      !iree_string_view_equal(lhs->interval_coverage, rhs->interval_coverage) ||
      !iree_string_view_equal(lhs->unknown_reason, rhs->unknown_reason) ||
      lhs->subgroup_size != rhs->subgroup_size ||
      lhs->lane_term_count != rhs->lane_term_count ||
      lhs->per_lane_packet_byte_count != rhs->per_lane_packet_byte_count ||
      lhs->linear_lane_byte_stride != rhs->linear_lane_byte_stride ||
      lhs->subgroup_requested_byte_count !=
          rhs->subgroup_requested_byte_count ||
      lhs->subgroup_unique_byte_count != rhs->subgroup_unique_byte_count ||
      lhs->subgroup_span_byte_count != rhs->subgroup_span_byte_count ||
      lhs->maximum_adjacent_lane_delta_bytes !=
          rhs->maximum_adjacent_lane_delta_bytes ||
      lhs->maximum_uncovered_byte_gap_bytes !=
          rhs->maximum_uncovered_byte_gap_bytes ||
      lhs->distinct_lane_address_count != rhs->distinct_lane_address_count ||
      lhs->contiguous_region_count != rhs->contiguous_region_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->lane_term_count; ++i) {
    if (lhs->lane_terms[i].divisor != rhs->lane_terms[i].divisor ||
        lhs->lane_terms[i].modulus != rhs->lane_terms[i].modulus ||
        lhs->lane_terms[i].byte_stride != rhs->lane_terms[i].byte_stride) {
      return false;
    }
  }
  return true;
}

static bool
loom_target_compile_report_source_low_subgroup_access_summaries_match(
    const loom_target_compile_report_source_low_subgroup_access_summary_t* lhs,
    const loom_target_compile_report_source_low_subgroup_access_summary_t*
        rhs) {
  return iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         iree_string_view_equal(lhs->source_op_name, rhs->source_op_name) &&
         lhs->source_op_kind == rhs->source_op_kind &&
         iree_string_view_equal(lhs->source_root_name, rhs->source_root_name) &&
         lhs->source_root_argument_index == rhs->source_root_argument_index &&
         iree_string_view_equal(lhs->memory_space, rhs->memory_space) &&
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind) &&
         iree_string_view_equal(lhs->packet_key, rhs->packet_key) &&
         iree_string_view_equal(lhs->strategy_key, rhs->strategy_key) &&
         loom_target_compile_report_subgroup_accesses_match(&lhs->access,
                                                            &rhs->access);
}

static loom_target_compile_report_source_low_subgroup_access_summary_t*
loom_target_compile_report_find_source_low_subgroup_access_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_subgroup_access_summary_t*
        row) {
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_subgroup_access_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_subgroup_access_summary_t* summaries =
        (loom_target_compile_report_source_low_subgroup_access_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      if (loom_target_compile_report_source_low_subgroup_access_summaries_match(
              &summaries[i], row)) {
        return &summaries[i];
      }
    }
  }
  return NULL;
}

static void loom_target_compile_report_accumulate_subgroup_access_summary(
    loom_target_compile_report_subgroup_access_summary_t* summary,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  const loom_target_compile_report_subgroup_access_t* access =
      &row->subgroup_access;
  if (iree_string_view_is_empty(access->proof)) {
    return;
  }

  ++summary->modeled_packet_count;
  if (!iree_string_view_equal(access->proof, IREE_SV("exact"))) {
    ++summary->unknown_packet_count;
    ++summary->unknown_dynamic_packet_count;
    return;
  }

  ++summary->exact_packet_count;
  const bool is_dense =
      iree_string_view_equal(access->interval_coverage, IREE_SV("dense"));
  const bool is_gapped =
      iree_string_view_equal(access->interval_coverage, IREE_SV("gapped"));
  const bool is_overlapping = access->subgroup_requested_byte_count >
                              access->subgroup_unique_byte_count;
  summary->dense_packet_count += is_dense ? 1 : 0;
  summary->gapped_packet_count += is_gapped ? 1 : 0;
  summary->overlapping_packet_count += is_overlapping ? 1 : 0;

  if (row->execution_count_plus_one ==
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN) {
    ++summary->unknown_dynamic_packet_count;
    return;
  }

  const uint64_t execution_count = row->execution_count_plus_one - 1;
  uint64_t new_dynamic_packet_count = summary->dynamic_packet_count;
  uint64_t new_dynamic_dense_packet_count = summary->dynamic_dense_packet_count;
  uint64_t new_dynamic_gapped_packet_count =
      summary->dynamic_gapped_packet_count;
  uint64_t new_dynamic_overlapping_packet_count =
      summary->dynamic_overlapping_packet_count;
  const bool accumulation_ok =
      iree_checked_add_u64(new_dynamic_packet_count, execution_count,
                           &new_dynamic_packet_count) &&
      (!is_dense ||
       iree_checked_add_u64(new_dynamic_dense_packet_count, execution_count,
                            &new_dynamic_dense_packet_count)) &&
      (!is_gapped ||
       iree_checked_add_u64(new_dynamic_gapped_packet_count, execution_count,
                            &new_dynamic_gapped_packet_count)) &&
      (!is_overlapping ||
       iree_checked_add_u64(new_dynamic_overlapping_packet_count,
                            execution_count,
                            &new_dynamic_overlapping_packet_count));
  if (!accumulation_ok) {
    ++summary->unknown_dynamic_packet_count;
    return;
  }

  ++summary->exact_dynamic_packet_count;
  summary->dynamic_packet_count = new_dynamic_packet_count;
  summary->dynamic_dense_packet_count = new_dynamic_dense_packet_count;
  summary->dynamic_gapped_packet_count = new_dynamic_gapped_packet_count;
  summary->dynamic_overlapping_packet_count =
      new_dynamic_overlapping_packet_count;
}

static loom_target_compile_report_source_low_subgroup_access_summary_t
loom_target_compile_report_source_low_subgroup_access_summary_from_row(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return (loom_target_compile_report_source_low_subgroup_access_summary_t){
      .function_name = row->function_name,
      .source_op_name = row->source_op_name,
      .source_op_kind = row->source_op_kind,
      .source_root_name = row->source_root_name,
      .source_root_argument_index = row->source_root_argument_index,
      .memory_space = row->memory_space,
      .operation_kind = row->operation_kind,
      .packet_key = row->packet_key,
      .strategy_key = row->strategy_key,
      .access = row->subgroup_access,
  };
}

static iree_status_t
loom_target_compile_report_record_source_low_subgroup_access_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_subgroup_access_summary_t*
        row) {
  loom_target_compile_report_source_low_subgroup_access_summary_t* summary =
      loom_target_compile_report_find_source_low_subgroup_access_summary(report,
                                                                         row);
  if (summary != NULL) {
    loom_target_compile_report_accumulate_subgroup_access_summaries(
        &summary->summary, &row->summary);
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_subgroup_access_summaries, sizeof(*row),
      report->allocator, row);
}

static iree_status_t
loom_target_compile_report_record_source_low_subgroup_access_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  if (iree_string_view_is_empty(row->subgroup_access.proof)) {
    return iree_ok_status();
  }
  loom_target_compile_report_source_low_subgroup_access_summary_t key =
      loom_target_compile_report_source_low_subgroup_access_summary_from_row(
          row);
  loom_target_compile_report_source_low_subgroup_access_summary_t* summary =
      loom_target_compile_report_find_source_low_subgroup_access_summary(report,
                                                                         &key);
  if (summary != NULL) {
    loom_target_compile_report_accumulate_subgroup_access_summary(
        &summary->summary, row);
    return iree_ok_status();
  }
  loom_target_compile_report_accumulate_subgroup_access_summary(&key.summary,
                                                                row);
  return loom_target_compile_report_row_list_append(
      &report->source_low_subgroup_access_summaries, sizeof(key),
      report->allocator, &key);
}

iree_status_t loom_target_compile_report_merge_source_low_memory_details(
    loom_target_compile_report_t* target,
    const loom_target_compile_report_t* source) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
      &target->source_low_memory_rows, &source->source_low_memory_rows,
      sizeof(loom_target_compile_report_source_low_memory_row_t),
      target->allocator));
  for (const loom_target_compile_report_vec_t* vec =
           source->source_low_memory_root_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_root_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_root_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_memory_root_summary_row(
              target, &rows[i]));
    }
  }
  for (const loom_target_compile_report_vec_t* vec =
           source->source_low_memory_argument_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_argument_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_argument_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_memory_argument_summary_row(
              target, &rows[i]));
    }
  }
  for (const loom_target_compile_report_vec_t* vec =
           source->source_low_memory_argument_packet_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        rows =
            (const loom_target_compile_report_source_low_memory_argument_packet_summary_t*)
                loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_memory_argument_packet_summary_row(
              target, &rows[i]));
    }
  }
  for (const loom_target_compile_report_vec_t* vec =
           source->source_low_memory_strategy_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_strategy_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_strategy_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_memory_strategy_summary_row(
              target, &rows[i]));
    }
  }
  for (const loom_target_compile_report_vec_t* vec =
           source->source_low_bank_service_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_bank_service_summary_t* rows =
        (const loom_target_compile_report_source_low_bank_service_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_bank_service_summary_row(
              target, &rows[i]));
    }
  }
  for (const loom_target_compile_report_vec_t* vec =
           source->source_low_subgroup_access_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_subgroup_access_summary_t* rows =
        (const loom_target_compile_report_source_low_subgroup_access_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_subgroup_access_summary_row(
              target, &rows[i]));
    }
  }
  return iree_ok_status();
}

static bool loom_target_compile_report_source_low_memory_row_is_load(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return iree_string_view_equal(row->operation_kind, IREE_SV("load"));
}

static bool loom_target_compile_report_source_low_memory_row_is_store(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return iree_string_view_equal(row->operation_kind, IREE_SV("store"));
}

static void loom_target_compile_report_accumulate_source_low_memory_summary(
    loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  const uint64_t lane_count = row->vector_lane_count;
  const uint64_t source_byte_count = lane_count * row->element_byte_count;
  const bool is_load =
      loom_target_compile_report_source_low_memory_row_is_load(row);
  const bool is_store =
      loom_target_compile_report_source_low_memory_row_is_store(row);
  ++summary->packet_count;
  summary->source_lane_count += lane_count;
  summary->source_byte_count += source_byte_count;
  if (is_load) {
    ++summary->load_packet_count;
    summary->read_byte_count += source_byte_count;
    loom_target_compile_report_accumulate_memory_interval_summary(
        &summary->read_interval_envelope, &row->source_interval,
        direction_unique_delta);
  } else if (is_store) {
    ++summary->store_packet_count;
    summary->write_byte_count += source_byte_count;
    loom_target_compile_report_accumulate_memory_interval_summary(
        &summary->write_interval_envelope, &row->source_interval,
        direction_unique_delta);
  }
  summary->issued_read_byte_count += row->issued_read_byte_count;
  summary->issued_write_byte_count += row->issued_write_byte_count;
  summary->issued_read_unknown_width_count +=
      row->issued_read_unknown_width_count;
  summary->issued_write_unknown_width_count +=
      row->issued_write_unknown_width_count;
  loom_target_compile_report_accumulate_memory_interval_summary(
      &summary->interval_envelope, &row->source_interval, unique_delta);
  if (row->execution_count_plus_one ==
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN) {
    ++summary->unknown_dynamic_packet_count;
  } else {
    const uint64_t execution_count = row->execution_count_plus_one - 1;
    uint64_t dynamic_source_byte_count = 0;
    uint64_t dynamic_issued_read_byte_count = 0;
    uint64_t dynamic_issued_write_byte_count = 0;
    uint64_t dynamic_issued_read_unknown_width_count = 0;
    uint64_t dynamic_issued_write_unknown_width_count = 0;
    const bool dynamic_counts_ok =
        iree_checked_mul_u64(source_byte_count, execution_count,
                             &dynamic_source_byte_count) &&
        iree_checked_mul_u64(row->issued_read_byte_count, execution_count,
                             &dynamic_issued_read_byte_count) &&
        iree_checked_mul_u64(row->issued_write_byte_count, execution_count,
                             &dynamic_issued_write_byte_count) &&
        iree_checked_mul_u64(row->issued_read_unknown_width_count,
                             execution_count,
                             &dynamic_issued_read_unknown_width_count) &&
        iree_checked_mul_u64(row->issued_write_unknown_width_count,
                             execution_count,
                             &dynamic_issued_write_unknown_width_count);
    uint64_t new_dynamic_packet_count = summary->dynamic_packet_count;
    uint64_t new_dynamic_source_byte_count = summary->dynamic_source_byte_count;
    uint64_t new_dynamic_read_byte_count = summary->dynamic_read_byte_count;
    uint64_t new_dynamic_write_byte_count = summary->dynamic_write_byte_count;
    uint64_t new_dynamic_issued_read_byte_count =
        summary->dynamic_issued_read_byte_count;
    uint64_t new_dynamic_issued_write_byte_count =
        summary->dynamic_issued_write_byte_count;
    uint64_t new_dynamic_issued_read_unknown_width_count =
        summary->dynamic_issued_read_unknown_width_count;
    uint64_t new_dynamic_issued_write_unknown_width_count =
        summary->dynamic_issued_write_unknown_width_count;
    bool dynamic_accumulation_ok =
        dynamic_counts_ok &&
        iree_checked_add_u64(new_dynamic_packet_count, execution_count,
                             &new_dynamic_packet_count) &&
        iree_checked_add_u64(new_dynamic_source_byte_count,
                             dynamic_source_byte_count,
                             &new_dynamic_source_byte_count) &&
        iree_checked_add_u64(new_dynamic_issued_read_byte_count,
                             dynamic_issued_read_byte_count,
                             &new_dynamic_issued_read_byte_count) &&
        iree_checked_add_u64(new_dynamic_issued_write_byte_count,
                             dynamic_issued_write_byte_count,
                             &new_dynamic_issued_write_byte_count) &&
        iree_checked_add_u64(new_dynamic_issued_read_unknown_width_count,
                             dynamic_issued_read_unknown_width_count,
                             &new_dynamic_issued_read_unknown_width_count) &&
        iree_checked_add_u64(new_dynamic_issued_write_unknown_width_count,
                             dynamic_issued_write_unknown_width_count,
                             &new_dynamic_issued_write_unknown_width_count);
    if (dynamic_accumulation_ok && is_load) {
      dynamic_accumulation_ok = iree_checked_add_u64(
          new_dynamic_read_byte_count, dynamic_source_byte_count,
          &new_dynamic_read_byte_count);
    } else if (dynamic_accumulation_ok && is_store) {
      dynamic_accumulation_ok = iree_checked_add_u64(
          new_dynamic_write_byte_count, dynamic_source_byte_count,
          &new_dynamic_write_byte_count);
    }
    if (!dynamic_accumulation_ok) {
      ++summary->unknown_dynamic_packet_count;
    } else {
      ++summary->exact_dynamic_packet_count;
      summary->dynamic_packet_count = new_dynamic_packet_count;
      summary->dynamic_source_byte_count = new_dynamic_source_byte_count;
      summary->dynamic_read_byte_count = new_dynamic_read_byte_count;
      summary->dynamic_write_byte_count = new_dynamic_write_byte_count;
      summary->dynamic_issued_read_byte_count =
          new_dynamic_issued_read_byte_count;
      summary->dynamic_issued_write_byte_count =
          new_dynamic_issued_write_byte_count;
      summary->dynamic_issued_read_unknown_width_count =
          new_dynamic_issued_read_unknown_width_count;
      summary->dynamic_issued_write_unknown_width_count =
          new_dynamic_issued_write_unknown_width_count;
    }
  }
  if (lane_count == 1) {
    ++summary->scalar_packet_count;
  } else if (lane_count > 1) {
    ++summary->vector_packet_count;
    if (row->element_byte_count == 0 || row->vector_lane_stride_bytes == 0) {
      ++summary->unknown_stride_vector_packet_count;
    } else if (row->vector_lane_stride_bytes == row->element_byte_count) {
      ++summary->contiguous_vector_packet_count;
    } else {
      ++summary->strided_vector_packet_count;
    }
  }
}

static void
loom_target_compile_report_accumulate_source_low_memory_root_summary(
    loom_target_compile_report_source_low_memory_root_summary_t* summary,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &summary->summary, row, unique_delta, direction_unique_delta);
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_root_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  loom_target_compile_report_source_low_memory_root_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_root_summary(
          report, row->function_name, row->source_root_name,
          row->source_root_argument_index, row->memory_space);
  if (summary != NULL) {
    loom_target_compile_report_accumulate_source_low_memory_root_summary(
        summary, row, unique_delta, direction_unique_delta);
    return iree_ok_status();
  } else if (iree_string_view_is_empty(row->source_root_name) &&
             row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }

  loom_target_compile_report_source_low_memory_root_summary_t new_summary = {
      .function_name = row->function_name,
      .source_root_name = row->source_root_name,
      .source_root_argument_index = row->source_root_argument_index,
      .memory_space = row->memory_space,
  };
  loom_target_compile_report_accumulate_source_low_memory_root_summary(
      &new_summary, row, unique_delta, direction_unique_delta);
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_root_summaries, sizeof(new_summary),
      report->allocator, &new_summary);
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_argument_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  if (row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  loom_target_compile_report_source_low_memory_argument_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_argument_summary(
          report, row->function_name, row->source_root_argument_index,
          row->memory_space);
  if (summary == NULL) {
    loom_target_compile_report_source_low_memory_argument_summary_t
        new_summary = {
            .function_name = row->function_name,
            .source_root_name = row->source_root_name,
            .source_root_argument_index = row->source_root_argument_index,
            .memory_space = row->memory_space,
        };
    loom_target_compile_report_accumulate_source_low_memory_summary(
        &new_summary.summary, row, unique_delta, direction_unique_delta);
    return loom_target_compile_report_row_list_append(
        &report->source_low_memory_argument_summaries, sizeof(new_summary),
        report->allocator, &new_summary);
  }
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &summary->summary, row, unique_delta, direction_unique_delta);
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_strategy_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  if (iree_string_view_is_empty(row->strategy_key)) {
    return iree_ok_status();
  }
  loom_target_compile_report_source_low_memory_strategy_summary_t key =
      loom_target_compile_report_source_low_memory_strategy_summary_from_row(
          row);
  loom_target_compile_report_source_low_memory_strategy_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_strategy_summary(report,
                                                                         &key);
  const loom_target_compile_report_memory_interval_unique_delta_t
      no_unique_delta = {0};
  if (summary != NULL) {
    loom_target_compile_report_accumulate_source_low_memory_summary(
        &summary->summary, row, no_unique_delta, no_unique_delta);
    return iree_ok_status();
  }
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &key.summary, row, no_unique_delta, no_unique_delta);
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_strategy_summaries, sizeof(key),
      report->allocator, &key);
}

iree_status_t loom_target_compile_report_record_source_low_memory_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  loom_target_compile_report_memory_interval_unique_delta_t unique_delta;
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_calculate_source_low_unique_interval_delta(
          report, row, /*match_operation_kind=*/false, &unique_delta));
  loom_target_compile_report_memory_interval_unique_delta_t
      direction_unique_delta;
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_calculate_source_low_unique_interval_delta(
          report, row, /*match_operation_kind=*/true, &direction_unique_delta));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
      &report->source_low_memory_rows, sizeof(*row), report->allocator, row));
  const loom_target_compile_report_memory_interval_unique_delta_t
      no_unique_delta = {0};
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &report->source_low_memory_summary, row, no_unique_delta,
      no_unique_delta);
  loom_target_compile_report_accumulate_bank_service_summary(
      &report->bank_service_summary, row);
  loom_target_compile_report_accumulate_subgroup_access_summary(
      &report->subgroup_access_summary, row);
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_root_summary(
          report, row, unique_delta, direction_unique_delta));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_argument_summary(
          report, row, unique_delta, direction_unique_delta));
  loom_target_compile_report_source_low_memory_argument_packet_summary_t
      argument_packet =
          loom_target_compile_report_source_low_memory_argument_packet_summary_from_row(
              row);
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &argument_packet.summary, row, no_unique_delta, no_unique_delta);
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_argument_packet_summary_row(
          report, &argument_packet));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_strategy_summary(
          report, row));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_bank_service_summary(report,
                                                                        row));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_subgroup_access_summary(
          report, row));
  return iree_ok_status();
}
