// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/compile_report_format.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/compile_report_planning_format.h"
#include "loom/util/json.h"

void loom_target_compile_report_format_options_initialize(
    loom_target_compile_report_format_options_t* out_options) {
  *out_options = (loom_target_compile_report_format_options_t){
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
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

static iree_string_view_t loom_target_compile_report_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE:
      return IREE_SV("vm-archive");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE:
      return IREE_SV("hal-executable");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY:
      return IREE_SV("hal-kernel-library");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT:
      return IREE_SV("target-artifact");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_source_low_selection_name(
    loom_target_compile_report_source_low_selection_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE:
      return IREE_SV("rule");
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN:
      return IREE_SV("plan");
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE:
    default:
      return IREE_SV("none");
  }
}

static bool loom_target_compile_report_checked_add_u64(uint64_t lhs,
                                                       uint64_t rhs,
                                                       uint64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(lhs, rhs, out_result);
#else
  if (UINT64_MAX - lhs < rhs) return false;
  *out_result = lhs + rhs;
  return true;
#endif  // IREE_HAVE_BUILTIN(__builtin_add_overflow)
}

static bool loom_target_compile_report_checked_mul_u64(uint64_t lhs,
                                                       uint64_t rhs,
                                                       uint64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(lhs, rhs, out_result);
#else
  if (lhs != 0 && rhs > UINT64_MAX / lhs) return false;
  *out_result = lhs * rhs;
  return true;
#endif  // IREE_HAVE_BUILTIN(__builtin_mul_overflow)
}

typedef struct loom_target_compile_report_string_field_t {
  const char* name;
  iree_string_view_t value;
} loom_target_compile_report_string_field_t;

enum {
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT = 9,
};

static iree_host_size_t
loom_target_compile_report_source_low_memory_storage_fields_from_values(
    iree_string_view_t element_format, iree_string_view_t scale_format,
    iree_string_view_t secondary_scale_format,
    iree_string_view_t payload_packing, iree_string_view_t scale_topology,
    iree_string_view_t affine_policy, iree_string_view_t rounding_policy,
    iree_string_view_t codebook_policy, iree_string_view_t sparsity_policy,
    loom_target_compile_report_string_field_t* fields) {
  fields[0] = (loom_target_compile_report_string_field_t){
      .name = "element_format",
      .value = element_format,
  };
  fields[1] = (loom_target_compile_report_string_field_t){
      .name = "scale_format",
      .value = scale_format,
  };
  fields[2] = (loom_target_compile_report_string_field_t){
      .name = "secondary_scale_format",
      .value = secondary_scale_format,
  };
  fields[3] = (loom_target_compile_report_string_field_t){
      .name = "payload_packing",
      .value = payload_packing,
  };
  fields[4] = (loom_target_compile_report_string_field_t){
      .name = "scale_topology",
      .value = scale_topology,
  };
  fields[5] = (loom_target_compile_report_string_field_t){
      .name = "affine_policy",
      .value = affine_policy,
  };
  fields[6] = (loom_target_compile_report_string_field_t){
      .name = "rounding_policy",
      .value = rounding_policy,
  };
  fields[7] = (loom_target_compile_report_string_field_t){
      .name = "codebook_policy",
      .value = codebook_policy,
  };
  fields[8] = (loom_target_compile_report_string_field_t){
      .name = "sparsity_policy",
      .value = sparsity_policy,
  };
  return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT;
}

static iree_host_size_t
loom_target_compile_report_source_low_memory_storage_fields(
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_string_field_t* fields) {
  return loom_target_compile_report_source_low_memory_storage_fields_from_values(
      row->storage_element_format, row->storage_scale_format,
      row->storage_secondary_scale_format, row->storage_payload_packing,
      row->storage_scale_topology, row->storage_affine_policy,
      row->storage_rounding_policy, row->storage_codebook_policy,
      row->storage_sparsity_policy, fields);
}

static iree_host_size_t
loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    loom_target_compile_report_string_field_t* fields) {
  return loom_target_compile_report_source_low_memory_storage_fields_from_values(
      summary->storage_element_format, summary->storage_scale_format,
      summary->storage_secondary_scale_format, summary->storage_payload_packing,
      summary->storage_scale_topology, summary->storage_affine_policy,
      summary->storage_rounding_policy, summary->storage_codebook_policy,
      summary->storage_sparsity_policy, fields);
}

static iree_host_size_t
loom_target_compile_report_source_low_memory_strategy_storage_fields(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    loom_target_compile_report_string_field_t* fields) {
  return loom_target_compile_report_source_low_memory_storage_fields_from_values(
      summary->storage_element_format, summary->storage_scale_format,
      summary->storage_secondary_scale_format, summary->storage_payload_packing,
      summary->storage_scale_topology, summary->storage_affine_policy,
      summary->storage_rounding_policy, summary->storage_codebook_policy,
      summary->storage_sparsity_policy, fields);
}

static iree_host_size_t loom_target_compile_report_first_non_empty_string_field(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count) {
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (!iree_string_view_is_empty(fields[i].value)) {
      return i;
    }
  }
  return field_count;
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_storage_fields_text(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count, iree_string_builder_t* builder) {
  const iree_host_size_t first_storage_field =
      loom_target_compile_report_first_non_empty_string_field(fields,
                                                              field_count);
  if (first_storage_field == field_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, " storage={"));
  bool first_field = true;
  for (iree_host_size_t i = first_storage_field; i < field_count; ++i) {
    if (iree_string_view_is_empty(fields[i].value)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%s:%.*s", first_field ? "" : ",", fields[i].name,
        (int)fields[i].value.size, fields[i].value.data));
    first_field = false;
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_storage_text(
    const loom_target_compile_report_source_low_memory_row_t* row,
    iree_string_builder_t* builder) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_storage_fields(row, fields);
  return loom_target_compile_report_append_source_low_memory_storage_fields_text(
      fields, field_count, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_strategy_storage_text(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    iree_string_builder_t* builder) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_strategy_storage_fields(
          summary, fields);
  return loom_target_compile_report_append_source_low_memory_storage_fields_text(
      fields, field_count, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_argument_packet_storage_text(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    iree_string_builder_t* builder) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
          summary, fields);
  return loom_target_compile_report_append_source_low_memory_storage_fields_text(
      fields, field_count, builder);
}

static bool loom_target_compile_report_memory_interval_has_range(
    const loom_target_compile_report_memory_interval_t* interval) {
  const loom_target_compile_report_memory_interval_flags_t range_flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE;
  return iree_all_bits_set(interval->flags, range_flags);
}

static iree_status_t loom_target_compile_report_append_memory_interval_text(
    const loom_target_compile_report_memory_interval_t* interval,
    iree_string_builder_t* builder) {
  if (!loom_target_compile_report_memory_interval_has_range(interval)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " source_interval={begin_min_bytes:%" PRId64 ",begin_max_bytes:%" PRId64
      ",end_min_bytes:%" PRId64 ",end_max_bytes:%" PRId64,
      interval->begin_min_bytes, interval->begin_max_bytes,
      interval->end_min_bytes, interval->end_max_bytes));
  if (iree_all_bits_set(
          interval->flags,
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",exact_length_bytes:%" PRIu64, interval->exact_length_bytes));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t
loom_target_compile_report_append_memory_interval_summary_text(
    const char* field_name,
    const loom_target_compile_report_memory_interval_summary_t* summary,
    iree_string_builder_t* builder) {
  if (summary->packet_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " %s={packets:%" PRIu64 ",begin_min_bytes:%" PRId64
      ",end_max_bytes:%" PRId64 ",byte_count:%" PRIu64,
      field_name, summary->packet_count, summary->envelope_begin_min_bytes,
      summary->envelope_end_max_bytes, summary->envelope_byte_count));
  if (summary->exact_static_packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",exact_static_packet_count:%" PRIu64,
        summary->exact_static_packet_count));
  }
  if (summary->exact_symbolic_packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",exact_symbolic_packet_count:%" PRIu64,
        summary->exact_symbolic_packet_count));
  }
  if (summary->exact_static_packet_count +
          summary->exact_symbolic_packet_count ==
      summary->packet_count) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",unique_byte_count:%" PRIu64, summary->unique_byte_count));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static bool loom_target_compile_report_source_low_memory_has_dynamic_evidence(
    const loom_target_compile_report_source_low_memory_summary_t* summary) {
  return summary->exact_dynamic_packet_count != 0 ||
         summary->unknown_dynamic_packet_count != 0;
}

static bool
loom_target_compile_report_source_low_memory_has_complete_dynamic_evidence(
    const loom_target_compile_report_source_low_memory_summary_t* summary) {
  return summary->packet_count == summary->exact_dynamic_packet_count &&
         summary->unknown_dynamic_packet_count == 0;
}

static bool loom_target_compile_report_source_low_memory_has_dynamic_delta(
    const loom_target_compile_report_source_low_memory_summary_t* summary) {
  return summary->unknown_dynamic_packet_count != 0 ||
         summary->dynamic_packet_count != summary->packet_count ||
         summary->dynamic_source_byte_count != summary->source_byte_count ||
         summary->dynamic_read_byte_count != summary->read_byte_count ||
         summary->dynamic_write_byte_count != summary->write_byte_count ||
         summary->dynamic_issued_read_byte_count !=
             summary->issued_read_byte_count ||
         summary->dynamic_issued_write_byte_count !=
             summary->issued_write_byte_count ||
         summary->dynamic_issued_read_unknown_width_count !=
             summary->issued_read_unknown_width_count ||
         summary->dynamic_issued_write_unknown_width_count !=
             summary->issued_write_unknown_width_count;
}

static bool loom_target_compile_report_source_low_memory_can_dispatch_scale(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload) {
  return iree_any_bit_set(
             workload->flags,
             LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT) &&
         (!loom_target_compile_report_source_low_memory_has_dynamic_evidence(
              summary) ||
          loom_target_compile_report_source_low_memory_has_complete_dynamic_evidence(
              summary));
}

static bool loom_target_compile_report_source_low_memory_should_print_dynamic(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload) {
  if (!loom_target_compile_report_source_low_memory_has_dynamic_evidence(
          summary)) {
    return false;
  }
  return loom_target_compile_report_source_low_memory_has_dynamic_delta(
             summary) ||
         loom_target_compile_report_source_low_memory_can_dispatch_scale(
             summary, workload);
}

typedef struct loom_target_compile_report_dispatch_memory_bytes_t {
  // Dispatch-scaled logical or issued read bytes.
  uint64_t read_byte_count;
  // Dispatch-scaled logical or issued write bytes.
  uint64_t write_byte_count;
  // Sum of dispatch-scaled read and write bytes.
  uint64_t total_byte_count;
} loom_target_compile_report_dispatch_memory_bytes_t;

static bool loom_target_compile_report_dispatch_memory_bytes(
    uint64_t read_byte_count, uint64_t write_byte_count,
    uint64_t dispatch_workitem_count,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes) {
  *out_bytes = (loom_target_compile_report_dispatch_memory_bytes_t){0};
  return loom_target_compile_report_checked_mul_u64(
             read_byte_count, dispatch_workitem_count,
             &out_bytes->read_byte_count) &&
         loom_target_compile_report_checked_mul_u64(
             write_byte_count, dispatch_workitem_count,
             &out_bytes->write_byte_count) &&
         loom_target_compile_report_checked_add_u64(
             out_bytes->read_byte_count, out_bytes->write_byte_count,
             &out_bytes->total_byte_count);
}

static bool loom_target_compile_report_source_low_memory_dispatch_source_bytes(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes) {
  if (!loom_target_compile_report_source_low_memory_can_dispatch_scale(
          summary, workload)) {
    return false;
  }
  const bool use_dynamic_counts =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(
          summary);
  const uint64_t read_byte_count = use_dynamic_counts
                                       ? summary->dynamic_read_byte_count
                                       : summary->read_byte_count;
  const uint64_t write_byte_count = use_dynamic_counts
                                        ? summary->dynamic_write_byte_count
                                        : summary->write_byte_count;
  return loom_target_compile_report_dispatch_memory_bytes(
      read_byte_count, write_byte_count, workload->dispatch_workitem_count,
      out_bytes);
}

static bool loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes) {
  if (!loom_target_compile_report_source_low_memory_can_dispatch_scale(
          summary, workload)) {
    return false;
  }
  const bool use_dynamic_counts =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(
          summary);
  const uint64_t read_byte_count = use_dynamic_counts
                                       ? summary->dynamic_issued_read_byte_count
                                       : summary->issued_read_byte_count;
  const uint64_t write_byte_count =
      use_dynamic_counts ? summary->dynamic_issued_write_byte_count
                         : summary->issued_write_byte_count;
  return loom_target_compile_report_dispatch_memory_bytes(
      read_byte_count, write_byte_count, workload->dispatch_workitem_count,
      out_bytes);
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_summary_text(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " packets=%" PRIu64 " loads=%" PRIu64 " stores=%" PRIu64
      " scalar_packets=%" PRIu64 " vector_packets=%" PRIu64
      " source_lanes=%" PRIu64 " source_bytes=%" PRIu64 " read_bytes=%" PRIu64
      " write_bytes=%" PRIu64 " issued_read_bytes=%" PRIu64
      " issued_write_bytes=%" PRIu64 " issued_read_unknown_widths=%" PRIu64
      " issued_write_unknown_widths=%" PRIu64
      " contiguous_vector_packets=%" PRIu64 " strided_vector_packets=%" PRIu64
      " unknown_stride_vector_packets=%" PRIu64,
      summary->packet_count, summary->load_packet_count,
      summary->store_packet_count, summary->scalar_packet_count,
      summary->vector_packet_count, summary->source_lane_count,
      summary->source_byte_count, summary->read_byte_count,
      summary->write_byte_count, summary->issued_read_byte_count,
      summary->issued_write_byte_count,
      summary->issued_read_unknown_width_count,
      summary->issued_write_unknown_width_count,
      summary->contiguous_vector_packet_count,
      summary->strided_vector_packet_count,
      summary->unknown_stride_vector_packet_count));
  if (loom_target_compile_report_source_low_memory_should_print_dynamic(
          summary, workload)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " exact_dynamic_packets=%" PRIu64 " unknown_dynamic_packets=%" PRIu64
        " dynamic_packets=%" PRIu64 " dynamic_source_bytes=%" PRIu64
        " dynamic_read_bytes=%" PRIu64 " dynamic_write_bytes=%" PRIu64
        " dynamic_issued_read_bytes=%" PRIu64
        " dynamic_issued_write_bytes=%" PRIu64
        " dynamic_issued_read_unknown_widths=%" PRIu64
        " dynamic_issued_write_unknown_widths=%" PRIu64,
        summary->exact_dynamic_packet_count,
        summary->unknown_dynamic_packet_count, summary->dynamic_packet_count,
        summary->dynamic_source_byte_count, summary->dynamic_read_byte_count,
        summary->dynamic_write_byte_count,
        summary->dynamic_issued_read_byte_count,
        summary->dynamic_issued_write_byte_count,
        summary->dynamic_issued_read_unknown_width_count,
        summary->dynamic_issued_write_unknown_width_count));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    loom_target_compile_report_dispatch_memory_bytes_t dispatch_source = {0};
    if (loom_target_compile_report_source_low_memory_dispatch_source_bytes(
            summary, workload, &dispatch_source)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " dispatch_source_read_bytes=%" PRIu64
          " dispatch_source_write_bytes=%" PRIu64
          " dispatch_source_total_bytes=%" PRIu64,
          dispatch_source.read_byte_count, dispatch_source.write_byte_count,
          dispatch_source.total_byte_count));
    }
    loom_target_compile_report_dispatch_memory_bytes_t dispatch_issued = {0};
    if (loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
            summary, workload, &dispatch_issued)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " dispatch_issued_read_bytes=%" PRIu64
          " dispatch_issued_write_bytes=%" PRIu64
          " dispatch_issued_total_bytes=%" PRIu64,
          dispatch_issued.read_byte_count, dispatch_issued.write_byte_count,
          dispatch_issued.total_byte_count));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_memory_interval_summary_text(
          "interval_envelope", &summary->interval_envelope, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_memory_interval_summary_text(
          "read_interval_envelope", &summary->read_interval_envelope, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_memory_interval_summary_text(
          "write_interval_envelope", &summary->write_interval_envelope,
          builder));
  return iree_ok_status();
}

static iree_string_view_t
loom_target_compile_report_allocation_failure_blocking_kind_name(
    loom_target_compile_report_allocation_failure_blocking_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET:
      return IREE_SV("interval-exceeds-budget");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT:
      return IREE_SV("active-assignment");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT:
      return IREE_SV("location-constraint");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION:
      return IREE_SV("no-assignable-location");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_pressure_origin_kind_name(
    loom_target_compile_report_pressure_origin_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT:
      return IREE_SV("block-argument");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONSTANT:
      return IREE_SV("constant");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_COPY:
      return IREE_SV("copy");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SLICE:
      return IREE_SV("slice");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONCAT:
      return IREE_SV("concat");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_STORAGE:
      return IREE_SV("storage");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SPILL_RELOAD:
      return IREE_SV("spill-reload");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_ALU:
      return IREE_SV("scalar-alu");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_VECTOR_ALU:
      return IREE_SV("vector-alu");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX:
      return IREE_SV("matrix");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT:
      return IREE_SV("dot");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY:
      return IREE_SV("global-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY:
      return IREE_SV("local-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_MEMORY:
      return IREE_SV("scalar-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY:
      return IREE_SV("private-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GENERIC_MEMORY:
      return IREE_SV("generic-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL:
      return IREE_SV("control");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BARRIER:
      return IREE_SV("barrier");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONVERSION:
      return IREE_SV("conversion");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE:
      return IREE_SV("register-move");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CACHE:
      return IREE_SV("cache");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION:
      return IREE_SV("operation");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_spill_row_kind_name(
    loom_target_compile_report_spill_row_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED:
      return IREE_SV("planned");
    case LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_MATERIALIZED:
      return IREE_SV("materialized");
    case LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_legalization_mode_name(
    loom_target_compile_report_legalization_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_EAGER:
      return IREE_SV("eager");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL:
      return IREE_SV("final");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_legalization_policy_name(
    loom_target_compile_report_legalization_policy_t policy) {
  switch (policy) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_PREFER_NATIVE:
      return IREE_SV("prefer-native");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY:
      return IREE_SV("reference-only");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REQUIRE_NATIVE:
      return IREE_SV("require-native");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_legalization_action_name(
    loom_target_compile_report_legalization_action_t action) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL:
      return IREE_SV("legal");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN:
      return IREE_SV("rewritten");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED:
      return IREE_SV("deferred");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR:
      return IREE_SV("reject-invalid-ir");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL:
      return IREE_SV("reject-unsupported-final");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED:
      return IREE_SV("unhandled");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_legalization_outcome_name(
    loom_target_compile_report_legalization_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_ALREADY_LEGAL:
      return IREE_SV("already-legal");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_TARGET_REWRITE:
      return IREE_SV("target-rewrite");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK:
      return IREE_SV("reference-fallback");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_DEFERRED:
      return IREE_SV("deferred");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_INVALID_IR:
      return IREE_SV("reject-invalid-ir");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_UNSUPPORTED:
      return IREE_SV("reject-unsupported");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_UNHANDLED:
      return IREE_SV("unhandled");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_contract_outcome_name(
    loom_target_compile_report_contract_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNHANDLED:
      return IREE_SV("unhandled");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_LEGAL:
      return IREE_SV("legal");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED:
      return IREE_SV("unsupported");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_INVALID_IR:
      return IREE_SV("invalid-ir");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_capability_value_kind_name(
    loom_target_compile_report_capability_value_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL:
      return IREE_SV("bool");
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64:
      return IREE_SV("u64");
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING:
      return IREE_SV("string");
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_legalizer_strategy_name(
    loom_target_compile_report_legalizer_strategy_t strategy) {
  switch (strategy) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET:
      return IREE_SV("target");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE:
      return IREE_SV("reference");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_target_compile_report_math_action_name(
    loom_target_compile_report_math_action_t action) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN:
      return IREE_SV("rewritten");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REJECTED:
      return IREE_SV("rejected");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_POLICY:
      return IREE_SV("missing-policy");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_RECIPE:
      return IREE_SV("missing-recipe");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_NONE:
    default:
      return IREE_SV("none");
  }
}

typedef struct loom_target_compile_report_move_cause_descriptor_t {
  // Residual move cause used as the report counter index.
  loom_target_compile_report_move_cause_t cause;
  // Stable text name emitted in compile reports.
  iree_string_view_t name;
} loom_target_compile_report_move_cause_descriptor_t;

static const loom_target_compile_report_move_cause_descriptor_t
    loom_target_compile_report_move_cause_descriptors[] = {
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
         IREE_SVL("constant_materialization")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, IREE_SVL("low_copy")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
         IREE_SVL("low_slice")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
         IREE_SVL("low_concat")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE,
         IREE_SVL("branch_edge")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION,
         IREE_SVL("operand_bank_materialization")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_CONSTRAINT_REPAIR,
         IREE_SVL("operand_constraint_repair")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_ABI_COPY, IREE_SVL("abi_copy")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_SPILL_RELOAD,
         IREE_SVL("spill_reload")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_PARTIAL_REGISTER_REPAIR,
         IREE_SVL("partial_register_repair")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_UNKNOWN, IREE_SVL("unknown")},
};
static_assert(
    IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors) ==
        LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT - 1,
    "move cause descriptor table must cover each reportable move cause");

static iree_string_view_t loom_target_compile_report_type_kind_name(
    uint32_t type_kind) {
  switch (type_kind) {
    case LOOM_TYPE_NONE:
      return IREE_SV("none");
    case LOOM_TYPE_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TYPE_TENSOR:
      return IREE_SV("tensor");
    case LOOM_TYPE_TILE:
      return IREE_SV("tile");
    case LOOM_TYPE_VECTOR:
      return IREE_SV("vector");
    case LOOM_TYPE_VIEW:
      return IREE_SV("view");
    case LOOM_TYPE_BUFFER:
      return IREE_SV("buffer");
    case LOOM_TYPE_FUNCTION:
      return IREE_SV("function");
    case LOOM_TYPE_ENCODING:
      return IREE_SV("encoding");
    case LOOM_TYPE_DIALECT:
      return IREE_SV("dialect");
    case LOOM_TYPE_GROUP:
      return IREE_SV("group");
    case LOOM_TYPE_POOL:
      return IREE_SV("pool");
    case LOOM_TYPE_REGISTER:
      return IREE_SV("register");
    case LOOM_TYPE_STORAGE:
      return IREE_SV("storage");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_scalar_type_name(
    uint32_t element_type) {
  const char* name = loom_scalar_type_name((loom_scalar_type_t)element_type);
  if (name == NULL) {
    return IREE_SV("unknown");
  }
  return iree_make_cstring_view(name);
}

static iree_string_view_t loom_target_compile_report_non_empty(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static iree_status_t loom_target_compile_report_append_optional_u32(
    iree_string_builder_t* builder, uint32_t value) {
  if (value == UINT32_MAX) {
    return iree_string_builder_append_string(builder, IREE_SV("-"));
  }
  return iree_string_builder_append_format(builder, "%u", value);
}

static void loom_target_compile_report_accumulate_move_cause(
    const loom_target_compile_report_move_cause_counts_t* counts,
    uint64_t* kind_count, uint64_t* packet_count, uint64_t* unit_count) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return;
  }
  ++*kind_count;
  *packet_count += counts->packet_count;
  *unit_count += counts->unit_count;
}

static void loom_target_compile_report_move_cause_counts_totals(
    const loom_target_compile_report_move_cause_counts_t* counts,
    uint64_t* out_kind_count, uint64_t* out_packet_count,
    uint64_t* out_unit_count) {
  *out_kind_count = 0;
  *out_packet_count = 0;
  *out_unit_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    loom_target_compile_report_accumulate_move_cause(
        &counts[descriptor->cause], out_kind_count, out_packet_count,
        out_unit_count);
  }
}

static void loom_target_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  loom_target_compile_report_move_cause_counts_totals(
      report->move_causes, out_kind_count, out_packet_count, out_unit_count);
}

static iree_status_t loom_target_compile_report_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value) {
  value = loom_target_compile_report_non_empty(value);
  return iree_string_builder_append_format(builder, " %.*s=%.*s",
                                           (int)name.size, name.data,
                                           (int)value.size, value.data);
}

static bool loom_target_compile_report_economics_has_memory(
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  return mix->memory_read_byte_count != 0 ||
         mix->memory_write_byte_count != 0 ||
         mix->memory_read_unknown_width_count != 0 ||
         mix->memory_write_unknown_width_count != 0;
}

static bool loom_target_compile_report_economics_has_operations(
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  return mix->scalar_alu_count != 0 || mix->vector_alu_count != 0 ||
         mix->matrix_count != 0 || mix->mfma_count != 0 ||
         mix->smfmac_count != 0 || mix->wmma_count != 0 ||
         mix->swmmac_count != 0 || mix->dot_count != 0 ||
         mix->atomic_count != 0 || mix->branch_count != 0 ||
         mix->barrier_count != 0 || mix->control_count != 0 ||
         mix->conversion_count != 0 || mix->cache_count != 0 ||
         mix->register_move_count != 0;
}

static bool loom_target_compile_report_has_economics(
    loom_target_compile_report_detail_flags_t detail_flags,
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload) {
  (void)workload;
  return iree_any_bit_set(
             detail_flags,
             LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX) &&
         (loom_target_compile_report_economics_has_memory(mix) ||
          loom_target_compile_report_economics_has_operations(mix));
}

static iree_status_t loom_target_compile_report_append_workload_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_workload_t* workload) {
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " workgroup_size=%" PRIu32 "x%" PRIu32 "x%" PRIu32,
        workload->workgroup_size.x, workload->workgroup_size.y,
        workload->workgroup_size.z));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " flat_workgroup_size=%" PRIu64,
        workload->flat_workgroup_size));
  }
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " workgroup_count=%" PRIu32 "x%" PRIu32 "x%" PRIu32,
        workload->workgroup_count.x, workload->workgroup_count.y,
        workload->workgroup_count.z));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " dispatch_workgroup_count=%" PRIu64,
        workload->dispatch_workgroup_count));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " dispatch_workitem_count=%" PRIu64,
        workload->dispatch_workitem_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_append_instruction_mix_fields(
    iree_string_builder_t* builder, iree_string_view_t name,
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  return iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: %.*s descriptors=%" PRIu64 " unknown=%" PRIu64
      " scalar_alu=%" PRIu64 " vector_alu=%" PRIu64 " matrix=%" PRIu64
      " mfma=%" PRIu64 " smfmac=%" PRIu64 " wmma=%" PRIu64 " swmmac=%" PRIu64
      " dot=%" PRIu64 " global_memory=%" PRIu64 " global_load=%" PRIu64
      " global_store=%" PRIu64 " buffer_load=%" PRIu64 " buffer_store=%" PRIu64
      " flat_memory=%" PRIu64 " local_memory=%" PRIu64 " scalar_memory=%" PRIu64
      " private_memory=%" PRIu64 " generic_memory=%" PRIu64
      " memory_read_unknown_width=%" PRIu64
      " memory_write_unknown_width=%" PRIu64 " memory_read_bytes=%" PRIu64
      " memory_write_bytes=%" PRIu64 " global_load_bytes=%" PRIu64
      " global_store_bytes=%" PRIu64 " buffer_load_bytes=%" PRIu64
      " buffer_store_bytes=%" PRIu64 " flat_read_bytes=%" PRIu64
      " flat_write_bytes=%" PRIu64 " local_read_bytes=%" PRIu64
      " local_write_bytes=%" PRIu64 " scalar_read_bytes=%" PRIu64
      " scalar_write_bytes=%" PRIu64 " private_read_bytes=%" PRIu64
      " private_write_bytes=%" PRIu64 " unclassified_read_bytes=%" PRIu64
      " unclassified_write_bytes=%" PRIu64 " atomic=%" PRIu64 " branch=%" PRIu64
      " barrier=%" PRIu64 " control=%" PRIu64 " conversion=%" PRIu64
      " cache=%" PRIu64 " register_move=%" PRIu64 "\n",
      (int)name.size, name.data, mix->descriptor_count, mix->unknown_count,
      mix->scalar_alu_count, mix->vector_alu_count, mix->matrix_count,
      mix->mfma_count, mix->smfmac_count, mix->wmma_count, mix->swmmac_count,
      mix->dot_count, mix->global_memory_count, mix->global_load_count,
      mix->global_store_count, mix->buffer_load_count, mix->buffer_store_count,
      mix->flat_memory_count, mix->local_memory_count, mix->scalar_memory_count,
      mix->private_memory_count, mix->generic_memory_count,
      mix->memory_read_unknown_width_count,
      mix->memory_write_unknown_width_count, mix->memory_read_byte_count,
      mix->memory_write_byte_count, mix->global_load_byte_count,
      mix->global_store_byte_count, mix->buffer_load_byte_count,
      mix->buffer_store_byte_count, mix->flat_read_byte_count,
      mix->flat_write_byte_count, mix->local_read_byte_count,
      mix->local_write_byte_count, mix->scalar_read_byte_count,
      mix->scalar_write_byte_count, mix->private_read_byte_count,
      mix->private_write_byte_count, mix->unclassified_read_byte_count,
      mix->unclassified_write_byte_count, mix->atomic_count, mix->branch_count,
      mix->barrier_count, mix->control_count, mix->conversion_count,
      mix->cache_count, mix->register_move_count);
}

static iree_status_t loom_target_compile_report_append_economics_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload) {
  uint64_t per_workitem_total = 0;
  if (loom_target_compile_report_checked_add_u64(mix->memory_read_byte_count,
                                                 mix->memory_write_byte_count,
                                                 &per_workitem_total)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " per_workitem_issued_read_bytes=%" PRIu64
        " per_workitem_issued_write_bytes=%" PRIu64
        " per_workitem_issued_total_bytes=%" PRIu64,
        mix->memory_read_byte_count, mix->memory_write_byte_count,
        per_workitem_total));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    uint64_t dispatch_read_bytes = 0;
    uint64_t dispatch_write_bytes = 0;
    uint64_t dispatch_total_bytes = 0;
    if (loom_target_compile_report_checked_mul_u64(
            mix->memory_read_byte_count, workload->dispatch_workitem_count,
            &dispatch_read_bytes) &&
        loom_target_compile_report_checked_mul_u64(
            mix->memory_write_byte_count, workload->dispatch_workitem_count,
            &dispatch_write_bytes) &&
        loom_target_compile_report_checked_add_u64(
            dispatch_read_bytes, dispatch_write_bytes, &dispatch_total_bytes)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " dispatch_read_bytes=%" PRIu64 " dispatch_write_bytes=%" PRIu64
          " dispatch_total_bytes=%" PRIu64,
          dispatch_read_bytes, dispatch_write_bytes, dispatch_total_bytes));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_append_target_resources_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_target_resources_t* resources) {
  const iree_string_view_t scalar_register_class =
      loom_target_compile_report_non_empty(resources->scalar_register_class);
  const iree_string_view_t vector_register_class =
      loom_target_compile_report_non_empty(resources->vector_register_class);
  const iree_string_view_t limiting_resource =
      loom_target_compile_report_non_empty(resources->limiting_resource);
  return iree_string_builder_append_format(
      builder,
      "scalar_register_class=%.*s scalar_final_registers=%" PRIu64
      " scalar_scheduled_pressure_peak=%" PRIu64
      " scalar_final_overhead=%" PRIu64
      " vector_register_class=%.*s vector_final_registers=%" PRIu64
      " vector_scheduled_pressure_peak=%" PRIu64
      " vector_final_overhead=%" PRIu64 " subgroup_size=%" PRIu32
      " resident_subgroups_per_simd=%" PRIu32 " max_subgroups_per_simd=%" PRIu32
      " occupancy_percent=%" PRIu32 " limiting=%.*s",
      (int)scalar_register_class.size, scalar_register_class.data,
      resources->scalar_register_count,
      resources->scalar_pressure_peak_live_units,
      resources->scalar_register_overhead_units,
      (int)vector_register_class.size, vector_register_class.data,
      resources->vector_register_count,
      resources->vector_pressure_peak_live_units,
      resources->vector_register_overhead_units, resources->subgroup_size,
      resources->resident_subgroups_per_simd, resources->max_subgroups_per_simd,
      resources->occupancy_percent, (int)limiting_resource.size,
      limiting_resource.data);
}

static iree_status_t loom_target_compile_report_format_summary(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("COMPILE-REPORT: summary")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("artifact"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " status=%s", iree_status_code_string(report->status_code)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("function"), report->function_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("export"), report->target_export_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("export_symbol"), report->target_export_symbol));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("config"), report->target_config_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("lowered"), report->lowered_symbol));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " artifact_bytes=%" PRIu64, report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("\n")));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule nodes=%" PRIu64 " scheduled=%" PRIu64
        " deps=%" PRIu64 " resources=%" PRIu64 " hazards=%" PRIu64
        " models=%" PRIu64 " pressure_classes=%" PRIu64
        " peak_live_units=%" PRIu64 "\n",
        report->schedule_node_count, report->scheduled_node_count,
        report->schedule_dependency_count, report->schedule_resource_use_count,
        report->schedule_hazard_gap_count, report->schedule_model_summary_count,
        report->register_pressure_summary_count,
        report->register_pressure_peak_live_units));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: planning")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_low_planning_text_fields(
            &report->low_planning, builder));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: workload")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_workload_fields(
        builder, &report->workload));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_instruction_mix_fields(
            builder, IREE_SV("static_instruction_mix"),
            &report->static_instruction_mix));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_instruction_mix_fields(
            builder, IREE_SV("dynamic_instruction_mix"),
            &report->dynamic_instruction_mix));
  }

  if (loom_target_compile_report_has_economics(report->detail_flags,
                                               &report->dynamic_instruction_mix,
                                               &report->workload)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: economics memory")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_economics_fields(
        builder, &report->dynamic_instruction_mix, &report->workload));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation assignments=%" PRIu64 " spills=%" PRIu64
        " spill_plans=%" PRIu64 " coalesced_copies=%" PRIu64
        " materialized_copies=%" PRIu64 " materialized_spill_storage=%" PRIu64
        " materialized_spill_storage_bytes=%" PRIu64
        " materialized_spill_stores=%" PRIu64
        " materialized_spill_store_bytes=%" PRIu64
        " materialized_reloads=%" PRIu64 " materialized_reload_bytes=%" PRIu64
        " storage_leases=%" PRIu64 " storage_lease_instances=%" PRIu64
        " storage_release_actions=%" PRIu64 "\n",
        report->allocation_assignment_count, report->allocation_spill_count,
        report->allocation_spill_plan_count,
        report->allocation_coalesced_copy_count,
        report->allocation_materialized_copy_count,
        report->allocation_materialized_spill_storage_count,
        report->allocation_materialized_spill_storage_bytes,
        report->allocation_materialized_spill_store_count,
        report->allocation_materialized_spill_store_bytes,
        report->allocation_materialized_reload_count,
        report->allocation_materialized_reload_bytes,
        report->allocation_storage_lease_count,
        report->allocation_storage_lease_instance_count,
        report->allocation_storage_release_action_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    const loom_target_compile_report_target_resources_t* resources =
        &report->target_resources;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: target_resources ")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_target_resources_fields(builder,
                                                                  resources));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    uint64_t kind_count = 0;
    uint64_t packet_count = 0;
    uint64_t unit_count = 0;
    loom_target_compile_report_move_cause_totals(report, &kind_count,
                                                 &packet_count, &unit_count);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: move_causes kinds=%" PRIu64 " packets=%" PRIu64
        " units=%" PRIu64 "\n",
        kind_count, packet_count, unit_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    const loom_target_compile_report_wait_plan_t* wait_plan =
        &report->wait_plan;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: wait_plan actions=%" PRIu64 " explicit=%" PRIu64
        " planned=%" PRIu64 " full_drains=%" PRIu64 " partial_waits=%" PRIu64
        " drained=%" PRIu64 " max_drained=%" PRIu64 " max_outstanding=%" PRIu64
        " max_full_drain_outstanding=%" PRIu64 " counter_rows=%" PRIhsz
        " reason_summary_rows=%" PRIhsz " action_rows=%" PRIhsz "\n",
        wait_plan->action_count, wait_plan->explicit_action_count,
        wait_plan->planned_action_count, wait_plan->full_drain_count,
        wait_plan->partial_wait_count, wait_plan->drained_count,
        wait_plan->max_drained_count, wait_plan->max_outstanding_before,
        wait_plan->max_full_drain_outstanding_before,
        report->wait_counter_rows.count, report->wait_reason_summary_rows.count,
        report->wait_action_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: target_capability_rows count=%" PRIhsz "\n",
        report->target_capability_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: emission instructions=%" PRIu64 " code_bytes=%" PRIu64
        " storage_bytes=%" PRIu64 "\n",
        report->emitted_instruction_count, report->emitted_code_byte_count,
        report->emitted_code_storage_byte_count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: math_legalization rewritten=%" PRIu64
        " rejected=%" PRIu64 " missing_policy=%" PRIu64
        " missing_recipe=%" PRIu64 " rows=%" PRIhsz "\n",
        report->math_legalization_rewritten_op_count,
        report->math_legalization_rejected_op_count,
        report->math_legalization_missing_policy_op_count,
        report->math_legalization_missing_recipe_op_count,
        report->math_legalization_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: source_low selected_ops=%" PRIu64
        " emitted_ops=%" PRIu64 " rows=%" PRIhsz "\n",
        report->source_low_selected_op_count,
        report->source_low_emitted_op_count, report->source_low_rows.count));
    if (report->source_low_memory_summary.packet_count != 0) {
      const loom_target_compile_report_source_low_memory_summary_t* summary =
          &report->source_low_memory_summary;
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory roots=%" PRIhsz
          " arguments=%" PRIhsz " argument_packets=%" PRIhsz
          " strategies=%" PRIhsz,
          report->source_low_memory_root_summaries.count,
          report->source_low_memory_argument_summaries.count,
          report->source_low_memory_argument_packet_summaries.count,
          report->source_low_memory_strategy_summaries.count));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_summary_text(
              summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: target_legalization legal=%" PRIu64
        " rewritten=%" PRIu64 " target_rewritten=%" PRIu64
        " reference_rewritten=%" PRIu64 " deferred=%" PRIu64
        " invalid_ir=%" PRIu64 " unsupported=%" PRIu64 " unhandled=%" PRIu64
        " rows=%" PRIhsz "\n",
        report->target_legalization_legal_op_count,
        report->target_legalization_rewritten_op_count,
        report->target_legalization_target_rewritten_op_count,
        report->target_legalization_reference_rewritten_op_count,
        report->target_legalization_deferred_op_count,
        report->target_legalization_invalid_ir_op_count,
        report->target_legalization_unsupported_op_count,
        report->target_legalization_unhandled_op_count,
        report->target_legalization_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: memory private_bytes=%" PRIu64 " local_bytes=%" PRIu64
        "\n",
        report->private_memory_bytes, report->local_memory_bytes));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pressure_rows count=%" PRIhsz "\n",
        report->pressure_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pressure_origin_rows count=%" PRIhsz "\n",
        report->pressure_origin_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: schedule_band_rows count=%" PRIhsz "\n",
        report->schedule_band_rows.count));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule_band_summary_rows count=%" PRIhsz "\n",
        report->schedule_band_summary_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: spill_rows count=%" PRIhsz "\n",
        report->spill_rows.count));
  }

  if (options->diagnostic_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: diagnostics count=%" PRIhsz "\n",
        options->diagnostic_count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: allocation_failure_rows count=%" PRIhsz "\n",
        report->allocation_failure_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation_high_water_rows count=%" PRIhsz "\n",
        report->allocation_high_water_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: entries count=%" PRIhsz "\n",
        report->entry_rows.count));
  }

  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_entry_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->entry_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        (const loom_target_compile_report_entry_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_entry_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_function_name =
          loom_target_compile_report_non_empty(row->source_function_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_export_name =
          loom_target_compile_report_non_empty(row->target_export_name);
      const iree_string_view_t target_export_symbol =
          loom_target_compile_report_non_empty(row->target_export_symbol);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      uint64_t move_kind_count = 0;
      uint64_t move_packet_count = 0;
      uint64_t move_unit_count = 0;
      loom_target_compile_report_move_cause_counts_totals(
          row->move_causes, &move_kind_count, &move_packet_count,
          &move_unit_count);
      const loom_target_compile_report_wait_plan_t* wait_plan = &row->wait_plan;
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: entry[%" PRIhsz
          "] function=%.*s source=%.*s bundle=%.*s export=%.*s "
          "export_symbol=%.*s config=%.*s schedule_nodes=%" PRIu64
          " scheduled=%" PRIu64 " resource_uses=%" PRIu64
          " hazard_gaps=%" PRIu64 " model_summaries=%" PRIu64
          " pressure_summaries=%" PRIu64 " peak_live=%" PRIu64
          " assignments=%" PRIu64 " spills=%" PRIu64 " spill_plans=%" PRIu64
          " coalesced_copies=%" PRIu64 " materialized_copies=%" PRIu64
          " materialized_spill_storage=%" PRIu64
          " materialized_spill_storage_bytes=%" PRIu64
          " materialized_spill_stores=%" PRIu64
          " materialized_spill_store_bytes=%" PRIu64
          " materialized_reloads=%" PRIu64 " materialized_reload_bytes=%" PRIu64
          " storage_leases=%" PRIu64 " storage_lease_instances=%" PRIu64
          " storage_release_actions=%" PRIu64 " move_kinds=%" PRIu64
          " move_packets=%" PRIu64 " move_units=%" PRIu64
          " wait_actions=%" PRIu64 " wait_explicit=%" PRIu64
          " wait_planned=%" PRIu64 " wait_full_drains=%" PRIu64
          " wait_partial=%" PRIu64 " wait_drained=%" PRIu64
          " wait_max_drained=%" PRIu64 " wait_max_outstanding=%" PRIu64
          " wait_max_full_drain_outstanding=%" PRIu64 " instructions=%" PRIu64
          " code_bytes=%" PRIu64 " storage_bytes=%" PRIu64
          " private_bytes=%" PRIu64 " local_bytes=%" PRIu64,
          row_index, (int)function_name.size, function_name.data,
          (int)source_function_name.size, source_function_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_export_name.size, target_export_name.data,
          (int)target_export_symbol.size, target_export_symbol.data,
          (int)target_config_name.size, target_config_name.data,
          row->schedule_node_count, row->scheduled_node_count,
          row->schedule_resource_use_count, row->schedule_hazard_gap_count,
          row->schedule_model_summary_count,
          row->register_pressure_summary_count,
          row->register_pressure_peak_live_units,
          row->allocation_assignment_count, row->allocation_spill_count,
          row->allocation_spill_plan_count,
          row->allocation_coalesced_copy_count,
          row->allocation_materialized_copy_count,
          row->allocation_materialized_spill_storage_count,
          row->allocation_materialized_spill_storage_bytes,
          row->allocation_materialized_spill_store_count,
          row->allocation_materialized_spill_store_bytes,
          row->allocation_materialized_reload_count,
          row->allocation_materialized_reload_bytes,
          row->allocation_storage_lease_count,
          row->allocation_storage_lease_instance_count,
          row->allocation_storage_release_action_count, move_kind_count,
          move_packet_count, move_unit_count, wait_plan->action_count,
          wait_plan->explicit_action_count, wait_plan->planned_action_count,
          wait_plan->full_drain_count, wait_plan->partial_wait_count,
          wait_plan->drained_count, wait_plan->max_drained_count,
          wait_plan->max_outstanding_before,
          wait_plan->max_full_drain_outstanding_before,
          row->emitted_instruction_count, row->emitted_code_byte_count,
          row->emitted_code_storage_byte_count, row->private_memory_bytes,
          row->local_memory_bytes));
      if (iree_any_bit_set(
              row->detail_flags,
              LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
        const loom_target_compile_report_target_resources_t* resources =
            &row->target_resources;
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV(" ")));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_target_resources_fields(
                builder, resources));
      }
      if (iree_any_bit_set(row->detail_flags,
                           LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
        IREE_RETURN_IF_ERROR(loom_target_compile_report_append_workload_fields(
            builder, &row->workload));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " pressure_rows=%" PRIhsz " pressure_origin_rows=%" PRIhsz
          " schedule_band_rows=%" PRIhsz " schedule_band_summary_rows=%" PRIhsz
          " spill_rows=%" PRIhsz " allocation_high_water_rows=%" PRIhsz
          " wait_counter_rows=%" PRIhsz " wait_reason_summary_rows=%" PRIhsz
          " wait_action_rows=%" PRIhsz " target_capability_rows=%" PRIhsz "\n",
          row->pressure_row_count, row->pressure_origin_row_count,
          row->schedule_band_row_count, row->schedule_band_summary_row_count,
          row->spill_row_count, row->allocation_high_water_row_count,
          row->wait_counter_row_count, row->wait_reason_summary_row_count,
          row->wait_action_row_count, row->target_capability_row_count));
      if (iree_any_bit_set(row->detail_flags,
                           LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "COMPILE-REPORT: entry[%" PRIhsz "].planning", row_index));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_low_planning_text_fields(
                &row->low_planning, builder));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("\n")));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_target_capability_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_capability_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        (const loom_target_compile_report_target_capability_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_target_capability_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t target_family_name =
          loom_target_compile_report_non_empty(row->target_family_name);
      const iree_string_view_t namespace_name =
          loom_target_compile_report_non_empty(row->namespace_name);
      const iree_string_view_t key =
          loom_target_compile_report_non_empty(row->key);
      const iree_string_view_t value_kind =
          loom_target_compile_report_capability_value_kind_name(
              row->value_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: target_capability[%" PRIhsz
          "] function=%.*s target_family=%.*s namespace=%.*s key=%.*s "
          "value_kind=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)target_family_name.size, target_family_name.data,
          (int)namespace_name.size, namespace_name.data, (int)key.size,
          key.data, (int)value_kind.size, value_kind.data));
      switch (row->value_kind) {
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " value=%s\n", row->value_u64 != 0 ? "true" : "false"));
          break;
        }
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " value=%" PRIu64 "\n", row->value_u64));
          break;
        }
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING: {
          const iree_string_view_t value =
              loom_target_compile_report_non_empty(row->value_string);
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " value=%.*s\n", (int)value.size, value.data));
          break;
        }
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE:
        default: {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(builder, "\n"));
          break;
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_move_cause(
    iree_string_builder_t* builder,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: move_cause[%.*s] packets=%" PRIu64 " units=%" PRIu64
      "\n",
      (int)descriptor->name.size, descriptor->name.data, counts->packet_count,
      counts->unit_count);
}

static iree_status_t loom_target_compile_report_format_move_causes(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause(
        builder, descriptor, &report->move_causes[descriptor->cause]));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_counter_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_counter_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_counter_row_t* rows =
        (const loom_target_compile_report_wait_counter_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_counter_row_t* row = &rows[i];
      const loom_target_compile_report_wait_plan_t* summary = &row->summary;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_counter[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32 " actions=%" PRIu64
          " explicit=%" PRIu64 " planned=%" PRIu64 " full_drains=%" PRIu64
          " partial_waits=%" PRIu64 " drained=%" PRIu64 " max_drained=%" PRIu64
          " max_outstanding=%" PRIu64 " max_full_drain_outstanding=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          summary->action_count, summary->explicit_action_count,
          summary->planned_action_count, summary->full_drain_count,
          summary->partial_wait_count, summary->drained_count,
          summary->max_drained_count, summary->max_outstanding_before,
          summary->max_full_drain_outstanding_before));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_reason_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_reason_summary_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_reason_summary_row_t* rows =
        (const loom_target_compile_report_wait_reason_summary_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_reason_summary_row_t* row =
          &rows[i];
      const loom_target_compile_report_wait_plan_t* summary = &row->summary;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_non_empty(row->reason_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_reason_summary[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32
          " reason=%.*s reason_id=%" PRIu32 " actions=%" PRIu64
          " explicit=%" PRIu64 " planned=%" PRIu64 " full_drains=%" PRIu64
          " partial_waits=%" PRIu64 " drained=%" PRIu64 " max_drained=%" PRIu64
          " max_outstanding=%" PRIu64 " max_full_drain_outstanding=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          (int)reason_name.size, reason_name.data, row->reason_id,
          summary->action_count, summary->explicit_action_count,
          summary->planned_action_count, summary->full_drain_count,
          summary->partial_wait_count, summary->drained_count,
          summary->max_drained_count, summary->max_outstanding_before,
          summary->max_full_drain_outstanding_before));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_action_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_action_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_action_row_t* rows =
        (const loom_target_compile_report_wait_action_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_action_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      const iree_string_view_t action_name =
          loom_target_compile_report_non_empty(row->action_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_non_empty(row->reason_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_action[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32
          " action=%.*s action_id=%" PRIu32 " reason=%.*s reason_id=%" PRIu32
          " block=%" PRIu32 " node=%" PRIu32 " ordinal=%" PRIu32
          " producer_node=",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          (int)action_name.size, action_name.data, row->action_id,
          (int)reason_name.size, reason_name.data, row->reason_id,
          row->block_index, row->node_index, row->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->producer_node));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" producer_ordinal=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->producer_scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_operation"),
          row->producer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_descriptor_key"),
          row->producer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_semantic_tag"),
          row->producer_semantic_tag));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" consumer_node=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->consumer_node));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" consumer_ordinal=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->consumer_scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_operation"),
          row->consumer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_descriptor_key"),
          row->consumer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_semantic_tag"),
          row->consumer_semantic_tag));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " target_count=%" PRIu32 " outstanding_before=%" PRIu32
          " outstanding_after=%" PRIu32 " drained=%" PRIu32 "\n",
          row->target_count, row->outstanding_before, row->outstanding_after,
          row->drained_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->pressure_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_row_t* rows =
        (const loom_target_compile_report_pressure_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_pressure_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(
              (loom_scalar_type_t)row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_non_empty(row->peak_operation_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pressure[%" PRIhsz
          "] function=%.*s class=%.*s type=%.*s element=%.*s "
          "peak_units=%" PRIu64 " peak_values=%" PRIu64
          " point=%u block=%.*s op=%.*s\n",
          row_index, (int)function_name.size, function_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->peak_live_units, row->peak_live_values, row->peak_point,
          (int)peak_block_name.size, peak_block_name.data,
          (int)peak_operation_name.size, peak_operation_name.data));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_origin_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->pressure_origin_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_origin_row_t* rows =
        (const loom_target_compile_report_pressure_origin_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_pressure_origin_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_non_empty(row->peak_operation_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pressure_origin[%" PRIhsz
          "] function=%.*s class=%.*s type=%.*s element=%.*s "
          "point=%u block=%.*s op=%.*s origin=%.*s origin_op=%.*s "
          "semantic=%.*s sample=%.*s live_units=%" PRIu64
          " live_values=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data, row->peak_point,
          (int)peak_block_name.size, peak_block_name.data,
          (int)peak_operation_name.size, peak_operation_name.data,
          (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data, row->live_units,
          row->live_values));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_schedule_band_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->schedule_band_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_schedule_band_row_t* rows =
        (const loom_target_compile_report_schedule_band_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_schedule_band_row_t* row = &rows[i];
      const loom_target_compile_report_static_instruction_mix_t* mix =
          &row->static_instruction_mix;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: schedule_band[%" PRIhsz
          "] function=%.*s block=%.*s block_index=%" PRIu32
          " first_packet=%" PRIu64 " first_ordinal=%" PRIu32 " nodes=%" PRIu32
          " origin=%.*s"
          " origin_op=%.*s semantic=%.*s sample=%.*s"
          " descriptors=%" PRIu64 " unknown=%" PRIu64 " scalar_alu=%" PRIu64
          " vector_alu=%" PRIu64 " matrix=%" PRIu64 " mfma=%" PRIu64
          " smfmac=%" PRIu64 " wmma=%" PRIu64 " swmmac=%" PRIu64 " dot=%" PRIu64
          " global_memory=%" PRIu64 " local_memory=%" PRIu64
          " scalar_memory=%" PRIu64 " generic_memory=%" PRIu64
          " atomic=%" PRIu64 " branch=%" PRIu64 " barrier=%" PRIu64
          " control=%" PRIu64 " conversion=%" PRIu64 " cache=%" PRIu64
          " register_move=%" PRIu64 " result_values=%" PRIu64
          " result_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->first_packet_index, row->first_scheduled_ordinal,
          row->node_count, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data,
          mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
          mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
          mix->smfmac_count, mix->wmma_count, mix->swmmac_count, mix->dot_count,
          mix->global_memory_count, mix->local_memory_count,
          mix->scalar_memory_count, mix->generic_memory_count,
          mix->atomic_count, mix->branch_count, mix->barrier_count,
          mix->control_count, mix->conversion_count, mix->cache_count,
          mix->register_move_count, row->result_value_count,
          row->result_unit_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->schedule_band_summary_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_schedule_band_summary_row_t* rows =
        (const loom_target_compile_report_schedule_band_summary_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_schedule_band_summary_row_t* row =
          &rows[i];
      const loom_target_compile_report_static_instruction_mix_t* mix =
          &row->static_instruction_mix;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: schedule_band_summary[%" PRIhsz
          "] function=%.*s block=%.*s block_index=%" PRIu32
          " first_packet=%" PRIu64 " bands=%" PRIu64 " nodes=%" PRIu64
          " max_band_nodes=%" PRIu32
          " origin=%.*s"
          " origin_op=%.*s semantic=%.*s sample=%.*s"
          " descriptors=%" PRIu64 " unknown=%" PRIu64 " scalar_alu=%" PRIu64
          " vector_alu=%" PRIu64 " matrix=%" PRIu64 " mfma=%" PRIu64
          " smfmac=%" PRIu64 " wmma=%" PRIu64 " swmmac=%" PRIu64 " dot=%" PRIu64
          " global_memory=%" PRIu64 " local_memory=%" PRIu64
          " scalar_memory=%" PRIu64 " generic_memory=%" PRIu64
          " atomic=%" PRIu64 " branch=%" PRIu64 " barrier=%" PRIu64
          " control=%" PRIu64 " conversion=%" PRIu64 " cache=%" PRIu64
          " register_move=%" PRIu64 " result_values=%" PRIu64
          " result_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->first_packet_index, row->band_count, row->node_count,
          row->max_band_node_count, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data,
          mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
          mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
          mix->smfmac_count, mix->wmma_count, mix->swmmac_count, mix->dot_count,
          mix->global_memory_count, mix->local_memory_count,
          mix->scalar_memory_count, mix->generic_memory_count,
          mix->atomic_count, mix->branch_count, mix->barrier_count,
          mix->control_count, mix->conversion_count, mix->cache_count,
          mix->register_move_count, row->result_value_count,
          row->result_unit_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_spill_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->spill_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_spill_row_t* rows =
        (const loom_target_compile_report_spill_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_spill_row_t* row = &rows[i];
      const iree_string_view_t kind =
          loom_target_compile_report_spill_row_kind_name(row->kind);
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t slot_space =
          loom_target_compile_report_non_empty(row->slot_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: spill[%" PRIhsz
          "] kind=%.*s function=%.*s value=%.*s class=%.*s type=%.*s "
          "element=%.*s "
          "origin=%.*s origin_op=%.*s semantic=%.*s "
          "assignment=%u slot=%u space=%.*s bytes=%" PRIu64 " align=%" PRIu64
          " stores=%" PRIu64 " store_bytes=%" PRIu64 " reloads=%" PRIu64
          " reload_bytes=%" PRIu64 "\n",
          row_index, (int)kind.size, kind.data, (int)function_name.size,
          function_name.data, (int)value_name.size, value_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data, row->assignment_index,
          row->slot_index, (int)slot_space.size, slot_space.data,
          row->byte_size, row->byte_alignment, row->store_count,
          row->store_bytes, row->reload_count, row->reload_bytes));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_allocation_failure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->allocation_failure_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_allocation_failure_row_t* rows =
        (const loom_target_compile_report_allocation_failure_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_allocation_failure_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t failure_code =
          loom_target_compile_report_non_empty(row->failure_code);
      const iree_string_view_t blocking_kind =
          loom_target_compile_report_allocation_failure_blocking_kind_name(
              row->blocking_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t origin_block_name =
          loom_target_compile_report_non_empty(row->origin_block_name);
      const iree_string_view_t location_kind =
          loom_target_compile_report_non_empty(row->location_kind);
      const iree_string_view_t conflict_value_name =
          loom_target_compile_report_non_empty(row->conflict_value_name);
      const iree_string_view_t conflict_location_kind =
          loom_target_compile_report_non_empty(row->conflict_location_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: allocation_failure[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s code=%.*s blocking=%.*s "
          "origin=%.*s block=%.*s start=%u end=%u required_units=%u "
          "budget_units=",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)failure_code.size, failure_code.data,
          (int)blocking_kind.size, blocking_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)origin_block_name.size, origin_block_name.data, row->start_point,
          row->end_point, row->required_unit_count));
      if (row->budget_units == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->budget_units));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " peak_live_units=%u location=%.*s[", row->peak_live_units,
          (int)location_kind.size, location_kind.data));
      if (row->location_base == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->location_base));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ":%u] conflict_assignment=", row->location_count));
      if (row->conflict_assignment_index == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_assignment_index));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " conflict_value=%.*s conflict_start=",
          (int)conflict_value_name.size, conflict_value_name.data));
      if (row->conflict_start_point == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_start_point));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" conflict_end=")));
      if (row->conflict_end_point == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_end_point));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " conflict_location=%.*s[", (int)conflict_location_kind.size,
          conflict_location_kind.data));
      if (row->conflict_location_base == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_location_base));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ":%u]\n", row->conflict_location_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->allocation_high_water_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_allocation_high_water_row_t* rows =
        (const loom_target_compile_report_allocation_high_water_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_allocation_high_water_row_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t location_kind =
          loom_target_compile_report_non_empty(row->location_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: allocation_high_water[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s type=%.*s element=%.*s "
          "assignment=%u origin=%.*s origin_op=%.*s semantic=%.*s start=%u "
          "end=%u required_units=%u location=%.*s[%u:%u] high_water=%" PRIu64
          " lower_free_units=%" PRIu64
          " lower_free_runs=%u "
          "lower_largest_free_run_units=%u"
          " lower_pressure_releasable_free_units=%" PRIu64
          " lower_pressure_releasable_free_runs=%u"
          " lower_pressure_releasable_largest_free_run_units=%u"
          " active_assignment_blockers=%u "
          "active_assignment_blocker_units=%" PRIu64
          " active_storage_lease_blockers=%u "
          "active_storage_lease_blocker_units=%" PRIu64
          " active_pressure_storage_lease_blockers=%u "
          "active_pressure_storage_lease_blocker_units=%" PRIu64
          " active_fallback_storage_lease_blockers=%u "
          "active_fallback_storage_lease_blocker_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->assignment_index, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data, row->start_point,
          row->end_point, row->required_unit_count, (int)location_kind.size,
          location_kind.data, row->location_base, row->location_count,
          row->high_water_units, row->lower_free_unit_count,
          row->lower_free_run_count, row->lower_largest_free_run_unit_count,
          row->lower_pressure_releasable_free_unit_count,
          row->lower_pressure_releasable_free_run_count,
          row->lower_pressure_releasable_largest_free_run_unit_count,
          row->active_assignment_blocker_count,
          row->active_assignment_blocker_units,
          row->active_storage_lease_blocker_count,
          row->active_storage_lease_blocker_units,
          row->active_pressure_storage_lease_blocker_count,
          row->active_pressure_storage_lease_blocker_units,
          row->active_fallback_storage_lease_blocker_count,
          row->active_fallback_storage_lease_blocker_units));
    }
  }
  return iree_ok_status();
}

static bool
loom_target_compile_report_source_low_selection_summary_has_dynamic_delta(
    const loom_target_compile_report_source_low_selection_summary_t* row) {
  return row->unknown_dynamic_op_count != 0 ||
         row->dynamic_selected_op_count != row->selected_op_count ||
         row->dynamic_emitted_low_op_count != row->emitted_low_op_count;
}

static iree_status_t loom_target_compile_report_append_source_low_descriptor(
    iree_string_view_t descriptor_key,
    iree_string_view_t descriptor_semantic_tag,
    iree_string_builder_t* builder) {
  if (!iree_string_view_is_empty(descriptor_key)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " descriptor_key=%.*s", (int)descriptor_key.size,
        descriptor_key.data));
  }
  if (!iree_string_view_is_empty(descriptor_semantic_tag)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " descriptor_semantic_tag=%.*s",
        (int)descriptor_semantic_tag.size, descriptor_semantic_tag.data));
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_append_source_low_descriptor_fields(
    const loom_target_compile_report_source_low_row_t* row,
    iree_string_builder_t* builder) {
  return loom_target_compile_report_append_source_low_descriptor(
      row->descriptor_key, row->descriptor_semantic_tag, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_selection_summary_descriptor(
    const loom_target_compile_report_source_low_selection_summary_t* row,
    iree_string_builder_t* builder) {
  return loom_target_compile_report_append_source_low_descriptor(
      row->descriptor_key, row->descriptor_semantic_tag, builder);
}

static iree_status_t
loom_target_compile_report_format_source_low_selection_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_selection_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_selection_summary_t* rows =
        (const loom_target_compile_report_source_low_selection_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_selection_summary_t* row =
          &rows[i];
      if (!loom_target_compile_report_source_low_selection_summary_has_dynamic_delta(
              row)) {
        continue;
      }
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t selection_name =
          loom_target_compile_report_source_low_selection_name(
              row->selection_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_selection[%" PRIhsz
          "] function=%.*s source_op=%.*s selection=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)selection_name.size, selection_name.data));
      if (!iree_string_view_is_empty(row->plan_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " plan_key=%.*s", (int)row->plan_key.size,
            row->plan_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_selection_summary_descriptor(
              row, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " selected_ops=%" PRIu64 " emitted_ops=%" PRIu64,
          row->selected_op_count, row->emitted_low_op_count));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " exact_dynamic_ops=%" PRIu64 " unknown_dynamic_ops=%" PRIu64
          " dynamic_selected_ops=%" PRIu64 " dynamic_emitted_ops=%" PRIu64,
          row->exact_dynamic_op_count, row->unknown_dynamic_op_count,
          row->dynamic_selected_op_count, row->dynamic_emitted_low_op_count));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_row_t* rows =
        (const loom_target_compile_report_source_low_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t selection_name =
          loom_target_compile_report_source_low_selection_name(
              row->selection_kind);
      const iree_string_view_t plan_key = row->plan_key;
      if (row->selection_kind ==
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low[%" PRIhsz
            "] function=%.*s source_op=%.*s selection=%.*s rule_set=%u "
            "rule=%u",
            row_index, (int)function_name.size, function_name.data,
            (int)source_op_name.size, source_op_name.data,
            (int)selection_name.size, selection_name.data, row->rule_set_index,
            row->rule_index));
        if (!iree_string_view_is_empty(plan_key)) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " plan_key=%.*s", (int)plan_key.size, plan_key.data));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_source_low_descriptor_fields(
                row, builder));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " emitted_ops=%u", row->emitted_low_op_count));
        if (row->execution_count_plus_one !=
                LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
            row->execution_count_plus_one != 2) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " execution_count=%" PRIu64,
              row->execution_count_plus_one - 1));
        }
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low[%" PRIhsz
          "] function=%.*s source_op=%.*s selection=%.*s plan=%" PRIu64,
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)selection_name.size, selection_name.data, row->plan_id));
      if (!iree_string_view_is_empty(plan_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " plan_key=%.*s", (int)plan_key.size, plan_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_descriptor_fields(
              row, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " emitted_ops=%u", row->emitted_low_op_count));
      if (row->execution_count_plus_one !=
              LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
          row->execution_count_plus_one != 2) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " execution_count=%" PRIu64,
            row->execution_count_plus_one - 1));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_target_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_target_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_target_row_t* rows =
        (const loom_target_compile_report_source_low_target_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_target_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t target_symbol_name =
          loom_target_compile_report_non_empty(row->target_symbol_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t target_source =
          loom_target_selection_source_name(row->target_source);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_target[%" PRIhsz
          "] function=%.*s source=%.*s target=%.*s bundle=%.*s config=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)target_source.size, target_source.data,
          (int)target_symbol_name.size, target_symbol_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data));
      if (row->target_subgroup_size != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " subgroup_size=%u", row->target_subgroup_size));
      }
      if (row->candidate_target_count != 0) {
        const iree_string_view_t candidate_symbol_name =
            loom_target_compile_report_non_empty(
                row->candidate_target_symbol_name);
        const iree_string_view_t candidate_bundle_name =
            loom_target_compile_report_non_empty(
                row->candidate_target_bundle_name);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            " candidate_targets=%u first_candidate=%.*s "
            "first_candidate_bundle=%.*s",
            row->candidate_target_count, (int)candidate_symbol_name.size,
            candidate_symbol_name.data, (int)candidate_bundle_name.size,
            candidate_bundle_name.data));
        if (row->candidate_target_subgroup_size != 0) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " first_candidate_subgroup_size=%u",
              row->candidate_target_subgroup_size));
        }
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_transform_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  if (report->source_low_transform_rows.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "COMPILE-REPORT: source_low_transforms rows=%" PRIhsz "\n",
      report->source_low_transform_rows.count));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_transform_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_transform_row_t* rows =
        (const loom_target_compile_report_source_low_transform_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_transform_row_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t transform_key =
          loom_target_compile_report_non_empty(row->transform_key);
      const iree_string_view_t outcome =
          loom_target_compile_report_non_empty(row->outcome);
      const iree_string_view_t reason =
          loom_target_compile_report_non_empty(row->reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_transform[%" PRIhsz
          "] function=%.*s source_op=%.*s transform=%.*s outcome=%.*s "
          "reason=%.*s candidates=%u selected=%u removed_loop_carried=%u",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)transform_key.size, transform_key.data, (int)outcome.size,
          outcome.data, (int)reason.size, reason.data,
          row->candidate_value_count, row->selected_value_count,
          row->removed_loop_carried_value_count));
      if (row->removed_loop_carried_payload_register_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " removed_payload_registers=%" PRIu64,
            row->removed_loop_carried_payload_register_count));
      }
      if (row->block_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " shape=%" PRIu64 "x%" PRIu64 "x%" PRIu64,
            row->block_count, row->row_count, row->column_count));
      } else if (row->row_count != 0 || row->column_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " shape=%" PRIu64 "x%" PRIu64, row->row_count,
            row->column_count));
      }
      if (row->workgroup_memory_byte_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " workgroup_bytes=%" PRIu64,
            row->workgroup_memory_byte_count));
      }
      if (row->inserted_load_op_count != 0 ||
          row->inserted_store_op_count != 0 ||
          row->inserted_barrier_op_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " inserted_loads=%u inserted_stores=%u barriers=%u",
            row->inserted_load_op_count, row->inserted_store_op_count,
            row->inserted_barrier_op_count));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_memory_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_row_t* rows =
        (const loom_target_compile_report_source_low_memory_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_non_empty(row->strategy_key);
      const iree_string_view_t address_form =
          loom_target_compile_report_non_empty(row->address_form);
      const iree_string_view_t dynamic_term_kind =
          loom_target_compile_report_non_empty(row->dynamic_term_kind);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
      const iree_string_view_t bank_conflict_kind =
          loom_target_compile_report_non_empty(row->bank_conflict_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory[%" PRIhsz
          "] function=%.*s source_op=%.*s memory_space=%.*s operation=%.*s "
          "packet=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)memory_space.size,
          memory_space.data, (int)operation_kind.size, operation_kind.data,
          (int)packet_key.size, packet_key.data));
      if (!iree_string_view_is_empty(row->source_root_name)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root=%.*s", (int)source_root_name.size,
            source_root_name.data));
      }
      if (row->source_root_argument_index != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root_argument_index=%u",
            row->source_root_argument_index));
      }
      if (!iree_string_view_is_empty(row->strategy_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " strategy=%.*s", (int)strategy_key.size,
            strategy_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_storage_text(
              row, builder));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_memory_interval_text(
              &row->source_interval, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " address_form=%.*s dynamic_term_kind=%.*s "
          "fallback_reason=%.*s static_offset_bytes=%" PRId64
          " element_bytes=%u "
          "vector_lanes=%u issued_read_bytes=%u issued_write_bytes=%u "
          "issued_read_unknown_widths=%u issued_write_unknown_widths=%u "
          "dynamic_stride_bytes=%u "
          "vector_lane_stride_bytes=%u bank_stride_words=%u "
          "bank_conflict_degree=%u bank_conflict_kind=%.*s\n",
          (int)address_form.size, address_form.data,
          (int)dynamic_term_kind.size, dynamic_term_kind.data,
          (int)fallback_reason.size, fallback_reason.data,
          row->static_offset_bytes, row->element_byte_count,
          row->vector_lane_count, row->issued_read_byte_count,
          row->issued_write_byte_count, row->issued_read_unknown_width_count,
          row->issued_write_unknown_width_count, row->dynamic_stride_bytes,
          row->vector_lane_stride_bytes, row->bank_stride_words,
          row->bank_conflict_degree, (int)bank_conflict_kind.size,
          bank_conflict_kind.data));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_root_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_root_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_root_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_root_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_root_summary_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_root[%" PRIhsz
          "] function=%.*s source_root=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_root_name.size, source_root_name.data));
      if (row->source_root_argument_index != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root_argument_index=%u",
            row->source_root_argument_index));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " memory_space=%.*s", (int)memory_space.size,
          memory_space.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_summary_text(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_argument_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_argument_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_argument_summary_t*
          row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_argument[%" PRIhsz
          "] function=%.*s",
          row_index, (int)function_name.size, function_name.data));
      if (!iree_string_view_is_empty(row->source_root_name)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root=%.*s", (int)source_root_name.size,
            source_root_name.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " source_root_argument_index=%u memory_space=%.*s",
          row->source_root_argument_index, (int)memory_space.size,
          memory_space.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_summary_text(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_packet_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_packet_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        rows =
            (const loom_target_compile_report_source_low_memory_argument_packet_summary_t*)
                loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
          row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_non_empty(row->strategy_key);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_argument_packet[%" PRIhsz
          "] function=%.*s",
          row_index, (int)function_name.size, function_name.data));
      if (!iree_string_view_is_empty(row->source_root_name)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root=%.*s", (int)source_root_name.size,
            source_root_name.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_root_argument_index=%u memory_space=%.*s operation=%.*s "
          "packet=%.*s",
          row->source_root_argument_index, (int)memory_space.size,
          memory_space.data, (int)operation_kind.size, operation_kind.data,
          (int)packet_key.size, packet_key.data));
      if (!iree_string_view_is_empty(row->strategy_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " strategy=%.*s", (int)strategy_key.size,
            strategy_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_argument_packet_storage_text(
              row, builder));
      if (!iree_string_view_is_empty(row->fallback_reason)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " fallback_reason=%.*s", (int)fallback_reason.size,
            fallback_reason.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_summary_text(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_strategy_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_strategy_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_strategy_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_strategy_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_strategy_summary_t*
          row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_non_empty(row->strategy_key);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_strategy[%" PRIhsz
          "] function=%.*s memory_space=%.*s operation=%.*s packet=%.*s "
          "strategy=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)memory_space.size, memory_space.data, (int)operation_kind.size,
          operation_kind.data, (int)packet_key.size, packet_key.data,
          (int)strategy_key.size, strategy_key.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_strategy_storage_text(
              row, builder));
      if (!iree_string_view_is_empty(row->fallback_reason)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " fallback_reason=%.*s", (int)fallback_reason.size,
            fallback_reason.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_summary_text(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_math_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->math_legalization_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_math_row_t* rows =
        (const loom_target_compile_report_math_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_math_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t policy_name =
          loom_target_compile_report_non_empty(row->policy_name);
      const iree_string_view_t constraint_key =
          loom_target_compile_report_non_empty(row->constraint_key);
      const iree_string_view_t action_name =
          loom_target_compile_report_math_action_name(row->action);
      const iree_string_view_t math_op_name =
          loom_target_math_op_name((loom_target_math_op_t)row->math_op);
      const iree_string_view_t lane_domain_name =
          loom_target_math_lane_domain_name(
              (loom_target_math_lane_domain_t)row->lane_domain);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t recipe_name =
          loom_target_math_recipe_name((loom_target_math_recipe_t)row->recipe);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: math_legalization[%" PRIhsz
          "] function=%.*s source_op=%.*s action=%.*s policy=%.*s "
          "math_op=%.*s domain=%.*s element=%.*s recipe=%.*s "
          "constraint=%.*s bundle=%.*s config=%.*s source_fastmath=0x%02x "
          "recipe_fastmath=0x%02x created_ops=%" PRIu64 " erased_ops=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)action_name.size,
          action_name.data, (int)policy_name.size, policy_name.data,
          (int)math_op_name.size, math_op_name.data, (int)lane_domain_name.size,
          lane_domain_name.data, (int)element_type_name.size,
          element_type_name.data, (int)recipe_name.size, recipe_name.data,
          (int)constraint_key.size, constraint_key.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data,
          (uint32_t)row->source_fastmath_flags,
          (uint32_t)row->recipe_fastmath_flags, row->created_op_count,
          row->erased_op_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_legalization_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_legalization_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_legalization_row_t* rows =
        (const loom_target_compile_report_legalization_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_legalization_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t legalizer_name =
          loom_target_compile_report_non_empty(row->legalizer_name);
      const iree_string_view_t strategy_name =
          loom_target_compile_report_legalizer_strategy_name(
              row->legalizer_strategy);
      const iree_string_view_t mode_name =
          loom_target_compile_report_legalization_mode_name(row->mode);
      const iree_string_view_t policy_name =
          loom_target_compile_report_legalization_policy_name(row->policy);
      const iree_string_view_t action_name =
          loom_target_compile_report_legalization_action_name(row->action);
      const iree_string_view_t legalization_outcome_name =
          loom_target_compile_report_legalization_outcome_name(
              row->legalization_outcome);
      const iree_string_view_t outcome_name =
          loom_target_compile_report_contract_outcome_name(
              row->contract_outcome);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: target_legalization[%" PRIhsz
          "] function=%.*s source_op=%.*s mode=%.*s policy=%.*s "
          "action=%.*s outcome=%.*s contract=%.*s legalizer=%.*s strategy=%.*s "
          "bundle=%.*s config=%.*s binding=%u case=%u rule_set=%u rule=%u "
          "diagnostic=%u",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)mode_name.size,
          mode_name.data, (int)policy_name.size, policy_name.data,
          (int)action_name.size, action_name.data,
          (int)legalization_outcome_name.size, legalization_outcome_name.data,
          (int)outcome_name.size, outcome_name.data, (int)legalizer_name.size,
          legalizer_name.data, (int)strategy_name.size, strategy_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data,
          row->binding_index, row->case_index, row->rule_set_index,
          row->rule_index, row->diagnostic_index));
      if (!iree_string_view_is_empty(row->descriptor_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " descriptor_key=%.*s", (int)row->descriptor_key.size,
            row->descriptor_key.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_rejections=0x%08" PRIx32 " source_rejection_detail=%" PRIu32
          " target_rejections=0x%08" PRIx32 " missing_features=0x%08" PRIx32
          " missing_facts=0x%08" PRIx32 " created_ops=%" PRIu64
          " erased_ops=%" PRIu64 "\n",
          row->source_rejection_bits, row->source_rejection_detail,
          row->target_rejection_bits, row->missing_feature_bits,
          row->missing_fact_bits, row->created_op_count, row->erased_op_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_json_write_optional_string_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_string_field(object, name, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u16_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint16_t value) {
  if (value == UINT16_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint64_t value) {
  if (value == UINT64_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint64_field(object, name, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u32_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint32_t value) {
  if (value == UINT32_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

static iree_status_t loom_target_compile_report_format_schedule_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("node_count"), report->schedule_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scheduled_node_count"), report->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("dependency_count"), report->schedule_dependency_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("resource_use_count"),
      report->schedule_resource_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("hazard_gap_count"), report->schedule_hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("model_summary_count"),
      report->schedule_model_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_summary_count"),
      report->register_pressure_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_peak_live_units"),
      report->register_pressure_peak_live_units));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_instruction_mix_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("descriptor_count"), mix->descriptor_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unknown_count"), mix->unknown_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_alu_count"), mix->scalar_alu_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("vector_alu_count"), mix->vector_alu_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("matrix_count"), mix->matrix_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("mfma_count"), mix->mfma_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("smfmac_count"), mix->smfmac_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("wmma_count"), mix->wmma_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("swmmac_count"), mix->swmmac_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("dot_count"), mix->dot_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_memory_count"), mix->global_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_load_count"), mix->global_load_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_store_count"), mix->global_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_load_count"), mix->buffer_load_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_store_count"), mix->buffer_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("flat_memory_count"), mix->flat_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_memory_count"), mix->local_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_memory_count"), mix->scalar_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_memory_count"), mix->private_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("generic_memory_count"), mix->generic_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_read_unknown_width_count"),
      mix->memory_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_write_unknown_width_count"),
      mix->memory_write_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_read_byte_count"), mix->memory_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_write_byte_count"),
      mix->memory_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_load_byte_count"), mix->global_load_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_store_byte_count"),
      mix->global_store_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_load_byte_count"), mix->buffer_load_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_store_byte_count"),
      mix->buffer_store_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("flat_read_byte_count"), mix->flat_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("flat_write_byte_count"), mix->flat_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_read_byte_count"), mix->local_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_write_byte_count"), mix->local_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_read_byte_count"), mix->scalar_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_write_byte_count"),
      mix->scalar_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_read_byte_count"),
      mix->private_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_write_byte_count"),
      mix->private_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unclassified_read_byte_count"),
      mix->unclassified_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unclassified_write_byte_count"),
      mix->unclassified_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("atomic_count"), mix->atomic_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("branch_count"), mix->branch_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("barrier_count"), mix->barrier_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("control_count"), mix->control_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("conversion_count"), mix->conversion_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("cache_count"), mix->cache_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_move_count"), mix->register_move_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_allocation_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("assignment_count"),
                                          report->allocation_assignment_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("spill_count"), report->allocation_spill_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("spill_plan_count"),
                                          report->allocation_spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("coalesced_copy_count"),
      report->allocation_coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_copy_count"),
      report->allocation_materialized_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_storage_count"),
      report->allocation_materialized_spill_storage_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_storage_bytes"),
      report->allocation_materialized_spill_storage_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_store_count"),
      report->allocation_materialized_spill_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_store_bytes"),
      report->allocation_materialized_spill_store_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_reload_count"),
      report->allocation_materialized_reload_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_reload_bytes"),
      report->allocation_materialized_reload_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("storage_lease_count"),
      report->allocation_storage_lease_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("storage_lease_instance_count"),
      report->allocation_storage_lease_instance_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("storage_release_action_count"),
      report->allocation_storage_release_action_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_move_cause_json(
    loom_json_array_writer_t* array,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(array));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(array->stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("cause"), descriptor->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("packet_count"), counts->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unit_count"), counts->unit_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_move_cause_counts_json(
    const loom_target_compile_report_move_cause_counts_t* counts,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  uint64_t kind_count = 0;
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  loom_target_compile_report_move_cause_counts_totals(
      counts, &kind_count, &packet_count, &unit_count);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("kind_count"), kind_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("packet_count"), packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unit_count"), unit_count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("causes")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
         ++i) {
      const loom_target_compile_report_move_cause_descriptor_t* descriptor =
          &loom_target_compile_report_move_cause_descriptors[i];
      IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause_json(
          &array, descriptor, &counts[descriptor->cause]));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_move_causes_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  return loom_target_compile_report_format_move_cause_counts_json(
      report->move_causes, mode, stream);
}

static iree_status_t loom_target_compile_report_format_wait_plan_json(
    const loom_target_compile_report_wait_plan_t* wait_plan,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("action_count"), wait_plan->action_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("explicit_action_count"),
      wait_plan->explicit_action_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("planned_action_count"),
      wait_plan->planned_action_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("full_drain_count"), wait_plan->full_drain_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("partial_wait_count"), wait_plan->partial_wait_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("drained_count"), wait_plan->drained_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max_drained_count"), wait_plan->max_drained_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max_outstanding_before"),
      wait_plan->max_outstanding_before));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max_full_drain_outstanding_before"),
      wait_plan->max_full_drain_outstanding_before));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_counter_row_json(
    const loom_target_compile_report_wait_counter_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("counter"), row->counter_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("counter_id"), row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_plan_json(&row->summary, stream));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_counter_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->wait_counter_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_counter_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_counter_row_t* rows =
          (const loom_target_compile_report_wait_counter_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_counter_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_wait_reason_summary_row_json(
    const loom_target_compile_report_wait_reason_summary_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("counter"), row->counter_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("counter_id"), row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("reason"), row->reason_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reason_id"), row->reason_id));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_plan_json(&row->summary, stream));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_wait_reason_summary_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  (void)mode;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->wait_reason_summary_rows.count));
  if (report->wait_reason_summary_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_reason_summary_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_reason_summary_row_t* rows =
          (const loom_target_compile_report_wait_reason_summary_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_reason_summary_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_action_row_json(
    const loom_target_compile_report_wait_action_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("counter"), row->counter_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("counter_id"), row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("action"), row->action_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("action_id"), row->action_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("reason"), row->reason_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reason_id"), row->reason_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node_index"), row->node_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("scheduled_ordinal"), row->scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("producer_node"), row->producer_node));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("producer_scheduled_ordinal"),
      row->producer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("producer_operation"),
          row->producer_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("producer_descriptor_key"),
          row->producer_descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("producer_semantic_tag"),
          row->producer_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("consumer_node"), row->consumer_node));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("consumer_scheduled_ordinal"),
      row->consumer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("consumer_operation"),
          row->consumer_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("consumer_descriptor_key"),
          row->consumer_descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("consumer_semantic_tag"),
          row->consumer_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("target_count"), row->target_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("outstanding_before"), row->outstanding_before));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("outstanding_after"), row->outstanding_after));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("drained_count"), row->drained_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_action_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->wait_action_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_action_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_action_row_t* rows =
          (const loom_target_compile_report_wait_action_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_action_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_capability_row_json(
    const loom_target_compile_report_target_capability_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_family"), row->target_family_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("namespace"), row->namespace_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("key"), row->key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("value_kind"),
      loom_target_compile_report_capability_value_kind_name(row->value_kind)));
  switch (row->value_kind) {
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
          &object, IREE_SV("value_bool"), row->value_u64 != 0));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("value_u64"), row->value_u64));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &object, IREE_SV("value_string"), row->value_string));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE:
    default:
      break;
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_capability_rows_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->target_capability_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_capability_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        (const loom_target_compile_report_target_capability_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_target_capability_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_emission_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("instruction_count"),
                                          report->emitted_instruction_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_byte_count"), report->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_storage_byte_count"),
      report->emitted_code_storage_byte_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_memory_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_bytes"), report->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_bytes"), report->local_memory_bytes));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_resource_registers_json(
    iree_string_view_t register_class, uint64_t final_register_count,
    uint64_t scheduled_pressure_peak_live_units, uint64_t final_overhead_units,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (!iree_string_view_is_empty(register_class)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("register_class"), register_class));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("final")));
  loom_json_object_writer_t final_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &final_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &final_object, IREE_SV("register_count"), final_register_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &final_object, IREE_SV("overhead_units"), final_overhead_units));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&final_object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("scheduled_pressure")));
  loom_json_object_writer_t pressure_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &pressure_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &pressure_object, IREE_SV("peak_live_units"),
      scheduled_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&pressure_object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_target_resources_json(
    const loom_target_compile_report_target_resources_t* resources,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("scalar")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_target_resource_registers_json(
          resources->scalar_register_class, resources->scalar_register_count,
          resources->scalar_pressure_peak_live_units,
          resources->scalar_register_overhead_units, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("vector")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_target_resource_registers_json(
          resources->vector_register_class, resources->vector_register_count,
          resources->vector_pressure_peak_live_units,
          resources->vector_register_overhead_units, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("subgroup_size"), resources->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("max_subgroups_per_simd"),
      resources->max_subgroups_per_simd));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("resident_subgroups_per_simd"),
      resources->resident_subgroups_per_simd));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("occupancy_percent"), resources->occupancy_percent));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("limiting_resource"), resources->limiting_resource));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_dimension3_json(
    uint32_t x, uint32_t y, uint32_t z, uint64_t flat, bool include_flat,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("x"), x));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("y"), y));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("z"), z));
  if (include_flat) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint64_field(&object, IREE_SV("flat"), flat));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_workload_json(
    const loom_target_compile_report_workload_t* workload,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workgroup_size")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_dimension3_json(
        workload->workgroup_size.x, workload->workgroup_size.y,
        workload->workgroup_size.z, workload->flat_workgroup_size,
        iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE),
        stream));
  }
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workgroup_count")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_dimension3_json(
        workload->workgroup_count.x, workload->workgroup_count.y,
        workload->workgroup_count.z, workload->dispatch_workgroup_count,
        iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT),
        stream));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dispatch_workitem_count"),
        workload->dispatch_workitem_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_json_write_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value) {
  if (value == 0) return iree_ok_status();
  return loom_json_object_write_uint64_field(object, key, value);
}

static iree_status_t
loom_target_compile_report_json_write_scaled_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value,
    uint64_t scale) {
  uint64_t scaled_value = 0;
  if (!loom_target_compile_report_checked_mul_u64(value, scale,
                                                  &scaled_value)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_json_write_nonzero_u64_field(object, key,
                                                                 scaled_value);
}

static iree_status_t loom_target_compile_report_format_memory_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  uint64_t read_byte_count = 0;
  const bool has_read_byte_count = loom_target_compile_report_checked_mul_u64(
      mix->memory_read_byte_count, scale, &read_byte_count);
  if (has_read_byte_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("read_bytes"), read_byte_count));
  }
  uint64_t write_byte_count = 0;
  const bool has_write_byte_count = loom_target_compile_report_checked_mul_u64(
      mix->memory_write_byte_count, scale, &write_byte_count);
  if (has_write_byte_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("write_bytes"), write_byte_count));
  }
  uint64_t total_byte_count = 0;
  if (has_read_byte_count && has_write_byte_count &&
      loom_target_compile_report_checked_add_u64(
          read_byte_count, write_byte_count, &total_byte_count)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("total_bytes"), total_byte_count));
  }

#define LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(field) \
  do {                                                                       \
    IREE_RETURN_IF_ERROR(                                                    \
        loom_target_compile_report_json_write_scaled_nonzero_u64_field(      \
            &object, IREE_SV(#field), mix->field, scale));                   \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      global_load_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      global_store_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      buffer_load_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      buffer_store_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      flat_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      local_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      scalar_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      private_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      generic_memory_count);
#undef LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_scaled_nonzero_u64_field(
          &object, IREE_SV("read_unknown_width_count"),
          mix->memory_read_unknown_width_count, scale));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_scaled_nonzero_u64_field(
          &object, IREE_SV("write_unknown_width_count"),
          mix->memory_write_unknown_width_count, scale));

  uint64_t byte_count = 0;
#define LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(field) \
  do {                                                                 \
    if (loom_target_compile_report_checked_mul_u64(mix->field, scale,  \
                                                   &byte_count)) {     \
      IREE_RETURN_IF_ERROR(                                            \
          loom_target_compile_report_json_write_nonzero_u64_field(     \
              &object, IREE_SV(#field), byte_count));                  \
    }                                                                  \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      global_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      global_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      buffer_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      buffer_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(flat_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      flat_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      local_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      local_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      scalar_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      scalar_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      private_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      private_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      unclassified_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      unclassified_write_byte_count);
#undef LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_operation_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
#define LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(field) \
  do {                                                                    \
    IREE_RETURN_IF_ERROR(                                                 \
        loom_target_compile_report_json_write_scaled_nonzero_u64_field(   \
            &object, IREE_SV(#field), mix->field, scale));                \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(scalar_alu_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(vector_alu_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(matrix_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(mfma_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(smfmac_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(wmma_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(swmmac_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(dot_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(atomic_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(branch_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(barrier_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(control_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(conversion_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(cache_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(
      register_move_count);
#undef LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (loom_target_compile_report_economics_has_operations(mix)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("operations")));
    loom_json_object_writer_t operations;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &operations));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&operations, IREE_SV("per_workitem")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_operation_economics_json(mix, 1,
                                                                   stream));
    if (iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&operations, IREE_SV("dispatch")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_operation_economics_json(
              mix, workload->dispatch_workitem_count, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&operations));
  }
  if (loom_target_compile_report_economics_has_memory(mix)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory")));
    loom_json_object_writer_t memory;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &memory));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&memory, IREE_SV("per_workitem_issued")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_memory_economics_json(mix, 1,
                                                                stream));
    if (iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&memory, IREE_SV("dispatch_issued")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_memory_economics_json(
              mix, workload->dispatch_workitem_count, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&memory));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_entry_json(
    const loom_target_compile_report_entry_t* row, iree_host_size_t row_index,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_function"), row->source_function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_snapshot"), row->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export"), row->target_export_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export_symbol"), row->target_export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("detail_flags"), row->detail_flags));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("planning")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_low_planning_json(
        &row->low_planning, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_node_count"), row->schedule_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scheduled_node_count"), row->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_dependency_count"),
      row->schedule_dependency_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_resource_use_count"),
      row->schedule_resource_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_hazard_gap_count"),
      row->schedule_hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_model_summary_count"),
      row->schedule_model_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_summary_count"),
      row->register_pressure_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_peak_live_units"),
      row->register_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_assignment_count"),
      row->allocation_assignment_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_spill_count"), row->allocation_spill_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_spill_plan_count"),
      row->allocation_spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_coalesced_copy_count"),
      row->allocation_coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_copy_count"),
      row->allocation_materialized_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_storage_count"),
      row->allocation_materialized_spill_storage_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_storage_bytes"),
      row->allocation_materialized_spill_storage_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_store_count"),
      row->allocation_materialized_spill_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_store_bytes"),
      row->allocation_materialized_spill_store_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_reload_count"),
      row->allocation_materialized_reload_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_reload_bytes"),
      row->allocation_materialized_reload_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_storage_lease_count"),
      row->allocation_storage_lease_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_storage_lease_instance_count"),
      row->allocation_storage_lease_instance_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_storage_release_action_count"),
      row->allocation_storage_release_action_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("move_causes")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause_counts_json(
      row->move_causes, mode, stream));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_plan")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_wait_plan_json(
        &row->wait_plan, stream));
  }
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workload")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_workload_json(
        &row->workload, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("instruction_count"), row->emitted_instruction_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_byte_count"), row->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_storage_byte_count"),
      row->emitted_code_storage_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_memory_bytes"), row->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_memory_bytes"), row->local_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("pressure_row_count"), row->pressure_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("pressure_origin_row_count"),
      row->pressure_origin_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("schedule_band_row_count"),
      row->schedule_band_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("schedule_band_summary_row_count"),
      row->schedule_band_summary_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("spill_row_count"), row->spill_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("allocation_high_water_row_count"),
      row->allocation_high_water_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("wait_counter_row_count"), row->wait_counter_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("wait_reason_summary_row_count"),
      row->wait_reason_summary_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("wait_action_row_count"), row->wait_action_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("target_capability_row_count"),
      row->target_capability_row_count));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_resources")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_resources_json(
            &row->target_resources, stream));
  }
  if (iree_any_bit_set(
          row->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("static_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->static_instruction_mix, stream));
  }
  if (iree_any_bit_set(
          row->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->dynamic_instruction_mix, stream));
  }
  if (loom_target_compile_report_has_economics(
          row->detail_flags, &row->dynamic_instruction_mix, &row->workload)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("economics")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_economics_json(
        &row->dynamic_instruction_mix, &row->workload, stream));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_entries_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->entry_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->entry_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        (const loom_target_compile_report_entry_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_format_entry_json(
          &rows[i], row_index, mode, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pressure_row_json(
    const loom_target_compile_report_pressure_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("peak_live_units"), row->peak_live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("peak_live_values"), row->peak_live_values));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("peak_point"), row->peak_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_block"), row->peak_block_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_operation"), row->peak_operation_name));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pressure_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->pressure_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->pressure_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_pressure_row_t* rows =
          (const loom_target_compile_report_pressure_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pressure_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pressure_origin_row_json(
    const loom_target_compile_report_pressure_origin_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("peak_point"), row->peak_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_block"), row->peak_block_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_operation"), row->peak_operation_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("sample_value"), row->sample_value_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("live_units"), row->live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("live_values"), row->live_values));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_pressure_origin_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->pressure_origin_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->pressure_origin_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_pressure_origin_row_t* rows =
          (const loom_target_compile_report_pressure_origin_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pressure_origin_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_schedule_band_row_json(
    const loom_target_compile_report_schedule_band_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("block"), row->block_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("first_packet_index"), row->first_packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("first_scheduled_ordinal"),
      row->first_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node_count"), row->node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("sample_value"), row->sample_value_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("static_instruction_mix")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
      &row->static_instruction_mix, stream));
  if (iree_all_bits_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->dynamic_instruction_mix, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_value_count"), row->result_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_unit_count"), row->result_unit_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_schedule_band_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->schedule_band_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->schedule_band_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_schedule_band_row_t* rows =
          (const loom_target_compile_report_schedule_band_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_schedule_band_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_row_json(
    const loom_target_compile_report_schedule_band_summary_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("block"), row->block_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("first_packet_index"), row->first_packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("band_count"), row->band_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("node_count"), row->node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("max_band_node_count"), row->max_band_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("sample_value"), row->sample_value_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("static_instruction_mix")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
      &row->static_instruction_mix, stream));
  if (iree_all_bits_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->dynamic_instruction_mix, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_value_count"), row->result_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_unit_count"), row->result_unit_count));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  (void)mode;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->schedule_band_summary_rows.count));
  if (report->schedule_band_summary_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->schedule_band_summary_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_schedule_band_summary_row_t* rows =
          (const loom_target_compile_report_schedule_band_summary_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_schedule_band_summary_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_spill_row_json(
    const loom_target_compile_report_spill_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      loom_target_compile_report_spill_row_kind_name(row->kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("value"), row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("assignment_index"), row->assignment_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("slot_index"), row->slot_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("slot_space"), row->slot_space));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_size"), row->byte_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_alignment"), row->byte_alignment));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("store_count"), row->store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("store_bytes"), row->store_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("reload_count"), row->reload_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("reload_bytes"), row->reload_bytes));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_spill_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->spill_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec = report->spill_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_spill_row_t* rows =
          (const loom_target_compile_report_spill_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_row_json(
            &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_failure_row_json(
    const loom_target_compile_report_allocation_failure_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("value"), row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("failure_code"), row->failure_code));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("blocking_kind"),
      loom_target_compile_report_allocation_failure_blocking_kind_name(
          row->blocking_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_block"), row->origin_block_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("start_point"), row->start_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("end_point"), row->end_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("required_unit_count"), row->required_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("budget_units"), row->budget_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("peak_live_units"), row->peak_live_units));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("location_kind"), row->location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("location_base"), row->location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("location_count"), row->location_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_assignment_index"),
      row->conflict_assignment_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("conflict_value"), row->conflict_value_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_start_point"), row->conflict_start_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_end_point"), row->conflict_end_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("conflict_location_kind"),
          row->conflict_location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_location_base"), row->conflict_location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("conflict_location_count"),
      row->conflict_location_count));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_failure_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->allocation_failure_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->allocation_failure_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_allocation_failure_row_t* rows =
          (const loom_target_compile_report_allocation_failure_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_allocation_failure_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_row_json(
    const loom_target_compile_report_allocation_high_water_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("value"), row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("assignment_index"), row->assignment_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("start_point"), row->start_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("end_point"), row->end_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("required_unit_count"), row->required_unit_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("location_kind"), row->location_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("location_base"), row->location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("location_count"), row->location_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("high_water_units"), row->high_water_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("lower_free_unit_count"), row->lower_free_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_free_run_count"), row->lower_free_run_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_largest_free_run_unit_count"),
      row->lower_largest_free_run_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("lower_pressure_releasable_free_unit_count"),
      row->lower_pressure_releasable_free_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_pressure_releasable_free_run_count"),
      row->lower_pressure_releasable_free_run_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_pressure_releasable_largest_free_run_unit_count"),
      row->lower_pressure_releasable_largest_free_run_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_assignment_blocker_count"),
      row->active_assignment_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_assignment_blocker_units"),
      row->active_assignment_blocker_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_storage_lease_blocker_count"),
      row->active_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_storage_lease_blocker_units"),
      row->active_storage_lease_blocker_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_pressure_storage_lease_blocker_count"),
      row->active_pressure_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_pressure_storage_lease_blocker_units"),
      row->active_pressure_storage_lease_blocker_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_fallback_storage_lease_blocker_count"),
      row->active_fallback_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_fallback_storage_lease_blocker_units"),
      row->active_fallback_storage_lease_blocker_units));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->allocation_high_water_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->allocation_high_water_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_allocation_high_water_row_t* rows =
          (const loom_target_compile_report_allocation_high_water_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_allocation_high_water_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_diagnostics_json(
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write(stream, options->diagnostic_json_objects));
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_target_compile_report_format_config_binding_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->config_binding_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_config_binding_row_t* rows =
        (const loom_target_compile_report_config_binding_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_config_binding_row_t* row = &rows[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: config_binding[%" PRIhsz "] key=%.*s value=%.*s\n",
          row_index, (int)row->key.size, row->key.data, (int)row->value.size,
          row->value.data));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_config_binding_row_json(
    const loom_target_compile_report_config_binding_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("key"), row->key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("value"), row->value));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_config_bindings_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->config_binding_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->config_binding_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_config_binding_row_t* rows =
        (const loom_target_compile_report_config_binding_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_config_binding_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_target_row_json(
    const loom_target_compile_report_source_low_target_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target_source"),
      loom_target_selection_source_name(row->target_source)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_symbol"), row->target_symbol_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_snapshot"), row->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  if (row->target_subgroup_size != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("target_subgroup_size"), row->target_subgroup_size));
  }
  if (row->target_source == LOOM_TARGET_SELECTION_SOURCE_INVOCATION) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("candidate_target_count"),
        row->candidate_target_count));
    if (row->candidate_target_count != 0) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_symbol"),
              row->candidate_target_symbol_name));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_bundle"),
              row->candidate_target_bundle_name));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_snapshot"),
              row->candidate_target_snapshot_name));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_config"),
              row->candidate_target_config_name));
      if (row->candidate_target_subgroup_size != 0) {
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &object, IREE_SV("candidate_target_subgroup_size"),
            row->candidate_target_subgroup_size));
      }
    }
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_target_rows_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_target_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_target_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_target_row_t* rows =
        (const loom_target_compile_report_source_low_target_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_target_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_selection_summary_row_json(
    const loom_target_compile_report_source_low_selection_summary_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("selection"),
      loom_target_compile_report_source_low_selection_name(
          row->selection_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("plan_key"), row->plan_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_key"), row->descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_semantic_tag"),
          row->descriptor_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("selected_op_count"), row->selected_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("emitted_low_op_count"), row->emitted_low_op_count));
  const bool has_dynamic_delta =
      row->unknown_dynamic_op_count != 0 ||
      row->dynamic_selected_op_count != row->selected_op_count ||
      row->dynamic_emitted_low_op_count != row->emitted_low_op_count;
  if (has_dynamic_delta) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("exact_dynamic_op_count"),
        row->exact_dynamic_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("unknown_dynamic_op_count"),
        row->unknown_dynamic_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dynamic_selected_op_count"),
        row->dynamic_selected_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dynamic_emitted_low_op_count"),
        row->dynamic_emitted_low_op_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_selection_summaries_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_selection_summaries.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_selection_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_selection_summary_t* rows =
        (const loom_target_compile_report_source_low_selection_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_selection_summary_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_source_low_row_json(
    const loom_target_compile_report_source_low_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("selection"),
      loom_target_compile_report_source_low_selection_name(
          row->selection_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_set_index"), row->rule_set_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_index"), row->rule_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u64_field(
      &object, IREE_SV("plan_id"), row->plan_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("plan_key"), row->plan_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_key"), row->descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_semantic_tag"),
          row->descriptor_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("emitted_low_op_count"), row->emitted_low_op_count));
  if (row->execution_count_plus_one !=
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
      row->execution_count_plus_one != 2) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint64_field(&object, IREE_SV("execution_count"),
                                            row->execution_count_plus_one - 1));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_transform_row_json(
    const loom_target_compile_report_source_low_transform_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("transform"), row->transform_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("outcome"), row->outcome));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("reason"), row->reason));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("candidate_value_count"), row->candidate_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("selected_value_count"), row->selected_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("removed_loop_carried_value_count"),
      row->removed_loop_carried_value_count));
  if (row->removed_loop_carried_payload_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("removed_loop_carried_payload_register_count"),
        row->removed_loop_carried_payload_register_count));
  }
  if (row->block_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("block_count"), row->block_count));
  }
  if (row->block_count != 0 || row->row_count != 0 || row->column_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("row_count"), row->row_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("column_count"), row->column_count));
  }
  if (row->workgroup_memory_byte_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("workgroup_memory_byte_count"),
        row->workgroup_memory_byte_count));
  }
  if (row->inserted_load_op_count != 0 || row->inserted_store_op_count != 0 ||
      row->inserted_barrier_op_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("inserted_load_op_count"),
        row->inserted_load_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("inserted_store_op_count"),
        row->inserted_store_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("inserted_barrier_op_count"),
        row->inserted_barrier_op_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_transforms_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_transform_rows.count));
  if (mode != LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    return loom_json_object_end(&object);
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_transform_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_transform_row_t* rows =
        (const loom_target_compile_report_source_low_transform_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_transform_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_memory_interval_json(
    const loom_target_compile_report_memory_interval_t* interval,
    loom_json_object_writer_t* object) {
  if (!loom_target_compile_report_memory_interval_has_range(interval)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("source_interval")));
  loom_json_object_writer_t interval_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(object->stream, &interval_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("begin_min_bytes"), interval->begin_min_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("begin_max_bytes"), interval->begin_max_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("end_min_bytes"), interval->end_min_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("end_max_bytes"), interval->end_max_bytes));
  if (iree_all_bits_set(
          interval->flags,
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &interval_object, IREE_SV("exact_length_bytes"),
        interval->exact_length_bytes));
  }
  return loom_json_object_end(&interval_object);
}

static iree_status_t
loom_target_compile_report_format_memory_interval_summary_json(
    const char* field_name,
    const loom_target_compile_report_memory_interval_summary_t* summary,
    loom_json_object_writer_t* object) {
  if (summary->packet_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, iree_make_cstring_view(field_name)));
  loom_json_object_writer_t summary_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &summary_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &summary_object, IREE_SV("packet_count"), summary->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &summary_object, IREE_SV("begin_min_bytes"),
      summary->envelope_begin_min_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &summary_object, IREE_SV("end_max_bytes"),
      summary->envelope_end_max_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &summary_object, IREE_SV("byte_count"), summary->envelope_byte_count));
  if (summary->exact_static_packet_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &summary_object, IREE_SV("exact_static_packet_count"),
        summary->exact_static_packet_count));
  }
  if (summary->exact_symbolic_packet_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &summary_object, IREE_SV("exact_symbolic_packet_count"),
        summary->exact_symbolic_packet_count));
  }
  if (summary->exact_static_packet_count +
          summary->exact_symbolic_packet_count ==
      summary->packet_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &summary_object, IREE_SV("unique_byte_count"),
        summary->unique_byte_count));
  }
  return loom_json_object_end(&summary_object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_storage_fields_json(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count, loom_json_object_writer_t* object) {
  const iree_host_size_t first_storage_field =
      loom_target_compile_report_first_non_empty_string_field(fields,
                                                              field_count);
  if (first_storage_field == field_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("storage")));
  loom_json_object_writer_t storage;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &storage));
  for (iree_host_size_t i = first_storage_field; i < field_count; ++i) {
    if (iree_string_view_is_empty(fields[i].value)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &storage, iree_make_cstring_view(fields[i].name), fields[i].value));
  }
  return loom_json_object_end(&storage);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_storage_json(
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_storage_fields(row, fields);
  return loom_target_compile_report_format_source_low_memory_storage_fields_json(
      fields, field_count, object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_strategy_storage_json(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_strategy_storage_fields(
          summary, fields);
  return loom_target_compile_report_format_source_low_memory_storage_fields_json(
      fields, field_count, object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_packet_storage_json(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
          summary, fields);
  return loom_target_compile_report_format_source_low_memory_storage_fields_json(
      fields, field_count, object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_row_json(
    const loom_target_compile_report_source_low_memory_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_root"), row->source_root_name));
  if (row->source_root_argument_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("source_root_argument_index"),
        row->source_root_argument_index));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("operation"), row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("strategy"), row->strategy_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("address_form"), row->address_form));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("dynamic_term_kind"), row->dynamic_term_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("fallback_reason"), row->fallback_reason));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("static_offset_bytes"), row->static_offset_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_bytes"), row->element_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vector_lanes"), row->vector_lane_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_read_byte_count"), row->issued_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_write_byte_count"),
      row->issued_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_read_unknown_width_count"),
      row->issued_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_write_unknown_width_count"),
      row->issued_write_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("dynamic_stride_bytes"), row->dynamic_stride_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vector_lane_stride_bytes"),
      row->vector_lane_stride_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("bank_stride_words"), row->bank_stride_words));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("bank_conflict_degree"), row->bank_conflict_degree));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("bank_conflict_kind"), row->bank_conflict_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_storage_json(
          row, &object));
  if (row->execution_count_plus_one !=
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
      row->execution_count_plus_one != 2) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint64_field(&object, IREE_SV("execution_count"),
                                            row->execution_count_plus_one - 1));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_memory_interval_json(
      &row->source_interval, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_dispatch_source_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_dispatch_memory_bytes_t dispatch_source = {0};
  if (!loom_target_compile_report_source_low_memory_dispatch_source_bytes(
          summary, workload, &dispatch_source)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("dispatch_source")));
  loom_json_object_writer_t dispatch_source_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(object->stream, &dispatch_source_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dispatch_source_object, IREE_SV("read_bytes"),
      dispatch_source.read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dispatch_source_object, IREE_SV("write_bytes"),
      dispatch_source.write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dispatch_source_object, IREE_SV("total_bytes"),
      dispatch_source.total_byte_count));
  return loom_json_object_end(&dispatch_source_object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_dispatch_issued_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_json_object_writer_t* object) {
  if (!loom_target_compile_report_source_low_memory_can_dispatch_scale(
          summary, workload)) {
    return iree_ok_status();
  }
  const uint64_t issued_read_unknown_width_count =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(summary)
          ? summary->dynamic_issued_read_unknown_width_count
          : summary->issued_read_unknown_width_count;
  const uint64_t issued_write_unknown_width_count =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(summary)
          ? summary->dynamic_issued_write_unknown_width_count
          : summary->issued_write_unknown_width_count;
  loom_target_compile_report_dispatch_memory_bytes_t dispatch_issued = {0};
  const bool has_dispatch_issued =
      loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
          summary, workload, &dispatch_issued);
  const bool has_unknown_width_count = issued_read_unknown_width_count != 0 ||
                                       issued_write_unknown_width_count != 0;
  if (!has_dispatch_issued && !has_unknown_width_count) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("dispatch_issued")));
  loom_json_object_writer_t dispatch_issued_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(object->stream, &dispatch_issued_object));
  if (has_dispatch_issued) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &dispatch_issued_object, IREE_SV("read_bytes"),
        dispatch_issued.read_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &dispatch_issued_object, IREE_SV("write_bytes"),
        dispatch_issued.write_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &dispatch_issued_object, IREE_SV("total_bytes"),
        dispatch_issued.total_byte_count));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_nonzero_u64_field(
      &dispatch_issued_object, IREE_SV("read_unknown_width_count"),
      issued_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_nonzero_u64_field(
      &dispatch_issued_object, IREE_SV("write_unknown_width_count"),
      issued_write_unknown_width_count));
  return loom_json_object_end(&dispatch_issued_object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_summary_fields_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("packet_count"), summary->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("load_packet_count"), summary->load_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("store_packet_count"), summary->store_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("scalar_packet_count"), summary->scalar_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("vector_packet_count"), summary->vector_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("source_lane_count"), summary->source_lane_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("source_byte_count"), summary->source_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("read_byte_count"), summary->read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("write_byte_count"), summary->write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_read_byte_count"),
      summary->issued_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_write_byte_count"),
      summary->issued_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_read_unknown_width_count"),
      summary->issued_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_write_unknown_width_count"),
      summary->issued_write_unknown_width_count));
  if (loom_target_compile_report_source_low_memory_should_print_dynamic(
          summary, workload)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("exact_dynamic_packet_count"),
        summary->exact_dynamic_packet_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("unknown_dynamic_packet_count"),
        summary->unknown_dynamic_packet_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_packet_count"),
        summary->dynamic_packet_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_source_byte_count"),
        summary->dynamic_source_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_read_byte_count"),
        summary->dynamic_read_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_write_byte_count"),
        summary->dynamic_write_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_read_byte_count"),
        summary->dynamic_issued_read_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_write_byte_count"),
        summary->dynamic_issued_write_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_read_unknown_width_count"),
        summary->dynamic_issued_read_unknown_width_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_write_unknown_width_count"),
        summary->dynamic_issued_write_unknown_width_count));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("contiguous_vector_packet_count"),
      summary->contiguous_vector_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("strided_vector_packet_count"),
      summary->strided_vector_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("unknown_stride_vector_packet_count"),
      summary->unknown_stride_vector_packet_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_dispatch_source_json(
          summary, workload, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_dispatch_issued_json(
          summary, workload, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_memory_interval_summary_json(
          "interval_envelope", &summary->interval_envelope, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_memory_interval_summary_json(
          "read_interval_envelope", &summary->read_interval_envelope, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_memory_interval_summary_json(
          "write_interval_envelope", &summary->write_interval_envelope,
          object));
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_root_summary_json(
    const loom_target_compile_report_source_low_memory_root_summary_t* row,
    const loom_target_compile_report_workload_t* workload,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_root"), row->source_root_name));
  if (row->source_root_argument_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("source_root_argument_index"),
        row->source_root_argument_index));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_summary_json(
    const loom_target_compile_report_source_low_memory_argument_summary_t* row,
    const loom_target_compile_report_workload_t* workload,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_root"), row->source_root_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_root_argument_index"),
      row->source_root_argument_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_packet_summary_json(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        row,
    const loom_target_compile_report_workload_t* workload,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_root"), row->source_root_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_root_argument_index"),
      row->source_root_argument_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("operation"), row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("strategy"), row->strategy_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("fallback_reason"), row->fallback_reason));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_argument_packet_storage_json(
          row, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_strategy_summary_json(
    const loom_target_compile_report_source_low_memory_strategy_summary_t* row,
    const loom_target_compile_report_workload_t* workload,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("operation"), row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("strategy"), row->strategy_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("fallback_reason"), row->fallback_reason));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_strategy_storage_json(
          row, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_summary_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  const loom_target_compile_report_source_low_memory_summary_t* summary =
      &report->source_low_memory_summary;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          summary, &report->workload, &object));
  if (report->source_low_memory_root_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("root_count"),
        report->source_low_memory_root_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("roots")));
    loom_json_array_writer_t array_0;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_0));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_root_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_root_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_root_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_0));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_root_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_0));
  }
  if (report->source_low_memory_argument_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("argument_count"),
        report->source_low_memory_argument_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("arguments")));
    loom_json_array_writer_t array_1;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_1));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_argument_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_argument_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_argument_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_1));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_argument_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_1));
  }
  if (report->source_low_memory_argument_packet_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("argument_packet_count"),
        report->source_low_memory_argument_packet_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("argument_packets")));
    loom_json_array_writer_t array_2;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_2));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_argument_packet_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
          rows =
              (const loom_target_compile_report_source_low_memory_argument_packet_summary_t*)
                  loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_2));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_argument_packet_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_2));
  }
  if (report->source_low_memory_strategy_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("strategy_count"),
        report->source_low_memory_strategy_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("strategies")));
    loom_json_array_writer_t array_3;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_3));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_strategy_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_strategy_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_strategy_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_3));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_strategy_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_3));
  }
  return loom_json_object_end(&object);
}

static bool loom_target_compile_report_has_report_economics(
    const loom_target_compile_report_t* report) {
  return loom_target_compile_report_has_economics(
             report->detail_flags, &report->dynamic_instruction_mix,
             &report->workload) ||
         report->source_low_memory_summary.packet_count != 0;
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_economics_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_report_economics_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  const bool has_dynamic_instruction_mix = iree_any_bit_set(
      report->detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX);
  const bool has_low_dynamic_operations =
      has_dynamic_instruction_mix &&
      loom_target_compile_report_economics_has_operations(
          &report->dynamic_instruction_mix);
  const bool has_low_dynamic_memory =
      has_dynamic_instruction_mix &&
      loom_target_compile_report_economics_has_memory(
          &report->dynamic_instruction_mix);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (has_low_dynamic_operations) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("operations")));
    loom_json_object_writer_t operations;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &operations));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&operations, IREE_SV("per_workitem")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_operation_economics_json(
            &report->dynamic_instruction_mix, 1, stream));
    if (iree_any_bit_set(
            report->workload.flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&operations, IREE_SV("dispatch")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_operation_economics_json(
              &report->dynamic_instruction_mix,
              report->workload.dispatch_workitem_count, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&operations));
  }
  if (has_low_dynamic_memory ||
      report->source_low_memory_summary.packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory")));
    loom_json_object_writer_t memory;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &memory));
    if (has_low_dynamic_memory) {
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
          &memory, IREE_SV("per_workitem_issued")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_memory_economics_json(
              &report->dynamic_instruction_mix, 1, stream));
      if (iree_any_bit_set(
              report->workload.flags,
              LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
        IREE_RETURN_IF_ERROR(
            loom_json_object_begin_field(&memory, IREE_SV("dispatch_issued")));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_memory_economics_json(
                &report->dynamic_instruction_mix,
                report->workload.dispatch_workitem_count, stream));
      }
    }
    if (report->source_low_memory_summary.packet_count != 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&memory, IREE_SV("source_low")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_memory_economics_json(
              &report->source_low_memory_summary, &report->workload, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&memory));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_source_low_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("selected_op_count"),
      report->source_low_selected_op_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("emitted_op_count"),
                                          report->source_low_emitted_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_rows.count));
  if (report->source_low_target_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_selections")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_target_rows_json(report,
                                                                      stream));
  }
  if (report->source_low_selection_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("selection_summaries")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_selection_summaries_json(
            report, stream));
  }
  if (report->source_low_transform_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("transforms")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_transforms_json(
            report, mode, stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("memory")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_json(report,
                                                                       stream));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array_0;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_0));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_row_t* rows =
          (const loom_target_compile_report_source_low_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_0));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_0));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory_rows")));
    loom_json_array_writer_t array_1;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_1));
    row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_row_t* rows =
          (const loom_target_compile_report_source_low_memory_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_1));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_1));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_math_row_json(
    const loom_target_compile_report_math_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("policy"), row->policy_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("constraint_key"), row->constraint_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("math_op"),
      loom_target_math_op_name((loom_target_math_op_t)row->math_op)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("lane_domain"),
      loom_target_math_lane_domain_name(
          (loom_target_math_lane_domain_t)row->lane_domain)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element_type"),
      loom_target_compile_report_scalar_type_name(
          (loom_scalar_type_t)row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("action"),
      loom_target_compile_report_math_action_name(row->action)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("recipe"),
      loom_target_math_recipe_name((loom_target_math_recipe_t)row->recipe)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_fastmath_flags"), row->source_fastmath_flags));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("recipe_fastmath_flags"), row->recipe_fastmath_flags));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("created_op_count"), row->created_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("erased_op_count"), row->erased_op_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_math_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("rewritten_op_count"),
      report->math_legalization_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("rejected_op_count"),
      report->math_legalization_rejected_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("missing_policy_op_count"),
      report->math_legalization_missing_policy_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("missing_recipe_op_count"),
      report->math_legalization_missing_recipe_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->math_legalization_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->math_legalization_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_math_row_t* rows =
          (const loom_target_compile_report_math_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_target_compile_report_format_math_row_json(
            &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_legalization_row_json(
    const loom_target_compile_report_legalization_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("legalizer"), row->legalizer_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("legalizer_strategy"),
      loom_target_compile_report_legalizer_strategy_name(
          row->legalizer_strategy)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("mode"),
      loom_target_compile_report_legalization_mode_name(row->mode)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("policy"),
      loom_target_compile_report_legalization_policy_name(row->policy)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("action"),
      loom_target_compile_report_legalization_action_name(row->action)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("legalization_outcome"),
      loom_target_compile_report_legalization_outcome_name(
          row->legalization_outcome)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("contract_outcome"),
      loom_target_compile_report_contract_outcome_name(row->contract_outcome)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("binding_index"), row->binding_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("case_index"), row->case_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_set_index"), row->rule_set_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_index"), row->rule_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("diagnostic_index"), row->diagnostic_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_key"), row->descriptor_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_rejection_bits"), row->source_rejection_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_rejection_detail"),
      row->source_rejection_detail));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("target_rejection_bits"), row->target_rejection_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("missing_feature_bits"), row->missing_feature_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("missing_fact_bits"), row->missing_fact_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("created_op_count"), row->created_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("erased_op_count"), row->erased_op_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_legalization_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("legal_op_count"),
      report->target_legalization_legal_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("rewritten_op_count"),
      report->target_legalization_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("target_rewritten_op_count"),
      report->target_legalization_target_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("reference_rewritten_op_count"),
      report->target_legalization_reference_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("deferred_op_count"),
      report->target_legalization_deferred_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("invalid_ir_op_count"),
      report->target_legalization_invalid_ir_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unsupported_op_count"),
      report->target_legalization_unsupported_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unhandled_op_count"),
      report->target_legalization_unhandled_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->target_legalization_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->target_legalization_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_legalization_row_t* rows =
          (const loom_target_compile_report_legalization_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_legalization_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_text(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_summary(report, options, builder));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_entry_rows(report, builder));
  }
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_move_causes(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_counter_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_reason_summary_rows(report,
                                                                   builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_action_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_capability_rows(report,
                                                                 builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_config_binding_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_rows(report, builder));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_origin_rows(
        report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_summary_rows(report,
                                                                     builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_spill_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_failure_rows(report,
                                                                  builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows(report,
                                                                     builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_math_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_target_rows(report,
                                                                 builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_selection_summary_rows(
            report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_transform_rows(report,
                                                                    builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_rows(report,
                                                                 builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_root_summaries(
            report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_argument_summaries(
            report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_argument_packet_summaries(
            report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_strategy_summaries(
            report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_legalization_rows(report, builder));
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_format_json(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("artifact_kind"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("status")));
  IREE_RETURN_IF_ERROR(loom_json_write_status_object(
      stream, report->status_code, iree_string_view_empty()));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("detail_flags"), report->detail_flags));
  if (report->config_binding_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("config_bindings")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_config_bindings_json(report, stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("module"), report->module_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), report->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_family"), report->target_family_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_key"), report->target_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_snapshot"), report->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export"), report->target_export_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export_symbol"),
          report->target_export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), report->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("lowered"), report->lowered_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("artifact_format"), report->artifact_format));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("artifact_size"), report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("entries")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_entries_json(
      report, options->mode, stream));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("planning")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_low_planning_json(
        &report->low_planning, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("schedule")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workload")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_workload_json(
        &report->workload, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("static_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &report->static_instruction_mix, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &report->dynamic_instruction_mix, stream));
  }
  if (loom_target_compile_report_has_report_economics(report)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("economics")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_report_economics_json(report,
                                                                stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("allocation")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("move_causes")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_causes_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_plan")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_wait_plan_json(
        &report->wait_plan, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_counter_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_counter_rows_json(
            report, options->mode, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("wait_reason_summary_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_reason_summary_rows_json(
            report, options->mode, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_action_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_action_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("target_capability_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_capability_rows_json(report,
                                                                      stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("emission")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_emission_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_memory_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_resources")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_resources_json(
            &report->target_resources, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("pressure_rows")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_rows_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("pressure_origin_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_origin_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("schedule_band_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("schedule_band_summary_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_summary_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("spill_rows")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_rows_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("allocation_failure_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_failure_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("allocation_high_water_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows_json(
            report, options->mode, stream));
  }
  if (options->diagnostic_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("diagnostic_count"), options->diagnostic_count));
    if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("diagnostics")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_diagnostics_json(options, stream));
    }
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("math_legalization")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_math_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("source_low")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_source_low_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_legalization")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_legalization_json(
        report, options->mode, stream));
  }
  return loom_json_object_end(&object);
}
