// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_event.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static const char* iree_hal_device_event_type_string(
    iree_hal_device_event_type_t type) {
  switch (type) {
    case IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE:
      return "driver_failure";
    case IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT:
      return "asan_report";
    case IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT:
      return "ubsan_report";
    case IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT:
      return "tsan_report";
    case IREE_HAL_DEVICE_EVENT_TYPE_PRINTF:
      return "printf";
    case IREE_HAL_DEVICE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    case IREE_HAL_DEVICE_EVENT_TYPE_NONE:
      return "none";
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_event_severity_string(
    iree_hal_device_event_severity_t severity) {
  switch (severity) {
    case IREE_HAL_DEVICE_EVENT_SEVERITY_TRACE:
      return "trace";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_INFO:
      return "info";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING:
      return "warning";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR:
      return "error";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_FATAL:
      return "fatal";
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_asan_access_kind_string(
    iree_hal_device_asan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ:
      return "read";
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE:
      return "write";
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC:
      return "atomic";
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_ubsan_check_kind_string(
    iree_hal_device_ubsan_check_kind_t check_kind) {
  switch (check_kind) {
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_INTEGER_OVERFLOW:
      return "integer_overflow";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO:
      return "divide_by_zero";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ALIGNMENT:
      return "alignment";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT:
      return "float_nan_contract";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNREACHABLE:
      return "unreachable";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_tsan_check_kind_string(
    iree_hal_device_tsan_check_kind_t check_kind) {
  switch (check_kind) {
    case IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE:
      return "data_race";
    case IREE_HAL_DEVICE_TSAN_CHECK_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_tsan_memory_space_string(
    iree_hal_device_tsan_memory_space_t memory_space) {
  switch (memory_space) {
    case IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_GLOBAL:
      return "global";
    case IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP:
      return "workgroup";
    case IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_PRIVATE:
      return "private";
    case IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_tsan_access_kind_string(
    iree_hal_device_tsan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ:
      return "read";
    case IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE:
      return "write";
    case IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ_WRITE:
      return "read_write";
    case IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC:
      return "atomic";
    case IREE_HAL_DEVICE_TSAN_ACCESS_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static bool iree_hal_device_event_payload_has_prefix(
    iree_const_byte_span_t payload, iree_host_size_t prefix_length) {
  return payload.data && payload.data_length >= prefix_length;
}

static void iree_hal_device_event_sink_stderr_print_string_or_na(
    iree_string_view_t value) {
  if (!value.data || value.size == 0) {
    fprintf(stderr, "n/a");
    return;
  }
  fprintf(stderr, "%.*s", (int)value.size, value.data);
}

static void iree_hal_device_event_sink_stderr_print_ordinal_or_na(
    uint32_t value) {
  if (value == UINT32_MAX) {
    fprintf(stderr, "n/a");
    return;
  }
  fprintf(stderr, "%u", value);
}

static void iree_hal_device_event_sink_stderr_print_source(
    const iree_hal_device_event_source_t* source) {
  fprintf(stderr, "  device: driver=");
  iree_hal_device_event_sink_stderr_print_string_or_na(source->driver_id);
  fprintf(stderr, " device=");
  iree_hal_device_event_sink_stderr_print_string_or_na(source->device_id);
  fprintf(stderr, " physical_device=");
  iree_hal_device_event_sink_stderr_print_ordinal_or_na(
      source->physical_device_ordinal);
  fprintf(stderr, " queue=");
  iree_hal_device_event_sink_stderr_print_ordinal_or_na(source->queue_ordinal);
  fprintf(stderr,
          " executable=0x%016" PRIx64 " export=", source->executable_id);
  iree_hal_device_event_sink_stderr_print_ordinal_or_na(source->export_ordinal);
  fprintf(stderr, "\n");
}

static void iree_hal_device_event_sink_stderr_print_site(
    const iree_hal_device_event_site_t* site) {
  if (!site) return;
  if (!iree_string_view_is_empty(site->source_file)) {
    fprintf(stderr, "  source: ");
    iree_hal_device_event_sink_stderr_print_string_or_na(site->source_file);
    fprintf(stderr, ":%u:%u-%u:%u\n", site->start_line, site->start_column,
            site->end_line, site->end_column);
  }
  fprintf(stderr, "  site: id=0x%016" PRIx64, site->site_id);
  if (!iree_string_view_is_empty(site->function_name)) {
    fprintf(stderr, " function=");
    iree_hal_device_event_sink_stderr_print_string_or_na(site->function_name);
  }
  if (!iree_string_view_is_empty(site->operation_name)) {
    fprintf(stderr, " operation=");
    iree_hal_device_event_sink_stderr_print_string_or_na(site->operation_name);
  }
  if (!iree_const_byte_span_is_empty(site->producer_payload)) {
    fprintf(stderr, " payload_bytes=%" PRIhsz,
            site->producer_payload.data_length);
  }
  fprintf(stderr, "\n");
}

static void iree_hal_device_event_sink_stderr_print_asan(
    const iree_hal_device_asan_report_t* report,
    const iree_hal_device_event_site_t* site) {
  if (!site) {
    fprintf(stderr, "  site: id=0x%016" PRIx64 " unresolved\n",
            report->site_id);
  }
  fprintf(stderr,
          "  asan: %s access of %" PRIu64 " %s at 0x%016" PRIx64
          "\n"
          "  shadow: address=0x%016" PRIx64 " value=0x%016" PRIx64
          "\n"
          "  work: workgroup=(%u,%u,%u) workitem=(%u,%u,%u) "
          "dispatch=0x%016" PRIx64 "\n",
          iree_hal_device_asan_access_kind_string(report->access_kind),
          report->access_length, report->access_length == 1 ? "byte" : "bytes",
          report->fault_address, report->shadow_address, report->shadow_value,
          report->workgroup_id[0], report->workgroup_id[1],
          report->workgroup_id[2], report->workitem_id[0],
          report->workitem_id[1], report->workitem_id[2],
          report->source_dispatch_ptr);
}

static void iree_hal_device_event_sink_stderr_print_ubsan(
    const iree_hal_device_ubsan_report_t* report,
    const iree_hal_device_event_site_t* site) {
  if (!site) {
    fprintf(stderr, "  site: id=0x%016" PRIx64 " unresolved\n",
            report->site_id);
  }
  fprintf(stderr,
          "  ubsan: check=%s operand0=0x%016" PRIx64 " operand1=0x%016" PRIx64
          "\n"
          "  work: workgroup=(%u,%u,%u) workitem=(%u,%u,%u) "
          "dispatch=0x%016" PRIx64 "\n",
          iree_hal_device_ubsan_check_kind_string(report->check_kind),
          report->operand0, report->operand1, report->workgroup_id[0],
          report->workgroup_id[1], report->workgroup_id[2],
          report->workitem_id[0], report->workitem_id[1],
          report->workitem_id[2], report->source_dispatch_ptr);
}

static void iree_hal_device_event_sink_stderr_print_tsan(
    const iree_hal_device_tsan_report_t* report,
    const iree_hal_device_event_site_t* site) {
  if (!site) {
    fprintf(stderr, "  site: id=0x%016" PRIx64 " unresolved\n",
            report->current_site_id);
  }
  const bool prior_workitem_linear = iree_all_bits_set(
      report->flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR);
  const bool current_workitem_linear = iree_all_bits_set(
      report->flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR);
  fprintf(stderr,
          "  tsan: check=%s memory=%s address=0x%016" PRIx64
          " access_length=%u\n"
          "  access: current=%s prior=%s prior_site=0x%016" PRIx64
          "\n"
          "  shadow: address=0x%016" PRIx64 " value=0x%016" PRIx64 "\n",
          iree_hal_device_tsan_check_kind_string(report->check_kind),
          iree_hal_device_tsan_memory_space_string(report->memory_space),
          report->memory_address, report->access_length,
          iree_hal_device_tsan_access_kind_string(report->current_access_kind),
          iree_hal_device_tsan_access_kind_string(report->prior_access_kind),
          report->prior_site_id, report->shadow_address, report->shadow_value);
  if (current_workitem_linear) {
    fprintf(stderr, "  current: workgroup=(%u,%u,%u) workitem_linear=%u\n",
            report->current_workgroup_id[0], report->current_workgroup_id[1],
            report->current_workgroup_id[2], report->current_workitem_id[0]);
  } else {
    fprintf(stderr, "  current: workgroup=(%u,%u,%u) workitem=(%u,%u,%u)\n",
            report->current_workgroup_id[0], report->current_workgroup_id[1],
            report->current_workgroup_id[2], report->current_workitem_id[0],
            report->current_workitem_id[1], report->current_workitem_id[2]);
  }
  if (prior_workitem_linear) {
    fprintf(stderr,
            "  prior: workgroup=(%u,%u,%u) workitem_linear=%u "
            "dispatch=0x%016" PRIx64 "\n",
            report->prior_workgroup_id[0], report->prior_workgroup_id[1],
            report->prior_workgroup_id[2], report->prior_workitem_id[0],
            report->source_dispatch_ptr);
  } else {
    fprintf(stderr,
            "  prior: workgroup=(%u,%u,%u) workitem=(%u,%u,%u) "
            "dispatch=0x%016" PRIx64 "\n",
            report->prior_workgroup_id[0], report->prior_workgroup_id[1],
            report->prior_workgroup_id[2], report->prior_workitem_id[0],
            report->prior_workitem_id[1], report->prior_workitem_id[2],
            report->source_dispatch_ptr);
  }
}

static void iree_hal_device_event_sink_stderr_print_printf(
    const iree_hal_device_printf_event_t* event) {
  if (!iree_string_view_is_empty(event->text)) {
    fprintf(stderr, "  printf: ");
    iree_hal_device_event_sink_stderr_print_string_or_na(event->text);
    fprintf(stderr, "\n");
  } else {
    fprintf(stderr,
            "  printf: format_id=0x%016" PRIx64 " argument_bytes=%" PRIhsz "\n",
            event->format_id, event->arguments.data_length);
  }
}

static void iree_hal_device_event_sink_stderr_print_driver_failure(
    const iree_hal_device_driver_failure_event_t* event) {
  fprintf(stderr,
          "  driver_failure: status=%s(%d)"
          " backend_result=0x%016" PRIx64,
          iree_status_code_string(event->status_code), event->status_code,
          event->backend_result_code);
  if (!iree_string_view_is_empty(event->message)) {
    fprintf(stderr, " message=\"");
    iree_hal_device_event_sink_stderr_print_string_or_na(event->message);
    fprintf(stderr, "\"");
  }
  fprintf(stderr, "\n");
}

static void iree_hal_device_event_sink_stderr_callback(
    void* user_data, const iree_hal_device_event_t* event) {
  (void)user_data;
  fprintf(stderr,
          "IREE HAL device event: type=%s severity=%s sequence=%" PRIu64 "\n",
          iree_hal_device_event_type_string(event->type),
          iree_hal_device_event_severity_string(event->severity),
          event->sequence);
  iree_hal_device_event_sink_stderr_print_site(event->site);

  switch (event->type) {
    case IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_asan_report_t))) {
        iree_hal_device_asan_report_t report = {0};
        memcpy(&report, event->payload.data, sizeof(report));
        iree_hal_device_event_sink_stderr_print_asan(&report, event->site);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_ubsan_report_t))) {
        iree_hal_device_ubsan_report_t report = {0};
        memcpy(&report, event->payload.data, sizeof(report));
        iree_hal_device_event_sink_stderr_print_ubsan(&report, event->site);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_tsan_report_t))) {
        iree_hal_device_tsan_report_t report = {0};
        memcpy(&report, event->payload.data, sizeof(report));
        iree_hal_device_event_sink_stderr_print_tsan(&report, event->site);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_PRINTF:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_printf_event_t))) {
        iree_hal_device_printf_event_t printf_event = {0};
        memcpy(&printf_event, event->payload.data, sizeof(printf_event));
        iree_hal_device_event_sink_stderr_print_printf(&printf_event);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_driver_failure_event_t))) {
        iree_hal_device_driver_failure_event_t failure_event = {0};
        memcpy(&failure_event, event->payload.data, sizeof(failure_event));
        iree_hal_device_event_sink_stderr_print_driver_failure(&failure_event);
      }
      break;
    default:
      if (!iree_const_byte_span_is_empty(event->payload)) {
        fprintf(stderr, "  payload: bytes=%" PRIhsz "\n",
                event->payload.data_length);
      }
      break;
  }

  iree_hal_device_event_sink_stderr_print_source(&event->source);
  fprintf(stderr, "\n");
}

IREE_API_EXPORT iree_hal_device_event_sink_t
iree_hal_device_event_sink_stderr(void) {
  iree_hal_device_event_sink_t sink;
  sink.fn = iree_hal_device_event_sink_stderr_callback;
  sink.user_data = NULL;
  return sink;
}
