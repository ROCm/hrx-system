// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/device_event.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/util/json.h"

static bool loom_testbench_device_event_copy_string(
    iree_string_view_t source, char* storage, iree_host_size_t storage_capacity,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return true;
  }
  if (source.size > storage_capacity) {
    return false;
  }
  memcpy(storage, source.data, source.size);
  *out_target = iree_make_string_view(storage, source.size);
  return true;
}

static bool loom_testbench_device_event_copy_bytes(
    iree_const_byte_span_t source, uint8_t* storage,
    iree_host_size_t storage_capacity, iree_const_byte_span_t* out_target) {
  *out_target = iree_const_byte_span_empty();
  if (iree_const_byte_span_is_empty(source)) {
    return true;
  }
  if (source.data_length > storage_capacity) {
    return false;
  }
  memcpy(storage, source.data, source.data_length);
  *out_target = iree_make_const_byte_span(storage, source.data_length);
  return true;
}

static bool loom_testbench_device_event_record_copy(
    const iree_hal_device_event_t* source,
    loom_testbench_device_event_record_t* target) {
  memset(target, 0, sizeof(*target));
  target->event = *source;
  if (!loom_testbench_device_event_copy_string(
          source->source.device_id, target->source_device_id_storage,
          sizeof(target->source_device_id_storage),
          &target->event.source.device_id)) {
    return false;
  }
  if (!loom_testbench_device_event_copy_string(
          source->source.driver_id, target->source_driver_id_storage,
          sizeof(target->source_driver_id_storage),
          &target->event.source.driver_id)) {
    return false;
  }
  if (!loom_testbench_device_event_copy_bytes(
          source->payload, target->payload_storage,
          sizeof(target->payload_storage), &target->event.payload)) {
    return false;
  }
  // Standard payloads can themselves reference callback-scoped storage.
  // Preserve any extension bytes while replacing their borrowed spans.
  if (source->type == IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE &&
      source->payload.data_length >=
          sizeof(iree_hal_device_driver_failure_event_t)) {
    iree_hal_device_driver_failure_event_t failure;
    memcpy(&failure, target->payload_storage, sizeof(failure));
    if (!loom_testbench_device_event_copy_string(
            failure.message, target->payload_string_storage,
            sizeof(target->payload_string_storage), &failure.message)) {
      return false;
    }
    memcpy(target->payload_storage, &failure, sizeof(failure));
  } else if (source->type == IREE_HAL_DEVICE_EVENT_TYPE_PRINTF &&
             source->payload.data_length >=
                 sizeof(iree_hal_device_printf_event_t)) {
    iree_hal_device_printf_event_t printf_event;
    memcpy(&printf_event, target->payload_storage, sizeof(printf_event));
    if (!loom_testbench_device_event_copy_string(
            printf_event.text, target->payload_string_storage,
            sizeof(target->payload_string_storage), &printf_event.text) ||
        !loom_testbench_device_event_copy_bytes(
            printf_event.arguments, target->payload_data_storage,
            sizeof(target->payload_data_storage), &printf_event.arguments)) {
      return false;
    }
    memcpy(target->payload_storage, &printf_event, sizeof(printf_event));
  }
  if (!loom_testbench_device_event_copy_bytes(
          source->implementation_payload,
          target->implementation_payload_storage,
          sizeof(target->implementation_payload_storage),
          &target->event.implementation_payload)) {
    return false;
  }

  target->has_site = source->site != NULL;
  target->event.site = NULL;
  if (!source->site) {
    return true;
  }
  target->site = *source->site;
  if (!loom_testbench_device_event_copy_string(
          source->site->source_file, target->site_source_file_storage,
          sizeof(target->site_source_file_storage),
          &target->site.source_file)) {
    return false;
  }
  if (!loom_testbench_device_event_copy_string(
          source->site->function_name, target->site_function_name_storage,
          sizeof(target->site_function_name_storage),
          &target->site.function_name)) {
    return false;
  }
  if (!loom_testbench_device_event_copy_string(
          source->site->operation_name, target->site_operation_name_storage,
          sizeof(target->site_operation_name_storage),
          &target->site.operation_name)) {
    return false;
  }
  if (!loom_testbench_device_event_copy_bytes(
          source->site->producer_payload, target->site_producer_payload_storage,
          sizeof(target->site_producer_payload_storage),
          &target->site.producer_payload)) {
    return false;
  }
  target->event.site = &target->site;
  return true;
}

static iree_string_view_t loom_testbench_device_event_rebind_string(
    iree_string_view_t source, const char* target_storage) {
  return iree_string_view_is_empty(source)
             ? iree_string_view_empty()
             : iree_make_string_view(target_storage, source.size);
}

static iree_const_byte_span_t loom_testbench_device_event_rebind_bytes(
    iree_const_byte_span_t source, const uint8_t* target_storage) {
  return iree_const_byte_span_is_empty(source)
             ? iree_const_byte_span_empty()
             : iree_make_const_byte_span(target_storage, source.data_length);
}

static void loom_testbench_device_event_record_clone(
    const loom_testbench_device_event_record_t* source,
    loom_testbench_device_event_record_t* target) {
  *target = *source;
  target->event.source.device_id = loom_testbench_device_event_rebind_string(
      source->event.source.device_id, target->source_device_id_storage);
  target->event.source.driver_id = loom_testbench_device_event_rebind_string(
      source->event.source.driver_id, target->source_driver_id_storage);
  target->event.payload = loom_testbench_device_event_rebind_bytes(
      source->event.payload, target->payload_storage);
  if (source->event.type == IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE &&
      source->event.payload.data_length >=
          sizeof(iree_hal_device_driver_failure_event_t)) {
    iree_hal_device_driver_failure_event_t failure;
    memcpy(&failure, target->payload_storage, sizeof(failure));
    failure.message = loom_testbench_device_event_rebind_string(
        failure.message, target->payload_string_storage);
    memcpy(target->payload_storage, &failure, sizeof(failure));
  } else if (source->event.type == IREE_HAL_DEVICE_EVENT_TYPE_PRINTF &&
             source->event.payload.data_length >=
                 sizeof(iree_hal_device_printf_event_t)) {
    iree_hal_device_printf_event_t printf_event;
    memcpy(&printf_event, target->payload_storage, sizeof(printf_event));
    printf_event.text = loom_testbench_device_event_rebind_string(
        printf_event.text, target->payload_string_storage);
    printf_event.arguments = loom_testbench_device_event_rebind_bytes(
        printf_event.arguments, target->payload_data_storage);
    memcpy(target->payload_storage, &printf_event, sizeof(printf_event));
  }
  target->event.implementation_payload =
      loom_testbench_device_event_rebind_bytes(
          source->event.implementation_payload,
          target->implementation_payload_storage);
  target->event.site = target->has_site ? &target->site : NULL;
  if (!target->has_site) {
    return;
  }
  target->site.source_file = loom_testbench_device_event_rebind_string(
      source->site.source_file, target->site_source_file_storage);
  target->site.function_name = loom_testbench_device_event_rebind_string(
      source->site.function_name, target->site_function_name_storage);
  target->site.operation_name = loom_testbench_device_event_rebind_string(
      source->site.operation_name, target->site_operation_name_storage);
  target->site.producer_payload = loom_testbench_device_event_rebind_bytes(
      source->site.producer_payload, target->site_producer_payload_storage);
}

static void loom_testbench_device_event_capture_callback(
    void* user_data, const iree_hal_device_event_t* event) {
  loom_testbench_device_event_capture_t* capture =
      (loom_testbench_device_event_capture_t*)user_data;
  iree_slim_mutex_lock(&capture->mutex);
  if (capture->record_count >= capture->record_capacity) {
    ++capture->dropped_count;
  } else if (loom_testbench_device_event_record_copy(
                 event, &capture->records[capture->record_count])) {
    ++capture->record_count;
  } else {
    ++capture->dropped_count;
  }
  iree_slim_mutex_unlock(&capture->mutex);
}

iree_status_t loom_testbench_device_event_capture_initialize(
    iree_host_size_t record_capacity, iree_allocator_t host_allocator,
    loom_testbench_device_event_capture_t* out_capture) {
  *out_capture = (loom_testbench_device_event_capture_t){0};
  if (record_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device event capture capacity must be positive");
  }
  out_capture->host_allocator = iree_allocator_is_null(host_allocator)
                                    ? iree_allocator_system()
                                    : host_allocator;
  iree_slim_mutex_initialize(&out_capture->mutex);
  out_capture->mutex_initialized = true;
  out_capture->record_capacity = record_capacity;
  iree_status_t status = iree_allocator_malloc_array(
      out_capture->host_allocator, record_capacity,
      sizeof(*out_capture->records), (void**)&out_capture->records);
  if (!iree_status_is_ok(status)) {
    loom_testbench_device_event_capture_deinitialize(out_capture);
  }
  return status;
}

void loom_testbench_device_event_capture_deinitialize(
    loom_testbench_device_event_capture_t* capture) {
  if (capture == NULL) {
    return;
  }
  iree_allocator_free(capture->host_allocator, capture->records);
  if (capture->mutex_initialized) {
    iree_slim_mutex_deinitialize(&capture->mutex);
  }
  *capture = (loom_testbench_device_event_capture_t){0};
}

void loom_testbench_device_event_capture_reset(
    loom_testbench_device_event_capture_t* capture) {
  iree_slim_mutex_lock(&capture->mutex);
  capture->record_count = 0;
  capture->dropped_count = 0;
  iree_slim_mutex_unlock(&capture->mutex);
}

iree_hal_device_event_sink_t loom_testbench_device_event_capture_sink(
    loom_testbench_device_event_capture_t* capture) {
  iree_hal_device_event_sink_t sink = {
      .fn = loom_testbench_device_event_capture_callback,
      .user_data = capture,
  };
  return sink;
}

void loom_testbench_device_event_capture_events(
    loom_testbench_device_event_capture_t* capture,
    loom_testbench_device_event_list_t* out_events) {
  iree_slim_mutex_lock(&capture->mutex);
  *out_events = (loom_testbench_device_event_list_t){
      .records = capture->records,
      .count = capture->record_count,
      .dropped_count = capture->dropped_count,
  };
  iree_slim_mutex_unlock(&capture->mutex);
}

void loom_testbench_device_event_capture_dump(
    loom_testbench_device_event_capture_t* capture) {
  iree_slim_mutex_lock(&capture->mutex);
  const iree_hal_device_event_sink_t sink = iree_hal_device_event_sink_stderr();
  for (iree_host_size_t i = 0; i < capture->record_count; ++i) {
    iree_hal_device_event_sink_publish(sink, &capture->records[i].event);
  }
  if (capture->dropped_count != 0) {
    fprintf(stderr,
            "device event capture: dropped %" PRIhsz
            " events exceeding capture capacity\n",
            capture->dropped_count);
  }
  iree_slim_mutex_unlock(&capture->mutex);
}

iree_status_t loom_testbench_device_event_capture_snapshot(
    loom_testbench_device_event_capture_t* capture,
    iree_allocator_t host_allocator,
    loom_testbench_device_event_snapshot_t* out_snapshot) {
  IREE_ASSERT(iree_allocator_is_null(out_snapshot->host_allocator) &&
              out_snapshot->records == NULL &&
              out_snapshot->events.records == NULL &&
              out_snapshot->events.count == 0 &&
              out_snapshot->events.dropped_count == 0);
  const iree_allocator_t resolved_allocator =
      iree_allocator_is_null(host_allocator) ? iree_allocator_system()
                                             : host_allocator;
  iree_slim_mutex_lock(&capture->mutex);
  loom_testbench_device_event_record_t* records = NULL;
  iree_status_t status = iree_ok_status();
  if (capture->record_count != 0) {
    status =
        iree_allocator_malloc_array(resolved_allocator, capture->record_count,
                                    sizeof(*records), (void**)&records);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < capture->record_count; ++i) {
      loom_testbench_device_event_record_clone(&capture->records[i],
                                               &records[i]);
    }
    *out_snapshot = (loom_testbench_device_event_snapshot_t){
        .host_allocator = resolved_allocator,
        .records = records,
        .events =
            {
                .records = records,
                .count = capture->record_count,
                .dropped_count = capture->dropped_count,
            },
    };
  }
  iree_slim_mutex_unlock(&capture->mutex);
  return status;
}

void loom_testbench_device_event_snapshot_deinitialize(
    loom_testbench_device_event_snapshot_t* snapshot) {
  if (snapshot == NULL) {
    return;
  }
  iree_allocator_free(snapshot->host_allocator, snapshot->records);
  *snapshot = (loom_testbench_device_event_snapshot_t){0};
}

iree_host_size_t loom_testbench_device_event_unhandled_error_count(
    const loom_testbench_device_event_list_t* events,
    const uint8_t* expected_events) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < events->count; ++i) {
    if (events->records[i].event.severity >=
            IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR &&
        (expected_events == NULL || !expected_events[i])) {
      ++count;
    }
  }
  return count;
}

static const char* loom_testbench_device_event_type_name(
    iree_hal_device_event_type_t type) {
  switch (type) {
    case IREE_HAL_DEVICE_EVENT_TYPE_NONE:
      return "none";
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
    default:
      return "unknown";
  }
}

static const char* loom_testbench_device_event_severity_name(
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

static iree_status_t loom_testbench_device_event_site_write_json(
    const iree_hal_device_event_site_t* site, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("site_id"), site->site_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("source_file"), site->source_file));
  if (site->start_line != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("start_line"), site->start_line));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("start_column"), site->start_column));
  }
  if (site->end_line != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("end_line"), site->end_line));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("end_column"), site->end_column));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("function"), site->function_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("operation"), site->operation_name));
  return loom_json_object_end(&object);
}

static iree_status_t loom_testbench_device_event_write_json(
    const iree_hal_device_event_t* event, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      iree_make_cstring_view(
          loom_testbench_device_event_type_name(event->type))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("severity"),
      iree_make_cstring_view(
          loom_testbench_device_event_severity_name(event->severity))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("driver"), event->source.driver_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("device"), event->source.device_id));
  if (event->site != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("site")));
    IREE_RETURN_IF_ERROR(
        loom_testbench_device_event_site_write_json(event->site, stream));
  }
  return loom_json_object_end(&object);
}

iree_status_t loom_testbench_device_event_failure_write_json(
    const loom_testbench_device_event_list_t* events,
    const uint8_t* expected_events, iree_host_size_t unhandled_error_count,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("captured_count"), events->count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("dropped_count"), events->dropped_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("unhandled_error_count"), unhandled_error_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("unhandled_errors")));
  loom_json_array_writer_t unhandled_errors;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &unhandled_errors));
  for (iree_host_size_t i = 0; i < events->count; ++i) {
    const iree_hal_device_event_t* event = &events->records[i].event;
    if (event->severity < IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR ||
        (expected_events != NULL && expected_events[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&unhandled_errors));
    IREE_RETURN_IF_ERROR(loom_testbench_device_event_write_json(event, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&unhandled_errors));
  return loom_json_object_end(&object);
}
