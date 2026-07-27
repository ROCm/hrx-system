// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/device_event.h"

#include <string.h>

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
  if (iree_status_is_ok(status)) {
    memset(out_capture->records, 0,
           record_capacity * sizeof(*out_capture->records));
  }
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
