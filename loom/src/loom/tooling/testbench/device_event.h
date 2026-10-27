// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sample-scoped HAL device event capture for check testbench execution.
//
// HAL device events are side effects of kernel launches rather than SSA
// values. This recorder provides a reusable sink that copies event envelopes
// and their borrowed payload storage so check expectations can match structured
// reports without scraping stderr.

#ifndef LOOM_TOOLING_TESTBENCH_DEVICE_EVENT_H_
#define LOOM_TOOLING_TESTBENCH_DEVICE_EVENT_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Default maximum number of retained device events per checked sample.
  LOOM_TESTBENCH_DEVICE_EVENT_DEFAULT_CAPACITY = 256,
  // Maximum copied byte length for each event string field.
  LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY = IREE_MAX_PATH,
  // Maximum copied byte length for event payloads and implementation payloads.
  LOOM_TESTBENCH_DEVICE_EVENT_PAYLOAD_CAPACITY = 512,
};

typedef struct loom_testbench_device_event_record_t {
  // Copied event envelope with string and byte spans rebound to record-owned
  // storage.
  iree_hal_device_event_t event;
  // Copied site metadata when |event.site| is non-NULL.
  iree_hal_device_event_site_t site;
  // True when |site| contains event-site metadata.
  bool has_site;
  // Inline storage for |event.source.device_id|.
  char source_device_id_storage[LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY];
  // Inline storage for |event.source.driver_id|.
  char source_driver_id_storage[LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY];
  // Inline storage for |site.source_file|.
  char site_source_file_storage[LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY];
  // Inline storage for |site.function_name|.
  char site_function_name_storage[LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY];
  // Inline storage for |site.operation_name|.
  char site_operation_name_storage[LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY];
  // Inline storage for |event.payload|. Preserves alignment for fixed-layout
  // typed HAL event payloads retained past the sink callback.
  iree_alignas(iree_max_align_t) uint8_t
      payload_storage[LOOM_TESTBENCH_DEVICE_EVENT_PAYLOAD_CAPACITY];
  // Inline storage for a driver failure message or device printf text.
  char payload_string_storage[LOOM_TESTBENCH_DEVICE_EVENT_STRING_CAPACITY];
  // Inline storage for encoded device printf arguments.
  uint8_t payload_data_storage[LOOM_TESTBENCH_DEVICE_EVENT_PAYLOAD_CAPACITY];
  // Inline storage for |event.implementation_payload|. Preserves alignment for
  // backend-native typed payloads retained past the sink callback.
  iree_alignas(iree_max_align_t) uint8_t implementation_payload_storage
      [LOOM_TESTBENCH_DEVICE_EVENT_PAYLOAD_CAPACITY];
  // Inline storage for |site.producer_payload|. Preserves alignment for
  // producer-specific typed payloads retained past the sink callback.
  iree_alignas(iree_max_align_t) uint8_t site_producer_payload_storage
      [LOOM_TESTBENCH_DEVICE_EVENT_PAYLOAD_CAPACITY];
} loom_testbench_device_event_record_t;

typedef struct loom_testbench_device_event_list_t {
  // Event records in capture order. Ownership is defined by the container
  // exposing the list.
  const loom_testbench_device_event_record_t* records;
  // Number of captured entries in |records|.
  iree_host_size_t count;
  // Number of events dropped because capture storage was exhausted or copying
  // failed inside the sink callback.
  iree_host_size_t dropped_count;
} loom_testbench_device_event_list_t;

typedef struct loom_testbench_device_event_capture_t {
  // Host allocator owning |records| and per-record retained storage.
  iree_allocator_t host_allocator;
  // Mutex protecting writes from HAL callback threads and testbench resets.
  iree_slim_mutex_t mutex;
  // Event record storage.
  loom_testbench_device_event_record_t* records;
  // Maximum number of records retained per sample.
  iree_host_size_t record_capacity;
  // Number of records captured since the last reset.
  iree_host_size_t record_count;
  // Number of events dropped since the last reset.
  iree_host_size_t dropped_count;
  // True when |mutex| has been initialized.
  bool mutex_initialized;
} loom_testbench_device_event_capture_t;

// Owned immutable copy of one completed capture interval.
typedef struct loom_testbench_device_event_snapshot_t {
  // Host allocator owning |records|.
  iree_allocator_t host_allocator;
  // Owned records backing |events|.
  loom_testbench_device_event_record_t* records;
  // Snapshot event list with spans rebound into |records|.
  loom_testbench_device_event_list_t events;
} loom_testbench_device_event_snapshot_t;

// Initializes event capture with storage for |record_capacity| events.
iree_status_t loom_testbench_device_event_capture_initialize(
    iree_host_size_t record_capacity, iree_allocator_t host_allocator,
    loom_testbench_device_event_capture_t* out_capture);

// Releases all storage owned by |capture|.
void loom_testbench_device_event_capture_deinitialize(
    loom_testbench_device_event_capture_t* capture);

// Clears captured events while retaining the record array.
void loom_testbench_device_event_capture_reset(
    loom_testbench_device_event_capture_t* capture);

// Returns a sink that appends events to |capture|.
iree_hal_device_event_sink_t loom_testbench_device_event_capture_sink(
    loom_testbench_device_event_capture_t* capture);

// Returns the current captured event list.
void loom_testbench_device_event_capture_events(
    loom_testbench_device_event_capture_t* capture,
    loom_testbench_device_event_list_t* out_events);

// Prints captured events and any dropped-event count to stderr. Used when a
// terminal execution failure prevents the normal structured result report.
void loom_testbench_device_event_capture_dump(
    loom_testbench_device_event_capture_t* capture);

// Copies the current capture interval into an owned immutable snapshot.
// |out_snapshot| must be empty and remains empty on failure.
iree_status_t loom_testbench_device_event_capture_snapshot(
    loom_testbench_device_event_capture_t* capture,
    iree_allocator_t host_allocator,
    loom_testbench_device_event_snapshot_t* out_snapshot);

// Releases all storage owned by |snapshot|.
void loom_testbench_device_event_snapshot_deinitialize(
    loom_testbench_device_event_snapshot_t* snapshot);

// Counts error-severity events that no positive expectation consumed.
iree_host_size_t loom_testbench_device_event_unhandled_error_count(
    const loom_testbench_device_event_list_t* events,
    const uint8_t* expected_events);

// Writes the device-event failure report shared by case and scenario results.
iree_status_t loom_testbench_device_event_failure_write_json(
    const loom_testbench_device_event_list_t* events,
    const uint8_t* expected_events, iree_host_size_t unhandled_error_count,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_DEVICE_EVENT_H_
