// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sample-scoped HAL device event capture for check testbench execution.
//
// HAL device events are side effects of actual invocations rather than SSA
// values. This recorder provides a reusable sink that copies event envelopes
// and fixed-layout payload bytes so check expectations can match structured
// reports without scraping stderr.

#ifndef LOOM_TOOLING_TESTBENCH_DEVICE_EVENT_H_
#define LOOM_TOOLING_TESTBENCH_DEVICE_EVENT_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
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
  // Borrowed event records in capture order.
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_DEVICE_EVENT_H_
