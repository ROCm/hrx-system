// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_EPOCH_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_EPOCH_H_

#include "loom/serve/service/text/media_input.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_serve_text_span_t;

// Source-visible LE u32 projected-row records, first epoch then continuation.
// Source maps each plan index and span-relative input into its device packet.
typedef enum loom_serve_text_media_epoch_field_e {
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_PLAN_INDEX = 0,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_SPAN_INPUT_OFFSET = 1,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET = 2,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH = 3,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_OFFSET = 4,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_LENGTH = 5,
  // Row within the complete image, independent of chunk and cohort ordinals.
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_INDEX = 6,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT = 7,
} loom_serve_text_media_epoch_field_t;

// Every logical input has one source-written record. Device feature payloads
// follow both epochs' complete capacities, including device-generated inputs.
enum {
  LOOM_SERVE_TEXT_MEDIA_PACKET_RECORD_BYTES = 32,
  LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT = 16,
};

// One media-bearing invocation. Text-only intersections allocate nothing.
// Preparation borrows results until enqueue; enqueue captures their rows and
// releases transfer descriptors. Source plan and host packet records remain
// owned until successful completion or actual terminal consumer retirement.
typedef struct loom_serve_text_media_epoch_t {
  // Allocation policy for temporary transfer descriptors.
  iree_allocator_t host_allocator;
  // Owned projected-row plan followed by copied source-private metadata.
  iree_vm_buffer_t* plan;
  // Owned source-written device packet records, never feature bytes.
  iree_vm_buffer_t* packet_records;
  // Stable writable packet mapping retained through transfer retirement.
  iree_byte_span_t packet_record_bytes;
  // Number of projected rows in the first epoch.
  uint32_t first_feature_count;
  // Number of projected rows in the continuation.
  uint32_t next_feature_count;
  // Source-declared first-epoch input capacity.
  uint32_t first_packet_capacity;
  // Source-declared continuation input capacity.
  uint32_t next_packet_capacity;
  // Exact device packet extent, including aligned gathered rows.
  uint32_t device_length;
  // Temporary enqueue state, borrowing result owners only during submit.
  struct {
    // One allocation containing copy operations followed by cohort pointers.
    iree_hal_transfer_operation_t* operations;
    // Coalesced device feature copies; source control uploads are independent.
    iree_host_size_t operation_count;
    // Borrowed producer results for intersecting images.
    const loom_serve_finite_result_t** results;
    // Number of populated producer references, not projected rows.
    iree_host_size_t result_count;
  } transfer;
} loom_serve_text_media_epoch_t;

// Consumes checked text partitions and projected media facts. The optional
// continuation uses source plan indices starting at
// LOOM_SERVE_TEXT_ROW_CAPACITY. Selects image ranges by binary search and
// copies only intersecting metadata; projected device rows remain borrowed
// until enqueue returns.
iree_status_t loom_serve_text_media_epoch_initialize(
    iree_host_size_t first_span_count,
    const struct loom_serve_text_span_t* first_spans,
    iree_host_size_t first_packet_capacity, iree_host_size_t next_span_count,
    const struct loom_serve_text_span_t* next_spans,
    iree_host_size_t next_packet_capacity, iree_allocator_t host_allocator,
    loom_serve_text_media_epoch_t* out_epoch);

// Imports producer readiness and gathers device rows into target.
// Source-written control records use the ordinary epoch upload plan. No host
// feature transfer or wait occurs. HAL captures device views before return.
// Host control storage stays owned by the invocation through successful
// completion or actual terminal retirement.
iree_status_t loom_serve_text_media_epoch_enqueue(
    loom_serve_text_media_epoch_t* epoch, loom_serve_execution_t* execution,
    iree_hal_buffer_t* target, uint64_t* out_value);

// Releases unsubmitted or retired metadata; accepts zero state.
void loom_serve_text_media_epoch_deinitialize(
    loom_serve_text_media_epoch_t* epoch);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_MEDIA_EPOCH_H_
