// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_EPOCH_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_EPOCH_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/buffer.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/service/text/media.h"

#ifdef __cplusplus
extern "C" {
#endif

// One scheduled known-input span presented to mixed-packet construction. The
// plan index identifies the source-visible text plan record; request offsets
// remain absolute so arbitrary epoch chunking can intersect image ranges.
typedef struct loom_serve_text_media_epoch_span_t {
  // Source-visible text plan record index.
  uint32_t plan_index;
  // Absolute first request input in this scheduled chunk.
  iree_host_size_t input_offset;
  // Number of contiguous known inputs in this scheduled chunk.
  iree_host_size_t input_count;
  // Borrowed projected request media, or NULL for an ordinary text span.
  const loom_serve_text_media_input_t* media;
} loom_serve_text_media_epoch_span_t;

// One projected row upload retained by its owning request through completion.
typedef struct loom_serve_text_media_epoch_feature_t {
  // Borrowed projected row payload.
  iree_const_byte_span_t source;
  // Byte offset of the row in the device mixed-input packet.
  uint32_t target_offset;
} loom_serve_text_media_epoch_feature_t;

// Exact host-side description of one media-bearing device epoch. Empty text
// epochs allocate nothing. The source writes packet_records before submission;
// projected row bytes remain borrowed from their request owner.
typedef struct loom_serve_text_media_epoch_t {
  // Host allocation policy owning features and VM buffers.
  iree_allocator_t host_allocator;
  // Owned source-visible projected-row plan and copied image metadata.
  iree_vm_buffer_t* plan;
  // Owned source-written fixed packet records.
  iree_vm_buffer_t* packet_records;
  // Stable writable mapping of packet_records.
  iree_byte_span_t packet_record_bytes;
  // Owned upload descriptors borrowing projected row payloads.
  loom_serve_text_media_epoch_feature_t* features;
  // Number of first-epoch projected-row records.
  uint32_t first_feature_count;
  // Number of continuation projected-row records.
  uint32_t next_feature_count;
  // Number of fixed packet records reserved for the first epoch.
  uint32_t first_packet_capacity;
  // Number of fixed packet records reserved for the continuation.
  uint32_t next_packet_capacity;
  // Total device packet bytes, including aligned projected row payloads.
  uint32_t device_length;
} loom_serve_text_media_epoch_t;

// Builds exact source plan and host record storage for all projected rows
// intersecting the scheduled spans. Empty intersections succeed without
// allocation. Packet capacities are the source-declared shape capacities, not
// merely the known IDs uploaded by native code, so device-generated inputs
// retain their ordinary model-specific slots.
iree_status_t loom_serve_text_media_epoch_initialize(
    iree_host_size_t first_span_count,
    const loom_serve_text_media_epoch_span_t* first_spans,
    iree_host_size_t first_packet_capacity, iree_host_size_t next_span_count,
    const loom_serve_text_media_epoch_span_t* next_spans,
    iree_host_size_t next_packet_capacity, iree_allocator_t host_allocator,
    loom_serve_text_media_epoch_t* out_epoch);

// Uploads source-written packet records and every projected row into target.
// The ordinary epoch metadata upload joins the same transfer transaction.
// Host payloads remain borrowed through the returned execution frontier.
iree_status_t loom_serve_text_media_epoch_upload(
    const loom_serve_text_media_epoch_t* epoch,
    loom_serve_execution_t* execution, iree_const_byte_span_t metadata,
    iree_hal_buffer_t* metadata_target, iree_hal_buffer_t* target,
    uint64_t* out_value);

// Releases exact host plan/record storage and upload descriptors. Projected
// row payloads remain owned by the request.
void loom_serve_text_media_epoch_deinitialize(
    loom_serve_text_media_epoch_t* epoch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_MEDIA_EPOCH_H_
