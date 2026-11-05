// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/snapshot.h"

#include "iree/base/internal/atomics.h"

struct loom_serve_snapshot_t {
  // Caller ownership and queue-captured views independently retain the payload.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning the header and packed payload.
  iree_allocator_t host_allocator;
  // Number of logical bytes in data.
  iree_host_size_t length;
  // Concatenated retained ranges, independent of physical placement.
  uint8_t data[];
};

static void loom_serve_snapshot_release_view(void* user_data,
                                             iree_hal_buffer_t* buffer) {
  loom_serve_snapshot_release(user_data);
}

static iree_status_t loom_serve_snapshot_transfer(
    loom_serve_snapshot_t* snapshot, iree_host_size_t buffer_count,
    const loom_serve_snapshot_buffer_t* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges,
    iree_hal_transfer_operation_type_t type) {
  iree_hal_buffer_t** views = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      snapshot->host_allocator, buffer_count, sizeof(*views), (void**)&views));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < buffer_count && iree_status_is_ok(status);
       ++i) {
    if (buffers[i].buffer) {
      status = iree_hal_subspan_buffer_create_with_callback(
          buffers[i].buffer, iree_hal_buffer_byte_offset(buffers[i].buffer),
          iree_hal_buffer_byte_length(buffers[i].buffer),
          (iree_hal_buffer_release_callback_t){loom_serve_snapshot_release_view,
                                               snapshot},
          snapshot->host_allocator, &views[i]);
      if (iree_status_is_ok(status)) {
        iree_atomic_ref_count_inc(&snapshot->ref_count);
      }
    }
  }
  uint8_t* data = snapshot->data;
  iree_hal_transfer_operation_t operations[64];
  iree_host_size_t operation_count = 0;
  uint64_t completion = 0;
  for (iree_host_size_t i = 0; i < range_count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_snapshot_range_t* range = &ranges[i];
    loom_serve_execution_t* execution = buffers[range->buffer_index].execution;
    if (type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
      operations[operation_count++] = (iree_hal_transfer_operation_t){
          .type = type,
          .upload = {.source = data,
                     .target_buffer = views[range->buffer_index],
                     .target_offset = range->offset,
                     .length = range->length}};
    } else {
      operations[operation_count++] = (iree_hal_transfer_operation_t){
          .type = type,
          .download = {.source_buffer = views[range->buffer_index],
                       .source_offset = range->offset,
                       .target = data,
                       .length = range->length}};
    }
    data += range->length;
    if (operation_count == IREE_ARRAYSIZE(operations) || i + 1 == range_count ||
        buffers[ranges[i + 1].buffer_index].execution != execution) {
      status = loom_serve_execution_transfer(execution, operation_count,
                                             operations, &completion);
      operation_count = 0;
    }
  }
  for (iree_host_size_t i = 0; i < buffer_count; ++i) {
    iree_hal_buffer_release(views[i]);
  }
  iree_allocator_free(snapshot->host_allocator, views);
  return status;
}

iree_status_t loom_serve_snapshot_capture(
    iree_host_size_t buffer_count, const loom_serve_snapshot_buffer_t* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges,
    iree_allocator_t host_allocator, loom_serve_snapshot_t** out_snapshot) {
  *out_snapshot = NULL;
  iree_host_size_t length = 0;
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    if (ranges[i].length >
        IREE_HOST_SIZE_MAX - sizeof(loom_serve_snapshot_t) - length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "snapshot exceeds host address space");
    }
    length += ranges[i].length;
  }
  loom_serve_snapshot_t* snapshot = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, sizeof(*snapshot) + length, (void**)&snapshot));
  iree_atomic_ref_count_init(&snapshot->ref_count);
  snapshot->host_allocator = host_allocator;
  snapshot->length = length;
  iree_status_t status = loom_serve_snapshot_transfer(
      snapshot, buffer_count, buffers, range_count, ranges,
      IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD);
  if (iree_status_is_ok(status)) {
    *out_snapshot = snapshot;
  } else {
    loom_serve_snapshot_release(snapshot);
  }
  return status;
}

iree_status_t loom_serve_snapshot_restore(
    loom_serve_snapshot_t* snapshot, iree_host_size_t buffer_count,
    const loom_serve_snapshot_buffer_t* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges) {
  return loom_serve_snapshot_transfer(snapshot, buffer_count, buffers,
                                      range_count, ranges,
                                      IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD);
}

iree_host_size_t loom_serve_snapshot_size(
    const loom_serve_snapshot_t* snapshot) {
  return snapshot ? snapshot->length : 0;
}

void loom_serve_snapshot_release(loom_serve_snapshot_t* snapshot) {
  if (snapshot && iree_atomic_ref_count_dec(&snapshot->ref_count) == 1) {
    iree_allocator_free(snapshot->host_allocator, snapshot);
  }
}
