// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/snapshot.h"

#include "iree/base/internal/atomics.h"

struct loom_serve_snapshot_t {
  // Caller ownership and queue-captured views independently retain the payload.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning the header and packed payload.
  iree_allocator_t allocator;
  // Number of logical bytes in data.
  iree_host_size_t length;
  // Concatenated retained ranges, independent of physical placement.
  uint8_t data[];
};

static void snapshot_release_view(void* user_data, iree_hal_buffer_t* buffer) {
  loom_serve_snapshot_release(user_data);
}

static iree_status_t snapshot_transfer(
    loom_serve_snapshot_t* snapshot, loom_serve_execution_t* execution,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges,
    iree_hal_transfer_operation_type_t type) {
  iree_hal_buffer_t** views = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      snapshot->allocator, buffer_count, sizeof(*views), (void**)&views));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < buffer_count && iree_status_is_ok(status);
       ++i) {
    if (buffers[i]) {
      status = iree_hal_subspan_buffer_create_with_callback(
          buffers[i], iree_hal_buffer_byte_offset(buffers[i]),
          iree_hal_buffer_byte_length(buffers[i]),
          (iree_hal_buffer_release_callback_t){snapshot_release_view, snapshot},
          snapshot->allocator, &views[i]);
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
    if (operation_count == IREE_ARRAYSIZE(operations)) {
      status = loom_serve_execution_transfer(execution, operation_count,
                                             operations, &completion);
      operation_count = 0;
    }
  }
  if (iree_status_is_ok(status) && operation_count) {
    status = loom_serve_execution_transfer(execution, operation_count,
                                           operations, &completion);
  }
  for (iree_host_size_t i = 0; i < buffer_count; ++i) {
    iree_hal_buffer_release(views[i]);
  }
  iree_allocator_free(snapshot->allocator, views);
  return status;
}

iree_status_t loom_serve_snapshot_capture(
    loom_serve_execution_t* execution, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges,
    loom_serve_snapshot_t** out_snapshot, iree_allocator_t host_allocator) {
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
  snapshot->allocator = host_allocator;
  snapshot->length = length;
  iree_status_t status =
      snapshot_transfer(snapshot, execution, buffer_count, buffers, range_count,
                        ranges, IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD);
  if (iree_status_is_ok(status)) {
    *out_snapshot = snapshot;
  } else {
    loom_serve_snapshot_release(snapshot);
  }
  return status;
}

iree_status_t loom_serve_snapshot_restore(
    loom_serve_snapshot_t* snapshot, loom_serve_execution_t* execution,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges) {
  return snapshot_transfer(snapshot, execution, buffer_count, buffers,
                           range_count, ranges,
                           IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD);
}

iree_host_size_t loom_serve_snapshot_size(
    const loom_serve_snapshot_t* snapshot) {
  return snapshot ? snapshot->length : 0;
}

void loom_serve_snapshot_release(loom_serve_snapshot_t* snapshot) {
  if (snapshot && iree_atomic_ref_count_dec(&snapshot->ref_count) == 1) {
    iree_allocator_free(snapshot->allocator, snapshot);
  }
}
