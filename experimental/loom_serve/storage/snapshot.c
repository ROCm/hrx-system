// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/snapshot.h"

#include "experimental/loom_serve/runtime/retirement.h"

struct loom_serve_snapshot_t {
  // Allocator owning the header and packed payload.
  iree_allocator_t allocator;
  // Number of logical bytes in data.
  iree_host_size_t length;
  // Concatenated retained ranges, independent of physical placement.
  uint8_t data[];
};

static iree_status_t snapshot_transfer(
    loom_serve_execution_t* execution, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges, uint8_t* data,
    iree_hal_transfer_operation_type_t type, iree_allocator_t allocator) {
  iree_hal_buffer_t** views = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, buffer_count, sizeof(*views), (void**)&views));
  loom_serve_retirement_t retirement;
  loom_serve_retirement_initialize(&retirement);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < buffer_count && iree_status_is_ok(status);
       ++i) {
    if (buffers[i]) {
      views[i] = buffers[i];
      iree_hal_buffer_retain(views[i]);
      status = loom_serve_retirement_track(&retirement, &views[i], allocator);
    }
  }
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
  status = iree_status_join(status, loom_serve_execution_drain(execution));
  for (iree_host_size_t i = 0; i < buffer_count; ++i) {
    iree_hal_buffer_release(views[i]);
  }
  // An errored semaphore can precede the final staging callback's write into
  // host data. Joining the exact transfer views protects that borrowed payload.
  loom_serve_retirement_deinitialize(&retirement);
  iree_allocator_free(allocator, views);
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
  snapshot->allocator = host_allocator;
  snapshot->length = length;
  iree_status_t status = snapshot_transfer(
      execution, buffer_count, buffers, range_count, ranges, snapshot->data,
      IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD, host_allocator);
  if (iree_status_is_ok(status)) {
    *out_snapshot = snapshot;
  } else {
    loom_serve_snapshot_destroy(snapshot);
  }
  return status;
}

iree_status_t loom_serve_snapshot_restore(
    const loom_serve_snapshot_t* snapshot, loom_serve_execution_t* execution,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges) {
  return snapshot_transfer(execution, buffer_count, buffers, range_count,
                           ranges, (uint8_t*)snapshot->data,
                           IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                           snapshot->allocator);
}

iree_host_size_t loom_serve_snapshot_size(
    const loom_serve_snapshot_t* snapshot) {
  return snapshot ? snapshot->length : 0;
}

void loom_serve_snapshot_destroy(loom_serve_snapshot_t* snapshot) {
  if (snapshot) {
    iree_allocator_free(snapshot->allocator, snapshot);
  }
}
