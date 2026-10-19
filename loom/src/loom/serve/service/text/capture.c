// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/capture.h"

struct loom_serve_text_capture_t {
  // Shared across original requests and independent consumers.
  iree_atomic_ref_count_t ref_count;
  // Owns this header and trailing checked ranges.
  iree_allocator_t host_allocator;
  // Number of spans in the submitted cohort.
  iree_host_size_t span_count;
  // Number of source ranges following the per-span ranges.
  iree_host_size_t range_count;
  // Exact concatenated output length in bytes.
  iree_host_size_t length;
  // Retained submission domain and shared pool, independent of the model.
  loom_serve_execution_t* execution;
  // Owned allocation root, not a model-retirement-tracked view.
  iree_hal_buffer_t* buffer;
  // Accepted gather frontier, whose semaphores are owned by execution.
  loom_serve_execution_frontier_t ready;
  // Destination span ranges followed by source residual ranges.
  iree_hal_buffer_range_t ranges[];
};

struct loom_serve_text_capture_snapshot_t {
  // Shared by the caller and the download's actual queue capture.
  iree_atomic_ref_count_t ref_count;
  // Owns this header, copied ranges, and downloaded bytes together.
  iree_allocator_t host_allocator;
  // Retained domain and source-pool ownership, independent of the model.
  loom_serve_execution_t* execution;
  // Number of span ranges preceding downloaded payload bytes.
  iree_host_size_t span_count;
  // Copied destination ranges followed by exact downloaded payload bytes.
  iree_hal_buffer_range_t ranges[];
};

iree_status_t loom_serve_text_capture_create(
    iree_vm_buffer_t* plan, iree_host_size_t span_count,
    iree_device_size_t residual_length, iree_allocator_t host_allocator,
    loom_serve_text_capture_t** out_capture) {
  *out_capture = NULL;
  iree_const_byte_span_t bytes;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_read(plan, 0, iree_vm_buffer_length(plan), &bytes));
  const iree_host_size_t range_count = bytes.data_length / 24;
  if (bytes.data_length % 24 || range_count < span_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture requires complete ranges for every span");
  }
  iree_host_size_t range_bytes = 0;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(span_count, range_count, &range_bytes) ||
      !iree_host_size_checked_mul(range_bytes, sizeof(iree_hal_buffer_range_t),
                                  &range_bytes) ||
      !iree_host_size_checked_add(sizeof(loom_serve_text_capture_t),
                                  range_bytes, &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "capture metadata exceeds host address space");
  }
  loom_serve_text_capture_t* capture = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&capture));
  iree_atomic_ref_count_init(&capture->ref_count);
  capture->host_allocator = host_allocator;
  capture->span_count = span_count;
  capture->range_count = range_count;
  iree_status_t status = iree_ok_status();
  iree_host_size_t current_span = 0;
  for (iree_host_size_t i = 0; i < range_count && iree_status_is_ok(status);
       ++i) {
    const uint8_t* record = bytes.data + i * 24;
    const uint64_t span = iree_unaligned_load_le_u64(record);
    const uint64_t offset = iree_unaligned_load_le_u64(record + 8);
    const uint64_t length = iree_unaligned_load_le_u64(record + 16);
    if (span >= span_count || (i == 0 && span != 0) || span < current_span ||
        span > current_span + 1 || !length || offset > residual_length ||
        length > residual_length - offset) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid source capture range %zu", i);
    } else if (length > IREE_HOST_SIZE_MAX - capture->length) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "captured bytes exceed host address space");
    } else {
      if (span != current_span) {
        current_span = (iree_host_size_t)span;
        capture->ranges[current_span].byte_offset = capture->length;
      }
      capture->ranges[current_span].byte_length += length;
      capture->ranges[span_count + i] =
          iree_hal_make_buffer_range(offset, length);
      capture->length += (iree_host_size_t)length;
    }
  }
  if (iree_status_is_ok(status) && current_span + 1 != span_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source capture omits a cohort span");
  }
  if (iree_status_is_ok(status)) {
    *out_capture = capture;
  } else {
    loom_serve_text_capture_release(capture);
  }
  return status;
}

iree_status_t loom_serve_text_capture_submit(loom_serve_text_capture_t* capture,
                                             loom_serve_execution_t* execution,
                                             iree_hal_buffer_t* residual) {
  capture->execution = execution;
  loom_serve_execution_retain(execution);
  IREE_RETURN_IF_ERROR(
      loom_serve_execution_reserve_workspace(execution, capture->length, 1));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = capture->length, .min_alignment = 1};
  iree_hal_buffer_t* output = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_execution_alloca(execution, &request, &output));
  capture->buffer = iree_hal_buffer_allocated_buffer(output);
  iree_hal_buffer_retain(capture->buffer);
  iree_hal_transfer_operation_t operations[64];
  iree_host_size_t operation_count = 0;
  iree_device_size_t offset = 0;
  uint64_t completion = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < capture->range_count && iree_status_is_ok(status); ++i) {
    const iree_hal_buffer_range_t range =
        capture->ranges[capture->span_count + i];
    operations[operation_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
        .copy = {.source_buffer = residual,
                 .source_offset = range.byte_offset,
                 .target_buffer = output,
                 .target_offset = offset,
                 .length = range.byte_length}};
    offset += range.byte_length;
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
  iree_hal_buffer_release(output);
  capture->ready = loom_serve_execution_frontier(execution);
  return status;
}

void loom_serve_text_capture_retain(loom_serve_text_capture_t* capture) {
  if (capture) {
    iree_atomic_ref_count_inc(&capture->ref_count);
  }
}

void loom_serve_text_capture_release(loom_serve_text_capture_t* capture) {
  if (capture && iree_atomic_ref_count_dec(&capture->ref_count) == 1) {
    iree_hal_buffer_release(capture->buffer);
    loom_serve_execution_release(capture->execution);
    iree_allocator_free(capture->host_allocator, capture);
  }
}

loom_serve_execution_frontier_t loom_serve_text_capture_ready(
    const loom_serve_text_capture_t* capture) {
  return capture->ready;
}

iree_hal_buffer_ref_t loom_serve_text_capture_span(
    const loom_serve_text_capture_t* capture, iree_host_size_t span) {
  return (iree_hal_buffer_ref_t){.buffer = capture->buffer,
                                 .offset = capture->ranges[span].byte_offset,
                                 .length = capture->ranges[span].byte_length};
}

void loom_serve_text_capture_snapshot_release(
    loom_serve_text_capture_snapshot_t* snapshot) {
  if (snapshot && iree_atomic_ref_count_dec(&snapshot->ref_count) == 1) {
    loom_serve_execution_release(snapshot->execution);
    iree_allocator_free(snapshot->host_allocator, snapshot);
  }
}

static void loom_serve_text_capture_snapshot_retired(
    void* user_data, iree_hal_buffer_t* buffer) {
  loom_serve_text_capture_snapshot_release(user_data);
}

iree_status_t loom_serve_text_capture_download(
    loom_serve_text_capture_t* capture,
    loom_serve_text_capture_snapshot_t** out_snapshot,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_snapshot = NULL;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  const iree_host_size_t range_bytes =
      capture->span_count * sizeof(iree_hal_buffer_range_t);
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(sizeof(loom_serve_text_capture_snapshot_t),
                                  range_bytes, &allocation_size) ||
      !iree_host_size_checked_add(allocation_size, capture->length,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "capture snapshot exceeds host address space");
  }
  loom_serve_text_capture_snapshot_t* snapshot = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      capture->host_allocator, allocation_size, (void**)&snapshot));
  iree_atomic_ref_count_init(&snapshot->ref_count);
  snapshot->host_allocator = capture->host_allocator;
  snapshot->execution = capture->execution;
  loom_serve_execution_retain(snapshot->execution);
  snapshot->span_count = capture->span_count;
  memcpy(snapshot->ranges, capture->ranges, range_bytes);
  iree_hal_buffer_t* source = NULL;
  iree_status_t status = iree_hal_subspan_buffer_create_with_callback(
      capture->buffer, 0, capture->length,
      (iree_hal_buffer_release_callback_t){
          loom_serve_text_capture_snapshot_retired, snapshot},
      capture->host_allocator, &source);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_inc(&snapshot->ref_count);
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = source,
                     .target = snapshot->ranges + snapshot->span_count,
                     .length = capture->length}};
    uint64_t value = 0;
    status =
        loom_serve_execution_feedback(capture->execution, 1, &download, &value);
  }
  iree_hal_buffer_release(source);
  if (iree_status_is_ok(status)) {
    *out_snapshot = snapshot;
    *out_frontier = loom_serve_execution_frontier(capture->execution);
  } else {
    loom_serve_text_capture_snapshot_release(snapshot);
  }
  return status;
}

iree_const_byte_span_t loom_serve_text_capture_snapshot_span(
    const loom_serve_text_capture_snapshot_t* snapshot, iree_host_size_t span) {
  const uint8_t* data =
      (const uint8_t*)(snapshot->ranges + snapshot->span_count);
  return iree_make_const_byte_span(
      data + snapshot->ranges[span].byte_offset,
      (iree_host_size_t)snapshot->ranges[span].byte_length);
}
