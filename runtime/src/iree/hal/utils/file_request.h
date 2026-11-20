// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_FILE_REQUEST_H_
#define IREE_HAL_UTILS_FILE_REQUEST_H_

#include "iree/async/operations/file.h"
#include "iree/hal/file.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns an owned terminal status after accepted file work retires. Buffer
// publication and queue completion remain the enclosing transfer's obligations.
typedef struct iree_hal_file_transfer_callback_t {
  // Consumes status and may reclaim the placed request.
  void (*fn)(void* user_data, iree_status_t status);
  // Borrowed enclosing transfer state, live until terminal completion.
  void* user_data;
} iree_hal_file_transfer_callback_t;

// One placed native request with independent partial-I/O progress. Its owner
// retains the file and span through completion; the request borrows their
// immutable geometry and native handles.
typedef struct iree_hal_file_request_t {
  // Absolute first byte in the file, independent of other requests' progress.
  uint64_t file_offset;
  // Completed prefix of the admitted extent, including on terminal failure.
  iree_host_size_t bytes_transferred;
  // Returns ownership after all accepted partial requests retire.
  iree_hal_file_transfer_callback_t callback;
  union {
    // Placed read; base.user_data borrows the file's read requirements.
    iree_async_file_read_operation_t read;
    // Placed write; base.user_data borrows the file's write requirements.
    iree_async_file_write_operation_t write;
  };
} iree_hal_file_request_t;

// Returns the whole admitted span, including its completed prefix. The native
// operation stores the remaining suffix with its fixed-registration identity.
// This is valid after preparation and while the caller owns the request.
static inline iree_async_span_t iree_hal_file_request_span(
    const iree_hal_file_request_t* request) {
  iree_async_span_t span =
      request->read.base.type == IREE_ASYNC_OPERATION_TYPE_FILE_READ
          ? request->read.buffer
          : request->write.buffer;
  span.offset -= request->bytes_transferred;
  span.length += request->bytes_transferred;
  return span;
}

// Prepares a nonempty chunk within a validated asynchronous file transfer.
// Access is exactly READ or WRITE. File geometry, span capacity, and fixed
// registration boundaries bound the admitted span; remaining_length is the
// enclosing transfer's remaining extent. The file has a native async handle.
// Failure leaves out_request unchanged. Reuse requires prior I/O retirement.
iree_status_t iree_hal_file_request_prepare(
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, uint64_t remaining_length, iree_async_span_t span,
    iree_hal_file_request_t* out_request);

// Binds terminal completion and returns the prepared operation for
// iree_hal_file_request_submit_batch. The caller owns the request until batch
// acceptance. Synchronous rejection leaves it available for reuse.
// All operations submitted together belong to the same proactor. The enclosing
// owner joins the submit call with every accepted callback before reclaiming
// request storage, as completion may run concurrently with submission.
// Partial I/O retains this callback until the exact prepared span retires.
static inline iree_async_operation_t* iree_hal_file_request_bind(
    iree_hal_file_request_t* request,
    iree_hal_file_transfer_callback_t callback) {
  request->callback = callback;
  return &request->read.base;
}

// Submits independently completable, bound file requests to their common
// proactor. OK transfers the whole window to its callbacks. A synchronous
// failure transfers none. The caller can drop the pointer list on return and
// retains each request, file, and span through its callback. When native slots
// are scarce, the helper copies the list for admission by the polling thread.
iree_status_t iree_hal_file_request_submit_batch(
    iree_async_proactor_t* proactor, iree_async_operation_list_t operations);

// Submits the prepared request and continues partial I/O until its exact span
// retires or fails. Rejection retains caller ownership and invokes no callback.
// Acceptance transfers ownership to one terminal callback, which may reclaim
// the request. The caller joins its own access if completion can precede this
// call returning. Independent requests may complete in any order.
iree_status_t iree_hal_file_request_submit(
    iree_hal_file_request_t* request,
    iree_hal_file_transfer_callback_t callback);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_UTILS_FILE_REQUEST_H_
