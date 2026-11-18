// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/file_request.h"

#include "iree/async/proactor.h"

static void iree_hal_file_request_complete(void* user_data,
                                           iree_async_operation_t* operation,
                                           iree_status_t status,
                                           iree_async_completion_flags_t flags);

static void iree_hal_file_request_initialize_native(
    iree_hal_file_request_t* request, iree_async_file_t* file,
    iree_async_operation_type_t type,
    const iree_async_file_transfer_requirements_t* requirements,
    iree_async_span_t span) {
  if (type == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
    iree_async_file_read_operation_t* read = &request->read;
    iree_async_operation_zero(&read->base, sizeof(*read));
    iree_async_operation_initialize(
        &read->base, type, IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_hal_file_request_complete, (void*)requirements);
    read->file = file;
    read->offset = request->file_offset + request->bytes_transferred;
    read->buffer = span;
  } else {
    iree_async_file_write_operation_t* write = &request->write;
    iree_async_operation_zero(&write->base, sizeof(*write));
    iree_async_operation_initialize(
        &write->base, type, IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_hal_file_request_complete, (void*)requirements);
    write->file = file;
    write->offset = request->file_offset + request->bytes_transferred;
    write->buffer = span;
  }
}

iree_status_t iree_hal_file_request_prepare(
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, uint64_t remaining_length, iree_async_span_t span,
    iree_hal_file_request_t* out_request) {
  // A registration can span several native slots. Keep the admitted request
  // within one slot even when its first byte lies in the middle of that slot.
  if (span.region && span.region->buffer_size) {
    span.length =
        iree_min(span.length, span.region->buffer_size -
                                  span.offset % span.region->buffer_size);
  }
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(file);
  const bool reading = file_access == IREE_HAL_MEMORY_ACCESS_READ;
  const iree_async_file_transfer_requirements_t* geometry =
      reading ? &requirements->read : &requirements->write;
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_prepare_transfer(
      geometry, file_offset, iree_async_span_ptr(span), remaining_length,
      span.length, &length));
  span.length = length;
  out_request->file_offset = file_offset;
  out_request->bytes_transferred = 0;
  iree_hal_file_request_initialize_native(
      out_request, iree_hal_file_async_handle(file),
      reading ? IREE_ASYNC_OPERATION_TYPE_FILE_READ
              : IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      geometry, span);
  return iree_ok_status();
}

static iree_status_t iree_hal_file_request_issue(
    iree_hal_file_request_t* request) {
  iree_async_operation_t* operation = &request->read.base;
  iree_async_file_t* file =
      operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ
          ? request->read.file
          : request->write.file;
  return iree_async_proactor_submit_one(file->proactor, operation);
}

static void iree_hal_file_request_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)flags;
  const iree_async_file_transfer_requirements_t* requirements = user_data;
  iree_hal_file_request_t* request =
      iree_containerof(operation, iree_hal_file_request_t, read.base);
  if (iree_status_is_ok(status)) {
    const bool reading = operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ;
    const iree_host_size_t count =
        reading ? request->read.bytes_read : request->write.bytes_written;
    iree_async_span_t* span =
        reading ? &request->read.buffer : &request->write.buffer;
    span->offset += count;
    span->length -= count;
    request->bytes_transferred += count;
    if (!count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "file %s stopped after %" PRIhsz " of %" PRIhsz " bytes",
          reading ? "read" : "write", request->bytes_transferred,
          request->bytes_transferred + span->length);
    } else if (span->length) {
      iree_host_size_t length = 0;
      status = iree_hal_file_prepare_transfer(
          requirements, request->file_offset + request->bytes_transferred,
          iree_async_span_ptr(*span), span->length, span->length, &length);
      if (iree_status_is_ok(status)) {
        iree_async_file_t* file =
            reading ? request->read.file : request->write.file;
        iree_hal_file_request_initialize_native(request, file, operation->type,
                                                requirements, *span);
        status = iree_hal_file_request_issue(request);
      }
      if (iree_status_is_ok(status)) {
        return;
      }
    }
  }
  request->callback.fn(request->callback.user_data, status);
}

iree_status_t iree_hal_file_request_submit(
    iree_hal_file_request_t* request,
    iree_hal_file_transfer_callback_t callback) {
  iree_hal_file_request_bind(request, callback);
  return iree_hal_file_request_issue(request);
}
