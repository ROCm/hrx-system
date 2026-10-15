// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/file_transfer.h"

#include "iree/async/file.h"
#include "iree/async/proactor.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/utils/memory_file.h"

iree_status_t iree_hal_amd_fabric_file_transfer_initialize(
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_allocator_t host_allocator,
    iree_hal_amd_fabric_file_transfer_t* out_transfer) {
  memset(out_transfer, 0, sizeof(*out_transfer));
  iree_async_file_t* async_file = iree_hal_file_async_handle(file);
  iree_byte_span_t file_contents = iree_byte_span_empty();
  if (!async_file) {
    if (!iree_hal_memory_file_isa(file)) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "AMD queue file transfers require a memory file or an async file "
          "handle");
    }
    IREE_RETURN_IF_ERROR(iree_hal_memory_file_contents(file, &file_contents));
  }
  out_transfer->host_allocator = host_allocator;
  out_transfer->file = file;
  out_transfer->buffer = buffer;
  out_transfer->file_access = file_access;
  out_transfer->file_offset = file_offset;
  out_transfer->buffer_offset = buffer_offset;
  out_transfer->length = length;
  out_transfer->file_contents = file_contents;
  out_transfer->async.file = async_file;
  iree_hal_file_retain(file);
  iree_hal_buffer_retain(buffer);
  return iree_ok_status();
}

void iree_hal_amd_fabric_file_transfer_deinitialize(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  iree_allocator_free_aligned(transfer->host_allocator,
                              transfer->async.staging.data);
  iree_hal_buffer_release(transfer->buffer);
  iree_hal_file_release(transfer->file);
}

iree_status_t iree_hal_amd_fabric_file_transfer_finish(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  const iree_host_size_t offset =
      (iree_host_size_t)transfer->async.chunk.offset;
  const iree_host_size_t length =
      transfer->async.file ? transfer->async.chunk.contents.data_length
                           : transfer->buffer_contents.data_length;
  iree_status_t status = iree_ok_status();
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    if (transfer->async.staging.data) {
      memcpy(transfer->buffer_contents.data + offset,
             transfer->async.staging.data, length);
    }
    status = iree_hal_amd_fabric_buffer_flush_host_range(
        transfer->buffer, transfer->buffer_offset + offset, length);
  }
  if (transfer->async.bytes_transferred == transfer->length) {
    iree_allocator_free_aligned(transfer->host_allocator,
                                transfer->async.staging.data);
    transfer->async.staging = iree_byte_span_empty();
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!transfer->buffer_contents.data) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_buffer_native_host_span(
                transfer->buffer, transfer->buffer_offset, transfer->length,
                &transfer->buffer_contents));
    if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amd_fabric_buffer_invalidate_host_range(
                  transfer->buffer, transfer->buffer_offset, transfer->length));
    }
  }
  iree_status_t status = iree_ok_status();
  if (!transfer->async.file) {
    uint8_t* file_data = transfer->file_contents.data + transfer->file_offset;
    if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
      memmove(transfer->buffer_contents.data, file_data,
              transfer->buffer_contents.data_length);
    } else {
      memmove(file_data, transfer->buffer_contents.data,
              transfer->buffer_contents.data_length);
    }
    status = iree_hal_amd_fabric_file_transfer_finish(transfer);
  } else {
    const iree_async_file_requirements_t* file_requirements =
        iree_hal_file_requirements(transfer->file);
    const iree_async_file_transfer_requirements_t* requirements =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? &file_requirements->read
            : &file_requirements->write;
    if (iree_any_bit_set(transfer->async.file->flags,
                         IREE_ASYNC_FILE_FLAG_DIRECT) &&
        !transfer->async.staging.data) {
      // Device mappings may be CPU-accessible without supporting the page
      // pins required by native uncached I/O. Keep that distinction private to
      // this transfer instead of changing the buffer's allocation contract.
      const iree_host_size_t capacity =
          iree_min(transfer->buffer_contents.data_length, 1024 * 1024);
      const iree_host_size_t alignment =
          iree_max(requirements->buffer_alignment, sizeof(void*));
      status = iree_allocator_malloc_aligned(
          transfer->host_allocator, capacity, alignment, 0,
          (void**)&transfer->async.staging.data);
      if (iree_status_is_ok(status)) {
        transfer->async.staging.data_length = capacity;
      }
    }
    const iree_host_size_t offset =
        (iree_host_size_t)transfer->async.bytes_transferred;
    uint8_t* data = transfer->async.staging.data
                        ? transfer->async.staging.data
                        : transfer->buffer_contents.data + offset;
    const iree_host_size_t capacity =
        transfer->async.staging.data
            ? transfer->async.staging.data_length
            : iree_min(transfer->buffer_contents.data_length - offset,
                       UINT32_C(1) << 30);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_file_transfer_prepare_chunk(
          transfer, iree_make_byte_span(data, capacity));
    }
    if (iree_status_is_ok(status)) {
      if (transfer->async.staging.data &&
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
        memcpy(data, transfer->buffer_contents.data + offset,
               transfer->async.chunk.contents.data_length);
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare_chunk(
    iree_hal_amd_fabric_file_transfer_t* transfer, iree_byte_span_t contents) {
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(transfer->file);
  const uint64_t offset = transfer->async.bytes_transferred;
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_prepare_transfer(
      transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
          ? &requirements->read
          : &requirements->write,
      transfer->file_offset + offset, contents.data, transfer->length - offset,
      contents.data_length, &length));
  transfer->async.chunk.offset = offset;
  transfer->async.chunk.contents = iree_make_byte_span(contents.data, length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_fabric_file_transfer_issue(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_async_completion_fn_t completion) {
  const uint64_t offset = transfer->async.bytes_transferred;
  const iree_host_size_t chunk_offset =
      (iree_host_size_t)(offset - transfer->async.chunk.offset);
  const iree_async_span_t span = iree_async_span_from_ptr(
      transfer->async.chunk.contents.data + chunk_offset,
      transfer->async.chunk.contents.data_length - chunk_offset);
  iree_async_operation_t* operation = NULL;
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    iree_async_file_read_operation_t* read = &transfer->async.read;
    iree_async_operation_zero(&read->base, sizeof(*read));
    iree_async_operation_initialize(
        &read->base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
        IREE_ASYNC_OPERATION_FLAG_NONE, completion, transfer);
    read->file = transfer->async.file;
    read->offset = transfer->file_offset + offset;
    read->buffer = span;
    operation = &read->base;
  } else {
    iree_async_file_write_operation_t* write = &transfer->async.write;
    iree_async_operation_zero(&write->base, sizeof(*write));
    iree_async_operation_initialize(
        &write->base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
        IREE_ASYNC_OPERATION_FLAG_NONE, completion, transfer);
    write->file = transfer->async.file;
    write->offset = transfer->file_offset + offset;
    write->buffer = span;
    operation = &write->base;
  }
  return iree_async_proactor_submit_one(transfer->async.file->proactor,
                                        operation);
}

static void iree_hal_amd_fabric_file_transfer_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_amd_fabric_file_transfer_t* transfer = user_data;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t byte_count =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? transfer->async.read.bytes_read
            : transfer->async.write.bytes_written;
    transfer->async.bytes_transferred += byte_count;
    if (byte_count == 0) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "file %s stopped after %" PRIu64 " of %" PRIu64 " bytes",
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ ? "read"
                                                               : "write",
          transfer->async.bytes_transferred, (uint64_t)transfer->length);
    } else if (transfer->async.bytes_transferred <
               transfer->async.chunk.offset +
                   transfer->async.chunk.contents.data_length) {
      const uint64_t offset = transfer->async.bytes_transferred;
      const iree_host_size_t chunk_offset =
          (iree_host_size_t)(offset - transfer->async.chunk.offset);
      const iree_host_size_t remaining =
          transfer->async.chunk.contents.data_length - chunk_offset;
      const iree_async_file_requirements_t* requirements =
          iree_hal_file_requirements(transfer->file);
      iree_host_size_t length = 0;
      status = iree_hal_file_prepare_transfer(
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
              ? &requirements->read
              : &requirements->write,
          transfer->file_offset + offset,
          transfer->async.chunk.contents.data + chunk_offset, remaining,
          remaining, &length);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_fabric_file_transfer_issue(
            transfer, iree_hal_amd_fabric_file_transfer_complete);
      }
      if (iree_status_is_ok(status)) {
        return;
      }
    }
  }
  transfer->async.callback.fn(transfer->async.callback.user_data, status);
}

iree_status_t iree_hal_amd_fabric_file_transfer_submit(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_hal_amd_fabric_file_transfer_callback_t callback) {
  transfer->async.callback = callback;
  return iree_hal_amd_fabric_file_transfer_issue(
      transfer, iree_hal_amd_fabric_file_transfer_complete);
}
