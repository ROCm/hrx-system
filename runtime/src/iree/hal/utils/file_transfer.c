// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/file_transfer.h"

#include "iree/hal/memory_scope.h"

static iree_status_t iree_hal_file_transfer_capture_buffer(
    iree_hal_queue_t* queue, iree_hal_memory_access_t access,
    iree_hal_buffer_binding_t binding, iree_hal_buffer_binding_t* out_binding) {
  if (!binding.buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file transfer buffer is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      0, iree_hal_buffer_byte_length(binding.buffer), binding.offset,
      binding.length, &binding.offset, &binding.length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding.buffer), access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_family_usage(
      binding.buffer, iree_hal_queue_family(queue),
      access == IREE_HAL_MEMORY_ACCESS_READ
          ? IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE
          : IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding.buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  *out_binding = binding;
  return iree_ok_status();
}

iree_status_t iree_hal_file_transfer_batch_storage_size(
    iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_host_size_t* out_size) {
  *out_size = 0;
  if (!transfers || (files.count && !files.bindings) ||
      (buffers.count && !buffers.bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file transfer list or binding storage is null");
  }
  if (transfers->type != IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST &&
      transfers->type != IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported file transfer list type");
  }
  if ((files.count && files.count - 1 > UINT32_MAX) ||
      (buffers.count && buffers.count - 1 > UINT32_MAX) ||
      (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT &&
       (files.count > 65536 || buffers.count > 65536))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file transfer binding table exceeds index width");
  }

  const iree_host_size_t host_count =
      transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST
          ? transfers->host.count
          : 0;
  if (host_count && !transfers->host.transfers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host file transfer storage is null");
  }
  return IREE_STRUCT_LAYOUT(
      0, out_size,
      IREE_STRUCT_FIELD_ALIGNED(files.count, iree_hal_file_binding_t,
                                iree_max_align_t, NULL),
      IREE_STRUCT_FIELD_ALIGNED(buffers.count, iree_hal_buffer_binding_t,
                                iree_max_align_t, NULL),
      IREE_STRUCT_FIELD_ALIGNED(host_count, iree_hal_file_transfer_t,
                                iree_max_align_t, NULL));
}

iree_status_t iree_hal_file_transfer_batch_capture(
    iree_hal_queue_t* queue, iree_hal_memory_access_t file_access,
    iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers, void* storage,
    iree_hal_file_transfer_batch_t* out_batch) {
  *out_batch = (iree_hal_file_transfer_batch_t){0};
  iree_hal_file_binding_t* captured_files = storage;
  iree_hal_buffer_binding_t* captured_buffers =
      (void*)((uint8_t*)storage +
              iree_host_align(files.count * sizeof(*captured_files),
                              iree_max_align_t));
  iree_hal_file_transfer_t* captured_transfers =
      (void*)((uint8_t*)captured_buffers +
              iree_host_align(buffers.count * sizeof(*captured_buffers),
                              iree_max_align_t));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    captured_files[i] = files.bindings[i];
    if (!captured_files[i].file) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "file transfer file is null");
    } else {
      status =
          iree_hal_file_validate_access(captured_files[i].file, file_access);
      if (iree_status_is_ok(status)) {
        status = iree_hal_file_validate_range(captured_files[i].file,
                                              captured_files[i].offset,
                                              captured_files[i].length);
      }
    }
  }
  const iree_hal_memory_access_t buffer_access =
      file_access == IREE_HAL_MEMORY_ACCESS_READ ? IREE_HAL_MEMORY_ACCESS_WRITE
                                                 : IREE_HAL_MEMORY_ACCESS_READ;
  for (iree_host_size_t i = 0; i < buffers.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_file_transfer_capture_buffer(
        queue, buffer_access, buffers.bindings[i], &captured_buffers[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  out_batch->files =
      (iree_hal_file_binding_table_t){files.count, captured_files};
  out_batch->buffers =
      (iree_hal_buffer_binding_table_t){buffers.count, captured_buffers};
  out_batch->transfers = *transfers;
  if (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST) {
    const iree_host_size_t count = transfers->host.count;
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      captured_transfers[i] = transfers->host.transfers[i];
      status = iree_hal_file_transfer_validate(
          out_batch->files, out_batch->buffers, &captured_transfers[i]);
    }
    out_batch->transfers.host.transfers = captured_transfers;
  } else {
    iree_hal_buffer_binding_t* records = &out_batch->transfers.indirect.records;
    IREE_RETURN_IF_ERROR(iree_hal_file_transfer_capture_buffer(
        queue, IREE_HAL_MEMORY_ACCESS_READ, *records, records));
    iree_hal_buffer_binding_t count = {transfers->indirect.count_buffer,
                                       transfers->indirect.count_offset, 8};
    IREE_RETURN_IF_ERROR(iree_hal_file_transfer_capture_buffer(
        queue, IREE_HAL_MEMORY_ACCESS_READ, count, &count));
    if (records->offset % 8 || records->length % 64 || count.offset % 8) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "indirect file records/count require aligned "
                              "words and whole 64-byte records");
    }
  }
  return status;
}

iree_status_t iree_hal_file_transfer_iterator_initialize(
    const iree_hal_file_transfer_batch_t* batch, uint64_t count,
    const iree_hal_file_transfer_record_t* records,
    iree_hal_file_transfer_iterator_t* out_iterator) {
  if (batch->transfers.type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST) {
    count = batch->transfers.host.count;
    records = NULL;
  } else if (count > batch->transfers.indirect.records.length /
                         sizeof(iree_hal_file_transfer_record_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "live file transfer count exceeds record capacity");
  }
  *out_iterator = (iree_hal_file_transfer_iterator_t){
      .batch = batch,
      .records = records,
      .count = (iree_host_size_t)count,
      .acquired = records || batch->transfers.type ==
                                 IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST
                      ? (iree_host_size_t)count
                      : 0};
  return iree_ok_status();
}

void iree_hal_file_transfer_iterator_acquire(
    iree_hal_file_transfer_iterator_t* iterator,
    const iree_hal_file_transfer_record_t* records, iree_host_size_t count) {
  iterator->records = records;
  iterator->acquired = iterator->index + count;
}

// Validation proves the addressed span fits both windows. When strides equal
// their extent, multiplying that extent cannot overflow. Unused strides are
// never consulted, allowing callers to initialize only meaningful dimensions.
static void iree_hal_file_transfer_normalize(
    iree_hal_file_transfer_t* transfer) {
  if (transfer->row_count > 1 &&
      transfer->file_strides.row_stride == transfer->row_length &&
      transfer->buffer_strides.row_stride == transfer->row_length) {
    transfer->row_length *= transfer->row_count;
    transfer->row_count = 1;
  }
  if (transfer->row_count == 1) {
    transfer->file_strides.row_stride = 0;
    transfer->buffer_strides.row_stride = 0;
    if (transfer->slice_count > 1 &&
        transfer->file_strides.slice_stride == transfer->row_length &&
        transfer->buffer_strides.slice_stride == transfer->row_length) {
      transfer->row_length *= transfer->slice_count;
      transfer->slice_count = 1;
    }
  }
  if (transfer->slice_count == 1) {
    transfer->file_strides.slice_stride = 0;
    transfer->buffer_strides.slice_stride = 0;
  }
}

iree_status_t iree_hal_file_transfer_iterator_peek(
    iree_hal_file_transfer_iterator_t* iterator,
    iree_hal_file_transfer_range_t* out_range) {
  out_range->length = 0;
  iree_status_t status = iree_ok_status();
  while (!iterator->current.row_length &&
         iterator->index < iterator->acquired && iree_status_is_ok(status)) {
    if (iterator->records) {
      status = iree_hal_file_transfer_record_decode(iterator->records++,
                                                    &iterator->current);
      if (iree_status_is_ok(status)) {
        status = iree_hal_file_transfer_validate(iterator->batch->files,
                                                 iterator->batch->buffers,
                                                 &iterator->current);
      }
    } else {
      iterator->current =
          iterator->batch->transfers.host.transfers[iterator->index];
    }
    ++iterator->index;
    if (iree_status_is_ok(status) && iterator->current.row_length) {
      if (!iterator->current.row_count || !iterator->current.slice_count) {
        iterator->current.row_length = 0;
      } else {
        iree_hal_file_transfer_normalize(&iterator->current);
      }
    }
    iterator->row = iterator->slice = iterator->position = 0;
  }
  if (iree_status_is_ok(status) && iterator->current.row_length) {
    const iree_hal_file_transfer_t* current = &iterator->current;
    *out_range = (iree_hal_file_transfer_range_t){
        .file_binding = current->file_binding,
        .buffer_binding = current->buffer_binding,
        .file_offset = current->file_offset +
                       iterator->slice * current->file_strides.slice_stride +
                       iterator->row * current->file_strides.row_stride +
                       iterator->position,
        .buffer_offset =
            (iree_device_size_t)(current->buffer_offset +
                                 iterator->slice *
                                     current->buffer_strides.slice_stride +
                                 iterator->row *
                                     current->buffer_strides.row_stride +
                                 iterator->position),
        .length = current->row_length - iterator->position,
    };
  }
  return status;
}

void iree_hal_file_transfer_iterator_advance(
    iree_hal_file_transfer_iterator_t* iterator, uint64_t length) {
  iterator->position += length;
  if (iterator->position == iterator->current.row_length) {
    iterator->position = 0;
    if (++iterator->row == iterator->current.row_count) {
      iterator->row = 0;
      if (++iterator->slice == iterator->current.slice_count) {
        iterator->current.row_length = 0;
      }
    }
  }
}
