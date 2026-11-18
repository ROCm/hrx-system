// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_file_transfer.h"

#include "iree/hal/utils/memory_file.h"
#include "iree/io/file_handle.h"

// One retained host allocation backs every captured-range binding in a group.
// Its lifetime follows native file ownership, independently of replay records.
typedef struct iree_hal_replay_file_bytes_t {
  // Allocator owning this header and the immediately trailing bytes.
  iree_allocator_t allocator;
} iree_hal_replay_file_bytes_t;

static void iree_hal_replay_file_bytes_release(
    void* user_data, iree_io_file_handle_primitive_t primitive) {
  iree_hal_replay_file_bytes_t* storage = user_data;
  iree_allocator_free(storage->allocator, storage);
}

static iree_status_t iree_hal_replay_file_bytes_wrap(
    iree_const_byte_span_t bytes, iree_allocator_t allocator,
    iree_hal_file_t** out_file) {
  *out_file = NULL;
  iree_host_size_t size = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_replay_file_bytes_t),
                                  bytes.data_length, &size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file capture size overflow");
  }
  iree_hal_replay_file_bytes_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, size, (void**)&storage));
  storage->allocator = allocator;
  if (bytes.data_length) {
    memcpy(storage + 1, bytes.data, bytes.data_length);
  }
  iree_io_file_handle_t* handle = NULL;
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(storage + 1, bytes.data_length),
      (iree_io_file_handle_release_callback_t){
          iree_hal_replay_file_bytes_release, storage},
      allocator, &handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_file_wrap(NULL, IREE_HAL_MEMORY_ACCESS_READ,
                                       handle, IREE_HAL_MEMORY_FILE_FLAG_NONE,
                                       allocator, out_file);
    iree_io_file_handle_release(handle);
  } else {
    iree_allocator_free(allocator, storage);
  }
  return status;
}

static iree_hal_file_transfer_t iree_hal_replay_file_transfer_decode(
    const uint8_t* data) {
  iree_hal_replay_file_transfer_payload_t row;
  memcpy(&row, data, sizeof(row));
  return (iree_hal_file_transfer_t){
      .file_binding = row.file_binding,
      .buffer_binding = row.buffer_binding,
      .file_offset = row.file_offset,
      .buffer_offset = row.buffer_offset,
      .row_length = row.row_length,
      .row_count = row.row_count,
      .slice_count = row.slice_count,
      .file_strides = {row.file_row_stride, row.file_slice_stride},
      .buffer_strides = {row.buffer_row_stride, row.buffer_slice_stride}};
}

iree_status_t iree_hal_replay_executor_queue_file_transfer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record,
    const iree_hal_queue_barriers_t* barriers) {
  const bool read = record->header.operation_code ==
                    IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_READ;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record,
      read ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ
           : IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE,
      sizeof(iree_hal_replay_queue_file_payload_t)));
  iree_hal_replay_queue_file_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_queue_file_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hal_replay_queue_file_layout(&payload, &layout));
  iree_hal_replay_object_entry_t* queue_entry = NULL;
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.device_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));

  iree_host_size_t size = 0, files_offset = 0, buffers_offset = 0,
                   rows_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD((iree_host_size_t)payload.file_count,
                        iree_hal_file_binding_t, &files_offset),
      IREE_STRUCT_FIELD((iree_host_size_t)payload.buffer_count,
                        iree_hal_buffer_binding_t, &buffers_offset),
      IREE_STRUCT_FIELD((iree_host_size_t)payload.transfer_count,
                        iree_hal_file_transfer_t, &rows_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executor->host_allocator, iree_max(size, 1), (void**)&storage));
  iree_hal_file_binding_t* files = (void*)(storage + files_offset);
  iree_hal_buffer_binding_t* buffers = (void*)(storage + buffers_offset);
  iree_hal_file_transfer_t* rows = (void*)(storage + rows_offset);
  iree_hal_replay_semaphore_list_storage_t waits, signals;
  iree_const_byte_span_t trailing;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, layout.total_size, &waits, &signals,
      &trailing);
  iree_hal_file_t* captured_file = NULL;
  for (iree_host_size_t i = 0;
       i < payload.file_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_file_binding_payload_t binding;
    memcpy(&binding, trailing.data + layout.files_offset + i * sizeof(binding),
           sizeof(binding));
    iree_hal_replay_object_entry_t* entry = NULL;
    status = iree_hal_replay_executor_lookup(
        executor, binding.file_id, IREE_HAL_REPLAY_OBJECT_TYPE_FILE, &entry);
    if (iree_status_is_ok(status) &&
        (binding.captured > 1 ||
         (binding.captured && (!read || payload.list_type != 0)))) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid captured file binding");
    }
    if (iree_status_is_ok(status) && binding.captured && !captured_file) {
      status = iree_hal_replay_file_bytes_wrap(
          iree_make_const_byte_span(
              trailing.data + layout.data_offset,
              (iree_host_size_t)payload.captured_data_length),
          executor->host_allocator, &captured_file);
    }
    if (iree_status_is_ok(status)) {
      files[i] = (iree_hal_file_binding_t){
          binding.captured ? captured_file : entry->value.file, binding.offset,
          binding.length};
      if (!files[i].file) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay file binding requires imported or captured data");
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.buffer_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_buffer_ref_payload_t binding;
    memcpy(&binding,
           trailing.data + layout.buffers_offset + i * sizeof(binding),
           sizeof(binding));
    iree_hal_buffer_ref_t ref;
    status = iree_hal_replay_executor_make_direct_buffer_ref(executor, &binding,
                                                             &ref);
    if (iree_status_is_ok(status)) {
      buffers[i] =
          (iree_hal_buffer_binding_t){ref.buffer, ref.offset, ref.length};
    }
  }
  iree_hal_file_transfer_list_t transfers = {.type = payload.list_type};
  if (iree_status_is_ok(status) && payload.list_type == 0) {
    for (iree_host_size_t i = 0; i < payload.transfer_count; ++i) {
      rows[i] = iree_hal_replay_file_transfer_decode(
          trailing.data + layout.transfers_offset +
          i * sizeof(iree_hal_replay_file_transfer_payload_t));
    }
    transfers.host.count = (iree_host_size_t)payload.transfer_count;
    transfers.host.transfers = rows;
  } else if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_t records, count;
    status = iree_hal_replay_executor_make_direct_buffer_ref(
        executor, &payload.records_ref, &records);
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_executor_make_direct_buffer_ref(
          executor, &payload.count_ref, &count);
    }
    if (iree_status_is_ok(status) && count.length != sizeof(uint64_t)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "indirect file count is not eight bytes");
    }
    if (iree_status_is_ok(status)) {
      transfers.indirect.records = (iree_hal_buffer_binding_t){
          records.buffer, records.offset, records.length};
      transfers.indirect.count_buffer = count.buffer;
      transfers.indirect.count_offset = count.offset;
    }
  }
  iree_hal_replay_queue_completion_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_prepare_queue_completion(
        executor, device_entry->value.device, queue_entry->value.queue,
        &signals, &completion);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_file_binding_table_t file_table = {
        (iree_host_size_t)payload.file_count, files};
    const iree_hal_buffer_binding_table_t buffer_table = {
        (iree_host_size_t)payload.buffer_count, buffers};
    status = read ? iree_hal_queue_read(queue_entry->value.queue, waits.list,
                                        signals.list, file_table, buffer_table,
                                        &transfers, barriers, payload.flags)
                  : iree_hal_queue_write(queue_entry->value.queue, waits.list,
                                         signals.list, buffer_table, file_table,
                                         &transfers, barriers, payload.flags);
  }
  status = iree_hal_replay_executor_finalize_queue_completion(
      executor, completion, false, status);
  iree_hal_file_release(captured_file);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signals,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&waits,
                                                      executor->host_allocator);
  iree_allocator_free(executor->host_allocator, storage);
  return status;
}
