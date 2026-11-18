// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_file_transfer.h"

#include "iree/hal/replay/recorder_barrier.h"
#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_file.h"
#include "iree/hal/replay/recorder_record.h"
#include "iree/hal/utils/file_transfer.h"

static iree_hal_replay_buffer_ref_payload_t iree_hal_replay_file_buffer_payload(
    iree_hal_replay_recorder_t* recorder, iree_hal_buffer_binding_t binding) {
  iree_hal_replay_buffer_ref_payload_t payload;
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(binding.buffer, binding.offset, binding.length),
      &payload);
  payload.buffer_id =
      iree_hal_replay_recorder_find_buffer_id(recorder, binding.buffer);
  return payload;
}

static iree_hal_replay_file_transfer_payload_t
iree_hal_replay_file_transfer_payload(iree_hal_file_transfer_t transfer) {
  return (iree_hal_replay_file_transfer_payload_t){
      .file_binding = transfer.file_binding,
      .buffer_binding = transfer.buffer_binding,
      .file_offset = transfer.file_offset,
      .buffer_offset = transfer.buffer_offset,
      .row_length = transfer.row_length,
      .row_count = transfer.row_count,
      .slice_count = transfer.slice_count,
      .file_row_stride = transfer.file_strides.row_stride,
      .file_slice_stride = transfer.file_strides.slice_stride,
      .buffer_row_stride = transfer.buffer_strides.row_stride,
      .buffer_slice_stride = transfer.buffer_strides.slice_stride};
}

// Capture cost is proportional to useful bytes, even for a sparse logical file.
// Producer-dependent reads cannot be sampled here and remain unsupported by the
// recorder's range policy; forwarding preserves their deferred acquisition.
static iree_status_t iree_hal_replay_file_capture_size(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_file_transfer_batch_t* batch, iree_host_size_t* out_size) {
  *out_size = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < batch->transfers.host.count && iree_status_is_ok(status); ++i) {
    const iree_hal_file_transfer_t* row = &batch->transfers.host.transfers[i];
    if (!row->row_length || !row->row_count || !row->slice_count ||
        !iree_hal_replay_recorder_file_uses_captured_ranges(
            queue->recorder, batch->files.bindings[row->file_binding].file)) {
      continue;
    }
    iree_host_size_t size = 0;
    if (row->row_length > IREE_HOST_SIZE_MAX ||
        row->row_count > IREE_HOST_SIZE_MAX ||
        row->slice_count > IREE_HOST_SIZE_MAX ||
        !iree_host_size_checked_mul((iree_host_size_t)row->row_length,
                                    (iree_host_size_t)row->row_count, &size) ||
        !iree_host_size_checked_mul(size, (iree_host_size_t)row->slice_count,
                                    &size) ||
        !iree_host_size_checked_add(*out_size, size, out_size)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "file range capture exceeds host address space");
    }
  }
  return status;
}

static iree_status_t iree_hal_replay_file_capture_rows(
    const iree_hal_file_transfer_batch_t* batch,
    iree_hal_replay_file_binding_payload_t* files,
    iree_hal_replay_file_transfer_payload_t* rows, uint8_t* data) {
  iree_host_size_t offset = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < batch->transfers.host.count && iree_status_is_ok(status); ++i) {
    const iree_hal_file_transfer_t* row = &batch->transfers.host.transfers[i];
    rows[i] = iree_hal_replay_file_transfer_payload(*row);
    if (!row->row_length || !row->row_count || !row->slice_count ||
        !files[row->file_binding].captured) {
      continue;
    }
    rows[i].file_offset = offset;
    rows[i].file_row_stride = row->row_length;
    rows[i].file_slice_stride = row->row_length * row->row_count;
    const iree_hal_file_binding_t file =
        batch->files.bindings[row->file_binding];
    // The common iterator coalesces adjacent rows and validates useful extents.
    const iree_hal_file_transfer_batch_t single = {
        .files = batch->files,
        .buffers = batch->buffers,
        .transfers = {.type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
                      .host = {1, row}}};
    iree_hal_file_transfer_iterator_t iterator;
    status =
        iree_hal_file_transfer_iterator_initialize(&single, 0, NULL, &iterator);
    iree_hal_file_transfer_range_t range;
    while (iree_status_is_ok(status)) {
      status = iree_hal_file_transfer_iterator_peek(&iterator, &range);
      if (!iree_status_is_ok(status) || !range.length) {
        break;
      }
      status = iree_hal_replay_recorder_file_capture_read_data(
          file.file, file.offset + range.file_offset,
          iree_make_byte_span(data + offset, (iree_host_size_t)range.length));
      offset += (iree_host_size_t)range.length;
      iree_hal_file_transfer_iterator_advance(&iterator, range.length);
    }
  }
  return status;
}

iree_status_t iree_hal_replay_recorder_queue_file_transfer(
    iree_hal_replay_recorder_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access,
    const iree_hal_queue_barriers_t* barriers, uint64_t flags) {
  const bool read = file_access == IREE_HAL_MEMORY_ACCESS_READ;
  iree_host_size_t capture_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_batch_storage_size(
      files, buffers, transfers, &capture_size));
  const iree_host_size_t metadata_count =
      transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT ? 2 : 0;
  iree_host_size_t temporary_count = 0, temporary_offset = 0, storage_size = 0;
  if (!iree_host_size_checked_add(buffers.count, metadata_count,
                                  &temporary_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file capture size overflow");
  }
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(capture_size, &storage_size,
                         IREE_STRUCT_FIELD(temporary_count, iree_hal_buffer_t*,
                                           &temporary_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, iree_max(storage_size, 1), (void**)&storage));
  memset(storage, 0, storage_size);
  iree_hal_buffer_t** temporary = (void*)(storage + temporary_offset);
  iree_hal_file_transfer_batch_t batch;
  iree_status_t status = iree_hal_file_transfer_batch_capture(
      &queue->base, file_access, files, buffers, transfers, storage, &batch);

  bool can_record = iree_hal_replay_recorder_has_captured_semaphores(
                        queue->recorder, waits) &&
                    iree_hal_replay_recorder_has_captured_semaphores(
                        queue->recorder, signals);
  bool captured_files = false;
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    can_record &= iree_hal_replay_recorder_find_file_id(
                      queue->recorder, files.bindings[i].file) != 0;
    captured_files |= iree_hal_replay_recorder_file_uses_captured_ranges(
        queue->recorder, files.bindings[i].file);
  }
  if (captured_files && (!read || metadata_count || waits.count)) {
    can_record = false;
  }
  iree_host_size_t data_length = 0;
  if (iree_status_is_ok(status) && can_record && captured_files) {
    status = iree_hal_replay_file_capture_size(queue, &batch, &data_length);
  }
  iree_hal_replay_queue_file_payload_t payload = {
      .flags = flags,
      .wait_semaphore_count = waits.count,
      .signal_semaphore_count = signals.count,
      .file_count = files.count,
      .buffer_count = buffers.count,
      .transfer_count = metadata_count ? 0 : transfers->host.count,
      .captured_data_length = data_length,
      .list_type = transfers->type};
  iree_hal_replay_queue_file_layout_t layout = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_queue_file_layout(&payload, &layout);
  }
  uint8_t* serialized = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(queue->host_allocator,
                                   iree_max(layout.total_size, 1),
                                   (void**)&serialized);
  }
  iree_hal_replay_file_binding_payload_t* file_payloads = (void*)serialized;
  iree_hal_replay_buffer_ref_payload_t* buffer_payloads =
      serialized ? (void*)(serialized + layout.buffers_offset) : NULL;
  iree_hal_file_binding_t* base_files = (void*)batch.files.bindings;
  iree_hal_buffer_binding_t* base_buffers = (void*)batch.buffers.bindings;
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    const bool captured =
        can_record && iree_hal_replay_recorder_file_uses_captured_ranges(
                          queue->recorder, files.bindings[i].file);
    file_payloads[i] = (iree_hal_replay_file_binding_payload_t){
        .file_id = iree_hal_replay_recorder_find_file_id(
            queue->recorder, files.bindings[i].file),
        .offset = captured ? 0 : files.bindings[i].offset,
        .length = captured ? data_length : files.bindings[i].length,
        .captured = captured};
  }
  if (iree_status_is_ok(status) && !metadata_count) {
    status = iree_hal_replay_file_capture_rows(
        &batch, file_payloads, (void*)(serialized + layout.transfers_offset),
        serialized + layout.data_offset);
  }
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    base_files[i].file =
        iree_hal_replay_recorder_file_base_or_self(base_files[i].file);
  }
  for (iree_host_size_t i = 0; i < buffers.count && iree_status_is_ok(status);
       ++i) {
    buffer_payloads[i] =
        iree_hal_replay_file_buffer_payload(queue->recorder, base_buffers[i]);
    can_record &= buffer_payloads[i].buffer_id != 0;
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        base_buffers[i].buffer, queue->host_allocator, &base_buffers[i].buffer,
        &temporary[i]);
  }
  if (iree_status_is_ok(status) && metadata_count) {
    payload.records_ref = iree_hal_replay_file_buffer_payload(
        queue->recorder, batch.transfers.indirect.records);
    payload.count_ref = iree_hal_replay_file_buffer_payload(
        queue->recorder,
        (iree_hal_buffer_binding_t){batch.transfers.indirect.count_buffer,
                                    batch.transfers.indirect.count_offset,
                                    sizeof(uint64_t)});
    can_record &=
        payload.records_ref.buffer_id != 0 && payload.count_ref.buffer_id != 0;
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        batch.transfers.indirect.records.buffer, queue->host_allocator,
        &batch.transfers.indirect.records.buffer, &temporary[buffers.count]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_recorder_buffer_unwrap_for_call(
          batch.transfers.indirect.count_buffer, queue->host_allocator,
          &batch.transfers.indirect.count_buffer,
          &temporary[buffers.count + 1]);
    }
  }
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t wait_size = 0, signal_size = 0;
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, waits, queue->host_allocator, &wait_payloads,
        &wait_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signals, queue->host_allocator, &signal_payloads,
        &signal_size);
  }
  iree_hal_replay_recorder_barriers_t barrier_storage = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_barriers_initialize(
        queue->recorder, barriers, queue->host_allocator, &barrier_storage);
    can_record &= barrier_storage.can_record;
  }
  iree_hal_replay_pending_record_t pending = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        read ? IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_READ
             : IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_WRITE,
        !can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE
                    : (read ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ
                            : IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE),
        &pending);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending);
  }
  if (iree_status_is_ok(status)) {
    status =
        read ? iree_hal_queue_read(queue->base_queue, waits, signals,
                                   batch.files, batch.buffers, &batch.transfers,
                                   &barrier_storage.base, flags)
             : iree_hal_queue_write(
                   queue->base_queue, waits, signals, batch.buffers,
                   batch.files, &batch.transfers, &barrier_storage.base, flags);
  }
  if (pending.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(wait_payloads, wait_size),
          iree_make_const_byte_span(signal_payloads, signal_size),
          iree_make_const_byte_span(serialized, layout.total_size),
          barrier_storage.payload};
      if (barrier_storage.payload.data_length) {
        pending.metadata.record_flags |=
            IREE_HAL_REPLAY_FILE_RECORD_FLAG_QUEUE_BARRIERS;
      }
      if (barrier_storage.has_transition_recipes) {
        pending.metadata.record_flags |=
            IREE_HAL_REPLAY_FILE_RECORD_FLAG_MEMORY_TRANSITION_RECIPES;
      }
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending, status);
    }
  }
  iree_hal_replay_recorder_barriers_deinitialize(queue->host_allocator,
                                                 &barrier_storage);
  iree_allocator_free(queue->host_allocator, signal_payloads);
  iree_allocator_free(queue->host_allocator, wait_payloads);
  iree_allocator_free(queue->host_allocator, serialized);
  for (iree_host_size_t i = 0; i < temporary_count; ++i) {
    iree_hal_replay_recorder_buffer_release_temporary(temporary[i]);
  }
  iree_allocator_free(queue->host_allocator, storage);
  return status;
}
