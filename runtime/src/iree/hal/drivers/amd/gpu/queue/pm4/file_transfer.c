// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/file_transfer.h"

#include "iree/async/semaphore.h"
#include "iree/hal/drivers/amd/fabric/file_transfer.h"

// The existing native transfer engine provides its own bounded GPU staging.
// Only platform files need this additional ordinary host span for native I/O.
#define IREE_HAL_AMD_GPU_PM4_FILE_STAGING_CAPACITY (1024u * 1024u)

typedef struct iree_hal_amd_gpu_pm4_file_transfer_t {
  // Initial admission or terminal I/O continuation on the existing service.
  iree_hal_memory_maintenance_entry_t entry;
  // Borrowed queue-owned services, dominated by the retained device.
  const iree_hal_amd_gpu_pm4_file_executor_t* executor;
  // Retained parent dominating every borrowed executor field.
  iree_hal_device_t* device;
  // Exact capture extent returned to the reusable metadata pool.
  iree_host_size_t allocation_size;
  // Retained file/buffer, partial I/O state, and optional owned host staging.
  iree_hal_amd_fabric_file_transfer_t transfer;
  // Copied and retained dependencies, joined before accessing payloads.
  iree_hal_semaphore_list_t waits;
  // Copied and retained public completion points.
  iree_hal_semaphore_list_t signals;
  // Prepared cache boundary preceding the first native buffer access.
  iree_hal_amd_gpu_pm4_barrier_t before;
  // Prepared cache boundary following every file and host effect.
  iree_hal_amd_gpu_pm4_barrier_t after;
  // I/O callback status consumed by the next service turn.
  iree_status_t status;
  // Host-completed native copy progress, private to this file operation.
  struct {
    // Owned timeline, reused monotonically across native chunks.
    iree_hal_semaphore_t* semaphore;
    // Current native copy completion value.
    uint64_t value;
    // One tracker registration, finalized before the next chunk starts.
    iree_hal_amd_gpu_host_wait_t wait;
    // Reusable tracker allocation sized for initial waits and single copies.
    iree_allocator_inline_storage_t storage;
  } completion;
} iree_hal_amd_gpu_pm4_file_transfer_t;

static void iree_hal_amd_gpu_pm4_file_transfer_free(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation) {
  iree_hal_device_t* device = operation->device;
  iree_hal_semaphore_release(operation->completion.semaphore);
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_gpu_capture_pool_release(
      operation->executor->captures,
      iree_make_byte_span(operation, operation->allocation_size));
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_pm4_file_transfer_retire(void* user_data,
                                                      iree_status_t status) {
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = user_data;
  if (operation->executor->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->executor->profile,
                                              (void*)(operation + 1), false, 0);
  }
  // Native copies, file requests, and the final cache boundary have retired.
  // Keep backing live until dirty cache lines can no longer refer to it.
  iree_hal_amd_fabric_file_transfer_deinitialize(&operation->transfer);
  if (iree_status_is_ok(status)) {
    // Every private point comes from the same queue. Preserve that completed
    // epoch while publishing only after the buffer/pool uses have ended.
    iree_async_single_frontier_t frontier;
    const uint8_t count = iree_async_semaphore_query_frontier(
        (iree_async_semaphore_t*)operation->completion.semaphore,
        iree_async_single_frontier_as_frontier(&frontier), 1);
    IREE_ASSERT_LE(count, 1);
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        iree_async_single_frontier_as_const_frontier(&frontier));
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(operation->executor->context,
                                           operation->device, status);
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_amd_gpu_pm4_file_transfer_free(operation);
}

static void iree_hal_amd_gpu_pm4_file_transfer_complete(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_retire(operation, status);
    return;
  }
  ++operation->completion.value;
  const iree_hal_semaphore_list_t completion = {
      .count = 1,
      .semaphores = &operation->completion.semaphore,
      .payload_values = &operation->completion.value,
  };
  status = iree_hal_amd_gpu_host_wait_begin(
      operation->executor->waits, operation->executor->service, completion,
      iree_hal_amd_gpu_pm4_file_transfer_retire, operation,
      iree_allocator_inline_arena(&operation->completion.storage),
      &operation->completion.wait);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_retire(operation, status);
    return;
  }
  // The service retires captured operands before public completion. Callers
  // may then immediately destroy their pools, which buffers only borrow.
  status = operation->executor->submit(
      operation->executor->queue, NULL, completion,
      (iree_hal_amd_gpu_pm4_barrier_t){0}, operation->after);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->completion.semaphore, status);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_advance(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation);
static void iree_hal_amd_gpu_pm4_file_transfer_copy(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation);

static void iree_hal_amd_gpu_pm4_file_transfer_io_complete(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = (void*)entry;
  iree_status_t status = operation->status;
  operation->status = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
  } else if (operation->transfer.file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    iree_hal_amd_gpu_pm4_file_transfer_copy(operation);
  } else {
    iree_hal_amd_gpu_pm4_file_transfer_advance(operation);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_resume(void* user_data,
                                                      iree_status_t status) {
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = user_data;
  operation->status = status;
  operation->entry.fn = iree_hal_amd_gpu_pm4_file_transfer_io_complete;
  iree_hal_memory_maintenance_enqueue(operation->executor->service,
                                      &operation->entry);
}

static void iree_hal_amd_gpu_pm4_file_transfer_issue(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation) {
  iree_status_t status = iree_hal_amd_fabric_file_transfer_submit(
      &operation->transfer, (iree_hal_amd_fabric_file_transfer_callback_t){
                                .fn = iree_hal_amd_gpu_pm4_file_transfer_resume,
                                .user_data = operation,
                            });
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_copy_complete(
    void* user_data, iree_status_t status) {
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = user_data;
  if (!iree_status_is_ok(status) || !operation->transfer.async.file) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
  } else if (operation->transfer.file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
    iree_hal_amd_gpu_pm4_file_transfer_issue(operation);
  } else {
    iree_hal_amd_gpu_pm4_file_transfer_advance(operation);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_copy(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation) {
  const iree_hal_amd_fabric_file_transfer_t* transfer = &operation->transfer;
  const iree_byte_span_t contents =
      transfer->async.file
          ? transfer->async.chunk.contents
          : iree_make_byte_span(
                transfer->file_contents.data + transfer->file_offset,
                (iree_host_size_t)transfer->length);
  const iree_device_size_t offset =
      transfer->buffer_offset + transfer->async.chunk.offset;
  iree_hal_transfer_operation_t copy = {0};
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    copy.upload.source = contents.data;
    copy.upload.target_buffer = transfer->buffer;
    copy.upload.target_offset = offset;
    copy.upload.length = contents.data_length;
  } else {
    copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    copy.download.source_buffer = transfer->buffer;
    copy.download.source_offset = offset;
    copy.download.target = contents.data;
    copy.download.length = contents.data_length;
  }
  ++operation->completion.value;
  const iree_hal_semaphore_list_t completion = {
      .count = 1,
      .semaphores = &operation->completion.semaphore,
      .payload_values = &operation->completion.value,
  };
  // Enroll before publishing the copy. A tracker allocation failure must not
  // orphan a native user of the operation's borrowed host staging.
  iree_status_t status = iree_hal_amd_gpu_host_wait_begin(
      operation->executor->waits, operation->executor->service, completion,
      iree_hal_amd_gpu_pm4_file_transfer_copy_complete, operation,
      iree_allocator_inline_arena(&operation->completion.storage),
      &operation->completion.wait);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
    return;
  }
  status = operation->executor->submit(
      operation->executor->queue, &copy, completion,
      transfer->async.chunk.offset == 0 ? operation->before
                                        : (iree_hal_amd_gpu_pm4_barrier_t){0},
      (iree_hal_amd_gpu_pm4_barrier_t){0});
  if (!iree_status_is_ok(status)) {
    // The enrolled callback consumes the failure on service and owns cleanup.
    iree_hal_semaphore_fail(operation->completion.semaphore, status);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_advance(
    iree_hal_amd_gpu_pm4_file_transfer_t* operation) {
  iree_hal_amd_fabric_file_transfer_t* transfer = &operation->transfer;
  if (transfer->async.bytes_transferred == transfer->length) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, iree_ok_status());
    return;
  }
  iree_status_t status = iree_hal_amd_fabric_file_transfer_prepare_chunk(
      transfer, transfer->async.staging);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
  } else if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    iree_hal_amd_gpu_pm4_file_transfer_issue(operation);
  } else {
    iree_hal_amd_gpu_pm4_file_transfer_copy(operation);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_start(void* user_data,
                                                     iree_status_t status) {
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
    return;
  }
  if (operation->executor->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->executor->profile,
                                              (void*)(operation + 1), true, 0);
  }
  iree_hal_amd_fabric_file_transfer_t* transfer = &operation->transfer;
  if (!transfer->async.file) {
    iree_hal_amd_gpu_pm4_file_transfer_copy(operation);
    return;
  }
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(transfer->file);
  const iree_host_size_t alignment = iree_max(
      sizeof(void*), transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
                         ? requirements->read.buffer_alignment
                         : requirements->write.buffer_alignment);
  const iree_host_size_t capacity =
      iree_min(transfer->length, IREE_HAL_AMD_GPU_PM4_FILE_STAGING_CAPACITY);
  status = iree_allocator_malloc_aligned(transfer->host_allocator, capacity,
                                         alignment, 0,
                                         (void**)&transfer->async.staging.data);
  if (iree_status_is_ok(status)) {
    transfer->async.staging.data_length = capacity;
    iree_hal_amd_gpu_pm4_file_transfer_advance(operation);
  } else {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
  }
}

static void iree_hal_amd_gpu_pm4_file_transfer_begin(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = (void*)entry;
  if (!operation->waits.count) {
    iree_hal_amd_gpu_pm4_file_transfer_start(operation, iree_ok_status());
    return;
  }
  iree_status_t status = iree_hal_amd_gpu_host_wait_begin(
      operation->executor->waits, operation->executor->service,
      operation->waits, iree_hal_amd_gpu_pm4_file_transfer_start, operation,
      iree_allocator_inline_arena(&operation->completion.storage),
      &operation->completion.wait);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_file_transfer_complete(operation, status);
  }
}

iree_status_t iree_hal_amd_gpu_pm4_file_transfer_submit(
    const iree_hal_amd_gpu_pm4_file_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_hal_amd_gpu_pm4_barrier_t before,
    iree_hal_amd_gpu_pm4_barrier_t after) {
  iree_host_size_t tracker_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_semaphore_wait_tracker_storage_size(
      iree_max(waits.count, 1), &tracker_size));
  iree_host_size_t total_size = 0;
  iree_host_size_t wait_values_offset = 0, wait_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0, signal_semaphores_offset = 0;
  iree_host_size_t tracker_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_pm4_file_transfer_t) +
          (executor->profile ? sizeof(iree_hal_amd_gpu_profile_operation_t)
                             : 0),
      &total_size,
      IREE_STRUCT_FIELD_ALIGNED(waits.count, uint64_t, iree_alignof(uint64_t),
                                &wait_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(waits.count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                &wait_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(signals.count, uint64_t, iree_alignof(uint64_t),
                                &signal_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(signals.count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                &signal_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(tracker_size, uint8_t,
                                iree_alignof(iree_max_align_t),
                                &tracker_offset)));
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
      executor->captures, total_size, &storage));
  iree_hal_amd_gpu_pm4_file_transfer_t* operation = (void*)storage.data;
  *operation = (iree_hal_amd_gpu_pm4_file_transfer_t){
      .entry = {.fn = iree_hal_amd_gpu_pm4_file_transfer_begin},
      .executor = executor,
      .device = executor->queue->queue_family->device,
      .allocation_size = storage.data_length,
      .waits = {.count = waits.count,
                .semaphores = (void*)(storage.data + wait_semaphores_offset),
                .payload_values = (void*)(storage.data + wait_values_offset)},
      .signals =
          {
              .count = signals.count,
              .semaphores = (void*)(storage.data + signal_semaphores_offset),
              .payload_values = (void*)(storage.data + signal_values_offset),
          },
      .before = before,
      .after = after,
      .completion = {.storage = {.capacity = tracker_size,
                                 .buffer = storage.data + tracker_offset}},
  };
  iree_hal_device_retain(operation->device);
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    operation->waits.semaphores[i] = waits.semaphores[i];
    operation->waits.payload_values[i] = waits.payload_values[i];
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  iree_status_t status = iree_hal_amd_fabric_file_transfer_initialize(
      file, file_access, file_offset, buffer, buffer_offset, length,
      executor->captures->host_allocator, &operation->transfer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        operation->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &operation->completion.semaphore);
  }
  if (iree_status_is_ok(status)) {
    if (executor->profile) {
      iree_hal_amd_gpu_profile_operation_t* profile_operation =
          (void*)(operation + 1);
      iree_hal_amd_gpu_profile_operation_initialize(
          executor->profile, executor->queue_ordinal,
          file_access == IREE_HAL_MEMORY_ACCESS_READ
              ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
              : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE,
          waits.count, signals.count, 1, profile_operation);
      profile_operation->event.payload_length = length;
      iree_hal_amd_gpu_profile_operation_defer(profile_operation);
    }
    iree_hal_memory_maintenance_enqueue(executor->service, &operation->entry);
  } else {
    iree_hal_amd_fabric_file_transfer_deinitialize(&operation->transfer);
    iree_hal_amd_gpu_pm4_file_transfer_free(operation);
  }
  return status;
}
