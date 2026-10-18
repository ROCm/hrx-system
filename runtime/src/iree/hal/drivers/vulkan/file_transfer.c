// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/file_transfer.h"

#include "iree/async/operations/file.h"
#include "iree/async/proactor.h"

// Request records, not payload slots. Each continuation advances by one stripe
// in the caller's final backing; no staging resource or copy submission exists.
#define IREE_HAL_VULKAN_FILE_TRANSFER_LANE_COUNT 4
#define IREE_HAL_VULKAN_FILE_TRANSFER_CHUNK_SIZE (1u << 20)

typedef struct iree_hal_vulkan_file_transfer_t iree_hal_vulkan_file_transfer_t;

typedef struct iree_hal_vulkan_file_transfer_lane_t {
  // Transfer retaining this placed continuation until all lanes retire.
  iree_hal_vulkan_file_transfer_t* transfer;
  // Start of this lane's current chunk in the prepared buffer span.
  iree_host_size_t offset;
  // Bytes completed within the current chunk.
  iree_host_size_t progress;
  // Native operation storage reused only after terminal completion.
  union {
    // File-to-buffer request.
    iree_async_file_read_operation_t read;
    // Buffer-to-file request.
    iree_async_file_write_operation_t write;
  } operation;
} iree_hal_vulkan_file_transfer_lane_t;

struct iree_hal_vulkan_file_transfer_t {
  // Retained by the entry boundary, accepted I/O and terminal publication.
  iree_hal_resource_t resource;
  // Queue owning dependency resolution and final native publication. Borrowed.
  iree_hal_vulkan_queue_t* queue;
  // Retained native file and its imported proactor handle.
  iree_hal_file_t* file;
  // Retained logical view preserving the prepared import owner.
  iree_hal_buffer_t* buffer;
  // Logical byte offset, resolved only after allocation dependencies retire.
  iree_device_size_t buffer_offset;
  // Exact byte length in the caller's logical view.
  iree_device_size_t length;
  // Original host allocation and its cache-maintenance coordinates.
  iree_hal_vulkan_buffer_host_span_t span;
  // Exactly READ or WRITE, from the file's perspective.
  iree_hal_memory_access_t file_access;
  // First file byte corresponding to the prepared span.
  uint64_t file_offset;
  // Geometry-qualified maximum bytes per continuation.
  iree_host_size_t chunk_size;
  // Active stripe count, fixed before any I/O is submitted.
  uint32_t lane_count;
  // Protects only terminal lane accounting and aggregate failure ownership.
  iree_slim_mutex_t mutex;
  // Lanes whose accepted I/O has not yet retired.
  uint32_t remaining_lanes;
  // Owned aggregate of native submission/completion failures.
  iree_status_t failure_status;
  // Captured completion timepoints, published after every lane retires.
  iree_hal_semaphore_list_t signals;
  // Caller-requested dependency after native I/O and cache maintenance.
  iree_hal_vulkan_queue_barriers_t publication;
  // Fixed native request capacity; independent of payload size.
  iree_hal_vulkan_file_transfer_lane_t
      lanes[IREE_HAL_VULKAN_FILE_TRANSFER_LANE_COUNT];
};

static void iree_hal_vulkan_file_transfer_release_captures(
    iree_hal_vulkan_file_transfer_t* transfer) {
  iree_hal_buffer_release(transfer->buffer);
  transfer->buffer = NULL;
  iree_hal_file_release(transfer->file);
  transfer->file = NULL;
}

static void iree_hal_vulkan_file_transfer_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_vulkan_file_transfer_t* transfer =
      (iree_hal_vulkan_file_transfer_t*)resource;
  iree_hal_vulkan_file_transfer_release_captures(transfer);
  iree_hal_semaphore_list_free(transfer->signals,
                               transfer->queue->host_allocator);
  iree_slim_mutex_deinitialize(&transfer->mutex);
  iree_allocator_free(transfer->queue->host_allocator, transfer);
}

static const iree_hal_resource_vtable_t iree_hal_vulkan_file_transfer_vtable = {
    .destroy = iree_hal_vulkan_file_transfer_destroy,
};

// Consumes the I/O lifetime reference. Caller resources are released before
// publication so a terminal observer may immediately retire its allocation.
static void iree_hal_vulkan_file_transfer_publish(
    iree_hal_vulkan_file_transfer_t* transfer, iree_status_t status) {
  iree_hal_vulkan_file_transfer_release_captures(transfer);
  if (iree_status_is_ok(status)) {
    // A native submission after the host writes performs the host-to-device
    // domain transition. Signaling a host timeline alone is not sufficient
    // for consumers already queued before those writes occurred.
    status = iree_hal_vulkan_queue_submit_barrier_with_action(
        transfer->queue, iree_hal_semaphore_list_empty(), transfer->signals,
        &transfer->publication,
        (iree_hal_vulkan_queue_completion_action_t){
            .resource = &transfer->resource,
        });
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(transfer->signals, status);
  }
  iree_hal_resource_release(&transfer->resource);
}

static void iree_hal_vulkan_file_transfer_lane_finish(
    iree_hal_vulkan_file_transfer_lane_t* lane, iree_status_t status) {
  iree_hal_vulkan_file_transfer_t* transfer = lane->transfer;
  iree_slim_mutex_lock(&transfer->mutex);
  transfer->failure_status = iree_status_join(transfer->failure_status, status);
  const bool finished = --transfer->remaining_lanes == 0;
  if (finished) {
    status = transfer->failure_status;
    transfer->failure_status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&transfer->mutex);
  if (!finished) {
    return;
  }
  if (iree_status_is_ok(status) &&
      transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ &&
      !transfer->span.coherent) {
    status = iree_hal_vulkan_buffer_flush_host_span(&transfer->span);
  }
  iree_hal_vulkan_file_transfer_publish(transfer, status);
}

static iree_status_t iree_hal_vulkan_file_transfer_issue(
    iree_hal_vulkan_file_transfer_lane_t* lane);

static void iree_hal_vulkan_file_transfer_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_vulkan_file_transfer_lane_t* lane = user_data;
  iree_hal_vulkan_file_transfer_t* transfer = lane->transfer;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t count =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? lane->operation.read.bytes_read
            : lane->operation.write.bytes_written;
    lane->progress += count;
    if (count == 0) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "native file transfer stopped before its "
                                "requested range completed");
    } else {
      const iree_host_size_t chunk_length =
          iree_min(transfer->chunk_size,
                   transfer->span.contents.data_length - lane->offset);
      if (lane->progress == chunk_length) {
        lane->offset += transfer->lane_count * transfer->chunk_size;
        lane->progress = 0;
      }
      if (lane->offset < transfer->span.contents.data_length) {
        status = iree_hal_vulkan_file_transfer_issue(lane);
        if (iree_status_is_ok(status)) {
          return;
        }
      }
    }
  }
  iree_hal_vulkan_file_transfer_lane_finish(lane, status);
}

static iree_status_t iree_hal_vulkan_file_transfer_issue(
    iree_hal_vulkan_file_transfer_lane_t* lane) {
  iree_hal_vulkan_file_transfer_t* transfer = lane->transfer;
  const bool reading = transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ;
  const iree_host_size_t offset = lane->offset + lane->progress;
  const iree_host_size_t chunk_length = iree_min(
      transfer->chunk_size, transfer->span.contents.data_length - lane->offset);
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(transfer->file);
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_prepare_transfer(
      reading ? &requirements->read : &requirements->write,
      transfer->file_offset + offset, transfer->span.contents.data + offset,
      chunk_length - lane->progress, transfer->chunk_size, &length));
  iree_async_file_t* file = iree_hal_file_async_handle(transfer->file);
  iree_async_operation_t* operation = NULL;
  if (reading) {
    iree_async_file_read_operation_t* read = &lane->operation.read;
    iree_async_operation_zero(&read->base, sizeof(*read));
    read->file = file;
    read->offset = transfer->file_offset + offset;
    read->buffer =
        iree_async_span_from_ptr(transfer->span.contents.data + offset, length);
    operation = &read->base;
  } else {
    iree_async_file_write_operation_t* write = &lane->operation.write;
    iree_async_operation_zero(&write->base, sizeof(*write));
    write->file = file;
    write->offset = transfer->file_offset + offset;
    write->buffer =
        iree_async_span_from_ptr(transfer->span.contents.data + offset, length);
    operation = &write->base;
  }
  iree_async_operation_initialize(operation,
                                  reading
                                      ? IREE_ASYNC_OPERATION_TYPE_FILE_READ
                                      : IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_hal_vulkan_file_transfer_complete, lane);
  return iree_async_proactor_submit_one(file->proactor, operation);
}

static void iree_hal_vulkan_file_transfer_start(
    void* user_data, const iree_status_t completion_status) {
  iree_hal_vulkan_file_transfer_t* transfer = user_data;
  iree_status_t status = iree_status_clone(completion_status);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_buffer_resolve_host_span(
        transfer->buffer, transfer->buffer_offset, transfer->length,
        &transfer->span);
  }
  if (iree_status_is_ok(status) && !transfer->span.allocation) {
    // The entry boundary already ran. Hand captured lifetimes to the staging
    // operation before starting it, without a second empty native submission.
    iree_hal_vulkan_queue_completion_action_t action;
    status = iree_hal_vulkan_queue_prepare_staged_file_transfer(
        transfer->queue, transfer->signals, transfer->file_access,
        transfer->file, transfer->file_offset, transfer->buffer,
        transfer->buffer_offset, transfer->length, &transfer->publication,
        &action);
    iree_hal_vulkan_file_transfer_release_captures(transfer);
    if (iree_status_is_ok(status)) {
      action.fn(action.user_data, iree_ok_status());
      iree_hal_resource_release(action.resource);
    } else {
      iree_hal_semaphore_list_fail(transfer->signals, status);
    }
    return;
  }
  if (iree_status_is_ok(status)) {
    const iree_async_file_requirements_t* requirements =
        iree_hal_file_requirements(transfer->file);
    status = iree_hal_file_prepare_transfer(
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? &requirements->read
            : &requirements->write,
        transfer->file_offset, transfer->span.contents.data,
        transfer->span.contents.data_length,
        IREE_HAL_VULKAN_FILE_TRANSFER_CHUNK_SIZE, &transfer->chunk_size);
  }
  if (iree_status_is_ok(status) &&
      transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
    status = iree_hal_vulkan_buffer_invalidate_host_span(&transfer->span);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_file_transfer_release_captures(transfer);
    iree_hal_semaphore_list_fail(transfer->signals, status);
    return;
  }
  transfer->lane_count = (uint32_t)iree_min(
      IREE_HAL_VULKAN_FILE_TRANSFER_LANE_COUNT,
      1 + (transfer->span.contents.data_length - 1) / transfer->chunk_size);
  transfer->remaining_lanes = transfer->lane_count;
  for (uint32_t i = 0; i < transfer->lane_count; ++i) {
    transfer->lanes[i].transfer = transfer;
    transfer->lanes[i].offset = i * transfer->chunk_size;
  }
  // One lifetime covers every accepted request and final cache publication. The
  // entry action additionally retains the object until this function returns,
  // including when all native completions race initial submission.
  iree_hal_resource_retain(&transfer->resource);
  for (uint32_t i = 0; i < transfer->lane_count; ++i) {
    status = iree_hal_vulkan_file_transfer_issue(&transfer->lanes[i]);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_file_transfer_lane_finish(&transfer->lanes[i], status);
    }
  }
}

iree_status_t iree_hal_vulkan_queue_submit_file_transfer(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_file_t* file,
    iree_hal_memory_access_t file_access, uint64_t file_offset,
    iree_hal_buffer_t* buffer, iree_device_size_t buffer_offset,
    iree_device_size_t length,
    const iree_hal_vulkan_queue_barriers_t* barriers) {
  if (length == 0) {
    return iree_hal_vulkan_queue_submit_barrier(
        queue, wait_semaphore_list, signal_semaphore_list, barriers);
  }
  iree_hal_vulkan_file_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*transfer), (void**)&transfer));
  iree_hal_resource_initialize(&iree_hal_vulkan_file_transfer_vtable,
                               &transfer->resource);
  transfer->queue = queue;
  transfer->file = file;
  iree_hal_file_retain(file);
  transfer->buffer = buffer;
  iree_hal_buffer_retain(buffer);
  transfer->buffer_offset = buffer_offset;
  transfer->length = length;
  transfer->file_access = file_access;
  transfer->file_offset = file_offset;
  iree_slim_mutex_initialize(&transfer->mutex);
  if (barriers) {
    transfer->publication.after = barriers->after;
  }
  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, queue->host_allocator, &transfer->signals);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_barriers_t entry = {
        .before = barriers ? barriers->before : (iree_hal_vulkan_barrier_t){0},
    };
    if (file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
      // Native I/O reads on the host even though the public operation is a
      // queue transfer. This implementation-owned domain transition remains
      // necessary when the caller omits optional payload visibility barriers.
      entry.before.source_stage_mask |=
          IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS;
      entry.before.source_access_mask |= VK_ACCESS_2_MEMORY_WRITE_BIT;
      entry.before.target_stage_mask |= IREE_HAL_EXECUTION_STAGE_HOST;
      entry.before.target_access_mask |= VK_ACCESS_2_HOST_READ_BIT;
    }
    status = iree_hal_vulkan_queue_submit_barrier_with_action(
        queue, wait_semaphore_list, iree_hal_semaphore_list_empty(), &entry,
        (iree_hal_vulkan_queue_completion_action_t){
            .fn = iree_hal_vulkan_file_transfer_start,
            .user_data = transfer,
            .resource = &transfer->resource,
        });
  }
  iree_hal_resource_release(&transfer->resource);
  return status;
}
