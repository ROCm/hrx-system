// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_transfer.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_staging.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_amdgpu_transfer_transaction_t
    iree_hal_amdgpu_transfer_transaction_t;

typedef struct iree_hal_amdgpu_transfer_child_t {
  // Transaction containing this child and its captured operation.
  iree_hal_amdgpu_transfer_transaction_t* transaction;
  // Index of the captured operation executed by this child.
  iree_host_size_t operation_index;
  // Prepared staging transfer for upload/download operations.
  iree_hal_amdgpu_staging_transfer_t* staging_transfer;
  // Owned completion status transferred to |completion_post_drain|.
  iree_status_t completion_status;
  // Continuation that aggregates hardware or staging completion.
  iree_hal_amdgpu_host_queue_post_drain_action_t completion_post_drain;
  // Continuation that retries native submission after capacity is reclaimed.
  iree_hal_amdgpu_host_queue_post_drain_action_t capacity_post_drain;
} iree_hal_amdgpu_transfer_child_t;

struct iree_hal_amdgpu_transfer_transaction_t {
  // Resource header retained across waits and child submissions.
  iree_hal_resource_t resource;
  // Arena containing this transaction and all captured variable-size data.
  iree_arena_allocator_t arena;
  // Queue receiving all transaction operations.
  iree_hal_amdgpu_host_queue_t* queue;
  // Retained buffers referenced by captured operations.
  iree_hal_resource_set_t* resource_set;
  // Serializes child completion and terminal status ownership.
  iree_slim_mutex_t mutex;
  // Captured operation array.
  iree_hal_transfer_operation_t* operations;
  // Per-operation child state parallel to |operations|.
  iree_hal_amdgpu_transfer_child_t* children;
  // Number of entries in |operations| and |children|.
  iree_host_size_t operation_count;
  // Number of non-empty children that have not completed.
  iree_host_size_t remaining_child_count;
  // Cloned signal list published after every child completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Owned aggregate of child and queue failures.
  iree_status_t failure_status;
  // Owned status passed from the wait barrier to |start_post_drain|.
  iree_status_t start_status;
  // Continuation that starts children after the wait barrier retires.
  iree_hal_amdgpu_host_queue_post_drain_action_t start_post_drain;
  // Continuation that retries final signal publication after capacity returns.
  iree_hal_amdgpu_host_queue_post_drain_action_t signal_post_drain;
};

static iree_device_size_t iree_hal_amdgpu_transfer_operation_length(
    const iree_hal_transfer_operation_t* operation) {
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      return operation->fill.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return operation->update.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      return operation->copy.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      return operation->upload.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      return operation->download.length;
    default:
      return 0;
  }
}

static void iree_hal_amdgpu_transfer_transaction_destroy(
    iree_hal_resource_t* base_resource) {
  iree_hal_amdgpu_transfer_transaction_t* transaction =
      (iree_hal_amdgpu_transfer_transaction_t*)base_resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (transaction->children) {
    for (iree_host_size_t i = 0; i < transaction->operation_count; ++i) {
      iree_hal_amdgpu_staging_transfer_release(
          transaction->children[i].staging_transfer);
      iree_status_free(transaction->children[i].completion_status);
    }
  }
  iree_status_free(transaction->failure_status);
  iree_status_free(transaction->start_status);
  iree_hal_semaphore_list_release(transaction->signal_semaphore_list);
  iree_hal_resource_set_free(transaction->resource_set);
  iree_slim_mutex_deinitialize(&transaction->mutex);
  iree_arena_allocator_t arena = transaction->arena;
  iree_arena_deinitialize(&arena);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_transfer_transaction_vtable = {
        .destroy = iree_hal_amdgpu_transfer_transaction_destroy,
};

static iree_status_t iree_hal_amdgpu_transfer_validate_buffer(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_buffer_t* buffer,
    iree_hal_semaphore_list_t wait_semaphore_list) {
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  if (!iree_hal_buffer_placement_is_undefined(placement) &&
      placement.device != queue->logical_device) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transfer buffer belongs to a different HAL device");
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (iree_hal_amdgpu_buffer_device_pointer(allocated_buffer)) {
    return iree_ok_status();
  }
  if (!iree_hal_amdgpu_transient_buffer_isa(allocated_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transfer buffer is not backed by the AMDGPU driver");
  }
  if (iree_hal_amdgpu_transient_buffer_is_deallocated(allocated_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transfer references a deallocated queue_alloca buffer");
  }
  if (wait_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transfer references an unstaged queue_alloca buffer without waiting "
        "on its allocation signal");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_transfer_capture_buffer(
    iree_hal_amdgpu_transfer_transaction_t* transaction,
    iree_hal_buffer_t* buffer, iree_hal_semaphore_list_t wait_semaphore_list) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transfer_validate_buffer(
      transaction->queue, buffer, wait_semaphore_list));
  return iree_hal_resource_set_insert(transaction->resource_set, 1, &buffer);
}

static iree_status_t iree_hal_amdgpu_transfer_capture_operation(
    iree_hal_amdgpu_transfer_transaction_t* transaction,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_host_size_t operation_index) {
  iree_hal_transfer_operation_t* operation =
      &transaction->operations[operation_index];
  iree_hal_amdgpu_transfer_child_t* child =
      &transaction->children[operation_index];
  child->transaction = transaction;
  child->operation_index = operation_index;

  if (iree_hal_amdgpu_transfer_operation_length(operation) == 0) {
    return iree_ok_status();
  }
  ++transaction->remaining_child_count;

  iree_status_t status = iree_ok_status();
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL: {
      void* pattern = NULL;
      status = iree_arena_allocate(&transaction->arena,
                                   operation->fill.pattern_length, &pattern);
      if (iree_status_is_ok(status)) {
        memcpy(pattern, operation->fill.pattern,
               operation->fill.pattern_length);
        operation->fill.pattern = pattern;
        status = iree_hal_amdgpu_transfer_capture_buffer(
            transaction, operation->fill.target_buffer, wait_semaphore_list);
      }
      break;
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE: {
      void* source_copy = NULL;
      status = iree_arena_allocate(&transaction->arena,
                                   (iree_host_size_t)operation->update.length,
                                   &source_copy);
      if (iree_status_is_ok(status)) {
        memcpy(source_copy,
               (const uint8_t*)operation->update.source_buffer +
                   operation->update.source_offset,
               (iree_host_size_t)operation->update.length);
        operation->update.source_buffer = source_copy;
        operation->update.source_offset = 0;
        status = iree_hal_amdgpu_transfer_capture_buffer(
            transaction, operation->update.target_buffer, wait_semaphore_list);
      }
      break;
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      status = iree_hal_amdgpu_transfer_capture_buffer(
          transaction, operation->copy.source_buffer, wait_semaphore_list);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdgpu_transfer_capture_buffer(
            transaction, operation->copy.target_buffer, wait_semaphore_list);
      }
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      status = iree_hal_amdgpu_transfer_capture_buffer(
          transaction, operation->upload.target_buffer, wait_semaphore_list);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdgpu_staging_transfer_create_upload(
            transaction->queue, operation->upload.source,
            operation->upload.target_buffer, operation->upload.target_offset,
            operation->upload.length, &child->staging_transfer);
      }
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      status = iree_hal_amdgpu_transfer_capture_buffer(
          transaction, operation->download.source_buffer, wait_semaphore_list);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdgpu_staging_transfer_create_download(
            transaction->queue, operation->download.source_buffer,
            operation->download.source_offset, operation->download.target,
            operation->download.length, &child->staging_transfer);
      }
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown transfer operation type %u",
                                (uint32_t)operation->type);
      break;
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        status, "AMDGPU transfer operation %" PRIhsz, operation_index);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_transfer_transaction_create(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_amdgpu_transfer_transaction_t** out_transaction) {
  IREE_ASSERT_ARGUMENT(out_transaction);

  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);
  iree_hal_amdgpu_transfer_transaction_t* transaction = NULL;
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*transaction), (void**)&transaction);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  memset(transaction, 0, sizeof(*transaction));
  iree_hal_resource_initialize(&iree_hal_amdgpu_transfer_transaction_vtable,
                               &transaction->resource);
  memcpy(&transaction->arena, &arena, sizeof(arena));
  transaction->queue = queue;
  transaction->operation_count = operation_count;
  iree_slim_mutex_initialize(&transaction->mutex);

  status = iree_hal_resource_set_allocate(queue->block_pool,
                                          &transaction->resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(
        &signal_semaphore_list, iree_arena_allocator(&transaction->arena),
        &transaction->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t operations_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &operations_size,
        IREE_STRUCT_FIELD(operation_count, iree_hal_transfer_operation_t,
                          NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&transaction->arena, operations_size,
                                   (void**)&transaction->operations);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t children_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &children_size,
        IREE_STRUCT_FIELD(operation_count, iree_hal_amdgpu_transfer_child_t,
                          NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&transaction->arena, children_size,
                                   (void**)&transaction->children);
    }
  }
  if (iree_status_is_ok(status)) {
    memcpy(transaction->operations, operations,
           operation_count * sizeof(*operations));
    memset(transaction->children, 0,
           operation_count * sizeof(*transaction->children));
    for (iree_host_size_t i = 0;
         i < operation_count && iree_status_is_ok(status); ++i) {
      status = iree_hal_amdgpu_transfer_capture_operation(
          transaction, wait_semaphore_list, i);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(transaction->resource_set);
    *out_transaction = transaction;
  } else {
    iree_hal_resource_release(&transaction->resource);
  }
  return status;
}

static void iree_hal_amdgpu_transfer_fail_signals(
    iree_hal_amdgpu_transfer_transaction_t* transaction, iree_status_t status) {
  if (iree_hal_semaphore_list_is_empty(transaction->signal_semaphore_list)) {
    iree_status_free(status);
  } else {
    iree_hal_semaphore_list_fail(transaction->signal_semaphore_list, status);
  }
}

static void iree_hal_amdgpu_transfer_signal_post_drain(void* user_data);

static void iree_hal_amdgpu_transfer_publish_signals(
    iree_hal_amdgpu_transfer_transaction_t* transaction) {
  iree_status_t status = transaction->failure_status;
  transaction->failure_status = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transfer_fail_signals(transaction, status);
    return;
  }
  if (iree_hal_semaphore_list_is_empty(transaction->signal_semaphore_list)) {
    return;
  }

  status = iree_hal_amdgpu_host_queue_clone_error_status(transaction->queue);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transfer_fail_signals(transaction, status);
    return;
  }

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  iree_slim_mutex_lock(&transaction->queue->locks.submission_mutex);
  bool ready = false;
  status = iree_hal_amdgpu_host_queue_try_submit_barrier(
      transaction->queue, &resolution, transaction->signal_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){0},
      /*operation_resources=*/NULL, /*operation_resource_count=*/0,
      /*profile_event_info=*/NULL,
      iree_hal_amdgpu_host_queue_post_commit_callback_null(),
      /*resource_set=*/NULL,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready,
      /*out_submission_id=*/NULL);
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&transaction->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        transaction->queue, &transaction->signal_post_drain,
        iree_hal_amdgpu_transfer_signal_post_drain, transaction);
  }
  iree_slim_mutex_unlock(&transaction->queue->locks.submission_mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transfer_fail_signals(transaction, status);
  }
}

static void iree_hal_amdgpu_transfer_signal_post_drain(void* user_data) {
  iree_hal_amdgpu_transfer_transaction_t* transaction =
      (iree_hal_amdgpu_transfer_transaction_t*)user_data;
  iree_hal_amdgpu_transfer_publish_signals(transaction);
  iree_hal_resource_release(&transaction->resource);
}

static void iree_hal_amdgpu_transfer_child_finish(
    iree_hal_amdgpu_transfer_child_t* child, iree_status_t status) {
  iree_hal_amdgpu_transfer_transaction_t* transaction = child->transaction;
  bool is_transaction_complete = false;
  iree_slim_mutex_lock(&transaction->mutex);
  if (!iree_status_is_ok(status)) {
    transaction->failure_status =
        iree_status_join(transaction->failure_status, status);
  }
  --transaction->remaining_child_count;
  is_transaction_complete = transaction->remaining_child_count == 0;
  iree_slim_mutex_unlock(&transaction->mutex);
  if (is_transaction_complete) {
    iree_hal_amdgpu_transfer_publish_signals(transaction);
  }
}

static void iree_hal_amdgpu_transfer_child_completion_post_drain(
    void* user_data) {
  iree_hal_amdgpu_transfer_child_t* child =
      (iree_hal_amdgpu_transfer_child_t*)user_data;
  iree_status_t status = child->completion_status;
  child->completion_status = iree_ok_status();
  iree_hal_amdgpu_transfer_child_finish(child, status);
  iree_hal_resource_release(&child->transaction->resource);
}

static void iree_hal_amdgpu_transfer_child_complete(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_transfer_child_t* child =
      (iree_hal_amdgpu_transfer_child_t*)user_data;
  child->completion_status =
      iree_status_is_ok(status) ? iree_ok_status() : iree_status_clone(status);
  iree_hal_resource_retain(&child->transaction->resource);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      child->transaction->queue, &child->completion_post_drain,
      iree_hal_amdgpu_transfer_child_completion_post_drain, child);
}

static iree_status_t iree_hal_amdgpu_transfer_try_submit_native_under_lock(
    iree_hal_amdgpu_transfer_child_t* child, bool* out_ready) {
  iree_hal_amdgpu_transfer_transaction_t* transaction = child->transaction;
  iree_hal_amdgpu_host_queue_t* queue = transaction->queue;
  const iree_hal_transfer_operation_t* operation =
      &transaction->operations[child->operation_index];
  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  const iree_hal_amdgpu_reclaim_action_t action = {
      .fn = iree_hal_amdgpu_transfer_child_complete,
      .user_data = child,
  };

  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL: {
      uint64_t pattern_bits = 0;
      memcpy(&pattern_bits, operation->fill.pattern,
             operation->fill.pattern_length);
      return iree_hal_amdgpu_host_queue_submit_fill_with_action(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          operation->fill.target_buffer, operation->fill.target_offset,
          operation->fill.length, pattern_bits, operation->fill.pattern_length,
          operation->fill.flags, action, &transaction->resource,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
          out_ready);
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return iree_hal_amdgpu_host_queue_submit_update_with_action(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          operation->update.source_buffer, operation->update.source_offset,
          operation->update.target_buffer, operation->update.target_offset,
          operation->update.length, operation->update.flags, action,
          &transaction->resource,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
          out_ready);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY: {
      iree_hal_resource_t* extra_resources[1] = {&transaction->resource};
      return iree_hal_amdgpu_host_queue_submit_copy_with_action(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          operation->copy.source_buffer, operation->copy.source_offset,
          operation->copy.target_buffer, operation->copy.target_offset,
          operation->copy.length, operation->copy.flags,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE, action,
          extra_resources, IREE_ARRAYSIZE(extra_resources),
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
          out_ready);
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "non-native transfer operation reached native "
                              "submission");
  }
}

static void iree_hal_amdgpu_transfer_child_capacity_post_drain(
    void* user_data) {
  iree_hal_amdgpu_transfer_child_t* child =
      (iree_hal_amdgpu_transfer_child_t*)user_data;
  iree_hal_amdgpu_transfer_transaction_t* transaction = child->transaction;
  iree_slim_mutex_lock(&transaction->queue->locks.submission_mutex);
  bool ready = false;
  iree_status_t status =
      iree_hal_amdgpu_transfer_try_submit_native_under_lock(child, &ready);
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&transaction->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        transaction->queue, &child->capacity_post_drain,
        iree_hal_amdgpu_transfer_child_capacity_post_drain, child);
  }
  iree_slim_mutex_unlock(&transaction->queue->locks.submission_mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transfer_child_finish(child, status);
  }
  iree_hal_resource_release(&transaction->resource);
}

static void iree_hal_amdgpu_transfer_enqueue_capacity_retry(
    iree_hal_amdgpu_transfer_child_t* child) {
  iree_hal_resource_retain(&child->transaction->resource);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      child->transaction->queue, &child->capacity_post_drain,
      iree_hal_amdgpu_transfer_child_capacity_post_drain, child);
}

static void iree_hal_amdgpu_transfer_start(
    iree_hal_amdgpu_transfer_transaction_t* transaction) {
  iree_status_t queue_status =
      iree_hal_amdgpu_host_queue_clone_error_status(transaction->queue);
  if (!iree_status_is_ok(queue_status)) {
    transaction->remaining_child_count = 0;
    transaction->failure_status = queue_status;
    iree_hal_amdgpu_transfer_publish_signals(transaction);
    return;
  }

  iree_slim_mutex_lock(&transaction->queue->locks.submission_mutex);
  for (iree_host_size_t i = 0; i < transaction->operation_count; ++i) {
    iree_hal_amdgpu_transfer_child_t* child = &transaction->children[i];
    const iree_hal_transfer_operation_t* operation =
        &transaction->operations[i];
    if (iree_hal_amdgpu_transfer_operation_length(operation) == 0 ||
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD ||
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      continue;
    }
    bool ready = false;
    iree_status_t status =
        iree_hal_amdgpu_transfer_try_submit_native_under_lock(child, &ready);
    if (!iree_status_is_ok(status)) {
      child->completion_status = status;
    } else if (!ready) {
      iree_hal_amdgpu_transfer_enqueue_capacity_retry(child);
    }
  }
  iree_slim_mutex_unlock(&transaction->queue->locks.submission_mutex);

  for (iree_host_size_t i = 0; i < transaction->operation_count; ++i) {
    iree_hal_amdgpu_transfer_child_t* child = &transaction->children[i];
    if (!iree_status_is_ok(child->completion_status)) {
      iree_status_t status = child->completion_status;
      child->completion_status = iree_ok_status();
      iree_hal_amdgpu_transfer_child_finish(child, status);
    }
  }

  for (iree_host_size_t i = 0; i < transaction->operation_count; ++i) {
    iree_hal_amdgpu_transfer_child_t* child = &transaction->children[i];
    if (!child->staging_transfer) continue;
    iree_hal_amdgpu_staging_transfer_t* staging_transfer =
        child->staging_transfer;
    child->staging_transfer = NULL;
    iree_status_t status = iree_hal_amdgpu_staging_transfer_start(
        staging_transfer,
        (iree_hal_amdgpu_reclaim_action_t){
            .fn = iree_hal_amdgpu_transfer_child_complete,
            .user_data = child,
        },
        &transaction->resource);
    iree_hal_amdgpu_staging_transfer_release(staging_transfer);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdgpu_transfer_child_finish(child, status);
    }
  }
}

static void iree_hal_amdgpu_transfer_start_post_drain(void* user_data) {
  iree_hal_amdgpu_transfer_transaction_t* transaction =
      (iree_hal_amdgpu_transfer_transaction_t*)user_data;
  iree_status_t status = transaction->start_status;
  transaction->start_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_transfer_start(transaction);
  } else {
    transaction->remaining_child_count = 0;
    transaction->failure_status = status;
    iree_hal_amdgpu_transfer_publish_signals(transaction);
  }
  iree_hal_resource_release(&transaction->resource);
}

static void iree_hal_amdgpu_transfer_waits_complete(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  iree_hal_amdgpu_transfer_transaction_t* transaction =
      (iree_hal_amdgpu_transfer_transaction_t*)user_data;
  // A NULL entry identifies failure of a deferred host action before it was
  // submitted. This callback is then outside notification-ring drain and must
  // terminate the transaction directly; no later drain is guaranteed to run
  // a queued continuation on a failed or shutting-down queue.
  if (!entry) {
    transaction->remaining_child_count = 0;
    transaction->failure_status = iree_status_clone(status);
    iree_hal_amdgpu_transfer_publish_signals(transaction);
    return;
  }
  transaction->start_status =
      iree_status_is_ok(status) ? iree_ok_status() : iree_status_clone(status);
  iree_hal_resource_retain(&transaction->resource);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      transaction->queue, &transaction->start_post_drain,
      iree_hal_amdgpu_transfer_start_post_drain, transaction);
}

iree_status_t iree_hal_amdgpu_host_queue_enqueue_transfer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_host_size_t active_operation_count = 0;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    active_operation_count +=
        iree_hal_amdgpu_transfer_operation_length(&operations[i]) != 0;
  }
  if (active_operation_count == 0) {
    return iree_hal_queue_barrier(&queue->base, wait_semaphore_list,
                                  signal_semaphore_list,
                                  IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  }

  iree_hal_amdgpu_transfer_transaction_t* transaction = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transfer_transaction_create(
      queue, wait_semaphore_list, signal_semaphore_list, operation_count,
      operations, &transaction));

  if (wait_semaphore_list.count == 0) {
    iree_hal_amdgpu_transfer_start(transaction);
    iree_hal_resource_release(&transaction->resource);
    return iree_ok_status();
  }

  iree_hal_resource_t* resources[1] = {&transaction->resource};
  iree_status_t status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_transfer_waits_complete,
          .user_data = transaction,
      },
      resources, IREE_ARRAYSIZE(resources));
  iree_hal_resource_release(&transaction->resource);
  return status;
}
