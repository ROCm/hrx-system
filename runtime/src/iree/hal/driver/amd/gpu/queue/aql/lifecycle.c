// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/state.h"

void iree_hal_amd_gpu_aql_queue_fail_issued(iree_hal_amd_gpu_aql_queue_t* queue,
                                            iree_status_code_t code) {
  if (queue->failure_delivered) {
    return;
  }
  const uint64_t begin = (uint64_t)iree_atomic_load(&queue->completed.count,
                                                    iree_memory_order_relaxed);
  const uint64_t end = (uint64_t)iree_atomic_load(&queue->producer.count,
                                                  iree_memory_order_acquire);
  for (uint64_t i = begin; i != end; ++i) {
    iree_hal_amd_gpu_aql_submission_t* submission =
        queue->issued[i & (queue->options.retirement_capacity - 1)].submission;
    iree_status_t status = iree_status_from_code(code);
    iree_hal_amd_gpu_aql_submission_fail(submission, status);
  }
  queue->failure_delivered = true;
}

bool iree_hal_amd_gpu_aql_queue_try_inactivate(iree_hal_queue_t* base,
                                               amdf_status_t* out_status) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (!iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher)) {
    return false;
  }
  *out_status = iree_hal_amd_gpu_user_queue_inactivate(&queue->storage);
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
    iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                        queue->options.source_ordinal);
  }
  return true;
}

bool iree_hal_amd_gpu_aql_queue_retire_failed(iree_hal_queue_t* base,
                                              iree_status_code_t code) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_ASSERT(queue->storage.inactivated,
              "failed retirement requires native access cessation");
  if (!iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher)) {
    return false;
  }
  iree_hal_amd_gpu_aql_queue_fail_issued(queue, code);
  uint64_t position = (uint64_t)iree_atomic_load(&queue->completed.count,
                                                 iree_memory_order_relaxed);
  const uint64_t end = (uint64_t)iree_atomic_load(&queue->producer.count,
                                                  iree_memory_order_acquire);
  for (; position != end; ++position) {
    // Native stop replaces uncommitted firmware completion for lifetime only.
    iree_hal_amd_gpu_aql_retirement_t* action =
        &queue->issued[position & (queue->options.retirement_capacity - 1)];
    iree_hal_amd_gpu_aql_submission_t* submission = action->submission;
    const uint64_t arguments = action->arguments;
    if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
      iree_atomic_store(&queue->completed.transfers, action->transfer.end,
                        iree_memory_order_release);
    }
    if (action->terminal) {
      iree_hal_amd_gpu_aql_submission_release_payload(submission);
    }
    iree_hal_amd_gpu_signal_retire(action->signal, code);
    iree_hal_amd_gpu_signal_release(action->signal);
    iree_hal_amd_gpu_aql_submission_release(submission);
    iree_atomic_store(&queue->completed.arguments, arguments,
                      iree_memory_order_release);
    iree_atomic_store(&queue->completed.count, position + 1,
                      iree_memory_order_release);
  }
  // Native stop and issued retirement precede failed frontier publication.
  // Public operation failures follow each capture's final payload release.
  if (end && queue->frontier.tracker) {
    iree_async_frontier_tracker_fail_axis(queue->frontier.tracker,
                                          queue->frontier.axis,
                                          iree_status_from_code(code));
  }
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
    iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                        queue->options.source_ordinal);
  }
  return true;
}

void iree_hal_amd_gpu_aql_queue_close(iree_hal_queue_t* base) {
  iree_atomic_store(&((iree_hal_amd_gpu_aql_queue_t*)base)->closed, 1,
                    iree_memory_order_release);
}

bool iree_hal_amd_gpu_aql_queue_is_host_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_slim_mutex_lock(&queue->pending.completion_mutex);
  const bool idle = !iree_atomic_load(&queue->pending.host_owners,
                                      iree_memory_order_acquire) &&
                    !iree_atomic_load(&queue->growth.pending_count,
                                      iree_memory_order_acquire) &&
                    !iree_hal_amd_gpu_aql_scratch_is_pending(&queue->scratch) &&
                    iree_hal_amd_gpu_publisher_is_idle(&queue->publisher);
  iree_slim_mutex_unlock(&queue->pending.completion_mutex);
  return idle;
}

bool iree_hal_amd_gpu_aql_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  return iree_hal_amd_gpu_aql_queue_is_host_idle(base) &&
         iree_atomic_load(&queue->producer.count, iree_memory_order_acquire) ==
             iree_atomic_load(&queue->completed.count,
                              iree_memory_order_acquire);
}

void iree_hal_amd_gpu_aql_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  queue->frontier.tracker = tracker;
  queue->frontier.axis = axis;
}

void iree_hal_amd_gpu_aql_queue_set_release(iree_hal_queue_t* base,
                                            bool (*release)(void*,
                                                            iree_hal_queue_t*),
                                            void* user_data) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  queue->release.fn = release;
  queue->release.user_data = user_data;
}

uint64_t iree_hal_amd_gpu_aql_queue_completed_epoch(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  return (uint64_t)iree_atomic_load(&queue->completed.count,
                                    iree_memory_order_acquire);
}

amdf_status_t iree_hal_amd_gpu_aql_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_ASSERT(iree_hal_amd_gpu_aql_queue_is_idle(base),
              "native shutdown requires joined accepted work");
  amdf_status_t status = iree_hal_amd_gpu_user_queue_deinitialize(
      &queue->storage, queue->producer.ring.reserved);
  if (!amdf_status_is_ok(status)) {
    iree_atomic_store(&queue->context->cleanup_blocked, 1,
                      iree_memory_order_relaxed);
    return status;
  }
  if (queue->scratch.growth) {
    iree_hal_amd_gpu_aql_scratch_deinitialize(&queue->scratch);
  }
  iree_hal_amd_gpu_grid_storage_deinitialize(queue->context, &queue->grid);
  iree_hal_amd_gpu_sample_storage_free(queue->context, queue->host_allocator,
                                       queue->samples);
  queue->samples = NULL;
  iree_atomic_store(&queue->options.signals->waiting, 0,
                    iree_memory_order_release);
  iree_hal_amd_gpu_signal_cache_trim(queue->context, 0, queue->options.signals);
  iree_hal_amd_gpu_byte_storage_t* active[] = {queue->producer.arguments,
                                               queue->producer.transfers};
  iree_hal_amd_gpu_byte_storage_t* retired[] = {
      queue->producer.retired_arguments, queue->producer.retired_transfers};
  queue->producer.arguments = queue->producer.retired_arguments = NULL;
  queue->producer.transfers = queue->producer.retired_transfers = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(active); ++i) {
    iree_hal_amd_gpu_byte_storage_t* storage = active[i];
    if (storage) {
      storage->next = retired[i];
    } else {
      storage = retired[i];
    }
    while (storage) {
      iree_hal_amd_gpu_byte_storage_t* next = storage->next;
      iree_hal_amd_gpu_byte_storage_free(queue->context, storage);
      storage = next;
    }
  }
  return AMDF_STATUS_OK;
}

void iree_hal_amd_gpu_aql_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (queue->release.fn && !queue->release.fn(queue->release.user_data, base)) {
    return;
  }
  IREE_ASSERT(!queue->storage.queue && !queue->grid.memory,
              "parent must join native ownership before final queue release");
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_hal_amd_gpu_host_wait_set_deinitialize(&queue->pending.waits);
  iree_slim_mutex_deinitialize(&queue->pending.completion_mutex);
  iree_async_semaphore_wait_context_deinitialize(&queue->wait_context);
  iree_hal_amd_gpu_capture_pool_deinitialize(&queue->captures);
  uint64_t* frontier = queue->producer.frontier;
  if (frontier != (void*)(queue->issued + queue->options.retirement_capacity)) {
    iree_allocator_free(queue->host_allocator, frontier);
  }
  iree_allocator_free_aligned(queue->host_allocator, queue);
}
