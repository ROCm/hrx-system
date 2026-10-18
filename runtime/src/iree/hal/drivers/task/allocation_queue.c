// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/allocation_queue.h"

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/task/process.h"

struct iree_hal_task_allocation_queue_t {
  // Device and native provider ownership; the device joins before releasing.
  iree_hal_resource_t resource;
  // Allocator for the process and queue body.
  iree_allocator_t host_allocator;
  // Existing executor retained until the process's final access.
  iree_task_executor_t* executor;
  // Selected progress service retained through all native cleanup callbacks.
  iree_async_proactor_pool_entry_t* progress_entry;
  // Sole consumer of native allocation batches.
  iree_task_process_t process;
  // Protects only queue linkage and closing state, never native calls.
  iree_slim_mutex_t mutex;
  // Wakes the device's outer teardown after the process's last access.
  iree_notification_t notification;
  // First pending batch; NULL when no native work is queued.
  iree_hal_slab_operation_t* head;
  // Last pending batch, valid only when head is non-NULL.
  iree_hal_slab_operation_t* tail;
  // Device teardown has requested process termination.
  bool closing;
  // Process release has completed; protected by mutex.
  bool closed;
};

typedef struct iree_hal_task_slab_provider_t {
  // Shared native provider interface and ownership.
  iree_hal_slab_provider_t base;
  // Retained native allocation domain shared with other memory classes.
  iree_hal_task_allocation_queue_t* queue;
  // Retained synchronous CPU allocation source.
  iree_hal_slab_provider_t* native_provider;
} iree_hal_task_slab_provider_t;

static iree_status_t iree_hal_task_allocation_queue_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  iree_hal_task_allocation_queue_t* queue = process->user_data;
  iree_slim_mutex_lock(&queue->mutex);
  iree_hal_slab_operation_t* operation = queue->head;
  if (operation) {
    queue->head = operation->next;
    if (!queue->head) {
      queue->tail = NULL;
    }
  }
  out_result->completed = !operation && queue->closing;
  iree_slim_mutex_unlock(&queue->mutex);
  out_result->did_work = operation != NULL;
  if (!operation) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  // Completion may drop the source pool's last provider reference. Keep the
  // forwarding provider and its native source alive through submission return.
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)operation->provider;
  iree_hal_slab_provider_retain(&provider->base);
  switch (operation->type) {
    case IREE_HAL_SLAB_OPERATION_ACQUIRE:
      iree_hal_slab_provider_acquire_slabs(
          provider->native_provider,
          (iree_hal_slab_acquire_request_t*)operation);
      break;
    case IREE_HAL_SLAB_OPERATION_RELEASE:
      iree_hal_slab_provider_release_slabs(
          provider->native_provider,
          (iree_hal_slab_release_request_t*)operation);
      break;
  }
  iree_hal_slab_provider_release(&provider->base);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_task_allocation_queue_process_release(
    iree_task_process_t* process) {
  iree_hal_task_allocation_queue_t* queue = process->user_data;
  iree_slim_mutex_lock(&queue->mutex);
  queue->closed = true;
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->mutex);
}

static bool iree_hal_task_allocation_queue_is_closed(void* user_data) {
  iree_hal_task_allocation_queue_t* queue = user_data;
  iree_slim_mutex_lock(&queue->mutex);
  const bool closed = queue->closed;
  iree_slim_mutex_unlock(&queue->mutex);
  return closed;
}

void iree_hal_task_allocation_queue_close(
    iree_hal_task_allocation_queue_t* queue) {
  if (!queue) {
    return;
  }
  iree_slim_mutex_lock(&queue->mutex);
  const bool closing = queue->closing;
  queue->closing = true;
  iree_slim_mutex_unlock(&queue->mutex);
  if (!closing) {
    iree_task_executor_schedule_process(queue->executor, &queue->process);
  }
  iree_notification_await(&queue->notification,
                          iree_hal_task_allocation_queue_is_closed, queue,
                          iree_infinite_timeout());
}

static void iree_hal_task_allocation_queue_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_task_allocation_queue_t* queue =
      (iree_hal_task_allocation_queue_t*)resource;
  IREE_ASSERT(queue->closed, "the device must join native allocation shutdown");
  iree_task_executor_t* executor = queue->executor;
  iree_async_proactor_pool_entry_release(queue->progress_entry);
  iree_notification_deinitialize(&queue->notification);
  iree_slim_mutex_deinitialize(&queue->mutex);
  iree_allocator_free_aligned(queue->host_allocator, queue);
  iree_task_executor_release(executor);
}

static const iree_hal_resource_vtable_t iree_hal_task_allocation_queue_vtable =
    {
        .destroy = iree_hal_task_allocation_queue_destroy,
};

iree_status_t iree_hal_task_allocation_queue_create(
    iree_task_executor_t* executor,
    iree_async_proactor_pool_entry_t* progress_entry,
    iree_allocator_t host_allocator,
    iree_hal_task_allocation_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_task_allocation_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*queue),
      iree_alignof(iree_hal_task_allocation_queue_t), /*offset=*/0,
      (void**)&queue));
  iree_hal_resource_initialize(&iree_hal_task_allocation_queue_vtable,
                               &queue->resource);
  queue->host_allocator = host_allocator;
  queue->executor = executor;
  iree_task_executor_retain(executor);
  queue->progress_entry = progress_entry;
  iree_async_proactor_pool_entry_retain(progress_entry);
  iree_slim_mutex_initialize(&queue->mutex);
  iree_notification_initialize(&queue->notification);
  iree_task_process_initialize(iree_hal_task_allocation_queue_drain,
                               /*suspend_count=*/0, /*wake_budget=*/1,
                               &queue->process);
  queue->process.user_data = queue;
  queue->process.release_fn = iree_hal_task_allocation_queue_process_release;
  *out_queue = queue;
  return iree_ok_status();
}

void iree_hal_task_allocation_queue_retain(
    iree_hal_task_allocation_queue_t* queue) {
  iree_hal_resource_retain(queue);
}

void iree_hal_task_allocation_queue_release(
    iree_hal_task_allocation_queue_t* queue) {
  iree_hal_resource_release(queue);
}

static void iree_hal_task_allocation_queue_enqueue(
    iree_hal_task_allocation_queue_t* queue,
    iree_hal_slab_operation_t* operation) {
  iree_slim_mutex_lock(&queue->mutex);
  if (queue->tail) {
    queue->tail->next = operation;
  } else {
    queue->head = operation;
  }
  queue->tail = operation;
  iree_slim_mutex_unlock(&queue->mutex);
  iree_task_executor_schedule_process(queue->executor, &queue->process);
}

static void iree_hal_task_slab_provider_destroy(
    iree_hal_slab_provider_t* base) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  iree_hal_task_allocation_queue_t* queue = provider->queue;
  iree_hal_slab_provider_release(provider->native_provider);
  iree_allocator_free(provider->base.host_allocator, provider);
  iree_hal_task_allocation_queue_release(queue);
}

static void iree_hal_task_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* base, iree_hal_slab_acquire_request_t* request) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  iree_hal_task_allocation_queue_enqueue(provider->queue, &request->operation);
}

static void iree_hal_task_slab_provider_release_slabs(
    iree_hal_slab_provider_t* base, iree_hal_slab_release_request_t* request) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  // Both supported CPU sources charge every owned native allocation. The only
  // uncharged slabs are portable host imports: they borrow an ordinary pointer
  // and have no native attachment to retire. Their return stays inline.
  for (iree_host_size_t i = 0; i < request->count; ++i) {
    if (request->slabs[i].native_allocation.byte_length) {
      iree_hal_task_allocation_queue_enqueue(provider->queue,
                                             &request->operation);
      return;
    }
  }
  iree_hal_slab_provider_release_slabs(provider->native_provider, request);
}

static iree_status_t iree_hal_task_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length,
    iree_hal_memory_access_t access, iree_hal_buffer_usage_t usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  return iree_hal_slab_provider_wrap_buffer(provider->native_provider, slab,
                                            offset, length, access, usage,
                                            release_callback, out_buffer);
}

static iree_status_t iree_hal_task_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_task_slab_provider_t* provider =
      (const iree_hal_task_slab_provider_t*)base;
  return iree_hal_slab_provider_validate_asan_options(provider->native_provider,
                                                      options);
}

static void iree_hal_task_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset, iree_hal_asan_range_advice_flags_t flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  iree_hal_slab_provider_advise_asan_range(provider->native_provider, slab,
                                           backing_offset, flags, layout);
}

static void iree_hal_task_slab_provider_prefault(iree_hal_slab_provider_t* base,
                                                 const iree_hal_slab_t* slab,
                                                 iree_device_size_t offset,
                                                 iree_device_size_t length) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  iree_hal_slab_provider_prefault(provider->native_provider, slab, offset,
                                  length);
}

static void iree_hal_task_slab_provider_trim(iree_hal_slab_provider_t* base,
                                             iree_hal_pool_trim_flags_t flags) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  iree_hal_slab_provider_trim(provider->native_provider, flags);
}

static void iree_hal_task_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base)) {
    return;
  }
  const iree_hal_task_slab_provider_t* provider =
      (const iree_hal_task_slab_provider_t*)base;
  iree_hal_slab_provider_query_stats(provider->native_provider, visited,
                                     out_stats);
}

static void iree_hal_task_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_task_slab_provider_t* provider =
      (const iree_hal_task_slab_provider_t*)base;
  iree_hal_slab_provider_query_properties(provider->native_provider,
                                          out_properties);
}

static iree_status_t iree_hal_task_slab_provider_import_slab(
    iree_hal_slab_provider_t* base, const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import) {
  iree_hal_task_slab_provider_t* provider =
      (iree_hal_task_slab_provider_t*)base;
  return iree_hal_slab_provider_import_slab(provider->native_provider, storage,
                                            out_import);
}

static bool iree_hal_task_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* base,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  const iree_hal_task_slab_provider_t* provider =
      (const iree_hal_task_slab_provider_t*)base;
  return iree_hal_slab_provider_query_import_requirements(
      provider->native_provider, storage_class, out_requirements);
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_task_slab_provider_vtable = {
        .destroy = iree_hal_task_slab_provider_destroy,
        .acquire_slabs = iree_hal_task_slab_provider_acquire_slabs,
        .release_slabs = iree_hal_task_slab_provider_release_slabs,
        .wrap_buffer = iree_hal_task_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_task_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_task_slab_provider_advise_asan_range,
        .prefault = iree_hal_task_slab_provider_prefault,
        .trim = iree_hal_task_slab_provider_trim,
        .query_stats = iree_hal_task_slab_provider_query_stats,
        .query_properties = iree_hal_task_slab_provider_query_properties,
        .import_slab = iree_hal_task_slab_provider_import_slab,
        .query_import_requirements =
            iree_hal_task_slab_provider_query_import_requirements,
};

iree_status_t iree_hal_task_slab_provider_create(
    iree_hal_task_allocation_queue_t* queue,
    iree_hal_slab_provider_t* native_provider, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  *out_provider = NULL;
  iree_hal_task_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  iree_hal_slab_provider_initialize(&iree_hal_task_slab_provider_vtable,
                                    host_allocator, &provider->base);
  provider->queue = queue;
  iree_hal_task_allocation_queue_retain(queue);
  provider->native_provider = native_provider;
  iree_hal_slab_provider_retain(native_provider);
  *out_provider = &provider->base;
  return iree_ok_status();
}
