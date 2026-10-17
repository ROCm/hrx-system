// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/allocation_queue.h"

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amd/fabric/slab_provider.h"

struct iree_hal_amd_fabric_allocation_queue_t {
  // References held by the fabric, devices, and native providers.
  iree_hal_resource_t resource;
  // Public teardown, worker exit, and active submitters own storage claims.
  iree_atomic_ref_count_t body_ref_count;
  // Allocator owning this queue and its native thread handle.
  iree_allocator_t host_allocator;
  // Native allocation worker; public teardown releases its handle.
  iree_thread_t* thread;
  // Protects only pending linkage and closing, never native calls.
  iree_slim_mutex_t mutex;
  // Wakes the worker when a batch arrives or final teardown begins.
  iree_notification_t notification;
  // First pending native batch, or NULL when idle.
  iree_hal_slab_operation_t* head;
  // Last pending native batch, valid only when head is non-NULL.
  iree_hal_slab_operation_t* tail;
  // Implementation-owned native backing requests, independent of HAL slabs.
  iree_hal_amd_fabric_memory_operation_t* memory_head;
  // Last pending backing request, valid only when memory_head is non-NULL.
  iree_hal_amd_fabric_memory_operation_t* memory_tail;
  // Alternates the two typed FIFOs when both contain ready native batches.
  bool prefer_memory;
  // Final public release has requested worker exit.
  bool closing;
};

static void iree_hal_amd_fabric_allocation_queue_release_body(
    iree_hal_amd_fabric_allocation_queue_t* queue) {
  if (iree_atomic_ref_count_dec(&queue->body_ref_count) != 1) {
    return;
  }
  iree_notification_deinitialize(&queue->notification);
  iree_slim_mutex_deinitialize(&queue->mutex);
  iree_allocator_free(queue->host_allocator, queue);
}

static bool iree_hal_amd_fabric_allocation_queue_is_ready(void* user_data) {
  iree_hal_amd_fabric_allocation_queue_t* queue = user_data;
  iree_slim_mutex_lock(&queue->mutex);
  const bool ready = queue->head || queue->memory_head || queue->closing;
  iree_slim_mutex_unlock(&queue->mutex);
  return ready;
}

static void iree_hal_amd_fabric_allocation_queue_acquire_memory_execute(
    iree_hal_amd_fabric_memory_acquire_request_t* request) {
  memset(request->memories, 0, request->count * sizeof(*request->memories));
  iree_status_t status = iree_ok_status();
  iree_host_size_t acquired_count = 0;
  for (iree_host_size_t i = 0; i < request->count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_fabric_memory_request_t* geometry =
        &request->requests[i];
    status = iree_hal_amd_fabric_memory_allocate(
        request->operation.api, geometry->options, geometry->byte_length,
        geometry->alignment, request->operation.event_sink,
        &request->memories[i]);
    if (iree_status_is_ok(status)) {
      ++acquired_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      iree_hal_amd_fabric_memory_deinitialize(request->operation.api,
                                              request->operation.event_sink,
                                              &request->memories[i]);
    }
    // Failed native detach is diagnosed and leaks its backing. No failed batch
    // publishes ownership or offers a consumed handle for another attempt.
    memset(request->memories, 0, request->count * sizeof(*request->memories));
  }
  request->completion.fn(request->completion.user_data, status);
}

static void iree_hal_amd_fabric_allocation_queue_release_memory_execute(
    iree_hal_amd_fabric_memory_release_request_t* request) {
  for (iree_host_size_t i = 0; i < request->count; ++i) {
    iree_hal_amd_fabric_memory_deinitialize(request->operation.api,
                                            request->operation.event_sink,
                                            &request->memories[i]);
  }
  memset(request->memories, 0, request->count * sizeof(*request->memories));
  request->completion.fn(request->completion.user_data);
}

static int iree_hal_amd_fabric_allocation_queue_main(void* user_data) {
  iree_hal_amd_fabric_allocation_queue_t* queue = user_data;
  while (true) {
    iree_notification_await(&queue->notification,
                            iree_hal_amd_fabric_allocation_queue_is_ready,
                            queue, iree_infinite_timeout());
    iree_slim_mutex_lock(&queue->mutex);
    iree_hal_slab_operation_t* operation = NULL;
    iree_hal_amd_fabric_memory_operation_t* memory_operation = NULL;
    if (queue->memory_head && (queue->prefer_memory || !queue->head)) {
      memory_operation = queue->memory_head;
      queue->memory_head = memory_operation->next;
      if (!queue->memory_head) {
        queue->memory_tail = NULL;
      }
      queue->prefer_memory = false;
    } else if (queue->head) {
      operation = queue->head;
      queue->head = operation->next;
      if (!queue->head) {
        queue->tail = NULL;
      }
      queue->prefer_memory = true;
    }
    const bool closing = queue->closing;
    iree_slim_mutex_unlock(&queue->mutex);
    if (!operation && !memory_operation) {
      if (closing) {
        break;
      }
      continue;
    }

    if (operation) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_fabric_slab_batch");
      // A completion can drop the pool's last provider reference. Keep its
      // native devices alive until the entire native call has returned.
      iree_hal_slab_provider_t* provider = operation->provider;
      iree_hal_slab_provider_retain(provider);
      iree_hal_amd_fabric_slab_provider_execute(operation);
      iree_hal_slab_provider_release(provider);
      IREE_TRACE_ZONE_END(z0);
    } else {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_fabric_memory_batch");
      switch (memory_operation->type) {
        case IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE:
          iree_hal_amd_fabric_allocation_queue_acquire_memory_execute(
              (iree_hal_amd_fabric_memory_acquire_request_t*)memory_operation);
          break;
        case IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_RELEASE:
          iree_hal_amd_fabric_allocation_queue_release_memory_execute(
              (iree_hal_amd_fabric_memory_release_request_t*)memory_operation);
          break;
      }
      // The callback may reclaim/reuse its descriptor and drop the final
      // external queue reference. Submission's claim joins our final access.
      iree_hal_amd_fabric_allocation_queue_release(queue);
      IREE_TRACE_ZONE_END(z0);
    }
  }
  iree_hal_amd_fabric_allocation_queue_release_body(queue);
  return 0;
}

static void iree_hal_amd_fabric_allocation_queue_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amd_fabric_allocation_queue_t* queue =
      (iree_hal_amd_fabric_allocation_queue_t*)resource;
  iree_slim_mutex_lock(&queue->mutex);
  queue->closing = true;
  iree_slim_mutex_unlock(&queue->mutex);
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  // Pending native work retains its provider and therefore this queue. Final
  // public release joins an idle worker, or detaches when called by the worker
  // itself. The independent body claim protects its remaining loop/exit access.
  iree_thread_release(queue->thread);
  iree_hal_amd_fabric_allocation_queue_release_body(queue);
}

static const iree_hal_resource_vtable_t
    iree_hal_amd_fabric_allocation_queue_vtable = {
        .destroy = iree_hal_amd_fabric_allocation_queue_destroy,
};

iree_status_t iree_hal_amd_fabric_allocation_queue_create(
    uint32_t node_id, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_allocation_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amd_fabric_allocation_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  iree_hal_resource_initialize(&iree_hal_amd_fabric_allocation_queue_vtable,
                               &queue->resource);
  iree_atomic_ref_count_init(&queue->body_ref_count);
  queue->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&queue->mutex);
  iree_notification_initialize(&queue->notification);
  iree_thread_create_params_t params = {
      .name = IREE_SV("libamdf-numa-any"),
      .create_suspended = true,
  };
  char thread_name[32];
  if (node_id != UINT32_MAX) {
    iree_snprintf(thread_name, sizeof(thread_name), "libamdf-numa-%u", node_id);
    params.name = iree_make_cstring_view(thread_name);
    iree_thread_affinity_set_group_any(node_id, &params.initial_affinity);
  }
  iree_status_t status =
      iree_thread_create(iree_hal_amd_fabric_allocation_queue_main, queue,
                         params, host_allocator, &queue->thread);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_inc(&queue->body_ref_count);
    iree_thread_resume(queue->thread);
    *out_queue = queue;
  } else {
    iree_hal_amd_fabric_allocation_queue_release_body(queue);
  }
  return status;
}

void iree_hal_amd_fabric_allocation_queue_retain(
    iree_hal_amd_fabric_allocation_queue_t* queue) {
  iree_hal_resource_retain(queue);
}

void iree_hal_amd_fabric_allocation_queue_release(
    iree_hal_amd_fabric_allocation_queue_t* queue) {
  iree_hal_resource_release(queue);
}

void iree_hal_amd_fabric_allocation_queue_enqueue(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_slab_operation_t* operation) {
  // An immediate completion may drop the last provider/queue reference before
  // this submitter posts its wake. Join that final producer access separately
  // from public ownership and worker exit.
  iree_atomic_ref_count_inc(&queue->body_ref_count);
  iree_slim_mutex_lock(&queue->mutex);
  const bool wake = !queue->head && !queue->memory_head;
  if (queue->tail) {
    queue->tail->next = operation;
  } else {
    queue->head = operation;
  }
  queue->tail = operation;
  iree_slim_mutex_unlock(&queue->mutex);
  if (wake) {
    iree_notification_post(&queue->notification, 1);
  }
  iree_hal_amd_fabric_allocation_queue_release_body(queue);
}

static void iree_hal_amd_fabric_allocation_queue_enqueue_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_operation_t* operation) {
  iree_atomic_ref_count_inc(&queue->body_ref_count);
  operation->next = NULL;
  iree_hal_amd_fabric_allocation_queue_retain(queue);
  iree_slim_mutex_lock(&queue->mutex);
  const bool wake = !queue->head && !queue->memory_head;
  if (queue->memory_tail) {
    queue->memory_tail->next = operation;
  } else {
    queue->memory_head = operation;
  }
  queue->memory_tail = operation;
  iree_slim_mutex_unlock(&queue->mutex);
  if (wake) {
    iree_notification_post(&queue->notification, 1);
  }
  iree_hal_amd_fabric_allocation_queue_release_body(queue);
}

void iree_hal_amd_fabric_allocation_queue_acquire_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_acquire_request_t* request) {
  request->operation.type = IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE;
  iree_hal_amd_fabric_allocation_queue_enqueue_memory(queue,
                                                      &request->operation);
}

void iree_hal_amd_fabric_allocation_queue_release_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_release_request_t* request) {
  request->operation.type = IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_RELEASE;
  iree_hal_amd_fabric_allocation_queue_enqueue_memory(queue,
                                                      &request->operation);
}
