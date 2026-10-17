// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/allocation_queue.h"

#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amd/fabric/slab_provider.h"

// Both request families lend their intrusive next pointer to this queue. One
// private tag distinguishes their layouts without allocating an adapter or
// adding scheduler state to the generic slab API. Tagged links are decoded
// before accessing the exact request type; no type-punned lvalue is used.
typedef uintptr_t iree_hal_amd_fabric_allocation_link_t;
#define IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY ((uintptr_t)1)
static_assert(iree_alignof(iree_hal_slab_operation_t) >= 2,
              "slab request linkage must reserve its tag bit");
static_assert(iree_alignof(iree_hal_amd_fabric_memory_operation_t) >= 2,
              "native request linkage must reserve its tag bit");

typedef enum iree_hal_amd_fabric_allocation_priority_e {
  IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_MAP = 0,
  IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_RELEASE,
  IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_ACQUIRE,
  IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_COUNT,
} iree_hal_amd_fabric_allocation_priority_t;

typedef struct iree_hal_amd_fabric_allocation_bucket_t {
  // First request, left at the head until all its native steps complete.
  iree_hal_amd_fabric_allocation_link_t head;
  // Last request, or zero for an empty bucket; protected by the queue mutex.
  iree_hal_amd_fabric_allocation_link_t tail;
  // Completed prefix of head, accessed only by the native worker.
  iree_host_size_t completed_count;
} iree_hal_amd_fabric_allocation_bucket_t;

struct iree_hal_amd_fabric_allocation_queue_t {
  // References held by the fabric, devices, providers, and accepted requests.
  iree_hal_resource_t resource;
  // Public teardown, worker exit, and active submitters own storage claims.
  iree_atomic_ref_count_t body_ref_count;
  // Allocator owning this queue and its native thread handle.
  iree_allocator_t host_allocator;
  // Native allocation worker; public teardown releases its handle.
  iree_thread_t* thread;
  // Protects only pending linkage and closing, never native calls.
  iree_slim_mutex_t mutex;
  // Wakes an idle worker on empty-to-nonempty or final teardown.
  iree_notification_t notification;
  // Same-priority FIFO batches; heads include the currently active step.
  iree_hal_amd_fabric_allocation_bucket_t
      buckets[IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_COUNT];
  // One bit per nonempty bucket, protected by mutex.
  uint32_t pending_mask;
  // Original failure of the one acquisition being rolled back. Release has
  // priority over acquisition, so another acquisition cannot begin meanwhile.
  iree_status_t rollback_status;
  // Acquired prefix of that failed batch, consumed at release priority.
  iree_host_size_t rollback_count;
  // Final public release has requested worker exit.
  bool closing;
};

static iree_hal_amd_fabric_memory_operation_t*
iree_hal_amd_fabric_allocation_memory(
    iree_hal_amd_fabric_allocation_link_t link) {
  return (
      iree_hal_amd_fabric_memory_operation_t*)(link &
                                               ~IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY);
}

static iree_hal_amd_fabric_allocation_link_t
iree_hal_amd_fabric_allocation_next(
    iree_hal_amd_fabric_allocation_link_t link) {
  if (link & IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY) {
    return (uintptr_t)iree_hal_amd_fabric_allocation_memory(link)->next;
  }
  return (uintptr_t)((iree_hal_slab_operation_t*)link)->next;
}

static void iree_hal_amd_fabric_allocation_set_next(
    iree_hal_amd_fabric_allocation_link_t link,
    iree_hal_amd_fabric_allocation_link_t next) {
  if (link & IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY) {
    iree_hal_amd_fabric_allocation_memory(link)->next =
        (iree_hal_amd_fabric_memory_operation_t*)next;
  } else {
    ((iree_hal_slab_operation_t*)link)->next = (iree_hal_slab_operation_t*)next;
  }
}

// Called under mutex; callbacks cannot observe or reclaim a linked request.
static void iree_hal_amd_fabric_allocation_append(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_allocation_priority_t priority,
    iree_hal_amd_fabric_allocation_link_t link) {
  iree_hal_amd_fabric_allocation_bucket_t* bucket = &queue->buckets[priority];
  iree_hal_amd_fabric_allocation_set_next(link, 0);
  if (bucket->tail) {
    iree_hal_amd_fabric_allocation_set_next(bucket->tail, link);
  } else {
    bucket->head = link;
  }
  bucket->tail = link;
  queue->pending_mask |= 1u << priority;
}

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
  const bool ready = queue->pending_mask || queue->closing;
  iree_slim_mutex_unlock(&queue->mutex);
  return ready;
}

static iree_status_t iree_hal_amd_fabric_allocation_map(
    iree_hal_amd_fabric_memory_map_request_t* request) {
  if (!request->batch) {
    return iree_ok_status();
  }
  iree_hal_memory_mapping_batch_t* batch = request->batch;
  iree_status_t status = iree_ok_status();
  // Check synchronous publication support for the entire resolved transaction
  // before mutating any target. Queue-qualified targets need their native
  // queue.
  for (iree_host_size_t i = 0;
       i < batch->entry_count && iree_status_is_ok(status); ++i) {
    const iree_hal_buffer_vtable_t* vtable =
        (const iree_hal_buffer_vtable_t*)batch->entries[i]
            .target->buffer.resource.vtable;
    if (!vtable->update_mappings) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "target requires native queue mapping");
    }
  }
  for (iree_host_size_t i = 0;
       i < batch->entry_count && iree_status_is_ok(status); ++i) {
    const iree_hal_memory_mapping_batch_entry_t* entry = &batch->entries[i];
    iree_hal_buffer_t* target = &entry->target->buffer;
    const iree_hal_buffer_vtable_t* vtable =
        (const iree_hal_buffer_vtable_t*)target->resource.vtable;
    status = vtable->update_mappings(target, entry->projection,
                                     entry->update_count, entry->updates);
  }
  return status;
}

// One indivisible native acquisition or release. Output initialization and
// rollback bookkeeping remain private until the batch's terminal callback.
static iree_status_t iree_hal_amd_fabric_allocation_memory_step(
    iree_hal_amd_fabric_memory_operation_t* operation, bool release,
    iree_host_size_t rollback_count, iree_host_size_t* completed_count,
    bool* out_complete) {
  const iree_host_size_t index = *completed_count;
  iree_status_t status = iree_ok_status();
  iree_host_size_t count = 0;
  iree_hal_amd_fabric_memory_t* memories = NULL;
  if (operation->type == IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE) {
    iree_hal_amd_fabric_memory_acquire_request_t* request = (void*)operation;
    count = release ? rollback_count : request->count;
    memories = request->memories;
    if (!release && index == 0 && count) {
      memset(memories, 0, count * sizeof(*memories));
    }
    if (!release && index < count) {
      const iree_hal_amd_fabric_memory_request_t* geometry =
          &request->requests[index];
      status = iree_hal_amd_fabric_memory_allocate(
          operation->api, geometry->options, geometry->byte_length,
          geometry->alignment, operation->event_sink, &memories[index]);
    }
  } else {
    iree_hal_amd_fabric_memory_release_request_t* request = (void*)operation;
    count = request->count;
    memories = request->memories;
  }
  if (release && index < count) {
    iree_hal_amd_fabric_memory_deinitialize(
        operation->api, operation->event_sink, &memories[index]);
    memset(&memories[index], 0, sizeof(memories[index]));
  }
  if (index < count && iree_status_is_ok(status)) {
    ++*completed_count;
  }
  *out_complete = *completed_count == count || !iree_status_is_ok(status);
  return status;
}

static iree_status_t iree_hal_amd_fabric_allocation_slab_step(
    iree_hal_slab_operation_t* operation, bool release,
    iree_host_size_t rollback_count, iree_host_size_t* completed_count,
    bool* out_complete) {
  const iree_host_size_t index = *completed_count;
  iree_status_t status = iree_ok_status();
  iree_host_size_t count = 0;
  iree_hal_slab_t* slabs = NULL;
  if (operation->type == IREE_HAL_SLAB_OPERATION_ACQUIRE) {
    iree_hal_slab_acquire_request_t* request = (void*)operation;
    count = release ? rollback_count : request->count;
    slabs = request->slabs;
    if (!release && index == 0 && count) {
      memset(slabs, 0, count * sizeof(*slabs));
    }
    if (!release && index < count) {
      status = iree_hal_amd_fabric_slab_provider_acquire_slab(
          operation->provider, request->requests[index].min_length,
          request->requests[index].min_alignment, &slabs[index]);
    }
  } else {
    iree_hal_slab_release_request_t* request = (void*)operation;
    count = request->count;
    slabs = request->slabs;
  }
  if (release && index < count) {
    iree_hal_amd_fabric_slab_provider_release_slab(operation->provider,
                                                   &slabs[index]);
    memset(&slabs[index], 0, sizeof(slabs[index]));
  }
  if (index < count && iree_status_is_ok(status)) {
    ++*completed_count;
  }
  *out_complete = *completed_count == count || !iree_status_is_ok(status);
  return status;
}

// Removes all scheduler borrows before invoking a callback that may immediately
// free or reuse its request. The retained provider keeps its native owners live
// through the callback. Accepted raw requests separately retain the queue.
static void iree_hal_amd_fabric_allocation_complete(
    iree_hal_amd_fabric_allocation_link_t link, iree_status_t status) {
  if (link & IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY) {
    iree_hal_amd_fabric_memory_operation_t* operation =
        iree_hal_amd_fabric_allocation_memory(link);
    switch (operation->type) {
      case IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE: {
        iree_hal_amd_fabric_memory_acquire_request_t* request =
            (void*)operation;
        request->completion.fn(request->completion.user_data, status);
        break;
      }
      case IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_RELEASE: {
        iree_hal_amd_fabric_memory_release_request_t* request =
            (void*)operation;
        request->completion.fn(request->completion.user_data);
        break;
      }
      case IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_MAP: {
        iree_hal_amd_fabric_memory_map_request_t* request = (void*)operation;
        request->completion.fn(request->completion.user_data, status);
        break;
      }
    }
  } else {
    iree_hal_slab_operation_t* operation = (void*)link;
    iree_hal_slab_provider_t* provider = operation->provider;
    if (operation->type == IREE_HAL_SLAB_OPERATION_ACQUIRE) {
      iree_hal_slab_acquire_request_t* request = (void*)operation;
      request->completion.fn(request->completion.user_data, status);
    } else {
      iree_hal_slab_release_request_t* request = (void*)operation;
      request->completion.fn(request->completion.user_data);
    }
    iree_hal_slab_provider_release(provider);
  }
}

static int iree_hal_amd_fabric_allocation_queue_main(void* user_data) {
  iree_hal_amd_fabric_allocation_queue_t* queue = user_data;
  while (true) {
    iree_slim_mutex_lock(&queue->mutex);
    if (!queue->pending_mask) {
      const bool closing = queue->closing;
      iree_slim_mutex_unlock(&queue->mutex);
      if (closing) {
        break;
      }
      iree_notification_await(&queue->notification,
                              iree_hal_amd_fabric_allocation_queue_is_ready,
                              queue, iree_infinite_timeout());
      continue;
    }
    const uint32_t priority =
        iree_math_count_trailing_zeros_u32(queue->pending_mask);
    iree_hal_amd_fabric_allocation_bucket_t* bucket = &queue->buckets[priority];
    const iree_hal_amd_fabric_allocation_link_t link = bucket->head;
    iree_slim_mutex_unlock(&queue->mutex);

    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_fabric_native_memory_step");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, priority);
    const bool release =
        priority == IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_RELEASE;
    bool complete = false;
    iree_status_t status = iree_ok_status();
    bool acquisition = false;
    if (link & IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY) {
      iree_hal_amd_fabric_memory_operation_t* operation =
          iree_hal_amd_fabric_allocation_memory(link);
      acquisition =
          operation->type == IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE;
      if (priority == IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_MAP) {
        status = iree_hal_amd_fabric_allocation_map((void*)operation);
        complete = true;
      } else {
        status = iree_hal_amd_fabric_allocation_memory_step(
            operation, release, queue->rollback_count, &bucket->completed_count,
            &complete);
      }
    } else {
      iree_hal_slab_operation_t* operation = (void*)link;
      acquisition = operation->type == IREE_HAL_SLAB_OPERATION_ACQUIRE;
      status = iree_hal_amd_fabric_allocation_slab_step(
          operation, release, queue->rollback_count, &bucket->completed_count,
          &complete);
    }
    IREE_TRACE_ZONE_END(z0);
    if (!complete) {
      continue;
    }

    // Only this worker removes heads. Producers may have appended while the
    // native call ran, so successor linkage is read under the same mutex.
    iree_slim_mutex_lock(&queue->mutex);
    bucket->head = iree_hal_amd_fabric_allocation_next(link);
    if (!bucket->head) {
      bucket->tail = 0;
      queue->pending_mask &= ~(1u << priority);
    }
    const iree_host_size_t completed_count = bucket->completed_count;
    bucket->completed_count = 0;
    const bool rollback = !release && acquisition && completed_count &&
                          !iree_status_is_ok(status);
    if (rollback) {
      queue->rollback_count = completed_count;
      queue->rollback_status = status;
      iree_hal_amd_fabric_allocation_append(
          queue, IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_RELEASE, link);
    }
    iree_slim_mutex_unlock(&queue->mutex);
    if (rollback) {
      continue;
    }
    if (release && acquisition) {
      status = queue->rollback_status;
      queue->rollback_status = iree_ok_status();
      queue->rollback_count = 0;
    }
    iree_hal_amd_fabric_allocation_complete(link, status);
    iree_hal_amd_fabric_allocation_queue_release(queue);
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

static void iree_hal_amd_fabric_allocation_queue_submit(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_allocation_priority_t priority,
    iree_hal_amd_fabric_allocation_link_t link) {
  // Completion can reclaim the last external owner before this submitter posts
  // its wake. A body claim joins that access independently of public teardown.
  iree_atomic_ref_count_inc(&queue->body_ref_count);
  iree_hal_amd_fabric_allocation_queue_retain(queue);
  iree_slim_mutex_lock(&queue->mutex);
  const bool wake = !queue->pending_mask;
  iree_hal_amd_fabric_allocation_append(queue, priority, link);
  iree_slim_mutex_unlock(&queue->mutex);
  if (wake) {
    iree_notification_post(&queue->notification, 1);
  }
  iree_hal_amd_fabric_allocation_queue_release_body(queue);
}

void iree_hal_amd_fabric_allocation_queue_enqueue(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_slab_operation_t* operation) {
  iree_hal_slab_provider_retain(operation->provider);
  iree_hal_amd_fabric_allocation_queue_submit(
      queue,
      operation->type == IREE_HAL_SLAB_OPERATION_ACQUIRE
          ? IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_ACQUIRE
          : IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_RELEASE,
      (uintptr_t)operation);
}

void iree_hal_amd_fabric_allocation_queue_acquire_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_acquire_request_t* request) {
  request->operation.type = IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE;
  iree_hal_amd_fabric_allocation_queue_submit(
      queue, IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_ACQUIRE,
      (uintptr_t)&request->operation |
          IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY);
}

void iree_hal_amd_fabric_allocation_queue_release_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_release_request_t* request) {
  request->operation.type = IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_RELEASE;
  iree_hal_amd_fabric_allocation_queue_submit(
      queue, IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_RELEASE,
      (uintptr_t)&request->operation |
          IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY);
}

void iree_hal_amd_fabric_allocation_queue_map_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_map_request_t* request) {
  request->operation.type = IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_MAP;
  iree_hal_amd_fabric_allocation_queue_submit(
      queue, IREE_HAL_AMD_FABRIC_ALLOCATION_PRIORITY_MAP,
      (uintptr_t)&request->operation |
          IREE_HAL_AMD_FABRIC_ALLOCATION_LINK_MEMORY);
}
