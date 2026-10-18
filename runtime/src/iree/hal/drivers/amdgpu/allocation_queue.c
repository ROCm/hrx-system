// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/allocation_queue.h"

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/slab_provider.h"

struct iree_hal_amdgpu_allocation_queue_t {
  // References held by the ROCr allocator and native providers.
  iree_hal_resource_t resource;
  // Public teardown and worker exit each own one storage claim.
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
  // Final public release has requested worker exit.
  bool closing;
};

// Native completion can release the final device reference on this worker.
// Teardown then drains the remaining ready native batches without a self-join.
static IREE_THREAD_LOCAL iree_hal_amdgpu_allocation_queue_t*
    iree_hal_amdgpu_current_allocation_queue = NULL;

static void iree_hal_amdgpu_allocation_queue_release_body(
    iree_hal_amdgpu_allocation_queue_t* queue) {
  if (iree_atomic_ref_count_dec(&queue->body_ref_count) != 1) {
    return;
  }
  iree_notification_deinitialize(&queue->notification);
  iree_slim_mutex_deinitialize(&queue->mutex);
  iree_allocator_free(queue->host_allocator, queue);
}

static bool iree_hal_amdgpu_allocation_queue_is_ready(void* user_data) {
  iree_hal_amdgpu_allocation_queue_t* queue = user_data;
  iree_slim_mutex_lock(&queue->mutex);
  const bool ready = queue->head || queue->closing;
  iree_slim_mutex_unlock(&queue->mutex);
  return ready;
}

static bool iree_hal_amdgpu_allocation_queue_run_one(
    iree_hal_amdgpu_allocation_queue_t* queue) {
  iree_slim_mutex_lock(&queue->mutex);
  iree_hal_slab_operation_t* operation = queue->head;
  if (operation) {
    queue->head = operation->next;
    if (!queue->head) {
      queue->tail = NULL;
    }
  }
  iree_slim_mutex_unlock(&queue->mutex);
  if (!operation) {
    return false;
  }
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amdgpu_slab_batch");
  // Completion can drop the pool's last provider reference. This reference
  // covers the callback's return, while the queue body covers the worker loop.
  iree_hal_slab_provider_t* provider = operation->provider;
  iree_hal_slab_provider_retain(provider);
  iree_hal_amdgpu_slab_provider_execute(operation);
  iree_hal_slab_provider_release(provider);
  IREE_TRACE_ZONE_END(z0);
  return true;
}

static int iree_hal_amdgpu_allocation_queue_main(void* user_data) {
  iree_hal_amdgpu_allocation_queue_t* queue = user_data;
  iree_hal_amdgpu_current_allocation_queue = queue;
  while (true) {
    iree_notification_await(&queue->notification,
                            iree_hal_amdgpu_allocation_queue_is_ready, queue,
                            iree_infinite_timeout());
    if (iree_hal_amdgpu_allocation_queue_run_one(queue)) {
      continue;
    }
    iree_slim_mutex_lock(&queue->mutex);
    const bool closing = queue->closing;
    iree_slim_mutex_unlock(&queue->mutex);
    if (closing) {
      break;
    }
  }
  iree_hal_amdgpu_current_allocation_queue = NULL;
  iree_hal_amdgpu_allocation_queue_release_body(queue);
  return 0;
}

void iree_hal_amdgpu_allocation_queue_close(
    iree_hal_amdgpu_allocation_queue_t* queue) {
  if (!queue || !queue->thread) {
    return;
  }
  iree_slim_mutex_lock(&queue->mutex);
  queue->closing = true;
  iree_slim_mutex_unlock(&queue->mutex);
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  if (iree_hal_amdgpu_current_allocation_queue == queue) {
    // Native calls precede completion. An owner destroyed by that completion
    // must finish other ready batches before tearing down its native context.
    while (iree_hal_amdgpu_allocation_queue_run_one(queue)) {
    }
  }
  iree_thread_t* thread = queue->thread;
  queue->thread = NULL;
  iree_thread_release(thread);
}

static void iree_hal_amdgpu_allocation_queue_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_allocation_queue_t* queue =
      (iree_hal_amdgpu_allocation_queue_t*)resource;
  iree_hal_amdgpu_allocation_queue_close(queue);
  iree_hal_amdgpu_allocation_queue_release_body(queue);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_allocation_queue_vtable = {
        .destroy = iree_hal_amdgpu_allocation_queue_destroy,
};

iree_status_t iree_hal_amdgpu_allocation_queue_create(
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amdgpu_allocation_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amdgpu_allocation_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  iree_hal_resource_initialize(&iree_hal_amdgpu_allocation_queue_vtable,
                               &queue->resource);
  iree_atomic_ref_count_init(&queue->body_ref_count);
  queue->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&queue->mutex);
  iree_notification_initialize(&queue->notification);
  iree_thread_create_params_t params = {
      .name = IREE_SV("hal-rocr-alloc"),
      .initial_affinity = affinity,
      .create_suspended = true,
  };
  iree_status_t status =
      iree_thread_create(iree_hal_amdgpu_allocation_queue_main, queue, params,
                         host_allocator, &queue->thread);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_inc(&queue->body_ref_count);
    iree_thread_resume(queue->thread);
    *out_queue = queue;
  } else {
    iree_hal_amdgpu_allocation_queue_release_body(queue);
  }
  return status;
}

void iree_hal_amdgpu_allocation_queue_retain(
    iree_hal_amdgpu_allocation_queue_t* queue) {
  iree_hal_resource_retain(queue);
}

void iree_hal_amdgpu_allocation_queue_release(
    iree_hal_amdgpu_allocation_queue_t* queue) {
  iree_hal_resource_release(queue);
}

void iree_hal_amdgpu_allocation_queue_enqueue(
    iree_hal_amdgpu_allocation_queue_t* queue,
    iree_hal_slab_operation_t* operation) {
  iree_slim_mutex_lock(&queue->mutex);
  const bool wake = queue->head == NULL;
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
}
