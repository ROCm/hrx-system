// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_service.h"

static bool iree_hal_amd_xdna_queue_service_is_ready(void* user_data) {
  iree_hal_amd_xdna_queue_service_t* service =
      (iree_hal_amd_xdna_queue_service_t*)user_data;
  if (iree_atomic_load(&service->stopping, iree_memory_order_acquire)) {
    return true;
  }
  iree_slim_mutex_lock(&service->mutex);
  const bool is_ready = !service->active && service->ready_head;
  iree_slim_mutex_unlock(&service->mutex);
  return is_ready;
}

static void iree_hal_amd_xdna_queue_service_handoff(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0,
                              "iree_hal_amd_xdna_queue_service_result_handoff");
  iree_hal_amd_xdna_queue_service_t* service =
      (iree_hal_amd_xdna_queue_service_t*)user_data;
  iree_hal_amd_xdna_queue_service_complete_fn_t complete =
      service->callbacks.complete;
  void* callback_user_data = service->user_data;
  iree_slim_mutex_lock(&service->mutex);
  iree_hal_amd_xdna_queue_service_item_t* item = service->active;
  iree_slim_mutex_unlock(&service->mutex);
  complete(callback_user_data, service, item, status);
  // The completion callback may release the final owner of |service|.
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amd_xdna_queue_service_abandon_all(
    iree_hal_amd_xdna_queue_service_t* service, iree_status_t status) {
  iree_slim_mutex_lock(&service->mutex);
  iree_hal_amd_xdna_queue_service_item_t* active = service->active;
  iree_hal_amd_xdna_queue_service_item_t* ready = service->ready_head;
  service->ready_head = NULL;
  service->ready_tail = NULL;
  iree_slim_mutex_unlock(&service->mutex);

  for (iree_hal_amd_xdna_queue_service_item_t* item = ready; item;) {
    iree_hal_amd_xdna_queue_service_item_t* next = item->next;
    item->next = NULL;
    service->callbacks.abandon(service->user_data, item,
                               iree_status_clone(status));
    item = next;
  }
  service->callbacks.abandon(service->user_data, active, status);
}

static int iree_hal_amd_xdna_queue_service_main(void* user_data) {
  iree_hal_amd_xdna_queue_service_t* service =
      (iree_hal_amd_xdna_queue_service_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  while (true) {
    // Classify the existing predicate for tracing without adding production
    // synchronization. An active service with no ready successor needs both
    // acknowledgement and new work before it can run again.
    IREE_TRACE(iree_zone_id_t wait_zone = 0);
    IREE_TRACE({
      iree_slim_mutex_lock(&service->mutex);
      const bool waiting_for_acknowledgement = service->active != NULL;
      const bool waiting_for_work = service->ready_head == NULL;
      iree_slim_mutex_unlock(&service->mutex);
      if (waiting_for_acknowledgement && waiting_for_work) {
        IREE_TRACE_ZONE_BEGIN_NAMED(
            z_wait,
            "iree_hal_amd_xdna_queue_service_wait_for_work_and_"
            "acknowledgement");
        wait_zone = z_wait;
      } else if (waiting_for_acknowledgement) {
        IREE_TRACE_ZONE_BEGIN_NAMED(
            z_wait, "iree_hal_amd_xdna_queue_service_wait_for_acknowledgement");
        wait_zone = z_wait;
      } else {
        IREE_TRACE_ZONE_BEGIN_NAMED(
            z_wait, "iree_hal_amd_xdna_queue_service_wait_for_work");
        wait_zone = z_wait;
      }
    });
    iree_notification_await(&service->notification,
                            iree_hal_amd_xdna_queue_service_is_ready, service,
                            iree_infinite_timeout());
    IREE_TRACE(IREE_TRACE_ZONE_END(wait_zone));
    if (iree_atomic_load(&service->stopping, iree_memory_order_acquire)) {
      break;
    }

    iree_slim_mutex_lock(&service->mutex);
    iree_hal_amd_xdna_queue_service_item_t* item = service->ready_head;
    if (item && !service->active) {
      service->ready_head = item->next;
      if (!service->ready_head) {
        service->ready_tail = NULL;
      }
      item->next = NULL;
      service->active = item;
    } else {
      item = NULL;
    }
    iree_slim_mutex_unlock(&service->mutex);
    if (!item) {
      continue;
    }

    IREE_TRACE_ZONE_BEGIN_NAMED(z_execute,
                                "iree_hal_amd_xdna_queue_service_execute");
    service->callbacks.execute(service->user_data, item);
    IREE_TRACE_ZONE_END(z_execute);
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_submit, "iree_hal_amd_xdna_queue_service_result_submit");
    iree_async_operation_initialize(
        &item->handoff, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_service_handoff,
        service);
    iree_status_t status =
        iree_async_proactor_submit_one(service->proactor, &item->handoff);
    IREE_TRACE_ZONE_END(z_submit);
    if (!iree_status_is_ok(status)) {
      iree_atomic_store(&service->stopping, 1, iree_memory_order_release);
      iree_hal_amd_xdna_queue_service_abandon_all(service, status);
      break;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return 0;
}

iree_status_t iree_hal_amd_xdna_queue_service_initialize(
    iree_string_view_t name, iree_async_proactor_t* proactor,
    iree_hal_amd_xdna_queue_service_callbacks_t callbacks, void* user_data,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_queue_service_t* out_service) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);
  memset(out_service, 0, sizeof(*out_service));
  out_service->proactor = proactor;
  out_service->user_data = user_data;
  out_service->callbacks = callbacks;
  iree_slim_mutex_initialize(&out_service->mutex);
  iree_notification_initialize(&out_service->notification);
  iree_atomic_store(&out_service->stopping, 0, iree_memory_order_relaxed);
  const iree_thread_create_params_t params = {
      .name = name,
      .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
  };
  iree_status_t status =
      iree_thread_create(iree_hal_amd_xdna_queue_service_main, out_service,
                         params, host_allocator, &out_service->thread);
  if (!iree_status_is_ok(status)) {
    iree_notification_deinitialize(&out_service->notification);
    iree_slim_mutex_deinitialize(&out_service->mutex);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amd_xdna_queue_service_deinitialize(
    iree_hal_amd_xdna_queue_service_t* service) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!service->thread) {
    IREE_TRACE_ZONE_END(z0);
    return;
  }
  iree_atomic_store(&service->stopping, 1, iree_memory_order_release);
  iree_notification_post(&service->notification, IREE_ALL_WAITERS);
  IREE_TRACE_ZONE_BEGIN_NAMED(z_join, "iree_hal_amd_xdna_queue_service_join");
  iree_thread_release(service->thread);
  IREE_TRACE_ZONE_END(z_join);
  service->thread = NULL;
  iree_notification_deinitialize(&service->notification);
  iree_slim_mutex_deinitialize(&service->mutex);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amd_xdna_queue_service_enqueue(
    iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item) {
  IREE_TRACE_ZONE_BEGIN(z0);
  item->next = NULL;
  iree_slim_mutex_lock(&service->mutex);
  const bool should_wake = !service->active && !service->ready_head;
  if (service->ready_tail) {
    service->ready_tail->next = item;
  } else {
    service->ready_head = item;
  }
  service->ready_tail = item;
  iree_slim_mutex_unlock(&service->mutex);
  if (should_wake) {
    iree_notification_post(&service->notification, 1);
  }
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amd_xdna_queue_service_acknowledge(
    iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&service->mutex);
  IREE_ASSERT(service->active == item);
  service->active = NULL;
  const bool should_wake = service->ready_head != NULL;
  iree_slim_mutex_unlock(&service->mutex);
  if (should_wake) {
    iree_notification_post(&service->notification, 1);
  }
  IREE_TRACE_ZONE_END(z0);
}
