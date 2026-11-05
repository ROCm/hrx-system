// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/warmup.h"

#include "iree/base/threading/notification.h"
#include "loom/serve/runtime/parameter/activation.h"

struct loom_serve_text_warmup_t {
  // Host ownership, independent of the preparing caller's lifetime.
  iree_allocator_t host_allocator;
  // Immutable physical-owner extent, shared by both coallocated arrays.
  iree_host_size_t count;
  // Independent parameter-ready timelines; never gate state initialization.
  loom_serve_execution_t** executions;
  // Complete invocation set protecting the model's physical parameter groups.
  loom_serve_residency_t** residencies;
  // Completion storage retained across preparation-to-service handoff.
  loom_serve_activation_t* activation;
  // Owned terminal outcome, preserved until model destruction.
  iree_status_t status;
};

iree_status_t loom_serve_text_warmup_allocate(
    iree_host_size_t pool_count, loom_serve_device_t* const* devices,
    loom_serve_pool_t* const* pools,
    loom_serve_residency_group_t* const* groups, void (*wake)(void* user_data),
    void* user_data, iree_allocator_t host_allocator,
    loom_serve_text_warmup_t** out_warmup) {
  *out_warmup = NULL;
  iree_host_size_t total = 0, executions_offset = 0, residencies_offset = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(loom_serve_text_warmup_t), &total,
                         IREE_STRUCT_FIELD(pool_count, loom_serve_execution_t*,
                                           &executions_offset),
                         IREE_STRUCT_FIELD(pool_count, loom_serve_residency_t*,
                                           &residencies_offset)));
  loom_serve_text_warmup_t* warmup = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total, (void**)&warmup));
  warmup->host_allocator = host_allocator;
  warmup->count = pool_count;
  warmup->executions =
      (loom_serve_execution_t**)((uint8_t*)warmup + executions_offset);
  warmup->residencies =
      (loom_serve_residency_t**)((uint8_t*)warmup + residencies_offset);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < pool_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_device_create_execution(devices[i], pools[i],
                                                &warmup->executions[i]);
    if (iree_status_is_ok(status)) {
      status = loom_serve_residency_allocate(
          loom_serve_pool_residency_cache(pools[i]), warmup->executions[i], 1,
          &groups[i], host_allocator, &warmup->residencies[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_activation_allocate(
        pool_count, warmup->residencies,
        loom_serve_host_proactor_pool(loom_serve_device_host(devices[0])), wake,
        user_data, host_allocator, &warmup->activation);
  }
  if (iree_status_is_ok(status)) {
    *out_warmup = warmup;
  } else {
    for (iree_host_size_t i = 0; i < pool_count; ++i) {
      loom_serve_residency_free(warmup->residencies[i]);
      loom_serve_execution_release(warmup->executions[i]);
    }
    iree_allocator_free(host_allocator, warmup);
  }
  return status;
}

void loom_serve_text_warmup_poll(loom_serve_text_warmup_t* warmup) {
  if (!warmup || !warmup->activation) {
    return;
  }
  loom_serve_activation_phase_t phase;
  warmup->status = loom_serve_activation_advance(warmup->activation, 0, &phase);
  if (phase == LOOM_SERVE_ACTIVATION_WAITING) {
    warmup->status = iree_status_join(
        warmup->status,
        loom_serve_activation_advance(
            warmup->activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  }
  if (phase == LOOM_SERVE_ACTIVATION_COMPLETE) {
    loom_serve_activation_free(warmup->activation);
    warmup->activation = NULL;
  }
}

void loom_serve_text_warmup_set_observer(loom_serve_text_warmup_t* warmup,
                                         void (*wake)(void* user_data),
                                         void* user_data) {
  if (warmup) {
    loom_serve_activation_set_observer(warmup->activation, wake, user_data);
  }
}

bool loom_serve_text_warmup_is_pending(const loom_serve_text_warmup_t* warmup) {
  return warmup && warmup->activation;
}

iree_status_t loom_serve_text_warmup_status(
    const loom_serve_text_warmup_t* warmup) {
  return warmup ? iree_status_clone(warmup->status) : iree_ok_status();
}

static void loom_serve_text_warmup_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

static bool loom_serve_text_warmup_complete(void* user_data) {
  loom_serve_text_warmup_t* warmup = user_data;
  loom_serve_text_warmup_poll(warmup);
  return !loom_serve_text_warmup_is_pending(warmup);
}

void loom_serve_text_warmup_await(loom_serve_text_warmup_t* warmup) {
  if (!loom_serve_text_warmup_is_pending(warmup)) {
    return;
  }
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_warmup_set_observer(warmup, loom_serve_text_warmup_notify,
                                      &notification);
  iree_notification_await(&notification, loom_serve_text_warmup_complete,
                          warmup, iree_infinite_timeout());
  loom_serve_text_warmup_set_observer(warmup, NULL, NULL);
  iree_notification_deinitialize(&notification);
}

void loom_serve_text_warmup_free(loom_serve_text_warmup_t* warmup) {
  if (!warmup) {
    return;
  }
  IREE_ASSERT(!warmup->activation);
  for (iree_host_size_t i = 0; i < warmup->count; ++i) {
    loom_serve_residency_free(warmup->residencies[i]);
    loom_serve_execution_release(warmup->executions[i]);
  }
  iree_status_free(warmup->status);
  iree_allocator_free(warmup->host_allocator, warmup);
}
