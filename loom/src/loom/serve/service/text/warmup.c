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
  // Independent parameter-ready timeline; never gates state initialization.
  loom_serve_execution_t* execution;
  // Owned invocation borrow protecting the model's shared parameter group.
  loom_serve_residency_t* residency;
  // Completion storage retained across preparation-to-service handoff.
  loom_serve_activation_t* activation;
  // Owned terminal outcome, preserved until model destruction.
  iree_status_t status;
};

iree_status_t loom_serve_text_warmup_allocate(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    loom_serve_residency_group_t* group, void (*wake)(void* user_data),
    void* user_data, iree_allocator_t host_allocator,
    loom_serve_text_warmup_t** out_warmup) {
  *out_warmup = NULL;
  loom_serve_text_warmup_t* warmup = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*warmup), (void**)&warmup));
  warmup->host_allocator = host_allocator;
  iree_status_t status =
      loom_serve_device_create_execution(device, pool, &warmup->execution);
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_allocate(
        loom_serve_pool_residency_cache(pool), warmup->execution, 1, &group,
        host_allocator, &warmup->residency);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_activation_allocate(
        warmup->residency,
        loom_serve_host_proactor_pool(loom_serve_device_host(device)), wake,
        user_data, host_allocator, &warmup->activation);
  }
  if (iree_status_is_ok(status)) {
    *out_warmup = warmup;
  } else {
    loom_serve_residency_free(warmup->residency);
    loom_serve_execution_release(warmup->execution);
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
  loom_serve_residency_free(warmup->residency);
  loom_serve_execution_release(warmup->execution);
  iree_status_free(warmup->status);
  iree_allocator_free(warmup->host_allocator, warmup);
}
