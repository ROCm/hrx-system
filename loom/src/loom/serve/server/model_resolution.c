// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/model_resolution.h"

#include "iree/base/threading/mutex.h"
#include "loomc/iree.h"

struct loom_serve_model_resolution_t {
  // Intrusive task retained by the shared worker sink until final publication.
  loomc_task_t task;
  // Allocator owning this operation and its resolved declaration.
  iree_allocator_t host_allocator;
  // Borrowed immutable declaration retained by the registration through take.
  const loom_serve_model_configuration_t* configuration;
  // Owned declaration produced privately by the worker.
  loom_serve_model_configuration_t resolved;
  // Owned result, transferred after the terminal callback publishes ready.
  iree_status_t status;
  // Protects publication and the notification owner's final access.
  iree_slim_mutex_t mutex;
  // Terminal callback has relinquished all borrowed configuration access.
  bool ready;
  // Borrowed serialized-owner notification, never an inline model operation.
  loom_serve_wake_callback_t observer;
};

static void loom_serve_model_resolution_execute(loomc_task_t* task,
                                                loomc_host_size_t worker) {
  loom_serve_model_resolution_t* resolution =
      (loom_serve_model_resolution_t*)task;
  resolution->status = loom_serve_model_configuration_resolve(
      resolution->configuration, resolution->host_allocator,
      &resolution->resolved);
}

static void loom_serve_model_resolution_release(loomc_task_t* task) {
  loom_serve_model_resolution_t* resolution =
      (loom_serve_model_resolution_t*)task;
  iree_slim_mutex_lock(&resolution->mutex);
  resolution->ready = true;
  if (resolution->observer.wake) {
    resolution->observer.wake(resolution->observer.user_data);
  }
  iree_slim_mutex_unlock(&resolution->mutex);
}

static const loomc_task_vtable_t model_resolution_vtable = {
    .execute = loom_serve_model_resolution_execute,
    .destroy = loom_serve_model_resolution_release};

iree_status_t loom_serve_model_resolution_start(
    loomc_task_sink_t sink,
    const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_model_resolution_t** out_resolution) {
  *out_resolution = NULL;
  loom_serve_model_resolution_t* resolution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*resolution), (void**)&resolution));
  resolution->host_allocator = host_allocator;
  resolution->configuration = configuration;
  resolution->observer = observer;
  iree_slim_mutex_initialize(&resolution->mutex);
  loomc_task_initialize(&model_resolution_vtable, &resolution->task);
  iree_status_t status =
      iree_status_from_loomc(loomc_task_sink_submit(sink, &resolution->task));
  if (iree_status_is_ok(status)) {
    *out_resolution = resolution;
  } else {
    // Rejected submissions retain exclusive task ownership and owe no wake.
    resolution->observer = (loom_serve_wake_callback_t){0};
    loomc_task_destroy(&resolution->task);
    iree_slim_mutex_deinitialize(&resolution->mutex);
    iree_allocator_free(host_allocator, resolution);
  }
  return status;
}

bool loom_serve_model_resolution_take(
    loom_serve_model_resolution_t** resolution_ptr, iree_status_t* status,
    loom_serve_model_configuration_t* out_configuration) {
  loom_serve_model_resolution_t* resolution = *resolution_ptr;
  iree_slim_mutex_lock(&resolution->mutex);
  const bool ready = resolution->ready;
  iree_slim_mutex_unlock(&resolution->mutex);
  if (!ready) {
    return false;
  }
  *status = iree_status_join(*status, resolution->status);
  *out_configuration = resolution->resolved;
  iree_slim_mutex_deinitialize(&resolution->mutex);
  iree_allocator_free(resolution->host_allocator, resolution);
  *resolution_ptr = NULL;
  return true;
}
