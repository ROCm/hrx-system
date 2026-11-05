// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/maintenance.h"

#include "iree/async/util/proactor_pool.h"
#include "loom/serve/runtime/host/host.h"

iree_status_t loom_serve_maintenance_initialize(
    loom_serve_device_group_t* group, iree_host_size_t policy_count,
    const loom_serve_pool_idle_policy_t* policies,
    loom_serve_maintenance_t* out_maintenance) {
  *out_maintenance =
      (loom_serve_maintenance_t){.group = group, .policies = policies};
  loom_serve_timer_initialize(&out_maintenance->timer);
  if (!policy_count) {
    return iree_ok_status();
  }
  if (!group || !policies ||
      policy_count != loom_serve_device_group_pool_count(group)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "idle policies must cover the entire pool inventory");
  }
  iree_duration_t interval = IREE_DURATION_INFINITE;
  for (iree_host_size_t i = 0; i < policy_count; ++i) {
    const loom_serve_pool_idle_policy_t policy = policies[i];
    if (policy.workspace_timeout < 0 || policy.parameter_timeout < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "idle timeouts must be nonnegative");
    }
    if (policy.workspace_timeout) {
      interval = iree_min(interval, policy.workspace_timeout);
    }
    if (policy.parameter_timeout) {
      interval = iree_min(interval, policy.parameter_timeout);
    }
  }
  if (interval == IREE_DURATION_INFINITE) {
    return iree_ok_status();
  }
  interval = iree_min(interval, iree_make_duration_ms(1000));
  loom_serve_host_t* host =
      loom_serve_device_host(loom_serve_device_group_at(group, 0));
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host), 0, &proactor));
  loom_serve_timer_configure(&out_maintenance->timer, proactor, interval);
  return iree_ok_status();
}

void loom_serve_maintenance_set_observer(
    loom_serve_maintenance_t* maintenance,
    loom_runtime_wake_callback_t observer) {
  loom_serve_timer_set_observer(&maintenance->timer, observer);
}

iree_status_t loom_serve_maintenance_advance(
    loom_serve_maintenance_t* maintenance, bool stopping, bool* out_progress) {
  *out_progress = false;
  if (!maintenance->timer.interval) {
    return iree_ok_status();
  }
  bool pending = false;
  iree_status_t status = iree_ok_status();
  if (!stopping) {
    IREE_TRACE_ZONE_BEGIN(z0);
    const iree_time_t now = iree_time_now();
    const iree_host_size_t count =
        loom_serve_device_group_pool_count(maintenance->group);
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      bool pool_pending = false;
      status = loom_serve_pool_trim_idle(
          loom_serve_device_group_pool_at(maintenance->group, i),
          maintenance->policies[i], now, &pool_pending);
      pending |= pool_pending;
    }
    IREE_TRACE_ZONE_END(z0);
  }
  bool tick = false;
  return iree_status_join(
      status, loom_serve_timer_advance(&maintenance->timer,
                                       stopping || !iree_status_is_ok(status),
                                       pending, out_progress, &tick));
}

bool loom_serve_maintenance_is_pending(loom_serve_maintenance_t* maintenance) {
  return maintenance->timer.interval &&
         loom_serve_timer_is_pending(&maintenance->timer);
}

void loom_serve_maintenance_deinitialize(
    loom_serve_maintenance_t* maintenance) {
  loom_serve_timer_deinitialize(&maintenance->timer);
}
