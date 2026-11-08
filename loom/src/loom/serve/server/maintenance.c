// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/maintenance.h"

#include "iree/async/util/proactor_pool.h"
#include "loom/runtime/host/host.h"

iree_status_t loom_serve_maintenance_initialize(
    loom_runtime_device_group_t* group, iree_host_size_t policy_count,
    const loom_runtime_pool_idle_policy_t* policies,
    loom_serve_maintenance_t* out_maintenance) {
  *out_maintenance =
      (loom_serve_maintenance_t){.group = group, .policies = policies};
  loom_runtime_timer_initialize(&out_maintenance->timer);
  loom_runtime_host_t* host =
      group && loom_runtime_device_group_count(group)
          ? loom_runtime_device_host(loom_runtime_device_group_at(group, 0))
          : NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_reclamation_initialize(
      host, iree_make_duration_ms(1000), &out_maintenance->compiler));
  if (!policy_count) {
    return iree_ok_status();
  }
  if (!group || !policies ||
      policy_count != loom_runtime_device_group_pool_count(group)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "idle policies must cover the entire pool inventory");
  }
  iree_duration_t interval = IREE_DURATION_INFINITE;
  for (iree_host_size_t i = 0; i < policy_count; ++i) {
    const loom_runtime_pool_idle_policy_t policy = policies[i];
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
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
      loom_runtime_host_proactor_pool(host), 0, &proactor));
  loom_runtime_timer_configure(&out_maintenance->timer, proactor, interval);
  return iree_ok_status();
}

void loom_serve_maintenance_set_observer(
    loom_serve_maintenance_t* maintenance,
    loom_runtime_wake_callback_t observer) {
  loom_runtime_timer_set_observer(&maintenance->timer, observer);
  loom_runtime_reclamation_set_observer(&maintenance->compiler, observer);
}

iree_status_t loom_serve_maintenance_advance(
    loom_serve_maintenance_t* maintenance, bool stopping, bool* out_progress) {
  iree_status_t status = loom_runtime_reclamation_advance(
      &maintenance->compiler, stopping, out_progress);
  if (!maintenance->timer.interval) {
    return status;
  }
  bool pending = false;
  if (!stopping && iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_BEGIN(z0);
    const iree_time_t now = iree_time_now();
    const iree_host_size_t count =
        loom_runtime_device_group_pool_count(maintenance->group);
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      bool pool_pending = false;
      status = loom_runtime_pool_trim_idle(
          loom_runtime_device_group_pool_at(maintenance->group, i),
          maintenance->policies[i], now, &pool_pending);
      pending |= pool_pending;
    }
    IREE_TRACE_ZONE_END(z0);
  }
  bool tick = false;
  bool timer_progress = false;
  iree_status_t timer_status = loom_runtime_timer_advance(
      &maintenance->timer, stopping || !iree_status_is_ok(status), pending,
      &timer_progress, &tick);
  status = iree_status_join(status, timer_status);
  *out_progress |= timer_progress;
  return status;
}

bool loom_serve_maintenance_is_pending(loom_serve_maintenance_t* maintenance) {
  return loom_runtime_reclamation_is_pending(&maintenance->compiler) ||
         (maintenance->timer.interval &&
          loom_runtime_timer_is_pending(&maintenance->timer));
}

void loom_serve_maintenance_deinitialize(
    loom_serve_maintenance_t* maintenance) {
  loom_runtime_reclamation_deinitialize(&maintenance->compiler);
  loom_runtime_timer_deinitialize(&maintenance->timer);
}
