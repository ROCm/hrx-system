// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/runtime.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"

static iree_status_t loom_run_hal_runtime_select_transfer_queue(
    iree_hal_device_t* device, iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
    const iree_hal_queue_family_spec_t* family_spec = &queue_spec->families[i];
    if (family_spec->provisioned_queue_count == 0 ||
        !iree_all_bits_set(family_spec->role_flags,
                           IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
      continue;
    }

    iree_hal_queue_t* queue = iree_hal_device_queue(
        device, (iree_hal_queue_family_ordinal_t)i, /*queue_ordinal=*/0);
    if (IREE_UNLIKELY(queue == NULL)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL device queue specification advertises provisioned transfer "
          "queue family %" PRIhsz " but the queue is unavailable",
          i);
    }
    *out_queue = queue;
    break;
  }
  return iree_ok_status();
}

void loom_run_hal_runtime_options_initialize(
    iree_string_view_t hal_driver_name,
    loom_run_hal_runtime_options_t* out_options) {
  *out_options = (loom_run_hal_runtime_options_t){
      .hal_driver_name = hal_driver_name,
      .event_sink = iree_hal_device_event_sink_stderr(),
  };
}

iree_hal_device_runtime_feature_flags_t
loom_run_hal_runtime_features_from_sanitizer_options(
    const loom_sanitizer_options_t* sanitizer_options) {
  if (!loom_sanitizer_options_is_enabled(sanitizer_options)) {
    return IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  }

  iree_hal_device_runtime_feature_flags_t runtime_features =
      sanitizer_options->reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
          ? IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE
          : IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
  if (iree_any_bit_set(sanitizer_options->checks,
                       LOOM_SANITIZER_CHECK_ACCESS)) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN;
  }
  if (iree_any_bit_set(sanitizer_options->checks, LOOM_SANITIZER_CHECK_RACE)) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN;
  }
  return runtime_features;
}

iree_status_t loom_run_hal_runtime_initialize(
    const loom_run_hal_runtime_options_t* options, iree_allocator_t allocator,
    loom_run_hal_runtime_t* out_runtime) {
  *out_runtime = (loom_run_hal_runtime_t){0};

  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), allocator, &proactor_pool);
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator,
        &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    create_params.event_sink = options->event_sink;
    create_params.runtime_features = options->runtime_features;
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), options->hal_driver_name,
        &create_params, allocator, &out_runtime->device);
  }
  iree_async_proactor_pool_release(proactor_pool);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_runtime_select_transfer_queue(
        out_runtime->device, &out_runtime->transfer_queue);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        out_runtime->device, frontier_tracker, allocator,
        &out_runtime->device_group);
  }
  iree_async_frontier_tracker_release(frontier_tracker);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_runtime_deinitialize(out_runtime);
  }
  return status;
}

void loom_run_hal_runtime_deinitialize(loom_run_hal_runtime_t* runtime) {
  if (runtime == NULL) {
    return;
  }
  runtime->transfer_queue = NULL;
  iree_hal_device_group_release(runtime->device_group);
  iree_hal_device_release(runtime->device);
  *runtime = (loom_run_hal_runtime_t){0};
}
