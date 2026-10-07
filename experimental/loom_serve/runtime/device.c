// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/init.h"
#include "iree/tooling/device_util.h"

struct loom_serve_device_t {
  // Allocation policy for the owner and its runtime resources.
  iree_allocator_t allocator;
  // Async I/O services outliving the HAL device.
  iree_async_proactor_pool_t* proactor_pool;
  // Completion registry outliving the device group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Owned device providing memory and queue resources.
  iree_hal_device_t* handle;
  // Owned semaphore namespace for this device.
  iree_hal_device_group_t* group;
  // Exact dispatch queue borrowed from the device.
  iree_hal_queue_t* dispatch;
  // Exact transfer queue borrowed from the device.
  iree_hal_queue_t* transfer;
  // Owned work and feedback timelines retaining both exact queues.
  loom_serve_execution_t* execution;
  // Owned physical domain shared by all model reservations; NULL when fixed.
  loom_serve_memory_pool_t* memory_pool;
  // Parameter admission and eviction policy borrowing execution and memory.
  loom_serve_residency_cache_t* residency_cache;
  // Device-wide profiling session enclosing every resident model.
  iree_hal_profiling_from_flags_t* profiling;
};

static iree_hal_queue_t* loom_serve_device_select_queue(
    iree_hal_device_t* device, iree_hal_queue_family_role_flags_t role) {
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    if (queues->families[i].provisioned_queue_count &&
        iree_all_bits_set(queues->families[i].role_flags, role)) {
      return iree_hal_device_queue(device, i, 0);
    }
  }
  return NULL;
}

static iree_status_t loom_serve_device_initialize(loom_serve_device_t* device,
                                                  iree_string_view_t uri) {
  iree_hal_driver_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_driver_registry_allocate(device->allocator, &registry));
  iree_status_t status = iree_hal_register_all_available_drivers(registry);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), NULL,
        iree_async_proactor_pool_options_default(), device->allocator,
        &device->proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = device->proactor_pool;
    params.event_sink = iree_hal_device_event_sink_stderr();
    status = iree_hal_create_device(registry, uri, &params, device->allocator,
                                    &device->handle);
  }
  iree_hal_driver_registry_free(registry);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), device->allocator,
      &device->frontier_tracker));
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      device->handle, device->frontier_tracker, device->allocator,
      &device->group));
  device->dispatch = loom_serve_device_select_queue(
      device->handle, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH);
  device->transfer = loom_serve_device_select_queue(
      device->handle, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER);
  if (!device->dispatch || !device->transfer) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device needs dispatch and transfer queues");
  }
  return loom_serve_execution_create(device->dispatch, device->transfer,
                                     device->allocator, &device->execution);
}

iree_status_t loom_serve_device_create(
    const loom_serve_device_options_t* options,
    loom_serve_device_t** out_device, iree_allocator_t host_allocator) {
  *out_device = NULL;
  if (options->backing != LOOM_SERVE_DEVICE_BACKING_FIXED &&
      options->backing != LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported physical backing strategy");
  }
  if (options->backing == LOOM_SERVE_DEVICE_BACKING_FIXED &&
      options->memory_limit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a physical budget requires elastic backing");
  }
  loom_serve_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*device), (void**)&device));
  device->allocator = host_allocator;
  iree_status_t status = loom_serve_device_initialize(device, options->uri);
  if (iree_status_is_ok(status) &&
      options->backing == LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
    status = loom_serve_memory_pool_create(
        iree_hal_device_allocator(device->handle),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, options->slab_size,
        options->memory_limit, &device->memory_pool, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_cache_create(
        device->memory_pool, device->execution, &device->residency_cache,
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        device->group, host_allocator, &device->profiling);
  }
  if (iree_status_is_ok(status)) {
    *out_device = device;
  } else {
    status = iree_status_join(status, loom_serve_device_destroy(device));
  }
  return status;
}

iree_status_t loom_serve_device_destroy(loom_serve_device_t* device) {
  if (!device) {
    return iree_ok_status();
  }
  if (device->memory_pool &&
      loom_serve_memory_pool_statistics(device->memory_pool).reserved_bytes) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "serving device still owns model reservations");
  }
  iree_status_t status = iree_hal_end_profiling_from_flags(device->profiling);
  loom_serve_residency_cache_destroy(device->residency_cache);
  loom_serve_memory_pool_destroy(device->memory_pool);
  loom_serve_execution_release(device->execution);
  iree_hal_device_group_release(device->group);
  iree_hal_device_release(device->handle);
  iree_async_frontier_tracker_release(device->frontier_tracker);
  iree_async_proactor_pool_release(device->proactor_pool);
  iree_allocator_free(device->allocator, device);
  return status;
}

loom_serve_device_memory_statistics_t loom_serve_device_memory_statistics(
    const loom_serve_device_t* device) {
  loom_serve_device_memory_statistics_t statistics = {
      .workspace =
          loom_serve_execution_workspace_statistics(device->execution)};
  if (device->memory_pool) {
    statistics.retained =
        loom_serve_memory_pool_statistics(device->memory_pool);
  }
  return statistics;
}

iree_status_t loom_serve_device_trim(loom_serve_device_t* device,
                                     uint64_t target_bytes) {
  loom_serve_device_memory_statistics_t memory =
      loom_serve_device_memory_statistics(device);
  if (memory.workspace.bytes_committed <= target_bytes &&
      memory.retained.committed_bytes <=
          target_bytes - memory.workspace.bytes_committed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_trim_workspace(device->execution));
  memory = loom_serve_device_memory_statistics(device);
  const uint64_t retained_target =
      memory.workspace.bytes_committed < target_bytes
          ? target_bytes - memory.workspace.bytes_committed
          : 0;
  return loom_serve_residency_cache_trim(device->residency_cache,
                                         retained_target);
}

iree_hal_device_t* loom_serve_device_handle(const loom_serve_device_t* device) {
  return device->handle;
}

iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device) {
  return device->group;
}

iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device) {
  return device->dispatch;
}

iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device) {
  return device->transfer;
}

loom_serve_execution_t* loom_serve_device_execution(
    const loom_serve_device_t* device) {
  return device->execution;
}

loom_serve_memory_pool_t* loom_serve_device_memory_pool(
    const loom_serve_device_t* device) {
  return device->memory_pool;
}

loom_serve_residency_cache_t* loom_serve_device_residency_cache(
    const loom_serve_device_t* device) {
  return device->residency_cache;
}
