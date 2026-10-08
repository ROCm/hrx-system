// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/init.h"
#include "iree/tooling/device_util.h"

struct loom_serve_device_t {
  // Borrowed enclosing application group, owning this record's storage.
  loom_serve_device_group_t* owner;
  // Owned device providing memory and queue resources.
  iree_hal_device_t* handle;
  // Exact dispatch queue borrowed from the device.
  iree_hal_queue_t* dispatch;
  // Exact transfer queue borrowed from the device.
  iree_hal_queue_t* transfer;
  // Shared private command backing, independent of model timelines.
  loom_serve_workspace_t* workspace;
  // Owned physical domain shared by all model reservations; NULL when fixed.
  loom_serve_memory_pool_t* memory_pool;
  // Parameter admission and eviction policy borrowing shared memory.
  loom_serve_residency_cache_t* residency_cache;
};

struct loom_serve_device_group_t {
  // Allocation policy for the group and its per-device resources.
  iree_allocator_t allocator;
  // Retained workers and I/O services outliving every HAL device.
  loom_serve_host_t* host;
  // Completion registry outliving the shared HAL group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Owned immutable topology and semaphore/memory namespace.
  iree_hal_device_group_t* handle;
  // One profiling session and sink enclosing every model on all devices.
  iree_hal_profiling_from_flags_t* profiling;
  // Number of initialized records, including nullable partial resources.
  iree_host_size_t device_count;
  // Owned records with separate pools and budgets, borrowed by model owners.
  loom_serve_device_t devices[];
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

static iree_status_t loom_serve_device_group_initialize(
    loom_serve_device_group_t* group,
    const loom_serve_device_options_t* options) {
  iree_hal_driver_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_driver_registry_allocate(group->allocator, &registry));
  iree_status_t status = iree_hal_register_all_available_drivers(registry);
  for (iree_host_size_t i = 0;
       i < group->device_count && iree_status_is_ok(status); ++i) {
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = loom_serve_host_proactor_pool(group->host);
    params.event_sink = iree_hal_device_event_sink_stderr();
    status =
        iree_hal_create_device(registry, options[i].uri, &params,
                               group->allocator, &group->devices[i].handle);
  }
  iree_hal_driver_registry_free(registry);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), group->allocator,
      &group->frontier_tracker));
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, group->frontier_tracker);
  for (iree_host_size_t i = 0;
       i < group->device_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_group_builder_add_device(&builder,
                                                      group->devices[i].handle);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(&builder, group->allocator,
                                                    &group->handle);
  }
  iree_hal_device_group_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_serve_device_initialize(
    loom_serve_device_t* device, const loom_serve_device_options_t* options) {
  loom_serve_device_group_t* group = device->owner;
  device->dispatch = loom_serve_device_select_queue(
      device->handle, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH);
  device->transfer = loom_serve_device_select_queue(
      device->handle, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER);
  if (!device->dispatch || !device->transfer) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device needs dispatch and transfer queues");
  }
  IREE_RETURN_IF_ERROR(loom_serve_workspace_create(
      group->handle, device->dispatch, device->transfer, &device->workspace,
      group->allocator));
  if (options->backing == LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_create(
        device->handle, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, options->slab_size,
        options->memory_limit, &device->memory_pool, group->allocator));
  }
  return loom_serve_residency_cache_create(
      device->memory_pool, &device->residency_cache, group->allocator);
}

iree_status_t loom_serve_device_group_create(
    loom_serve_host_t* host, iree_host_size_t device_count,
    const loom_serve_device_options_t* options,
    loom_serve_device_group_t** out_group, iree_allocator_t host_allocator) {
  *out_group = NULL;
  if (!device_count || device_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "serving group needs between 1 and %d devices",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    if (options[i].backing != LOOM_SERVE_DEVICE_BACKING_FIXED &&
        options[i].backing != LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported physical backing strategy");
    }
    if (options[i].backing == LOOM_SERVE_DEVICE_BACKING_FIXED &&
        options[i].memory_limit) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "a physical budget requires elastic backing");
    }
  }
  loom_serve_device_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*group) + device_count * sizeof(group->devices[0]),
      (void**)&group));
  group->allocator = host_allocator;
  group->host = host;
  group->device_count = device_count;
  loom_serve_host_retain(host);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    group->devices[i].owner = group;
  }
  iree_status_t status = loom_serve_device_group_initialize(group, options);
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_device_initialize(&group->devices[i], &options[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        group->handle, host_allocator, &group->profiling);
  }
  if (iree_status_is_ok(status)) {
    *out_group = group;
  } else {
    status = iree_status_join(status, loom_serve_device_group_destroy(group));
  }
  return status;
}

iree_status_t loom_serve_device_group_destroy(
    loom_serve_device_group_t* group) {
  if (!group) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < group->device_count; ++i) {
    if (group->devices[i].memory_pool &&
        loom_serve_memory_pool_statistics(group->devices[i].memory_pool)
            .reserved_bytes) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "serving device %" PRIhsz " still owns model reservations", i);
    }
  }
  iree_status_t status = iree_hal_end_profiling_from_flags(group->profiling);
  for (iree_host_size_t i = 0; i < group->device_count; ++i) {
    loom_serve_device_t* device = &group->devices[i];
    loom_serve_residency_cache_destroy(device->residency_cache);
    loom_serve_memory_pool_destroy(device->memory_pool);
    loom_serve_workspace_release(device->workspace);
  }
  iree_hal_device_group_release(group->handle);
  for (iree_host_size_t i = 0; i < group->device_count; ++i) {
    iree_hal_device_release(group->devices[i].handle);
  }
  iree_async_frontier_tracker_release(group->frontier_tracker);
  loom_serve_host_release(group->host);
  iree_allocator_free(group->allocator, group);
  return status;
}

iree_host_size_t loom_serve_device_group_count(
    const loom_serve_device_group_t* group) {
  return group->device_count;
}

loom_serve_device_t* loom_serve_device_group_at(
    loom_serve_device_group_t* group, iree_host_size_t device_index) {
  return &group->devices[device_index];
}

loom_serve_device_memory_statistics_t loom_serve_device_memory_statistics(
    const loom_serve_device_t* device) {
  loom_serve_device_memory_statistics_t statistics = {
      .workspace = loom_serve_workspace_statistics(device->workspace)};
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
  loom_serve_workspace_trim(device->workspace);
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

loom_serve_host_t* loom_serve_device_host(const loom_serve_device_t* device) {
  return device->owner->host;
}

iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device) {
  return device->owner->handle;
}

iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device) {
  return device->dispatch;
}

iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device) {
  return device->transfer;
}

iree_status_t loom_serve_device_create_execution(
    loom_serve_device_t* device, loom_serve_execution_t** out_execution) {
  return loom_serve_execution_create(device->dispatch, device->transfer,
                                     device->workspace,
                                     device->owner->allocator, out_execution);
}

loom_serve_memory_pool_t* loom_serve_device_memory_pool(
    const loom_serve_device_t* device) {
  return device->memory_pool;
}

loom_serve_residency_cache_t* loom_serve_device_residency_cache(
    const loom_serve_device_t* device) {
  return device->residency_cache;
}
