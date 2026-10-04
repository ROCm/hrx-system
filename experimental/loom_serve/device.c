// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/init.h"

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

iree_status_t loom_serve_device_create(iree_string_view_t uri,
                                       iree_allocator_t host_allocator,
                                       loom_serve_device_t** out_device) {
  *out_device = NULL;
  loom_serve_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*device), (void**)&device));
  device->allocator = host_allocator;
  iree_status_t status = loom_serve_device_initialize(device, uri);
  if (iree_status_is_ok(status)) {
    *out_device = device;
  } else {
    loom_serve_device_destroy(device);
  }
  return status;
}

void loom_serve_device_destroy(loom_serve_device_t* device) {
  if (!device) {
    return;
  }
  loom_serve_execution_release(device->execution);
  iree_hal_device_group_release(device->group);
  iree_hal_device_release(device->handle);
  iree_async_frontier_tracker_release(device->frontier_tracker);
  iree_async_proactor_pool_release(device->proactor_pool);
  iree_allocator_free(device->allocator, device);
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
