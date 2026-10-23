// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/options.h"

#include <string.h>

#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/tooling/device_flags.h"

IREE_FLAG(int32_t, workers, 8,
          "Shared CPU workers for compilation, scheduling and media jobs.");
IREE_FLAG(string, pool_backing, "elastic",
          "elastic shares demand-committed parameter/state/workspace slabs; "
          "fixed backs retained storage eagerly and supports device address "
          "sanitization.");
IREE_FLAG(int64_t, weight_slab_bytes, 1073741824,
          "Reusable physical slab size for elastic parameters and shared "
          "workspace (default 1 GiB). Coarse slabs reduce native allocation "
          "and mapping work; loading streams through independent bounded "
          "read windows. Zero selects "
          "the allocator recommendation. Ignored for fixed backing.");
IREE_FLAG(int64_t, state_slab_bytes, 67108864,
          "Physical commitment/reclamation granularity for elastic mutable "
          "state (default 64 MiB). Logical tensor and KV block sizes are "
          "unchanged. Zero selects the allocator recommendation. Ignored "
          "for fixed backing.");
IREE_FLAG(int64_t, memory_bytes, 0,
          "Shared physical parameter/state/workspace budget; a nonzero limit "
          "requires elastic backing and one device. Zero imposes no serving "
          "limit. Host snapshots and driver overhead are outside this budget.");

iree_status_t loom_serve_host_create_from_flags(
    const loom_serve_compiler_options_t* compiler_options,
    iree_allocator_t host_allocator, loom_serve_host_t** out_host) {
  return loom_serve_host_create((iree_host_size_t)FLAG_workers,
                                compiler_options, host_allocator, out_host);
}

iree_status_t loom_serve_device_group_allocate_from_flags(
    loom_serve_host_t* host, iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, loom_serve_device_group_t** out_group) {
  *out_group = NULL;
  const iree_string_view_list_t devices = iree_hal_device_flag_list();
  if (!devices.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device is required; specify a HAL device URI");
  }
  if (devices.count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at most %d --device values are supported",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  for (iree_host_size_t i = 0; i < devices.count; ++i) {
    if (!devices.values[i].size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device is required; device %zu has an empty URI", i);
    }
  }
  if ((strcmp(FLAG_pool_backing, "elastic") &&
       strcmp(FLAG_pool_backing, "fixed")) ||
      FLAG_weight_slab_bytes < 0 || FLAG_state_slab_bytes < 0 ||
      FLAG_memory_bytes < 0 ||
      (FLAG_memory_bytes && !strcmp(FLAG_pool_backing, "fixed"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool_backing must be elastic or fixed; slab sizes "
                            "and memory_bytes must be nonnegative; a physical "
                            "budget requires elastic backing");
  }
  if (devices.count > 1 && FLAG_memory_bytes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--memory_bytes requires one device; independent device pools cannot "
        "enforce a shared byte limit");
  }
  iree_status_t status = iree_ok_status();
  struct {
    // URI scheme borrowed from the device flag list through this function.
    iree_string_view_t name;
    // Retained configured driver released after group creation.
    iree_hal_driver_t* handle;
  } drivers[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT] = {0};
  iree_host_size_t driver_count = 0;
  loom_serve_device_options_t options[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT] = {0};
  for (iree_host_size_t i = 0; i < devices.count && iree_status_is_ok(status);
       ++i) {
    const iree_string_view_t name = iree_uri_schema(devices.values[i]);
    iree_host_size_t driver_index = 0;
    while (driver_index < driver_count &&
           !iree_string_view_equal(name, drivers[driver_index].name)) {
      ++driver_index;
    }
    if (driver_index == driver_count) {
      drivers[driver_index].name = name;
      if (iree_string_view_equal(name, IREE_SV("task"))) {
        drivers[driver_index].handle = loom_serve_host_task_driver(host);
        iree_hal_driver_retain(drivers[driver_index].handle);
      } else if (!driver_registry) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "no linked HAL driver for URI scheme '%.*s'",
                                  (int)name.size, name.data);
      } else {
        status = iree_hal_driver_registry_acquire_driver(
            driver_registry, name, &drivers[driver_index].handle);
      }
      if (iree_status_is_ok(status)) {
        ++driver_count;
      }
    }
    options[i] = (loom_serve_device_options_t){
        .driver = drivers[driver_index].handle,
        .uri = devices.values[i],
        .backing = !strcmp(FLAG_pool_backing, "elastic")
                       ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                       : LOOM_SERVE_DEVICE_BACKING_FIXED,
        .weight_slab_size = (iree_device_size_t)FLAG_weight_slab_bytes,
        .state_slab_size = (iree_device_size_t)FLAG_state_slab_bytes,
        .memory_limit = (uint64_t)FLAG_memory_bytes,
    };
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_group_allocate(host, devices.count, options,
                                              host_allocator, out_group);
  }
  for (iree_host_size_t i = 0; i < driver_count; ++i) {
    iree_hal_driver_release(drivers[i].handle);
  }
  return status;
}
