// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/init.h"

#include <stdio.h>   // for sscanf
#include <string.h>  // for memset, memcpy

#include "streaming/internal.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
// TODO(pyre): remove when init.c is replaced by libpyre global state
// #include "iree/hal/drivers/hip/registration/driver_module.h"
#include "iree/hal/remote/client/registration/driver_module.h"
#include "hsa_driver/registration/driver_module.h"
//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Global device registry.
static iree_hal_streaming_device_registry_t*
    iree_hal_streaming_global_registry = NULL;

// Accessor function for the global device registry.
iree_hal_streaming_device_registry_t* iree_hal_streaming_device_registry(void) {
  return iree_hal_streaming_global_registry;
}

//===----------------------------------------------------------------------===//
// Device enumeration and management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_deinitialize_device(
    iree_hal_streaming_device_t* device);

iree_hal_streaming_device_t* iree_hal_streaming_device_entry(
    iree_hal_streaming_device_ordinal_t ordinal) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  iree_hal_streaming_device_t* device = NULL;
  if (!device_registry || ordinal >= device_registry->device_count) {
    device = NULL;
  } else {
    device = &device_registry->devices[ordinal];
  }
  return device;
}

// Queries device info and populates device properties.
static iree_status_t iree_hal_streaming_query_device_info(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device->hal_device);

  // Query compute capability from the device architecture string.
  // AMD GCN/CDNA/RDNA architectures map to HIP compute capability:
  //   gfx900 -> 9.0, gfx906 -> 9.0, gfx908 -> 9.0, gfx90a -> 9.0
  //   gfx942 -> 9.4, gfx950 -> 9.5
  //   gfx1030 -> 10.3, gfx1100 -> 11.0
  char arch_name[64] = {0};
  iree_status_t arch_status = iree_hal_device_query_string(
      device->hal_device, IREE_SV("hal.device"), IREE_SV("architecture"),
      sizeof(arch_name), arch_name);
  if (iree_status_is_ok(arch_status) && arch_name[0] != '\0') {
    // Parse "gfxNNNN" to extract major.minor.
    // gfx9xx -> major=9, minor=x (e.g., gfx942 -> 9.4)
    // gfx10xx -> major=10, minor=x
    // gfx11xx -> major=11, minor=x
    int gfx_num = 0;
    if (sscanf(arch_name, "gfx%d", &gfx_num) == 1) {
      if (gfx_num >= 1000) {
        device->compute_capability_major = gfx_num / 100;
        device->compute_capability_minor = (gfx_num / 10) % 10;
      } else if (gfx_num >= 900) {
        device->compute_capability_major = gfx_num / 100;
        device->compute_capability_minor = (gfx_num / 10) % 10;
      } else {
        device->compute_capability_major = 7;
        device->compute_capability_minor = 5;
      }
    } else {
      device->compute_capability_major = 7;
      device->compute_capability_minor = 5;
    }
    // Store the architecture name for hipGetDeviceProperties.
    size_t name_len = strlen(arch_name);
    if (name_len >= sizeof(device->gcn_arch_name)) {
      name_len = sizeof(device->gcn_arch_name) - 1;
    }
    memcpy(device->gcn_arch_name, arch_name, name_len);
    device->gcn_arch_name[name_len] = '\0';
  } else {
    iree_status_ignore(arch_status);
    device->compute_capability_major = 9;
    device->compute_capability_minor = 4;
    memcpy(device->gcn_arch_name, "gfx942", 7);
  }

  // Query memory info from the HAL device.
  int64_t total_memory = 0;
  iree_status_t status = iree_hal_device_query_i64(
      device->hal_device, IREE_SV("hal.device"), IREE_SV("memory.total"),
      &total_memory);
  if (iree_status_is_ok(status) && total_memory > 0) {
    device->total_memory = (iree_device_size_t)total_memory;
  } else {
    // Fall back to 8GB default if query fails.
    iree_status_ignore(status);
    device->total_memory = 8ULL * 1024 * 1024 * 1024;
  }
  device->free_memory = device->total_memory;

  // Query cooperative launch support.
  // TODO: Query from actual device properties.
  // Cooperative launch requires Pascal (SM 6.0) or newer.
  device->supports_cooperative_launch = (device->compute_capability_major >= 6);

  // Query thread/block limits.
  device->max_threads_per_block = 1024;
  device->max_block_dim[0] = 1024;
  device->max_block_dim[1] = 1024;
  device->max_block_dim[2] = 64;
  device->max_grid_dim[0] = 2147483647;
  device->max_grid_dim[1] = 65535;
  device->max_grid_dim[2] = 65535;

  // Query warp/wavefront size from the HAL device.
  // AMD GPUs use 64, NVIDIA uses 32, RDNA may use 32 or 64.
  int64_t warp_size = 0;
  status = iree_hal_device_query_i64(
      device->hal_device, IREE_SV("hal.device"), IREE_SV("warp_size"),
      &warp_size);
  if (iree_status_is_ok(status) && warp_size > 0) {
    device->warp_size = (uint32_t)warp_size;
  } else {
    // Fall back to 64 for AMD (HIP) or 32 for CUDA.
    // Since this streaming layer is primarily used with HIP/AMD, default to 64.
    iree_status_ignore(status);
    device->warp_size = 64;
  }
  
  // Query multiprocessor (compute unit) count from the device.
  int64_t mp_count = 0;
  status = iree_hal_device_query_i64(
      device->hal_device, IREE_SV("hal.dispatch"), IREE_SV("concurrency"),
      &mp_count);
  if (iree_status_is_ok(status) && mp_count > 0) {
    device->multiprocessor_count = (uint32_t)mp_count;
  } else {
    // Fall back to generic value if query fails.
    // The HSA backend supports hal.dispatch.concurrency and will return
    // the actual CU count. For HIP backend, we use a generic fallback
    // which may cause different kernel variants to be selected.
    iree_status_ignore(status);
    device->multiprocessor_count = 80;
  }

  // Query occupancy calculation properties.
  // These are typical values for modern GPUs.
  device->max_threads_per_multiprocessor = 2048;
  device->max_blocks_per_multiprocessor = 32;
  device->max_registers_per_multiprocessor = 65536;
  device->max_shared_memory_per_multiprocessor = 49152;  // 48KB.
  device->max_registers_per_block = 65536;
  device->max_shared_memory_per_block = 49152;  // 48KB.

  return iree_ok_status();
}

// Initializes a single device.
static iree_status_t iree_hal_streaming_initialize_device(
    iree_hal_streaming_device_registry_t* registry, iree_hal_driver_t* driver,
    const iree_hal_device_info_t* device_info,
    iree_hal_streaming_device_t* out_device) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(device_info);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clear the entry.
  memset(out_device, 0, sizeof(*out_device));

  // Create HAL device from driver.
  // Let the HAL create its own dispatch stream rather than using the
  // default stream (0). The default stream (NULL) can cause issues with
  // GPU cache coherency for atomic operations in split-K GEMM kernels.
  // A dedicated stream created with hipStreamNonBlocking ensures proper
  // ordering and visibility.
  {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = registry->proactor_pool;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_driver_create_device_by_id(
                driver, device_info->device_id,
                /*param_count=*/0, /*params=*/NULL, &create_params,
                registry->host_allocator, &out_device->hal_device));
  }

  // Set driver and retain.
  out_device->driver = driver;
  iree_hal_driver_retain(driver);

  // Copy device info with deep copies of strings.
  // The device_info struct contains iree_string_view_t fields that point into
  // a buffer that will be freed after device enumeration, so we need to make
  // our own copies.
  out_device->info.device_id = device_info->device_id;

  // Allocate and copy the path string.
  if (device_info->path.size > 0) {
    char* path_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(registry->host_allocator,
                                  device_info->path.size + 1,
                                  (void**)&path_copy));
    memcpy(path_copy, device_info->path.data, device_info->path.size);
    path_copy[device_info->path.size] = '\0';
    out_device->info.path =
        iree_make_string_view(path_copy, device_info->path.size);
  } else {
    out_device->info.path = iree_string_view_empty();
  }

  // Allocate and copy the name string.
  if (device_info->name.size > 0) {
    char* name_copy = NULL;
    iree_status_t name_status = iree_allocator_malloc(
        registry->host_allocator, device_info->name.size + 1,
        (void**)&name_copy);
    if (!iree_status_is_ok(name_status)) {
      // Clean up path if name allocation fails.
      if (out_device->info.path.size > 0) {
        iree_allocator_free(registry->host_allocator,
                            (void*)out_device->info.path.data);
      }
      IREE_TRACE_ZONE_END(z0);
      return name_status;
    }
    memcpy(name_copy, device_info->name.data, device_info->name.size);
    name_copy[device_info->name.size] = '\0';
    out_device->info.name =
        iree_make_string_view(name_copy, device_info->name.size);
  } else {
    out_device->info.name = iree_string_view_empty();
  }

  // Query and initialize all device properties.
  iree_status_t status = iree_hal_streaming_query_device_info(out_device);

  // Initialize primary context flags with defaults.
  out_device->primary_context_flags.scheduling_mode =
      IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  out_device->primary_context_flags.map_host_memory = true;
  out_device->primary_context_flags.resize_local_mem_to_max = false;

  // Initialize primary context mutex for lazy initialization.
  iree_slim_mutex_initialize(&out_device->primary_context_mutex);

  // Initialize primary context reference count to 0.
  out_device->primary_context_ref_count = 0;

  // Initialize the arena block pool for graph allocations.
  // Use 64KB blocks as a good balance.
  if (iree_status_is_ok(status)) {
    const iree_host_size_t block_size = 64 * 1024;  // 64KB blocks
    iree_arena_block_pool_initialize(block_size, registry->host_allocator,
                                     &out_device->block_pool);
    status = iree_arena_block_pool_preallocate(&out_device->block_pool, 16);
  }

  // Primary context is NOT created here - it will be created lazily on first
  // access. This matches CUDA/HIP behavior where the primary context is not
  // active after init.
  out_device->primary_context = NULL;

  // Memory pools will be created when the primary context is created.
  out_device->default_mem_pool = NULL;
  out_device->current_mem_pool = NULL;

  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_deinitialize_device(out_device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Deinitializes a device, releasing all its resources.
static void iree_hal_streaming_deinitialize_device(
    iree_hal_streaming_device_t* device) {
  if (!device) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get allocator from global registry for freeing string copies.
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_device_registry();
  iree_allocator_t host_allocator = registry ? registry->host_allocator
                                             : iree_allocator_system();

  // Free the device name and path strings that were allocated during
  // initialization.
  if (device->info.path.size > 0 && device->info.path.data) {
    iree_allocator_free(host_allocator, (void*)device->info.path.data);
  }
  if (device->info.name.size > 0 && device->info.name.data) {
    iree_allocator_free(host_allocator, (void*)device->info.name.data);
  }
  device->info.path = iree_string_view_empty();
  device->info.name = iree_string_view_empty();

  // Release memory pools.
  if (device->current_mem_pool) {
    iree_hal_streaming_mem_pool_release(device->current_mem_pool);
    device->current_mem_pool = NULL;
  }
  if (device->default_mem_pool) {
    iree_hal_streaming_mem_pool_release(device->default_mem_pool);
    device->default_mem_pool = NULL;
  }

  // Release primary context (may not exist if never accessed).
  if (device->primary_context) {
    iree_hal_streaming_context_release(device->primary_context);
    device->primary_context = NULL;
  }

  // Deinitialize primary context mutex.
  iree_slim_mutex_deinitialize(&device->primary_context_mutex);

  // Deinitialize the arena block pool.
  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Release HAL device if created.
  if (device->hal_device) {
    iree_hal_device_release(device->hal_device);
    device->hal_device = NULL;
  }

  // Release driver.
  if (device->driver) {
    iree_hal_driver_release(device->driver);
    device->driver = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

// Queries device P2P capabilities and populates topology.
static iree_status_t iree_hal_streaming_query_p2p_capabilities(
    iree_hal_streaming_device_registry_t* registry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate P2P topology array.
  registry->p2p_link_count = registry->device_count * registry->device_count;
  const iree_host_size_t topology_size =
      registry->p2p_link_count * sizeof(iree_hal_streaming_p2p_link_t);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(registry->host_allocator,
                                             topology_size,
                                             (void**)&registry->p2p_topology));
  memset(registry->p2p_topology, 0, topology_size);

  // Populate P2P links for all device pairs.
  iree_host_size_t link_index = 0;
  for (iree_host_size_t i = 0; i < registry->device_count; ++i) {
    for (iree_host_size_t j = 0; j < registry->device_count; ++j) {
      iree_hal_streaming_p2p_link_t* link =
          &registry->p2p_topology[link_index++];
      link->src_device = i;
      link->dst_device = j;
      if (i == j) {
        // Device can always access itself with best performance.
        link->access_supported = true;
        link->native_atomic_supported = true;
        link->cuda_array_access_supported = true;
        link->performance_rank = 100;   // Highest rank for same device.
        link->bandwidth_mbps = 900000;  // 900 GB/s typical for device memory.
        link->latency_ns = 10;          // Very low latency.
      } else if (registry->devices[i].driver == registry->devices[j].driver) {
        // Same driver might support P2P.
        // TODO: Query actual P2P capabilities from driver.
        link->access_supported = false;  // Conservative default.
        link->native_atomic_supported = false;
        link->cuda_array_access_supported = false;
        link->performance_rank = 0;
        link->bandwidth_mbps = 0;
        link->latency_ns = 0;
      } else {
        // Different drivers: no P2P.
        link->access_supported = false;
        link->native_atomic_supported = false;
        link->cuda_array_access_supported = false;
        link->performance_rank = -1;  // Not supported.
        link->bandwidth_mbps = 0;
        link->latency_ns = 0;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Enumerates and registers all available devices.
static iree_status_t iree_hal_streaming_enumerate_devices(
    iree_hal_streaming_device_registry_t* registry) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t driver_info_count = 0;
  iree_hal_driver_info_t* driver_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
      registry->driver_registry, registry->host_allocator, &driver_info_count,
      &driver_infos));

  // Initialize device array (fixed-size in struct, no allocation needed).
  memset(registry->devices, 0,
         IREE_HAL_STREAMING_MAX_DEVICES * sizeof(iree_hal_streaming_device_t));
  registry->device_count = 0;

  // DO NOT SUBMIT rewrite this

  // Enumerate and register all devices in a single pass.
  for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
    iree_hal_driver_t* driver = NULL;
    iree_status_t status = iree_hal_driver_registry_try_create(
        registry->driver_registry, driver_infos[i].driver_name,
        registry->host_allocator, &driver);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    iree_host_size_t device_info_count = 0;
    iree_hal_device_info_t* device_infos = NULL;
    status = iree_hal_driver_query_available_devices(
        driver, registry->host_allocator, &device_info_count, &device_infos);
    if (!iree_status_is_ok(status)) {
      iree_hal_driver_release(driver);
      iree_status_ignore(status);
      continue;
    }

    // Register each device.
    for (iree_host_size_t j = 0; j < device_info_count; ++j) {
      // Check if we've exceeded the maximum device count.
      if (registry->device_count >= IREE_HAL_STREAMING_MAX_DEVICES) {
        // TODO: make this part of a common path.
        iree_allocator_free(registry->host_allocator, device_infos);
        iree_hal_driver_release(driver);
        iree_allocator_free(registry->host_allocator, driver_infos);

        // Clean up already initialized devices.
        for (iree_host_size_t k = 0; k < registry->device_count; ++k) {
          iree_hal_streaming_deinitialize_device(&registry->devices[k]);
        }
        registry->device_count = 0;

        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "too many devices found (limit is %d)",
                                IREE_HAL_STREAMING_MAX_DEVICES);
      }

      iree_hal_streaming_device_t* device =
          &registry->devices[registry->device_count];

      // Set the device ordinal.
      device->ordinal = registry->device_count;

      // Initialize the device.
      status = iree_hal_streaming_initialize_device(registry, driver,
                                                    &device_infos[j], device);
      if (!iree_status_is_ok(status)) {
        // DO NOT SUBMIT
        iree_status_ignore(status);
        // Skip this device if we can't initialize it.
        continue;
      }

      registry->device_count++;
    }

    iree_allocator_free(registry->host_allocator, device_infos);
    iree_hal_driver_release(driver);
  }

  iree_allocator_free(registry->host_allocator, driver_infos);

  // Check if we found any devices.
  if (registry->device_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND, "no HAL devices found");
  }

  // Query P2P capabilities and populate topology.
  IREE_RETURN_IF_ERROR(iree_hal_streaming_query_p2p_capabilities(registry));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Context registration
//===----------------------------------------------------------------------===//

void iree_hal_streaming_register_context(
    iree_hal_streaming_context_t* context) {
  if (!context) return;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device_registry->context_list.mutex);

  // Add to tail of list.
  context->context_list_entry.prev = device_registry->context_list.tail;
  context->context_list_entry.next = NULL;

  if (device_registry->context_list.tail) {
    device_registry->context_list.tail->context_list_entry.next = context;
  } else {
    // First context in list.
    device_registry->context_list.head = context;
  }
  device_registry->context_list.tail = context;

  // Retain for the global list.
  iree_hal_streaming_context_retain(context);

  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_unregister_context(
    iree_hal_streaming_context_t* context) {
  if (!context) return;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device_registry->context_list.mutex);

  // Check if the context is actually in the list.
  // A context might not be in the list if it failed during initialization
  // before it could be registered, or if this is called multiple times.
  // A context is in the list if it's either the head/tail or has neighbors.
  const bool was_in_list = context == device_registry->context_list.head ||
                           context == device_registry->context_list.tail ||
                           context->context_list_entry.prev ||
                           context->context_list_entry.next;
  if (was_in_list) {
    // Remove from list.
    if (context->context_list_entry.prev) {
      context->context_list_entry.prev->context_list_entry.next =
          context->context_list_entry.next;
    } else if (context == device_registry->context_list.head) {
      // Was head of list.
      device_registry->context_list.head = context->context_list_entry.next;
    }

    if (context->context_list_entry.next) {
      context->context_list_entry.next->context_list_entry.prev =
          context->context_list_entry.prev;
    } else if (context == device_registry->context_list.tail) {
      // Was tail of list.
      device_registry->context_list.tail = context->context_list_entry.prev;
    }

    // Clear list pointers.
    context->context_list_entry.next = NULL;
    context->context_list_entry.prev = NULL;
  }

  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  // Only release the global list reference if the context was actually in the
  // list.
  if (was_in_list) {
    iree_hal_streaming_context_release(context);
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Global initialization
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT
iree_status_t iree_hal_local_sync_driver_module_register(
    iree_hal_driver_registry_t* registry);
iree_status_t iree_hal_local_task_driver_module_register(
    iree_hal_driver_registry_t* registry);

// Creates a remote device via the remote-tcp driver.
// The |driver_uri| should be like "remote-tcp://host:port".
static iree_status_t iree_hal_streaming_create_remote_device(
    iree_hal_streaming_device_registry_t* registry,
    const char* driver_name_str, const char* server_address) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(driver_name_str);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse driver name and device path from the URI.
  // "remote-tcp://host:port" → driver_name="remote-tcp", device_path="host:port"
  iree_string_view_t driver_name = iree_make_cstring_view("remote-tcp");
  const char* sep = strstr(server_address, "://");
  const char* path_str = sep ? sep + 3 : server_address;
  iree_string_view_t device_path = iree_make_cstring_view(path_str);

  // Create the driver.
  iree_hal_driver_t* driver = NULL;
  iree_status_t status = iree_hal_driver_registry_try_create(
      registry->driver_registry, driver_name, registry->host_allocator,
      &driver);

  // Create the device by path (just the host:port portion).
  iree_hal_streaming_device_t* out_device = NULL;
  if (iree_status_is_ok(status)) {
    if (registry->device_count >= IREE_HAL_STREAMING_MAX_DEVICES) {
      iree_hal_driver_release(driver);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many devices");
    }
    out_device = &registry->devices[registry->device_count];
    memset(out_device, 0, sizeof(*out_device));
    out_device->ordinal = registry->device_count;

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = registry->proactor_pool;
    create_params.frontier.tracker = &registry->client_tracker;

    status = iree_hal_driver_create_device_by_path(
        driver, driver_name, device_path,
        /*param_count=*/0, /*params=*/NULL, &create_params,
        registry->host_allocator, &out_device->hal_device);
  }

  if (iree_status_is_ok(status)) {
    out_device->driver = driver;
    iree_hal_driver_retain(driver);

    // Set basic device info.
    out_device->info.device_id = 0;
    out_device->info.path = iree_string_view_empty();
    out_device->info.name = iree_string_view_empty();

    // Query device info (architecture, memory, etc.) from the remote server.
    iree_status_t query_status =
        iree_hal_streaming_query_device_info(out_device);
    iree_status_ignore(query_status);  // Non-fatal.

    // Use the queried architecture name as the device name so that
    // hipDeviceGetName / torch.cuda.get_device_name returns the GPU ISA
    // (e.g. "gfx942") instead of a generic "Remote Device".  This is
    // also what rocBLAS uses to locate TensileLibrary files.
    if (out_device->gcn_arch_name[0] != '\0') {
      size_t arch_len = strlen(out_device->gcn_arch_name);
      char* name_copy = NULL;
      iree_status_t name_status = iree_allocator_malloc(
          registry->host_allocator, arch_len + 1, (void**)&name_copy);
      if (iree_status_is_ok(name_status)) {
        memcpy(name_copy, out_device->gcn_arch_name, arch_len + 1);
        out_device->info.name =
            iree_make_string_view(name_copy, arch_len);
      } else {
        iree_status_ignore(name_status);
      }
    }
    // Fallback if architecture query didn't produce a name.
    if (out_device->info.name.size == 0) {
      char* name_copy = NULL;
      iree_status_t name_status = iree_allocator_malloc(
          registry->host_allocator, strlen("Remote Device") + 1,
          (void**)&name_copy);
      if (iree_status_is_ok(name_status)) {
        memcpy(name_copy, "Remote Device", strlen("Remote Device") + 1);
        out_device->info.name =
            iree_make_string_view(name_copy, strlen("Remote Device"));
      } else {
        iree_status_ignore(name_status);
      }
    }

    // Initialize primary context state.
    out_device->primary_context_flags.scheduling_mode =
        IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    out_device->primary_context_flags.map_host_memory = true;
    out_device->primary_context_flags.resize_local_mem_to_max = false;
    iree_slim_mutex_initialize(&out_device->primary_context_mutex);
    out_device->primary_context_ref_count = 0;
    iree_arena_block_pool_initialize(64 * 1024, registry->host_allocator,
                                     &out_device->block_pool);
    out_device->primary_context = NULL;
    out_device->default_mem_pool = NULL;
    out_device->current_mem_pool = NULL;

    registry->device_count++;
  }

  iree_hal_driver_release(driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_init_global(
    iree_hal_streaming_init_flags_t flags, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (iree_hal_streaming_global_registry &&
      iree_hal_streaming_global_registry->initialized) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Create global registry.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                sizeof(iree_hal_streaming_device_registry_t),
                                (void**)&iree_hal_streaming_global_registry));

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  device_registry->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&device_registry->mutex);

  // Initialize context list.
  iree_slim_mutex_initialize(&device_registry->context_list.mutex);
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;

  // Create HAL driver registry.
  iree_status_t status = iree_hal_driver_registry_allocate(
      host_allocator, &device_registry->driver_registry);

  // Create proactor pool for async operations.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), host_allocator,
        &device_registry->proactor_pool);
  }

  // Initialize frontier tracker for remote HAL.
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_initialize(
        &device_registry->client_tracker,
        device_registry->client_axis_entries,
        IREE_ARRAYSIZE(device_registry->client_axis_entries),
        host_allocator);
  }

  // Register all available HAL drivers.
  if (iree_status_is_ok(status)) {
    // DO NOT SUBMIT env vars?
    // status = iree_hal_register_all_available_drivers(
    //     device_registry->driver_registry);
    const char* driver_name = getenv("IREE_HAL_DRIVER");
    if (driver_name && strcmp(driver_name, "hip") == 0) {
      // TODO(pyre): HIP driver removed, will be replaced by libpyre
      status = iree_make_status(IREE_STATUS_UNAVAILABLE, "HIP driver not available");
    } else if (driver_name && strcmp(driver_name, "hsa") == 0) {
      status =
          iree_hal_hsa_driver_module_register(device_registry->driver_registry);
    } else if (driver_name && strcmp(driver_name, "local-sync") == 0) {
      status = iree_hal_local_sync_driver_module_register(
          device_registry->driver_registry);
    } else if (driver_name &&
               strncmp(driver_name, "remote-tcp", 10) == 0) {
      status = iree_hal_remote_client_driver_module_register(
          device_registry->driver_registry);
    } else if (!driver_name ||
               (driver_name && strcmp(driver_name, "local-task") == 0)) {
      status = iree_hal_local_task_driver_module_register(
          device_registry->driver_registry);
    }
  }

  // Enumerate devices (or create remote device).
  if (iree_status_is_ok(status)) {
    const char* driver_name = getenv("IREE_HAL_DRIVER");
    if (driver_name && strncmp(driver_name, "remote-tcp", 10) == 0) {
      status = iree_hal_streaming_create_remote_device(
          device_registry, driver_name, driver_name);
    } else {
      status = iree_hal_streaming_enumerate_devices(device_registry);
    }
  }

  if (iree_status_is_ok(status)) {
    device_registry->initialized = true;
  } else {
    iree_hal_streaming_cleanup_global();
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_streaming_cleanup_global(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Clear the TLS current context first to avoid dangling references.
  // This is important because contexts may be destroyed below.
  iree_hal_streaming_context_set_current(NULL);

  // Force destroy all remaining contexts from the global list.
  // This ensures all contexts are properly destroyed even if the user
  // forgot to explicitly release them.
  // We grab the whole context list locally so we aren't in the lock (which
  // _shouldn't_ matter, but calling synchronize within a lock is bad form).
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  iree_hal_streaming_context_t* context_head =
      device_registry->context_list.head;
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  while (context_head) {
    iree_hal_streaming_context_t* context = context_head;

    // Move to next before releasing (release may modify the list).
    context_head = context->context_list_entry.next;

    // Clear the context's list pointers.
    context->context_list_entry.next = NULL;
    context->context_list_entry.prev = NULL;

    // Synchronize the context to ensure all operations complete.
    iree_status_ignore(iree_hal_streaming_context_synchronize(context));

    // Release the global list's reference.
    // Note: The user should have already released their reference via
    // hipCtxDestroy/cuCtxDestroy. If not, this is a user error but we
    // still need to clean up our reference.
    iree_hal_streaming_context_release(context);
  }

  iree_slim_mutex_lock(&device_registry->mutex);

  iree_slim_mutex_deinitialize(&device_registry->context_list.mutex);

  // Release all device resources.
  for (iree_host_size_t i = 0; i < device_registry->device_count; ++i) {
    iree_hal_streaming_device_t* device = &device_registry->devices[i];
    iree_hal_streaming_deinitialize_device(device);
  }

  // Free P2P topology array (devices array is fixed-size in struct).
  iree_allocator_free(device_registry->host_allocator,
                      device_registry->p2p_topology);

  // Release driver registry.
  iree_hal_driver_registry_free(device_registry->driver_registry);

  // Deinitialize frontier tracker.
  iree_async_frontier_tracker_deinitialize(&device_registry->client_tracker);

  // Release proactor pool.
  iree_async_proactor_pool_release(device_registry->proactor_pool);
  device_registry->proactor_pool = NULL;

  iree_slim_mutex_unlock(&device_registry->mutex);
  iree_slim_mutex_deinitialize(&device_registry->mutex);

  iree_allocator_free(device_registry->host_allocator, device_registry);
  iree_hal_streaming_global_registry = NULL;
  IREE_TRACE_ZONE_END(z0);
}
