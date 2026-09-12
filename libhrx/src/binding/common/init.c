// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "common/internal.h"
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

static uint32_t iree_hal_streaming_u32_or_default(uint64_t value,
                                                  uint32_t default_value) {
  if (value == 0) return default_value;
  if (value > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)value;
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
  iree_status_t arch_status = hrx_to_iree_status(hrx_device_get_property(
      device->hrx_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch_name,
      sizeof(arch_name)));
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

  // Query total memory from the immutable HAL device spec.
  uint64_t total_memory = 0;
  iree_status_t status = HRX_CALL(hrx_device_get_property(
      device->hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total_memory,
      sizeof(total_memory)));
  if (!iree_status_is_ok(status)) return status;
  if (total_memory > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HRX device total memory exceeds the representable "
                            "iree_device_size_t range");
  }
  device->total_memory = (iree_device_size_t)total_memory;
  iree_atomic_store(&device->free_memory, device->total_memory,
                    iree_memory_order_relaxed);

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(device->hal_device);
  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  const iree_hal_device_launch_spec_t* launch =
      dispatch ? &dispatch->launch : NULL;
  const iree_hal_device_subgroup_spec_t* subgroup =
      dispatch ? &dispatch->subgroup : NULL;
  const iree_hal_device_execution_spec_t* execution =
      dispatch ? &dispatch->execution : NULL;
  const bool is_gfx1100 = strncmp(device->gcn_arch_name, "gfx1100", 7) == 0;
  const bool is_gfx942 = strncmp(device->gcn_arch_name, "gfx942", 6) == 0;

  // Cooperative launch is a property of the queue family selected for HIP
  // streams, not of a marketing architecture number. A provisioned
  // cooperative queue can be used directly; otherwise the family must support
  // both the feature and dynamic acquisition of the specialized realization.
  iree_hal_queue_t* primary_queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_device_select_primary_queue(device, &primary_queue));
  const iree_hal_queue_family_spec_t* primary_queue_family_spec =
      iree_hal_queue_family_spec(iree_hal_queue_family(primary_queue));
  device->supports_cooperative_launch =
      iree_all_bits_set(iree_hal_queue_features(primary_queue),
                        IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH) ||
      (iree_all_bits_set(primary_queue_family_spec->supported_queue_features,
                         IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH) &&
       iree_all_bits_set(primary_queue_family_spec->flags,
                         IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION));

  device->max_threads_per_block = iree_hal_streaming_u32_or_default(
      launch ? launch->maximum_workgroup_invocations : 0, 1024);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(device->max_block_dim); ++i) {
    device->max_block_dim[i] = iree_hal_streaming_u32_or_default(
        launch ? launch->maximum_workgroup_size[i] : 0, i == 2 ? 64 : 1024);
    device->max_grid_dim[i] = iree_hal_streaming_u32_or_default(
        launch ? launch->maximum_workgroup_count[i] : 0,
        i == 0 ? 2147483647u : 65535u);
  }

  device->warp_size = iree_hal_streaming_u32_or_default(
      subgroup ? subgroup->default_size : 0, 64);
  if (is_gfx1100) {
    // HIP reports wave32 as the warp size for RDNA3 devices. IREE/HSA may
    // expose the hardware wavefront width instead, which in turn prevents the
    // CU-to-WGP compatibility adjustment below.
    device->warp_size = 32;
  }

  uint32_t multiprocessor_count = iree_hal_streaming_u32_or_default(
      execution ? execution->unit_count : 0, 80);
  // IREE/HSA reports raw compute units, while HIP reports RDNA devices in
  // WGP-like units. Keep this HIP-compatible because rocBLAS/hipBLASLt query
  // the physical multiprocessor count when selecting GEMM solutions.
  if (device->compute_capability_major >= 10 && device->warp_size == 32 &&
      multiprocessor_count > 1 && (multiprocessor_count % 2) == 0) {
    multiprocessor_count /= 2;
  }
  device->multiprocessor_count = multiprocessor_count;

  device->max_threads_per_multiprocessor = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_resident_invocation_count : 0, 2048);
  device->max_blocks_per_multiprocessor = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_resident_workgroup_count : 0,
      is_gfx942 ? 2 : 32);
  uint64_t maximum_register_count =
      execution ? execution->maximum_register_count : 0;
  if (maximum_register_count == 0 && execution) {
    maximum_register_count = execution->maximum_workgroup_register_count;
  }
  device->max_registers_per_multiprocessor =
      iree_hal_streaming_u32_or_default(maximum_register_count, 65536);
  uint64_t maximum_local_memory_size =
      execution ? execution->maximum_local_memory_size : 0;
  if (maximum_local_memory_size == 0 && execution) {
    maximum_local_memory_size = execution->maximum_workgroup_local_memory_size;
  }
  device->max_shared_memory_per_multiprocessor =
      iree_hal_streaming_u32_or_default(maximum_local_memory_size,
                                        is_gfx942 ? 19922944u : 49152u);
  device->max_registers_per_block = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_workgroup_register_count : 0, 65536);
  device->max_shared_memory_per_block = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_workgroup_local_memory_size : 0,
      (is_gfx942 || is_gfx1100) ? 65536u : 49152u);
  device->max_shared_memory_per_block_optin = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_workgroup_local_memory_size_optin : 0,
      device->max_shared_memory_per_block);

  return iree_ok_status();
}

// Initializes a single device from a pyre device handle.
static iree_status_t iree_hal_streaming_initialize_device(
    iree_hal_streaming_device_registry_t* registry, hrx_device_t hrx_dev,
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_hal_streaming_device_t* out_device) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(hrx_dev);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_device, 0, sizeof(*out_device));
  // The entry was just zeroed, so the ordinal MUST be (re)assigned here.
  // Per-device pools, peer lookups, and context->device_ordinal all key off
  // it; if it stays 0 every device aliases device 0.
  out_device->ordinal = ordinal;

  // Store pyre device and extract HAL device for direct HAL usage.
  out_device->hrx_device = hrx_dev;
  out_device->hal_device = hrx_device_hal(hrx_dev);

  // Get device name from pyre.
  char name_buf[128] = {0};
  iree_status_t status = HRX_CALL(hrx_device_get_property(
      hrx_dev, HRX_DEVICE_PROPERTY_NAME, name_buf, sizeof(name_buf)));
  if (iree_status_is_ok(status) && name_buf[0] != '\0') {
    size_t len = strlen(name_buf);
    char* name_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(registry->host_allocator, len + 1,
                                  (void**)&name_copy));
    memcpy(name_copy, name_buf, len + 1);
    out_device->info.name = iree_make_string_view(name_copy, len);
  } else {
    iree_status_ignore(status);
    out_device->info.name = iree_string_view_empty();
  }

  // Use architecture as path.
  char arch_buf[64] = {0};
  status = HRX_CALL(hrx_device_get_property(
      hrx_dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch_buf, sizeof(arch_buf)));
  if (iree_status_is_ok(status) && arch_buf[0] != '\0') {
    size_t len = strlen(arch_buf);
    char* path_copy = NULL;
    iree_status_t path_status = iree_allocator_malloc(
        registry->host_allocator, len + 1, (void**)&path_copy);
    if (iree_status_is_ok(path_status)) {
      memcpy(path_copy, arch_buf, len + 1);
      out_device->info.path = iree_make_string_view(path_copy, len);
    } else {
      iree_status_ignore(path_status);
      out_device->info.path = iree_string_view_empty();
    }
  } else {
    iree_status_ignore(status);
    out_device->info.path = iree_string_view_empty();
  }

  // Query and initialize all device properties.
  status = iree_hal_streaming_query_device_info(out_device);

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

  iree_slim_mutex_initialize(&out_device->graph_memory_mutex);
  out_device->graph_memory_used_current = 0;
  out_device->graph_memory_used_high = 0;
  out_device->graph_memory_reserved_current = 0;
  out_device->graph_memory_reserved_high = 0;
  out_device->graph_memory_reusable_size_entries = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_execution_resource_table_initialize(
        out_device->hal_device, registry->host_allocator,
        &out_device->execution_resource_table);
  }

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
  iree_allocator_t host_allocator =
      registry ? registry->host_allocator : iree_allocator_system();

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
  hrx_mem_pool_release(device->current_mem_pool);
  device->current_mem_pool = NULL;
  hrx_mem_pool_release(device->default_mem_pool);
  device->default_mem_pool = NULL;

  // Release primary context (may not exist if never accessed).
  iree_hal_streaming_context_release(device->primary_context);
  device->primary_context = NULL;

  iree_hal_streaming_execution_resource_table_deinitialize(
      &device->execution_resource_table);

  iree_hal_streaming_graph_memory_size_entry_t* graph_memory_entry =
      device->graph_memory_reusable_size_entries;
  device->graph_memory_reusable_size_entries = NULL;
  while (graph_memory_entry) {
    iree_hal_streaming_graph_memory_size_entry_t* next_entry =
        graph_memory_entry->next;
    iree_allocator_free(host_allocator, graph_memory_entry);
    graph_memory_entry = next_entry;
  }
  iree_slim_mutex_deinitialize(&device->graph_memory_mutex);

  // Deinitialize primary context mutex.
  iree_slim_mutex_deinitialize(&device->primary_context_mutex);

  // Deinitialize the arena block pool.
  iree_arena_block_pool_deinitialize(&device->block_pool);

  // HAL device and driver are owned by pyre — don't release here.
  // hrx_gpu_shutdown() handles cleanup.
  device->hal_device = NULL;
  device->hrx_device = NULL;

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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(registry->host_allocator, topology_size,
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
      } else {
        // TODO: Query actual P2P capabilities from pyre/HSA.
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

//===----------------------------------------------------------------------===//
// Context registration
//===----------------------------------------------------------------------===//

// Unlinks |context| while |device_registry->context_list.mutex| is held.
static bool iree_hal_streaming_context_list_unlink_locked(
    iree_hal_streaming_device_registry_t* device_registry,
    iree_hal_streaming_context_t* context) {
  const bool was_in_list =
      (iree_hal_streaming_device_registry_t*)iree_atomic_load(
          &context->context_registry, iree_memory_order_relaxed) ==
          device_registry &&
      (context == device_registry->context_list.head ||
       context == device_registry->context_list.tail ||
       context->context_list_entry.prev || context->context_list_entry.next);
  if (!was_in_list) return false;

  if (context->context_list_entry.prev) {
    context->context_list_entry.prev->context_list_entry.next =
        context->context_list_entry.next;
  } else {
    IREE_ASSERT(context == device_registry->context_list.head);
    device_registry->context_list.head = context->context_list_entry.next;
  }
  if (context->context_list_entry.next) {
    context->context_list_entry.next->context_list_entry.prev =
        context->context_list_entry.prev;
  } else {
    IREE_ASSERT(context == device_registry->context_list.tail);
    device_registry->context_list.tail = context->context_list_entry.prev;
  }

  context->context_list_entry.next = NULL;
  context->context_list_entry.prev = NULL;
  iree_atomic_store(&context->context_registry, (intptr_t)NULL,
                    iree_memory_order_release);
  return true;
}

void iree_hal_streaming_register_context(
    iree_hal_streaming_device_registry_t* device_registry,
    iree_hal_streaming_context_t* context) {
  if (!device_registry || !context) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  IREE_ASSERT(iree_atomic_load(&context->context_registry,
                               iree_memory_order_relaxed) == (intptr_t)NULL);
  IREE_ASSERT(context->context_list_entry.prev == NULL);
  IREE_ASSERT(context->context_list_entry.next == NULL);

  // Publish weak membership only after the context is fully initialized. The
  // creator's reference, not this list, keeps it alive during insertion.
  iree_atomic_store(&context->context_registry, (intptr_t)device_registry,
                    iree_memory_order_release);
  context->context_list_entry.prev = device_registry->context_list.tail;
  context->context_list_entry.next = NULL;

  if (device_registry->context_list.tail) {
    device_registry->context_list.tail->context_list_entry.next = context;
  } else {
    // First context in list.
    device_registry->context_list.head = context;
  }
  device_registry->context_list.tail = context;

  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_unregister_context(
    iree_hal_streaming_context_t* context) {
  if (!context) return;

  iree_hal_streaming_device_registry_t* device_registry =
      (iree_hal_streaming_device_registry_t*)iree_atomic_load(
          &context->context_registry, iree_memory_order_acquire);
  if (!device_registry) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device_registry->context_list.mutex);

  // Final destruction is the usual caller. A context may already be detached
  // if creation failed before registration or global cleanup promoted it while
  // it was still live.
  if (iree_hal_streaming_context_list_unlink_locked(device_registry, context)) {
    // Publish while holding the list mutex so a cleanup waiter that
    // subsequently observes an empty list also observes all preceding context
    // teardown.
    iree_notification_post(&device_registry->context_list.changed,
                           IREE_ALL_WAITERS);
  }

  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_streaming_context_t*
iree_hal_streaming_context_list_try_retain_registered(
    iree_hal_streaming_device_registry_t* device_registry,
    const iree_hal_streaming_context_t* handle) {
  if (!device_registry || !handle) return NULL;

  iree_hal_streaming_context_t* retained_context = NULL;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    if (context == handle) {
      if (iree_hal_streaming_context_try_retain(context)) {
        retained_context = context;
      }
      break;
    }
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return retained_context;
}

iree_hal_streaming_context_t* iree_hal_streaming_context_list_retain_next(
    iree_hal_streaming_device_registry_t* device_registry,
    const iree_hal_streaming_context_t* previous_context) {
  if (!device_registry) return NULL;

  iree_hal_streaming_context_t* retained_context = NULL;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  const iree_hal_streaming_device_registry_t* previous_registry =
      previous_context
          ? (iree_hal_streaming_device_registry_t*)iree_atomic_load(
                &previous_context->context_registry, iree_memory_order_relaxed)
          : NULL;
  iree_hal_streaming_context_t* context =
      previous_context && previous_registry == device_registry
          ? previous_context->context_list_entry.next
      : previous_context ? NULL
                         : device_registry->context_list.head;
  while (context) {
    iree_hal_streaming_context_t* next = context->context_list_entry.next;
    if (iree_hal_streaming_context_try_retain(context)) {
      retained_context = context;
      break;
    }
    context = next;
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return retained_context;
}

iree_status_t iree_hal_streaming_context_list_drain(
    iree_hal_streaming_device_registry_t* device_registry,
    iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(device_registry);
  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  for (;;) {
    iree_slim_mutex_lock(&device_registry->context_list.mutex);
    iree_hal_streaming_context_t* context = device_registry->context_list.head;
    if (!context) {
      iree_slim_mutex_unlock(&device_registry->context_list.mutex);
      return iree_ok_status();
    }

    if (iree_hal_streaming_context_try_retain(context)) {
      // Promotion must precede detachment: after membership is cleared, a final
      // release no longer has to rendezvous with the list mutex before freeing.
      const bool was_unlinked = iree_hal_streaming_context_list_unlink_locked(
          device_registry, context);
      IREE_ASSERT(was_unlinked);
      (void)was_unlinked;
      iree_notification_post(&device_registry->context_list.changed,
                             IREE_ALL_WAITERS);
      iree_slim_mutex_unlock(&device_registry->context_list.mutex);

      iree_status_ignore(iree_hal_streaming_context_synchronize(context));
      iree_hal_streaming_context_release(context);
      continue;
    }

    // A zero-reference entry is owned by its final destructor. Keep its
    // membership intact: the destructor will unlink only after all context and
    // device teardown, then post |changed| before freeing the storage.
    IREE_ASSERT_EQ(iree_atomic_ref_count_load(&context->ref_count), 0,
                   "failed to promote a positive-reference context");
    const iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&device_registry->context_list.changed);
    iree_slim_mutex_unlock(&device_registry->context_list.mutex);
    if (!iree_notification_commit_wait(&device_registry->context_list.changed,
                                       wait_token, IREE_DURATION_ZERO,
                                       deadline_ns)) {
      return iree_make_status(
          IREE_STATUS_DEADLINE_EXCEEDED,
          "timed out waiting for final context destruction");
    }
  }
}

//===----------------------------------------------------------------------===//
// Global initialization via pyre
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_init_global(
    const iree_hal_device_create_params_extension_t* device_extensions,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (iree_hal_streaming_global_registry &&
      iree_hal_streaming_global_registry->initialized) {
    if (IREE_UNLIKELY(iree_hal_streaming_global_registry->device_extensions !=
                      device_extensions)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "streaming runtime is already initialized with a different HAL "
          "device extension configuration");
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Initialize pyre GPU subsystem (idempotent — handles HSA init,
  // driver registration, device enumeration).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, HRX_CALL(hrx_gpu_initialize_with_device_extensions(
              /*flags=*/0, device_extensions)));

  // Create global registry.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                sizeof(iree_hal_streaming_device_registry_t),
                                (void**)&iree_hal_streaming_global_registry));

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  memset(device_registry, 0, sizeof(*device_registry));
  device_registry->host_allocator = host_allocator;
  device_registry->device_extensions = device_extensions;
  iree_slim_mutex_initialize(&device_registry->mutex);

  // Initialize context list.
  iree_slim_mutex_initialize(&device_registry->context_list.mutex);
  iree_notification_initialize(&device_registry->context_list.changed);
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;
  iree_hal_streaming_ipc_memory_registry_initialize(
      &device_registry->ipc_memory_registry, host_allocator);

  // Enumerate GPU devices from pyre.
  int gpu_count = 0;
  iree_status_t status = HRX_CALL(hrx_gpu_device_count(&gpu_count));

  if (iree_status_is_ok(status)) {
    memset(device_registry->devices, 0, sizeof(device_registry->devices));
    device_registry->device_count = 0;

    for (int i = 0; i < gpu_count && i < IREE_HAL_STREAMING_MAX_DEVICES; ++i) {
      hrx_device_t hrx_dev = NULL;
      iree_status_t dev_status = HRX_CALL(hrx_gpu_device_get(i, &hrx_dev));
      if (!iree_status_is_ok(dev_status)) {
        iree_status_ignore(dev_status);
        continue;
      }

      iree_hal_streaming_device_t* device =
          &device_registry->devices[device_registry->device_count];

      dev_status = iree_hal_streaming_initialize_device(
          device_registry, hrx_dev, device_registry->device_count, device);
      if (!iree_status_is_ok(dev_status)) {
        iree_status_ignore(dev_status);
        continue;
      }

      device_registry->device_count++;
    }
  }

  // Must have at least one device.
  if (iree_status_is_ok(status) && device_registry->device_count == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no GPU devices found via pyre");
  }

  // Query P2P capabilities.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_query_p2p_capabilities(device_registry);
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
  iree_hal_streaming_context_set_current(NULL);

  // Drain one entry at a time without allocating. Live entries are retained
  // before detachment; zero-reference entries remain linked until their final
  // destructors finish and publish completion.
  iree_status_ignore(iree_hal_streaming_context_list_drain(
      device_registry, iree_infinite_timeout()));

  iree_hal_streaming_ipc_memory_registry_deinitialize(
      &device_registry->ipc_memory_registry);

  iree_slim_mutex_lock(&device_registry->mutex);

  // Release all device resources.
  for (iree_host_size_t i = 0; i < device_registry->device_count; ++i) {
    iree_hal_streaming_deinitialize_device(&device_registry->devices[i]);
  }
  // A device's final primary-context release may run its destructor. Drain
  // detached every positive-reference entry, so it will not touch this mutex.
  iree_notification_deinitialize(&device_registry->context_list.changed);
  iree_slim_mutex_deinitialize(&device_registry->context_list.mutex);

  // Free P2P topology.
  iree_allocator_free(device_registry->host_allocator,
                      device_registry->p2p_topology);

  // Shutdown pyre GPU subsystem.
  hrx_status_ignore(hrx_gpu_shutdown());

  iree_slim_mutex_unlock(&device_registry->mutex);
  iree_slim_mutex_deinitialize(&device_registry->mutex);

  iree_allocator_free(device_registry->host_allocator, device_registry);
  iree_hal_streaming_global_registry = NULL;
  IREE_TRACE_ZONE_END(z0);
}
