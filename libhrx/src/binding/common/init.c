// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/init.h"

#include <stdlib.h>
#include <string.h>

#include "common/amdgpu_architecture.h"
#include "common/context.h"
#include "common/device.h"
#include "common/graph_memory.h"
#include "common/hrx_bridge.h"
#include "common/mem_pool.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"
#include "libhrx/src/libhrx/device.h"
#include "libhrx/src/libhrx/device_group.h"
#include "libhrx/src/libhrx/runtime.h"

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
  if (value == 0) {
    return default_value;
  }
  if (value > UINT32_MAX) {
    return UINT32_MAX;
  }
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
  if (!iree_status_is_ok(arch_status)) {
    return arch_status;
  }
  iree_hal_streaming_amdgpu_architecture_t architecture = {0};
  if (!iree_hal_streaming_parse_amdgpu_architecture(arch_name, &architecture)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device architecture '%s' is not a valid exact "
                            "AMDGPU target identifier",
                            arch_name);
  }
  device->compute_capability_major = architecture.major;
  device->compute_capability_minor = architecture.minor;
  memcpy(device->gcn_arch_name, arch_name, strlen(arch_name) + 1);

  // Query total memory from the immutable HAL device spec.
  uint64_t total_memory = 0;
  iree_status_t status = HRX_CALL(hrx_device_get_property(
      device->hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total_memory,
      sizeof(total_memory)));
  if (!iree_status_is_ok(status)) {
    return status;
  }
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

// Initializes a single device from a HRX device handle.
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
  out_device->host_callbacks = &registry->host_callbacks;

  // Store HRX device and extract HAL device for direct HAL usage.
  out_device->hrx_device = hrx_dev;
  out_device->hal_device = hrx_device_hal(hrx_dev);

  // Get device name from HRX.
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
  out_device->primary_context_flags.map_host_memory = false;
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
  // access. The HIP primary context is not active after init.
  out_device->primary_context = NULL;

  // Memory pools will be created when the primary context is created.
  out_device->default_mem_pool = NULL;
  out_device->current_mem_pool = NULL;

  iree_slim_mutex_initialize(&out_device->graph_memory_mutex);
  out_device->graph_memory_used_current = 0;
  out_device->graph_memory_used_high = 0;
  out_device->graph_memory_reserved_current = 0;
  out_device->graph_memory_reserved_high = 0;
  out_device->graph_memory_blocks = NULL;

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
  if (!device) {
    return;
  }
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
  iree_hal_streaming_mem_pool_release(device->current_mem_pool);
  device->current_mem_pool = NULL;
  iree_hal_streaming_mem_pool_release(device->default_mem_pool);
  device->default_mem_pool = NULL;

  // Release primary context (may not exist if never accessed).
  iree_hal_streaming_context_release(device->primary_context);
  device->primary_context = NULL;

  iree_hal_streaming_execution_resource_table_deinitialize(
      &device->execution_resource_table);

  iree_hal_streaming_graph_memory_deinitialize(device);
  iree_slim_mutex_deinitialize(&device->graph_memory_mutex);

  // Deinitialize primary context mutex.
  iree_slim_mutex_deinitialize(&device->primary_context_mutex);

  // Deinitialize the arena block pool.
  iree_arena_block_pool_deinitialize(&device->block_pool);

  // The registry releases its sealed group after every borrowed device view.
  device->hal_device = NULL;
  device->hrx_device = NULL;

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Context registration
//===----------------------------------------------------------------------===//

void iree_hal_streaming_register_context(
    iree_hal_streaming_context_t* context) {
  if (!context) {
    return;
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return;
  }

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
  if (!context) {
    return;
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return;
  }

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
// Global initialization via HRX
//===----------------------------------------------------------------------===//

static const char* iree_hal_streaming_gpu_driver_name(void) {
  const char* value = getenv("HRX_GPU_DRIVER");
  return (value && value[0]) ? value : "amdgpu";
}

static const char* iree_hal_streaming_profile_file_path(void) {
  const char* value = getenv("HRX_PROFILE_FILE");
  return (value && value[0]) ? value : NULL;
}

static iree_status_t iree_hal_streaming_runtime_features(
    iree_hal_device_runtime_feature_flags_t* out_runtime_features) {
  *out_runtime_features = IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  const char* sanitizer = getenv("HRX_HAL_SANITIZER");
  if (!sanitizer || !sanitizer[0]) {
    return iree_ok_status();
  }

  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
  if (strcmp(sanitizer, "asan") == 0) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN;
  } else if (strcmp(sanitizer, "tsan") == 0) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN;
  } else if (strcmp(sanitizer, "asan,tsan") == 0 ||
             strcmp(sanitizer, "tsan,asan") == 0) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN |
                        IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported HRX_HAL_SANITIZER '%s'", sanitizer);
  }
  *out_runtime_features = runtime_features;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_profile_file_sink_create(
    const char* file_path, iree_allocator_t host_allocator,
    iree_hal_profile_sink_t** out_sink) {
  *out_sink = NULL;
  if (!file_path || !file_path[0]) {
    return iree_ok_status();
  }

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_create(
      IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
          IREE_IO_FILE_MODE_SHARE_READ,
      iree_make_cstring_view(file_path), /*initial_size=*/0, host_allocator,
      &file_handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_sink_create(file_handle, host_allocator,
                                               out_sink);
  }
  iree_io_file_handle_release(file_handle);
  return status;
}

static iree_status_t iree_hal_streaming_profile_data_families(
    iree_hal_device_profiling_data_families_t* out_data_families) {
  const char* value = getenv("HRX_PROFILE_MODE");
  if (!value || !value[0] || strcmp(value, "queue") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
  } else if (strcmp(value, "dispatch") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  } else if (strcmp(value, "executable") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
  } else if (strcmp(value, "all") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported HRX_PROFILE_MODE '%s'", value);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_device_group_create(
    hrx_runtime_t runtime,
    const iree_hal_streaming_device_group_options_t* options,
    hrx_device_group_t* out_group) {
  *out_group = NULL;
  hrx_device_group_t group = NULL;
  const char* driver_name = iree_hal_streaming_gpu_driver_name();
  if (strcmp(driver_name, "amdgpu") != 0 && strcmp(driver_name, "rdna") != 0 &&
      strcmp(driver_name, "cdna") != 0 &&
      strcmp(driver_name, "legacy-amdgpu") != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unknown HRX_GPU_DRIVER '%s' (expected amdgpu, rdna, cdna, or "
        "legacy-amdgpu)",
        driver_name);
  }
  const iree_allocator_t allocator = runtime->host_allocator;
  const iree_string_view_t scheme = iree_make_cstring_view(driver_name);
  iree_hal_driver_t* driver = NULL;
  iree_status_t status = hrx_runtime_acquire_driver(runtime, scheme, &driver);
  iree_host_size_t info_count = 0;
  iree_hal_device_info_t* infos = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_query_available_devices(driver, scheme, allocator,
                                                     &info_count, &infos);
  }
  iree_host_size_t selected_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < info_count; ++i) {
      if (strcmp(driver_name, "legacy-amdgpu") == 0 && !infos[i].path.size) {
        continue;
      }
      infos[selected_count++] = infos[i];
    }
    if (options->device_selector.select) {
      status =
          options->device_selector.select(options->device_selector.self, scheme,
                                          allocator, &selected_count, infos);
    }
  }
  if (iree_status_is_ok(status) && !selected_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no physical GPU devices selected");
  }
  if (iree_status_is_ok(status) &&
      selected_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "selected device count exceeds topology capacity");
  }
  iree_hal_device_runtime_feature_flags_t features = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_runtime_features(&features);
  }

  hrx_device_create_params_t params[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT] = {0};
  iree_string_builder_t uris[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  size_t uri_count = 0;
  for (size_t i = 0; i < selected_count && iree_status_is_ok(status); ++i) {
    iree_string_builder_initialize(allocator, &uris[i]);
    ++uri_count;
    status = iree_string_builder_append_format(
        &uris[i], "%s://%.*s", driver_name, (int)infos[i].path.size,
        infos[i].path.data);
    params[i].uri = iree_string_builder_buffer(&uris[i]);
    params[i].completion_wait_flags = HRX_WAIT_FLAG_YIELD;
  }
  if (iree_status_is_ok(status)) {
    status = hrx_device_group_create_with_extensions(
        runtime, selected_count, params, options->device_extensions, features,
        &group);
  }
  for (size_t i = 0; i < uri_count; ++i) {
    iree_string_builder_deinitialize(&uris[i]);
  }
  iree_allocator_free(allocator, infos);
  iree_hal_driver_release(driver);

  iree_hal_profile_sink_t* profile_sink = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_profile_file_sink_create(
        iree_hal_streaming_profile_file_path(), allocator, &profile_sink);
  }
  iree_hal_device_profiling_options_t profile_options = {.sink = profile_sink};
  if (iree_status_is_ok(status) && profile_sink) {
    status = iree_hal_streaming_profile_data_families(
        &profile_options.data_families);
  }
  for (size_t i = 0; profile_sink && i < hrx_device_group_device_count(group) &&
                     iree_status_is_ok(status);
       ++i) {
    status = hrx_device_profile_begin(hrx_device_group_get_device(group, i),
                                      &profile_options);
  }
  iree_hal_profile_sink_release(profile_sink);
  if (!iree_status_is_ok(status)) {
    hrx_device_group_release(group);
  } else {
    *out_group = group;
  }
  return status;
}

iree_status_t iree_hal_streaming_init_global(
    const iree_hal_streaming_init_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (iree_hal_streaming_global_registry &&
      iree_hal_streaming_global_registry->initialized) {
    if (IREE_UNLIKELY(iree_hal_streaming_global_registry->device_extensions !=
                      options->devices.device_extensions)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "streaming runtime is already initialized with a different HAL "
          "device extension configuration");
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Frontend storage is acquired before activating native devices.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                sizeof(iree_hal_streaming_device_registry_t),
                                (void**)&iree_hal_streaming_global_registry));

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  memset(device_registry, 0, sizeof(*device_registry));
  device_registry->host_allocator = host_allocator;
  device_registry->device_extensions = options->devices.device_extensions;
  device_registry->context_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  iree_hal_streaming_host_callback_queue_initialize(
      host_allocator, &device_registry->host_callbacks);

  // Initialize context list.
  iree_slim_mutex_initialize(&device_registry->context_list.mutex);
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;

  // Each frontend owns its runtime and sealed group. Native HRX callers may
  // independently create and release their own groups in the same process.
  hrx_host_allocator_t runtime_allocator;
  memcpy(&runtime_allocator, &host_allocator, sizeof(runtime_allocator));
  iree_status_t status = HRX_CALL(hrx_runtime_create(
      &options->runtime, runtime_allocator, &device_registry->runtime));
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_tls_key_create(
        iree_hal_streaming_context_cleanup_thread,
        &device_registry->context_key);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_device_group_create(
        device_registry->runtime, &options->devices, &device_registry->group);
  }
  const size_t gpu_count =
      hrx_device_group_device_count(device_registry->group);
  if (iree_status_is_ok(status) && gpu_count > IREE_HAL_STREAMING_MAX_DEVICES) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "selected device count exceeds registry capacity");
  }
  for (size_t i = 0; i < gpu_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_streaming_initialize_device(
        device_registry, hrx_device_group_get_device(device_registry->group, i),
        i, &device_registry->devices[i]);
    if (iree_status_is_ok(status)) {
      ++device_registry->device_count;
    }
  }

  // Must have at least one device.
  if (iree_status_is_ok(status) && device_registry->device_count == 0) {
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND, "no GPU devices found via HRX");
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

  // Release current and stacked context references before registry ownership.
  iree_hal_streaming_context_cleanup_thread(
      iree_hal_streaming_tls_get(device_registry->context_key));

  // Detach registry ownership; caller-owned context references remain live.
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  iree_hal_streaming_context_t* context_head =
      device_registry->context_list.head;
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context = context_head; context;
       context = context->context_list_entry.next) {
    iree_status_ignore(iree_hal_streaming_context_synchronize(context));
  }
  // Timeline completion may precede the callback worker's final CPU access.
  // Join that tail while registry/context owners remain live. Closed callback
  // state stays valid through native teardown to reject late error callbacks.
  iree_hal_streaming_host_callback_queue_stop(&device_registry->host_callbacks);
  while (context_head) {
    iree_hal_streaming_context_t* context = context_head;
    context_head = context->context_list_entry.next;
    context->context_list_entry.next = NULL;
    context->context_list_entry.prev = NULL;
    iree_hal_streaming_context_release(context);
  }

  iree_slim_mutex_deinitialize(&device_registry->context_list.mutex);

  // Release all device resources.
  for (iree_host_size_t i = 0; i < device_registry->device_count; ++i) {
    iree_hal_streaming_deinitialize_device(&device_registry->devices[i]);
  }

  // Release only this frontend's execution services, after its contexts and
  // compatibility metadata have finished using the group's device views.
  hrx_device_group_release(device_registry->group);
  hrx_runtime_release(device_registry->runtime);
  iree_hal_streaming_host_callback_queue_deinitialize(
      &device_registry->host_callbacks);

  iree_hal_streaming_tls_key_delete(device_registry->context_key);

  iree_allocator_free(device_registry->host_allocator, device_registry);
  iree_hal_streaming_global_registry = NULL;
  IREE_TRACE_ZONE_END(z0);
}
