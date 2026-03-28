// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hsa_driver/hsa_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/thread.h"
#include "iree/base/tracing.h"
#include "hsa_driver/hsa_allocator.h"
#include "hsa_driver/hsa_buffer.h"
#include "hsa_driver/native_executable.h"
#include "hsa_driver/nop_executable_cache.h"
#include "hsa_driver/per_device_information.h"
#include "hsa_driver/status_util.h"
#include "hsa_driver/hsa_semaphore.h"
#include "hsa_driver/stream_command_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"

#define IREE_HAL_DEVICE_TRANSFER_DEFAULT_BUFFER_SIZE (128 * 1024 * 1024)
#define IREE_HAL_DEVICE_MAX_TRANSFER_DEFAULT_CHUNK_SIZE (64 * 1024 * 1024)

//===----------------------------------------------------------------------===//
// iree_hal_hsa_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hsa_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size.
  iree_arena_block_pool_t block_pool;

  iree_hal_driver_t* driver;


  // Parameters used to control device behavior.
  iree_hal_hsa_device_params_t params;

  iree_allocator_t host_allocator;

  // Device memory allocator.
  iree_hal_allocator_t* device_allocator;

  // Per-device information.
  iree_hal_hsa_per_device_info_t device_info;

  // Proactor pool for async I/O. Retained from create_params.
  iree_async_proactor_pool_t* proactor_pool;
  // Proactor selected for this device (from the pool).
  iree_async_proactor_t* proactor;

  // Device topology information (assigned by the framework).
  iree_hal_device_topology_info_t topology_info;
} iree_hal_hsa_device_t;

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value);

static const iree_hal_device_vtable_t iree_hal_hsa_device_vtable;

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_device_vtable);
  return (iree_hal_hsa_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hsa_device_params_initialize(
    iree_hal_hsa_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->stream_tracing = 0;
  out_params->async_allocations = true;
  out_params->file_transfer_buffer_size =
      IREE_HAL_DEVICE_TRANSFER_DEFAULT_BUFFER_SIZE;
  out_params->file_transfer_chunk_size =
      IREE_HAL_DEVICE_MAX_TRANSFER_DEFAULT_CHUNK_SIZE;
  out_params->allow_inline_execution = false;
}

static iree_status_t iree_hal_hsa_device_check_params(
    const iree_hal_hsa_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one queue is required");
  }
  return iree_ok_status();
}

// Callback for finding memory pools on an agent.
typedef struct iree_hal_hsa_memory_pool_search_t {
  iree_hal_hsa_per_device_info_t* device_info;
  hsa_agent_t agent;
} iree_hal_hsa_memory_pool_search_t;

static hsa_status_t iree_hal_hsa_find_memory_pools_callback(
    hsa_amd_memory_pool_t pool, void* data) {
  iree_hal_hsa_memory_pool_search_t* search =
      (iree_hal_hsa_memory_pool_search_t*)data;

  // Check if pool is valid for allocations.
  bool alloc_allowed = false;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS || !alloc_allowed) {
    return HSA_STATUS_SUCCESS;  // Skip this pool.
  }

  // Get pool segment type.
  hsa_amd_segment_t segment;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_SUCCESS;  // Skip this pool.
  }

  // Get global flags for global segment.
  if (segment == HSA_AMD_SEGMENT_GLOBAL) {
    uint32_t global_flags = 0;
    status = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flags);
    if (status != HSA_STATUS_SUCCESS) {
      return HSA_STATUS_SUCCESS;
    }

    // Check if this is a coarse-grained (device local) pool.
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
        !search->device_info->device_local_memory_pool_valid) {
      search->device_info->device_local_memory_pool = pool;
      search->device_info->device_local_memory_pool_valid = true;
    }

    // Check if this is a fine-grained (host visible) pool.
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        !search->device_info->host_visible_memory_pool_valid) {
      search->device_info->host_visible_memory_pool = pool;
      search->device_info->host_visible_memory_pool_valid = true;
    }

    // Check if this is a kernarg pool (using KERNARG_INIT which is the newer
    // flag).
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) &&
        !search->device_info->kernarg_memory_pool_valid) {
      search->device_info->kernarg_memory_pool = pool;
      search->device_info->kernarg_memory_pool_valid = true;
    }

    // Also use fine-grained pool for kernarg if KERNARG_INIT not available.
    // Fine-grained pools are host-visible and can be used for kernel arguments.
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        !search->device_info->kernarg_memory_pool_valid) {
      search->device_info->kernarg_memory_pool = pool;
      search->device_info->kernarg_memory_pool_valid = true;
    }
  }

  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_hsa_device_initialize_memory_pools(
    iree_hal_hsa_device_t* device, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent) {
  // Find memory pools on the GPU agent.
  iree_hal_hsa_memory_pool_search_t gpu_search = {
      .device_info = &device->device_info,
      .agent = gpu_agent,
  };
  IREE_HSA_RETURN_IF_ERROR(
      hsa_amd_agent_iterate_memory_pools(
          gpu_agent, iree_hal_hsa_find_memory_pools_callback, &gpu_search),
      "hsa_amd_agent_iterate_memory_pools (GPU)");

  // Find fine-grained memory pool on CPU agent if not found on GPU.
  if (!device->device_info.host_visible_memory_pool_valid) {
    iree_hal_hsa_memory_pool_search_t cpu_search = {
        .device_info = &device->device_info,
        .agent = cpu_agent,
    };
    IREE_HSA_RETURN_IF_ERROR(
        hsa_amd_agent_iterate_memory_pools(
            cpu_agent, iree_hal_hsa_find_memory_pools_callback, &cpu_search),
        "hsa_amd_agent_iterate_memory_pools (CPU)");
  }

  if (!device->device_info.device_local_memory_pool_valid) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no device local memory pool found");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_device = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_hsa_device_check_params(params));

  iree_hal_hsa_device_t* device = NULL;
  const iree_host_size_t total_size =
      iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_hsa_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  memcpy(&device->params, params, sizeof(*params));
  device->host_allocator = host_allocator;

  // Initialize per-device info.
  memset(&device->device_info, 0, sizeof(device->device_info));
  device->device_info.agent = gpu_agent;
  device->device_info.cpu_agent = cpu_agent;

  // Initialize topology info.
  memset(&device->topology_info, 0, sizeof(device->topology_info));

  // Retain proactor pool and get a proactor for this device.
  device->proactor_pool = NULL;
  device->proactor = NULL;
  if (create_params && create_params->proactor_pool) {
    iree_async_proactor_pool_retain(create_params->proactor_pool);
    device->proactor_pool = create_params->proactor_pool;
    iree_status_t proactor_status = iree_async_proactor_pool_get(
        device->proactor_pool, /*index=*/0, &device->proactor);
    if (!iree_status_is_ok(proactor_status)) {
      iree_allocator_free(host_allocator, device);
      IREE_TRACE_ZONE_END(z0);
      return proactor_status;
    }
  }

  iree_status_t status = iree_ok_status();

  // Initialize memory pools.
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_device_initialize_memory_pools(device, gpu_agent,
                                                         cpu_agent);
  }

  // Create HSA queue for dispatch.
  if (iree_status_is_ok(status)) {
    uint32_t queue_size = 0;
    status = IREE_HSA_CALL_TO_STATUS(
        hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                           &queue_size),
        "hsa_agent_get_info(QUEUE_MAX_SIZE)");
    if (iree_status_is_ok(status)) {
      // Use a reasonable default queue size.
      if (queue_size > 4096) queue_size = 4096;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL,
                           NULL, UINT32_MAX, UINT32_MAX,
                           &device->device_info.queue),
          "hsa_queue_create");
    }
  }

  // Create completion signal.
  if (iree_status_is_ok(status)) {
    status = IREE_HSA_CALL_TO_STATUS(
        hsa_signal_create(1, 0, NULL, &device->device_info.completion_signal),
        "hsa_signal_create");
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_initialize(&device->device_info.completion_signal_mutex);
    }
  }

  // Create device allocator.
  if (iree_status_is_ok(status)) {
    iree_hal_hsa_device_topology_t topology = {
        .count = 1,
        .devices = &device->device_info,
    };
    status = iree_hal_hsa_allocator_create((iree_hal_device_t*)device,
                                           topology, host_allocator,
                                           &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}


static void iree_hal_hsa_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release device allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Wait for any pending work to complete before destroying resources.
  // This ensures the queue is idle before destruction.
  if (device->device_info.queue && device->device_info.completion_signal.handle) {
    // Wait for the completion signal to reach its expected value.
    // A short timeout helps avoid hanging if something went wrong.
    hsa_signal_wait_scacquire(
        device->device_info.completion_signal,
        HSA_SIGNAL_CONDITION_EQ, 1,
        /*timeout_hint=*/1000000000ull,  // 1 second timeout
        HSA_WAIT_STATE_BLOCKED);
  }

  // Destroy completion signal.
  if (device->device_info.completion_signal.handle) {
    iree_slim_mutex_deinitialize(&device->device_info.completion_signal_mutex);
    IREE_HSA_IGNORE_ERROR(
        hsa_signal_destroy(device->device_info.completion_signal));
  }

  // Destroy queue.
  if (device->device_info.queue) {
    IREE_HSA_IGNORE_ERROR(
                          hsa_queue_destroy(device->device_info.queue));
  }

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Release proactor pool.
  iree_async_proactor_pool_release(device->proactor_pool);

  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_hsa_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_hsa_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_hsa_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_hsa_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hsa_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  // HSA backend does not support channels yet.
}

static iree_status_t iree_hal_hsa_device_trim(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_hsa_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value = iree_string_view_match_pattern(device->identifier, key) ? 1
                                                                         : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("rocm-hsaco-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      // For HSA, device concurrency is 1 (single device).
      *out_value = 1;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("memory.total"))) {
      // Query total memory from the device-local memory pool.
      if (!device->device_info.device_local_memory_pool_valid) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "no device local memory pool found");
      }
      size_t pool_size = 0;
      IREE_RETURN_IF_ERROR(IREE_HSA_CALL_TO_STATUS(
          hsa_amd_memory_pool_get_info(
              device->device_info.device_local_memory_pool,
              HSA_AMD_MEMORY_POOL_INFO_SIZE, &pool_size),
          "hsa_amd_memory_pool_get_info(SIZE)"));
      *out_value = (int64_t)pool_size;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("warp_size"))) {
      // Query wavefront (warp) size from the HSA agent.
      // AMD GPUs typically use 64, RDNA may use 32.
      uint32_t wavefront_size = 0;
      IREE_RETURN_IF_ERROR(IREE_HSA_CALL_TO_STATUS(
          hsa_agent_get_info(device->device_info.agent,
                             HSA_AGENT_INFO_WAVEFRONT_SIZE,
                             &wavefront_size),
          "hsa_agent_get_info(WAVEFRONT_SIZE)"));
      *out_value = (int64_t)wavefront_size;
      return iree_ok_status();
    }
  }

  if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      // Query compute unit count from the HSA agent.
      uint32_t compute_unit_count = 0;
      IREE_RETURN_IF_ERROR(IREE_HSA_CALL_TO_STATUS(
          hsa_agent_get_info(
              device->device_info.agent,
              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
              &compute_unit_count),
          "hsa_agent_get_info(COMPUTE_UNIT_COUNT)"));
      *out_value = (int64_t)compute_unit_count;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hsa_device_query_string(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, iree_host_size_t out_string_size,
    char* out_string) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  if (out_string_size == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output string too small");
  }
  out_string[0] = '\0';

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("architecture"))) {
      // Get the device name from HSA which is the architecture (e.g., "gfx942").
      char device_name[64] = {0};
      IREE_RETURN_IF_ERROR(IREE_HSA_CALL_TO_STATUS(
          hsa_agent_get_info(device->device_info.agent, HSA_AGENT_INFO_NAME,
                             device_name),
          "hsa_agent_get_info(HSA_AGENT_INFO_NAME)"));
      size_t name_len = strlen(device_name);
      if (out_string_size <= name_len) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output string too small");
      }
      memcpy(out_string, device_name, name_len + 1);
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hsa_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HSA collective channels not yet implemented");
}

static iree_status_t iree_hal_hsa_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);

  if (iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) &&
      iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION) &&
      device->params.allow_inline_execution) {
    // Use stream command buffer for inline execution.
    return iree_hal_hsa_stream_command_buffer_create(
        device->device_allocator,
        device->device_info.tracing_context, mode, command_categories,
        queue_affinity, binding_capacity, &device->device_info,
        &device->block_pool, device->host_allocator, out_command_buffer);
  }

  // Default to deferred command buffer.
  return iree_hal_deferred_command_buffer_create(
      device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, &device->block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_hsa_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "events not yet implemented");
}

static iree_status_t iree_hal_hsa_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_hal_hsa_device_topology_t topology = {
      .count = 1,
      .devices = &device->device_info,
  };
  return iree_hal_hsa_nop_executable_cache_create(
      identifier, topology, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_hsa_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      iree_hal_hsa_device_allocator(base_device), queue_affinity, access, handle,
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hsa_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_semaphore_create(device->proactor, initial_value,
                                       device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hsa_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_hsa_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue alloca not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue dealloca not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue read not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue write not yet implemented");
}

// Executes a command buffer inline (no wait semaphores to wait on).
static iree_status_t iree_hal_hsa_device_queue_execute_inline(
    iree_hal_hsa_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_status_t status = iree_ok_status();

  if (command_buffer != NULL) {
    if (iree_hal_hsa_stream_command_buffer_isa(command_buffer)) {
      // Stream command buffer - already executed inline.
    } else if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      // Create a stream command buffer and replay the deferred commands.
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      status = iree_hal_hsa_stream_command_buffer_create(
          device->device_allocator,
          device->device_info.tracing_context,
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          iree_hal_command_buffer_allowed_categories(command_buffer),
          queue_affinity, /*binding_capacity=*/0, &device->device_info,
          &device->block_pool, device->host_allocator, &stream_command_buffer);

      if (iree_status_is_ok(status)) {
        status = iree_hal_deferred_command_buffer_apply(
            command_buffer, stream_command_buffer, binding_table);
      }

      iree_hal_command_buffer_release(stream_command_buffer);
    }
  }

  return status;
}

// State for deferred (async) queue execution on a background thread.
typedef struct iree_hal_hsa_deferred_queue_execute_state_t {
  iree_hal_hsa_device_t* device;
  iree_hal_queue_affinity_t queue_affinity;
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t binding_table;

  // Thread handle — released by the thread function itself.
  iree_thread_t* thread;

  // Copies of the semaphore lists (we need to own the memory).
  iree_host_size_t wait_count;
  iree_host_size_t signal_count;
  // Flexible array: [wait_semaphores, signal_semaphores, wait_values, signal_values]
  // Laid out as:
  //   iree_hal_semaphore_t* wait_semaphores[wait_count]
  //   iree_hal_semaphore_t* signal_semaphores[signal_count]
  //   uint64_t wait_values[wait_count]
  //   uint64_t signal_values[signal_count]
  uint8_t payload[];
} iree_hal_hsa_deferred_queue_execute_state_t;

static int iree_hal_hsa_deferred_queue_execute_main(void* param) {
  iree_hal_hsa_deferred_queue_execute_state_t* state =
      (iree_hal_hsa_deferred_queue_execute_state_t*)param;

  // Reconstruct lists from payload.
  iree_hal_semaphore_t** wait_sems =
      (iree_hal_semaphore_t**)state->payload;
  iree_hal_semaphore_t** signal_sems =
      (iree_hal_semaphore_t**)(state->payload +
                               state->wait_count * sizeof(iree_hal_semaphore_t*));
  uint64_t* wait_vals =
      (uint64_t*)(state->payload +
                  (state->wait_count + state->signal_count) *
                      sizeof(iree_hal_semaphore_t*));
  uint64_t* signal_vals = wait_vals + state->wait_count;

  iree_hal_semaphore_list_t wait_list = {
      .count = state->wait_count,
      .semaphores = wait_sems,
      .payload_values = wait_vals,
  };
  iree_hal_semaphore_list_t signal_list = {
      .count = state->signal_count,
      .semaphores = signal_sems,
      .payload_values = signal_vals,
  };

  // Wait for all wait semaphores.
  iree_status_t status = iree_hal_semaphore_list_wait(
      wait_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);

  // Execute the command buffer.
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_device_queue_execute_inline(
        state->device, state->queue_affinity, state->command_buffer,
        state->binding_table);
  }

  // Signal or fail the signal semaphores.
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_signal(signal_list);
  } else {
    iree_hal_semaphore_list_fail(signal_list, status);
  }

  // Release retained resources.
  for (iree_host_size_t i = 0; i < state->wait_count; ++i) {
    iree_hal_semaphore_release(wait_sems[i]);
  }
  for (iree_host_size_t i = 0; i < state->signal_count; ++i) {
    iree_hal_semaphore_release(signal_sems[i]);
  }
  iree_hal_command_buffer_release(state->command_buffer);

  // Save the allocator and thread handle before releasing the device and
  // freeing state (avoids use-after-free on state->device->host_allocator).
  iree_allocator_t host_allocator = state->device->host_allocator;
  iree_thread_t* thread = state->thread;
  iree_hal_device_release((iree_hal_device_t*)state->device);
  iree_allocator_free(host_allocator, state);

  // Release the thread handle (the thread owns itself).
  iree_thread_release(thread);
  return 0;
}

static iree_status_t iree_hal_hsa_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);

  // Fast path: if no wait semaphores or all are already satisfied, execute
  // inline.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list)) {
    iree_status_t status = iree_hal_hsa_device_queue_execute_inline(
        device, queue_affinity, command_buffer, binding_table);
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_list_signal(signal_semaphore_list);
    } else {
      iree_hal_semaphore_list_fail(signal_semaphore_list, status);
    }
    return status;
  }

  // Slow path: spawn a thread to wait and then execute.
  iree_host_size_t payload_size =
      (wait_semaphore_list.count + signal_semaphore_list.count) *
          sizeof(iree_hal_semaphore_t*) +
      (wait_semaphore_list.count + signal_semaphore_list.count) *
          sizeof(uint64_t);
  iree_hal_hsa_deferred_queue_execute_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      device->host_allocator,
      sizeof(*state) + payload_size, (void**)&state));

  state->device = device;
  iree_hal_device_retain(base_device);
  state->queue_affinity = queue_affinity;
  state->command_buffer = command_buffer;
  iree_hal_command_buffer_retain(command_buffer);
  state->binding_table = binding_table;
  state->wait_count = wait_semaphore_list.count;
  state->signal_count = signal_semaphore_list.count;

  // Copy and retain semaphores.
  iree_hal_semaphore_t** wait_sems =
      (iree_hal_semaphore_t**)state->payload;
  iree_hal_semaphore_t** signal_sems =
      (iree_hal_semaphore_t**)(state->payload +
                               wait_semaphore_list.count * sizeof(iree_hal_semaphore_t*));
  uint64_t* wait_vals =
      (uint64_t*)(state->payload +
                  (wait_semaphore_list.count + signal_semaphore_list.count) *
                      sizeof(iree_hal_semaphore_t*));
  uint64_t* signal_vals = wait_vals + wait_semaphore_list.count;

  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    wait_sems[i] = wait_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(wait_sems[i]);
    wait_vals[i] = wait_semaphore_list.payload_values[i];
  }
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    signal_sems[i] = signal_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(signal_sems[i]);
    signal_vals[i] = signal_semaphore_list.payload_values[i];
  }

  // Launch worker thread. The thread handle is stored in state and released
  // by the thread function itself (matching iree_hal_device_queue_emulated_host_call).
  iree_thread_create_params_t thread_params;
  memset(&thread_params, 0, sizeof(thread_params));
  thread_params.name = iree_make_cstring_view("hsa_deferred_exec");
  iree_status_t status = iree_thread_create(
      iree_hal_hsa_deferred_queue_execute_main, state, thread_params,
      device->host_allocator, &state->thread);
  if (!iree_status_is_ok(status)) {
    // Cleanup on failure.
    for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
      iree_hal_semaphore_release(wait_sems[i]);
    }
    for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
      iree_hal_semaphore_release(signal_sems[i]);
    }
    iree_hal_command_buffer_release(state->command_buffer);
    iree_hal_device_release(base_device);
    iree_allocator_free(device->host_allocator, state);
  }

  return status;
}

static iree_status_t iree_hal_hsa_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Nothing to flush in our implementation.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_hsa_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_hsa_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  memcpy(&device->topology_info, topology_info, sizeof(*topology_info));
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Profiling not yet implemented.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_end(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_transfer_h2d_raw(
    iree_hal_device_t* base_device, const void* source,
    uint64_t target_device_ptr, iree_device_size_t data_length,
    iree_timeout_t timeout) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Use hsa_memory_copy for synchronous host-to-device transfer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_HSA_CALL_TO_STATUS(
                               hsa_memory_copy((void*)target_device_ptr, source,
                                               data_length),
                               "hsa_memory_copy(H2D)"));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_transfer_d2h_raw(
    iree_hal_device_t* base_device, uint64_t source_device_ptr, void* target,
    iree_device_size_t data_length, iree_timeout_t timeout) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Use hsa_memory_copy for synchronous device-to-host transfer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_HSA_CALL_TO_STATUS(
          hsa_memory_copy(target, (void*)source_device_ptr, data_length),
          "hsa_memory_copy(D2H)"));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_hsa_device_vtable = {
    .destroy = iree_hal_hsa_device_destroy,
    .id = iree_hal_hsa_device_id,
    .host_allocator = iree_hal_hsa_device_host_allocator,
    .device_allocator = iree_hal_hsa_device_allocator,
    .replace_device_allocator = iree_hal_hsa_replace_device_allocator,
    .replace_channel_provider = iree_hal_hsa_replace_channel_provider,
    .trim = iree_hal_hsa_device_trim,
    .query_i64 = iree_hal_hsa_device_query_i64,
    .query_capabilities = iree_hal_hsa_device_query_capabilities,
    .topology_info = iree_hal_hsa_device_topology_info,
    .refine_topology_edge = iree_hal_hsa_device_refine_topology_edge,
    .assign_topology_info = iree_hal_hsa_device_assign_topology_info,
    .query_string = iree_hal_hsa_device_query_string,
    .create_channel = iree_hal_hsa_device_create_channel,
    .create_command_buffer = iree_hal_hsa_device_create_command_buffer,
    .create_event = iree_hal_hsa_device_create_event,
    .create_executable_cache = iree_hal_hsa_device_create_executable_cache,
    .import_file = iree_hal_hsa_device_import_file,
    .create_semaphore = iree_hal_hsa_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hsa_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_hsa_device_queue_alloca,
    .queue_dealloca = iree_hal_hsa_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_hsa_device_queue_read,
    .queue_write = iree_hal_hsa_device_queue_write,
    .queue_host_call = iree_hal_device_queue_emulated_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_hsa_device_queue_execute,
    .queue_flush = iree_hal_hsa_device_queue_flush,
    .profiling_begin = iree_hal_hsa_device_profiling_begin,
    .profiling_flush = iree_hal_hsa_device_profiling_flush,
    .profiling_end = iree_hal_hsa_device_profiling_end,
    .transfer_h2d_raw = iree_hal_hsa_device_transfer_h2d_raw,
    .transfer_d2h_raw = iree_hal_hsa_device_transfer_d2h_raw,
};
