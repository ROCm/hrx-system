// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/amdgpu/api.h"

IREE_FLAG_LIST(string, amdgpu_libhsa_search_path,
               "Search path (directory or file) for the ROCR-Runtime library "
               "(`libhsa-runtime64.so`, etc).");

IREE_FLAG(int64_t, amdgpu_host_block_pool_small_size, 0,
          "Size in bytes of a small host block in the pool. Must be a power of "
          "two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_host_block_pool_large_size, 0,
          "Size in bytes of a large host block in the pool. Must be a power of "
          "two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_host_block_pool_command_buffer_size, 0,
          "Usable size in bytes of a command-buffer recording block in the "
          "host block pool. Must be a power of two or 0 for the default.");

IREE_FLAG(int64_t, amdgpu_device_block_pool_small_size, 0,
          "Size in bytes of a small device block in the pool. Must be a power "
          "of two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_device_block_pool_small_capacity, 0,
          "Initial small block pool block allocation count in blocks or 0 for "
          "the default.");
IREE_FLAG(int64_t, amdgpu_device_block_pool_large_size, 0,
          "Size in bytes of a large device block in the pool. Must be a power "
          "of two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_device_block_pool_large_capacity, 0,
          "Initial large block pool block allocation count in blocks or 0 for "
          "the default.");

IREE_FLAG(int64_t, amdgpu_default_pool_range_length, 0,
          "Logical byte length of the default TLSF queue-allocation pool per "
          "physical device or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_default_pool_alignment, 0,
          "Minimum byte alignment for default-pool reservations. Must be a "
          "power of two or 0 for the default.");
IREE_FLAG(int32_t, amdgpu_default_pool_frontier_capacity, 0,
          "Maximum death-frontier entry count stored per free default-pool "
          "block or 0 for the default.");

IREE_FLAG(int64_t, amdgpu_file_staging_slot_size, 0,
          "Byte length of each queue_read/queue_write file staging slot. Must "
          "be a power of two or 0 for the default.");
IREE_FLAG(int32_t, amdgpu_file_staging_slot_count, 0,
          "Number of queue_read/queue_write file staging slots per physical "
          "device. Must be a power of two or 0 for the default.");

IREE_FLAG(string, amdgpu_queue_placement, "any",
          "Device queue placement: 'any' (currently host), 'host', or "
          "'device' (reserved and currently unsupported).");

IREE_FLAG(string, amdgpu_command_buffer_mode, "aql",
          "Command-buffer execution mode: 'aql', 'pm4', or 'auto'.");
IREE_FLAG(
    string, amdgpu_pm4_command_buffer_publication_mode, "host-copy",
    "PM4 command-buffer resident publication mode: 'host-copy' writes host "
    "staging builders and publishes populated segments with hsa_memory_copy; "
    "'host-async-copy' publishes one contiguous staging image with "
    "hsa_amd_memory_async_copy and waits in end(); "
    "'host-async-copy-nonblocking' publishes the same staging image and makes "
    "queue execution wait on publication completion.");

IREE_FLAG(bool, amdgpu_preallocate_pools, true,
          "Preallocates a reasonable number of resources in pools to reduce "
          "initial execution latency.");

IREE_FLAG(bool, amdgpu_exclusive_execution, false,
          "Reserved for exclusive queue scheduling; currently unsupported.");

IREE_FLAG(
    bool, amdgpu_force_wait_barrier_defer, false,
    "Forces cross-queue wait barriers through the software deferral path "
    "instead of using the device-side strategy selected from the GPU ISA.");

IREE_FLAG(bool, amdgpu_asan, false,
          "Enables AMDGPU ASAN runtime state and config global publication.");
IREE_FLAG(string, amdgpu_asan_report_policy, "report-only",
          "AMDGPU ASAN report policy: 'report-only' emits device events and "
          "keeps the logical device usable; 'fail-device' emits device events "
          "and then fails the logical device.");
IREE_FLAG(string, amdgpu_asan_shadow_mode, "sparse",
          "AMDGPU ASAN shadow mapping mode: 'sparse' maps precise shadow slabs "
          "on demand; 'premapped' aliases a shared poisoned slab across the "
          "full shadow reservation so arbitrary covered shadow reads report "
          "instead of faulting.");
IREE_FLAG(
    string, amdgpu_asan_shadow_backing, "device-local",
    "AMDGPU ASAN physical shadow slab backing: 'device-local' backs shadow "
    "slabs with GPU VRAM; 'host-local' backs shadow slabs with nearest-CPU "
    "fine-grained host memory and relies on queue dependency edges for "
    "dispatch-boundary shadow visibility.");
IREE_FLAG(
    int64_t, amdgpu_asan_quarantine_size,
    IREE_HAL_AMDGPU_ASAN_DEFAULT_QUARANTINE_SIZE,
    "Freed ASAN allocation mapping budget in bytes kept resident and poisoned "
    "for stale-pointer checks. Set to 0 to release freed mappings "
    "immediately.");
IREE_FLAG(bool, amdgpu_tsan, false,
          "Enables AMDGPU TSAN runtime state and config global publication.");
IREE_FLAG(string, amdgpu_tsan_report_policy, "fail-device",
          "AMDGPU TSAN report policy: all policies emit device events and "
          "stop the offending kernel path; 'report-only' keeps the logical "
          "device usable, while 'fail-device' then fails the logical device.");
IREE_FLAG(
    int32_t, amdgpu_tsan_workgroup_local_memory_size,
    IREE_HAL_AMDGPU_TSAN_DEFAULT_WORKGROUP_LOCAL_MEMORY_SIZE,
    "Local-memory byte capacity represented by each AMDGPU TSAN workgroup "
    "shadow. Zero uses the backend-selected group segment limit.");
IREE_FLAG(int32_t, amdgpu_tsan_workgroup_capacity,
          IREE_HAL_AMDGPU_TSAN_DEFAULT_WORKGROUP_CAPACITY,
          "Maximum workgroup ordinals represented by one AMDGPU TSAN dispatch "
          "shadow.");

IREE_FLAG(bool, amdgpu_suppress_device_fine_memory, false,
          "Suppresses fine-grained GPU-local memory pools even when reported "
          "by HSA. This validates the coarse-grained device-local memory path "
          "used on GPUs that do not expose host-coherent VRAM.");

IREE_FLAG(int64_t, amdgpu_wait_active_for_ns, 0,
          "Reserved for future HSA active-wait tuning. Must be 0 today.");

static iree_status_t iree_hal_amdgpu_flag_int64_to_host_size(
    const char* flag_name, int64_t flag_value, iree_host_size_t* out_value) {
  if (flag_value < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s must be non-negative (got %" PRIi64 ")",
                            flag_name, flag_value);
  }
  *out_value = (iree_host_size_t)flag_value;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_flag_int64_to_device_size(
    const char* flag_name, int64_t flag_value, iree_device_size_t* out_value) {
  if (flag_value < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s must be non-negative (got %" PRIi64 ")",
                            flag_name, flag_value);
  }
  *out_value = (iree_device_size_t)flag_value;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("amdgpu"),
      .full_name = IREE_SVL("AMD GPU Driver (HSA/ROCR)"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("amdgpu"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  iree_hal_amdgpu_logical_device_options_t* device_options =
      &options.default_device_options;

  options.libhsa_search_paths = FLAG_amdgpu_libhsa_search_path_list();

  if (FLAG_amdgpu_host_block_pool_small_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_host_size(
        "--amdgpu_host_block_pool_small_size",
        FLAG_amdgpu_host_block_pool_small_size,
        &device_options->host_block_pools.small.block_size));
  }
  if (FLAG_amdgpu_host_block_pool_large_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_host_size(
        "--amdgpu_host_block_pool_large_size",
        FLAG_amdgpu_host_block_pool_large_size,
        &device_options->host_block_pools.large.block_size));
  }
  if (FLAG_amdgpu_host_block_pool_command_buffer_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_host_size(
        "--amdgpu_host_block_pool_command_buffer_size",
        FLAG_amdgpu_host_block_pool_command_buffer_size,
        &device_options->host_block_pools.command_buffer.usable_block_size));
  }

  if (FLAG_amdgpu_device_block_pool_small_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_device_size(
        "--amdgpu_device_block_pool_small_size",
        FLAG_amdgpu_device_block_pool_small_size,
        &device_options->device_block_pools.small.block_size));
  }
  if (FLAG_amdgpu_device_block_pool_small_capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_host_size(
        "--amdgpu_device_block_pool_small_capacity",
        FLAG_amdgpu_device_block_pool_small_capacity,
        &device_options->device_block_pools.small.initial_capacity));
  }
  if (FLAG_amdgpu_device_block_pool_large_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_device_size(
        "--amdgpu_device_block_pool_large_size",
        FLAG_amdgpu_device_block_pool_large_size,
        &device_options->device_block_pools.large.block_size));
  }
  if (FLAG_amdgpu_device_block_pool_large_capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_host_size(
        "--amdgpu_device_block_pool_large_capacity",
        FLAG_amdgpu_device_block_pool_large_capacity,
        &device_options->device_block_pools.large.initial_capacity));
  }

  if (FLAG_amdgpu_default_pool_range_length) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_device_size(
        "--amdgpu_default_pool_range_length",
        FLAG_amdgpu_default_pool_range_length,
        &device_options->default_pool.range_length));
  }
  if (FLAG_amdgpu_default_pool_alignment) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_device_size(
        "--amdgpu_default_pool_alignment", FLAG_amdgpu_default_pool_alignment,
        &device_options->default_pool.alignment));
  }
  if (FLAG_amdgpu_default_pool_frontier_capacity) {
    if (FLAG_amdgpu_default_pool_frontier_capacity < 0) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "default pool frontier capacity must be non-negative (got %d)",
          FLAG_amdgpu_default_pool_frontier_capacity);
    }
    if (FLAG_amdgpu_default_pool_frontier_capacity > UINT8_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "default pool frontier capacity %d exceeds maximum %u",
          FLAG_amdgpu_default_pool_frontier_capacity, UINT8_MAX);
    }
    device_options->default_pool.frontier_capacity =
        (uint8_t)FLAG_amdgpu_default_pool_frontier_capacity;
  }
  if (FLAG_amdgpu_file_staging_slot_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_host_size(
        "--amdgpu_file_staging_slot_size", FLAG_amdgpu_file_staging_slot_size,
        &device_options->file_staging.slot_size));
  }
  if (FLAG_amdgpu_file_staging_slot_count) {
    if (FLAG_amdgpu_file_staging_slot_count < 0) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "file staging slot count must be non-negative (got %d)",
          FLAG_amdgpu_file_staging_slot_count);
    }
    device_options->file_staging.slot_count =
        (uint32_t)FLAG_amdgpu_file_staging_slot_count;
  }

  if (strcmp(FLAG_amdgpu_queue_placement, "any") == 0) {
    device_options->queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY;
  } else if (strcmp(FLAG_amdgpu_queue_placement, "host") == 0) {
    device_options->queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST;
  } else if (strcmp(FLAG_amdgpu_queue_placement, "device") == 0) {
    device_options->queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized queue placement: '%s'",
                            FLAG_amdgpu_queue_placement);
  }

  if (strcmp(FLAG_amdgpu_command_buffer_mode, "aql") == 0) {
    device_options->command_buffer_mode =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  } else if (strcmp(FLAG_amdgpu_command_buffer_mode, "pm4") == 0) {
    device_options->command_buffer_mode =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;
  } else if (strcmp(FLAG_amdgpu_command_buffer_mode, "auto") == 0) {
    device_options->command_buffer_mode =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized command buffer mode: '%s'",
                            FLAG_amdgpu_command_buffer_mode);
  }

  if (strcmp(FLAG_amdgpu_pm4_command_buffer_publication_mode, "host-copy") ==
      0) {
    device_options->pm4_command_buffer_publication_mode =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_COPY;
  } else if (strcmp(FLAG_amdgpu_pm4_command_buffer_publication_mode,
                    "host-async-copy") == 0) {
    device_options->pm4_command_buffer_publication_mode =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_ASYNC_COPY;
  } else if (strcmp(FLAG_amdgpu_pm4_command_buffer_publication_mode,
                    "host-async-copy-nonblocking") == 0) {
    device_options->pm4_command_buffer_publication_mode =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_ASYNC_COPY_NONBLOCKING;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unrecognized PM4 command-buffer publication mode: '%s'",
        FLAG_amdgpu_pm4_command_buffer_publication_mode);
  }

  device_options->preallocate_pools = FLAG_amdgpu_preallocate_pools;

  device_options->exclusive_execution = FLAG_amdgpu_exclusive_execution;

  device_options->force_wait_barrier_defer =
      FLAG_amdgpu_force_wait_barrier_defer;

  device_options->asan.enabled =
      device_options->asan.enabled || FLAG_amdgpu_asan;
  if (strcmp(FLAG_amdgpu_asan_report_policy, "report-only") == 0) {
    device_options->asan.report_policy =
        IREE_HAL_AMDGPU_ASAN_REPORT_POLICY_REPORT_ONLY;
  } else if (strcmp(FLAG_amdgpu_asan_report_policy, "fail-device") == 0) {
    device_options->asan.report_policy =
        IREE_HAL_AMDGPU_ASAN_REPORT_POLICY_FAIL_DEVICE;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized ASAN report policy: '%s'",
                            FLAG_amdgpu_asan_report_policy);
  }
  if (strcmp(FLAG_amdgpu_asan_shadow_mode, "sparse") == 0) {
    device_options->asan.shadow_mode = IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_SPARSE;
  } else if (strcmp(FLAG_amdgpu_asan_shadow_mode, "premapped") == 0) {
    device_options->asan.shadow_mode =
        IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_PREMAPPED;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized ASAN shadow mode: '%s'",
                            FLAG_amdgpu_asan_shadow_mode);
  }
  if (strcmp(FLAG_amdgpu_asan_shadow_backing, "device-local") == 0) {
    device_options->asan.shadow_backing =
        IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL;
  } else if (strcmp(FLAG_amdgpu_asan_shadow_backing, "host-local") == 0) {
    device_options->asan.shadow_backing =
        IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_HOST_LOCAL;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized ASAN shadow backing: '%s'",
                            FLAG_amdgpu_asan_shadow_backing);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_flag_int64_to_device_size(
      "amdgpu_asan_quarantine_size", FLAG_amdgpu_asan_quarantine_size,
      &device_options->asan.quarantine_size));
  device_options->tsan.enabled =
      device_options->tsan.enabled || FLAG_amdgpu_tsan;
  if (strcmp(FLAG_amdgpu_tsan_report_policy, "report-only") == 0) {
    device_options->tsan.report_policy =
        IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_REPORT_ONLY;
  } else if (strcmp(FLAG_amdgpu_tsan_report_policy, "fail-device") == 0) {
    device_options->tsan.report_policy =
        IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_FAIL_DEVICE;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized TSAN report policy: '%s'",
                            FLAG_amdgpu_tsan_report_policy);
  }
  if (FLAG_amdgpu_tsan_workgroup_local_memory_size < 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdgpu_tsan_workgroup_local_memory_size must be non-negative "
        "(got %d)",
        FLAG_amdgpu_tsan_workgroup_local_memory_size);
  }
  device_options->tsan.workgroup_local_memory_size =
      (uint32_t)FLAG_amdgpu_tsan_workgroup_local_memory_size;
  if (FLAG_amdgpu_tsan_workgroup_capacity < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdgpu_tsan_workgroup_capacity must be "
                            "non-negative (got %d)",
                            FLAG_amdgpu_tsan_workgroup_capacity);
  }
  device_options->tsan.workgroup_capacity =
      (uint32_t)FLAG_amdgpu_tsan_workgroup_capacity;

  device_options->suppress_device_fine_memory =
      FLAG_amdgpu_suppress_device_fine_memory;

  if (FLAG_amdgpu_wait_active_for_ns < 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "--amdgpu_wait_active_for_ns must be non-negative (got %" PRIi64 ")",
        FLAG_amdgpu_wait_active_for_ns);
  }
  device_options->wait_active_for_ns = FLAG_amdgpu_wait_active_for_ns;

  iree_status_t status = iree_hal_amdgpu_driver_create(
      driver_name, &options, host_allocator, out_driver);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_amdgpu_driver_factory_enumerate,
      .try_create = iree_hal_amdgpu_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
