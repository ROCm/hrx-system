// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/device_spec_builder.h"

#include <string.h>

#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/utils/device_spec_builder.h"

static iree_status_t iree_hal_local_device_spec_verify_params(
    const iree_hal_local_device_spec_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(params->queue_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local device queue count %" PRIhsz
                            " exceeds uint32_t range",
                            params->queue_count);
  }
  if (IREE_UNLIKELY(params->default_queue_worker_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local device worker count %" PRIhsz
                            " exceeds uint32_t range",
                            params->default_queue_worker_count);
  }
  if (IREE_UNLIKELY(params->loader_count && !params->loaders)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local device executable loader list is NULL");
  }
  return iree_ok_status();
}

static iree_host_size_t iree_hal_local_device_spec_find_executable_format(
    iree_host_size_t format_count,
    const iree_hal_executable_format_spec_t* formats,
    iree_string_view_t format) {
  for (iree_host_size_t i = 0; i < format_count; ++i) {
    if (iree_string_view_equal(formats[i].format, format)) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_host_size_t iree_hal_local_device_spec_find_executable_target(
    iree_host_size_t target_count, const iree_hal_executable_target_t* targets,
    const iree_hal_executable_target_t* target) {
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    if (targets[i].kind == target->kind &&
        targets[i].priority == target->priority &&
        iree_string_view_equal(targets[i].family, target->family) &&
        iree_string_view_equal(targets[i].target_key, target->target_key)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t iree_hal_local_device_spec_collect_executables(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_allocator_t host_allocator,
    iree_hal_executable_format_spec_t** out_formats,
    iree_host_size_t* out_format_count,
    iree_hal_executable_target_t** out_targets,
    iree_host_size_t* out_target_count) {
  *out_formats = NULL;
  *out_format_count = 0;
  *out_targets = NULL;
  *out_target_count = 0;

  iree_host_size_t format_capacity = 0;
  iree_host_size_t target_capacity = 0;
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_device_executable_spec_t executable_spec;
    iree_hal_executable_loader_query_spec(loaders[i], &executable_spec);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            format_capacity, executable_spec.format_count, &format_capacity))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "local executable loader format count overflow");
    }
    if (IREE_UNLIKELY(executable_spec.format_count &&
                      !executable_spec.formats)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "local executable loader returned %" PRIhsz
                              " formats with NULL storage",
                              executable_spec.format_count);
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            target_capacity, executable_spec.target_count, &target_capacity))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "local executable loader target count overflow");
    }
    if (IREE_UNLIKELY(executable_spec.target_count &&
                      !executable_spec.targets)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "local executable loader returned %" PRIhsz
                              " targets with NULL storage",
                              executable_spec.target_count);
    }
  }

  iree_hal_executable_format_spec_t* formats = NULL;
  iree_status_t status = iree_ok_status();
  if (format_capacity != 0) {
    status = iree_allocator_malloc_array(host_allocator, format_capacity,
                                         sizeof(*formats), (void**)&formats);
  }
  if (iree_status_is_ok(status) && format_capacity != 0) {
    memset(formats, 0, format_capacity * sizeof(*formats));
  }

  iree_hal_executable_target_t* targets = NULL;
  if (iree_status_is_ok(status) && target_capacity != 0) {
    status = iree_allocator_malloc_array(host_allocator, target_capacity,
                                         sizeof(*targets), (void**)&targets);
  }
  if (iree_status_is_ok(status) && target_capacity != 0) {
    memset(targets, 0, target_capacity * sizeof(*targets));
  }

  iree_host_size_t format_count = 0;
  iree_host_size_t target_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < loader_count; ++i) {
      iree_hal_device_executable_spec_t executable_spec;
      iree_hal_executable_loader_query_spec(loaders[i], &executable_spec);
      for (iree_host_size_t j = 0; j < executable_spec.format_count; ++j) {
        const iree_hal_executable_format_spec_t* source_format =
            &executable_spec.formats[j];
        iree_host_size_t existing_ordinal =
            iree_hal_local_device_spec_find_executable_format(
                format_count, formats, source_format->format);
        if (existing_ordinal != IREE_HOST_SIZE_MAX) {
          formats[existing_ordinal].caching_modes |=
              source_format->caching_modes;
          formats[existing_ordinal].flags |= source_format->flags;
          continue;
        }
        formats[format_count++] = *source_format;
      }
      for (iree_host_size_t j = 0; j < executable_spec.target_count; ++j) {
        const iree_hal_executable_target_t* source_target =
            &executable_spec.targets[j];
        iree_host_size_t existing_ordinal =
            iree_hal_local_device_spec_find_executable_target(
                target_count, targets, source_target);
        if (existing_ordinal != IREE_HOST_SIZE_MAX) {
          targets[existing_ordinal].physical_device_affinity |=
              source_target->physical_device_affinity;
          targets[existing_ordinal].flags |= source_target->flags;
          continue;
        }
        targets[target_count++] = *source_target;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_formats = formats;
    *out_format_count = format_count;
    *out_targets = targets;
    *out_target_count = target_count;
  } else {
    iree_allocator_free(host_allocator, targets);
    iree_allocator_free(host_allocator, formats);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_local_device_spec_create(
    const iree_hal_local_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_local_device_spec_verify_params(params));

  iree_hal_executable_format_spec_t* executable_formats = NULL;
  iree_host_size_t executable_format_count = 0;
  iree_hal_executable_target_t* executable_targets = NULL;
  iree_host_size_t executable_target_count = 0;
  iree_status_t status = iree_hal_local_device_spec_collect_executables(
      params->loader_count, params->loaders, host_allocator,
      &executable_formats, &executable_format_count, &executable_targets,
      &executable_target_count);

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);

  iree_hal_physical_device_spec_t physical_device = {
      .identity =
          {
              .display_name = params->display_name,
              .backend_path = params->logical_device_id,
          },
      .physical_device_affinity = 1ull,
  };
  iree_hal_device_identity_spec_t identity = {
      .logical_device_id = params->logical_device_id,
      .display_name = params->display_name,
      .driver_id = params->driver_id,
      .backend_id = params->backend_id,
      .physical_device_count = 1,
      .physical_devices = &physical_device,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_identity(&builder, &identity);
  }

  iree_hal_memory_heap_spec_t memory_heap = {
      .name = IREE_SV("host"),
      .allocation_granularity = 1,
      .allocation_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      .physical_device_affinity = 1ull,
      .flags = IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN |
               IREE_HAL_MEMORY_HEAP_SPEC_FLAG_MAXIMUM_ALLOCATION_SIZE_UNKNOWN,
  };
  iree_hal_memory_type_spec_t memory_type = {
      .heap_index = 0,
      .memory_type = IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE,
      .allowed_buffer_usage = IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE,
      .allowed_memory_access = IREE_HAL_MEMORY_ACCESS_ALL,
      .minimum_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      .optimal_transfer_granularity = 1,
      .flags = IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
  };
  iree_hal_device_memory_spec_t memory = {
      .heap_count = 1,
      .heaps = &memory_heap,
      .memory_type_count = 1,
      .memory_types = &memory_type,
      .flags = IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_memory(&builder, &memory);
  }

  iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("default"),
      .queue_count = (uint32_t)params->queue_count,
      .priority_count = 1,
      .physical_device_affinity = 1ull,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING,
      .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
  };
  iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &queue_family,
      .flags = IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_queues(&builder, &queues);
  }

  iree_hal_device_dispatch_spec_t dispatch = {
      .launch.maximum_workgroup_invocations = UINT32_MAX,
      .launch.maximum_workgroup_size = {UINT32_MAX, UINT32_MAX, UINT32_MAX},
      .launch.maximum_workgroup_count = {UINT32_MAX, UINT32_MAX, UINT32_MAX},
      .subgroup.default_size = 1,
      .subgroup.minimum_size = 1,
      .subgroup.maximum_size = 1,
      .subgroup.supported_size_mask = 1ull << 1,
      .execution.unit_count = (uint32_t)params->default_queue_worker_count,
      .execution.group_count = 1,
      .addressing.pointer_size_bits = 8u * sizeof(void*),
      .addressing.address_space_bits = 8u * sizeof(void*),
      .flags = IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  }

  if (iree_status_is_ok(status) &&
      (executable_format_count != 0 || executable_target_count != 0)) {
    iree_hal_device_executable_spec_t executables = {
        .format_count = executable_format_count,
        .formats = executable_formats,
        .target_count = executable_target_count,
        .targets = executable_targets,
        .flags = IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
    };
    status =
        iree_hal_device_spec_builder_set_executables(&builder, &executables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }

  iree_hal_device_spec_builder_deinitialize(&builder);
  iree_allocator_free(host_allocator, executable_targets);
  iree_allocator_free(host_allocator, executable_formats);
  return status;
}
