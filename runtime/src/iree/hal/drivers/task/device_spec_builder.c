// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/device_spec_builder.h"

#include <string.h>

#include "iree/base/internal/cpu.h"
#include "iree/hal/drivers/task/atomic.h"
#include "iree/hal/drivers/task/device_spec.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/utils/device_spec_builder.h"

#define IREE_HAL_TASK_QUEUE_AXIS_CAPACITY ((iree_host_size_t)UINT8_MAX + 1)

static iree_status_t iree_hal_task_device_spec_verify_params(
    const iree_hal_task_device_spec_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(params->queue_count == 0 ||
                    params->queue_count > IREE_HAL_TASK_QUEUE_AXIS_CAPACITY)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "task device queue count must be in [1, %" PRIhsz "] (got %" PRIhsz ")",
        IREE_HAL_TASK_QUEUE_AXIS_CAPACITY, params->queue_count);
  }
  if (IREE_UNLIKELY(params->default_queue_worker_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "task device worker count %" PRIhsz
                            " exceeds uint32_t range",
                            params->default_queue_worker_count);
  }
  if (IREE_UNLIKELY(params->loader_count && !params->loaders)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "task device executable loader list is NULL");
  }
  return iree_ok_status();
}

static iree_host_size_t iree_hal_task_device_spec_find_executable_target(
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

static iree_status_t iree_hal_task_device_spec_collect_executables(
    iree_host_size_t seed_target_count,
    const iree_hal_executable_target_t* seed_targets,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_allocator_t host_allocator, iree_hal_executable_target_t** out_targets,
    iree_host_size_t* out_target_count) {
  *out_targets = NULL;
  *out_target_count = 0;

  iree_host_size_t target_capacity = seed_target_count;
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_device_executable_spec_t executable_spec;
    iree_hal_executable_loader_query_spec(loaders[i], &executable_spec);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            target_capacity, executable_spec.target_count, &target_capacity))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable loader target count overflow");
    }
    if (IREE_UNLIKELY(executable_spec.target_count &&
                      !executable_spec.targets)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "executable loader returned %" PRIhsz
                              " targets with NULL storage",
                              executable_spec.target_count);
    }
  }

  iree_status_t status = iree_ok_status();
  iree_hal_executable_target_t* targets = NULL;
  if (target_capacity != 0) {
    status = iree_allocator_malloc_array(host_allocator, target_capacity,
                                         sizeof(*targets), (void**)&targets);
  }
  if (iree_status_is_ok(status) && target_capacity != 0) {
    memset(targets, 0, target_capacity * sizeof(*targets));
  }

  iree_host_size_t target_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < seed_target_count; ++i) {
      targets[target_count++] = seed_targets[i];
    }
    for (iree_host_size_t i = 0; i < loader_count; ++i) {
      iree_hal_device_executable_spec_t executable_spec;
      iree_hal_executable_loader_query_spec(loaders[i], &executable_spec);
      for (iree_host_size_t j = 0; j < executable_spec.target_count; ++j) {
        const iree_hal_executable_target_t* source_target =
            &executable_spec.targets[j];
        iree_host_size_t existing_ordinal =
            iree_hal_task_device_spec_find_executable_target(
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
    *out_targets = targets;
    *out_target_count = target_count;
  } else {
    iree_allocator_free(host_allocator, targets);
  }
  return status;
}

static iree_status_t iree_hal_task_device_spec_add_cpu_facet(
    const iree_cpu_data_t* cpu_data, iree_hal_device_spec_builder_t* builder) {
  iree_hal_cpu_device_spec_t cpu_spec = {
      .cpu_data = *cpu_data,
      .flags = IREE_HAL_CPU_DEVICE_SPEC_FLAG_NONE,
  };
  const iree_host_size_t payload_size = iree_hal_cpu_device_spec_payload_size();
  uint8_t* payload_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      builder->host_allocator, payload_size, (void**)&payload_storage));

  iree_status_t status = iree_hal_cpu_device_spec_encode(
      &cpu_spec, iree_make_byte_span(payload_storage, payload_size));
  if (iree_status_is_ok(status)) {
    const iree_hal_device_spec_facet_t facet = {
        .schema_id = iree_make_cstring_view(IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_ID),
        .schema_version = IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION,
        .payload = iree_make_const_byte_span(payload_storage, payload_size),
    };
    status = iree_hal_device_spec_builder_add_facet(builder, &facet);
  }

  iree_allocator_free(builder->host_allocator, payload_storage);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_task_device_spec_create(
    const iree_hal_task_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_device_spec_verify_params(params));

  iree_cpu_data_t cpu_data;
  iree_cpu_query_data(host_allocator, &cpu_data);

  iree_string_builder_t exact_target_key_builder;
  iree_string_builder_initialize(host_allocator, &exact_target_key_builder);
  iree_status_t status =
      iree_cpu_data_append_target_key(&cpu_data, &exact_target_key_builder);
  const iree_hal_executable_target_t available_cpu_targets[] = {
      {
          .family = IREE_SV("cpu"),
          .target_key = iree_string_builder_view(&exact_target_key_builder),
          .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          .priority = 100,
          .physical_device_affinity = 1ull,
          .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
      {
          .family = IREE_SV("cpu"),
          .target_key = iree_cpu_architecture_name(cpu_data.architecture),
          .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
          .priority = 50,
          .physical_device_affinity = 1ull,
          .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };

  iree_hal_executable_target_t
      cpu_targets[IREE_ARRAYSIZE(available_cpu_targets)];
  iree_host_size_t cpu_target_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(available_cpu_targets); ++i) {
    if (iree_hal_query_any_executable_loader_target_support(
            params->loader_count, params->loaders, &available_cpu_targets[i])) {
      cpu_targets[cpu_target_count++] = available_cpu_targets[i];
    }
  }

  iree_hal_executable_target_t* executable_targets = NULL;
  iree_host_size_t executable_target_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_device_spec_collect_executables(
        cpu_target_count, cpu_targets, params->loader_count, params->loaders,
        host_allocator, &executable_targets, &executable_target_count);
  }

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
      .atomic_operations = iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL),
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

  iree_hal_queue_family_role_flags_t queue_role_flags =
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING;
  const iree_hal_atomic_operation_capabilities_t* atomic_operations =
      &params->atomic_capabilities.operations;
  if (atomic_operations->device_scope_32 ||
      atomic_operations->device_scope_64 ||
      atomic_operations->system_scope_32 ||
      atomic_operations->system_scope_64) {
    queue_role_flags |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  }
  iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("default"),
      .provisioned_queue_count = (uint32_t)params->queue_count,
      .priority_count = 1,
      .physical_device_affinity = 1ull,
      .role_flags = queue_role_flags,
      .atomic_capabilities = params->atomic_capabilities,
      .zero_compute_atomic_capabilities =
          params->zero_compute_atomic_capabilities,
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

  if (iree_status_is_ok(status) && executable_target_count != 0) {
    iree_hal_device_executable_spec_t executables = {
        .target_count = executable_target_count,
        .targets = executable_targets,
        .flags = IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
    };
    status =
        iree_hal_device_spec_builder_set_executables(&builder, &executables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_device_spec_add_cpu_facet(&cpu_data, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }

  iree_hal_device_spec_builder_deinitialize(&builder);
  iree_allocator_free(host_allocator, executable_targets);
  iree_string_builder_deinitialize(&exact_target_key_builder);
  return status;
}
