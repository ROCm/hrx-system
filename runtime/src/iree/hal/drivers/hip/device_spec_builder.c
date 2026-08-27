// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/device_spec_builder.h"

#include <string.h>

#include "iree/hal/executable/amdgpu/executable_target.h"

static iree_status_t iree_hal_hip_device_spec_verify_params(
    const iree_hal_hip_device_spec_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(params->physical_device_count == 0 ||
                    !params->physical_devices)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP device spec requires physical devices");
  }
  if (IREE_UNLIKELY(params->physical_device_count >
                    IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HIP device spec physical device count %" PRIhsz
                            " exceeds physical-device affinity capacity %d",
                            params->physical_device_count,
                            IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT);
  }
  if (IREE_UNLIKELY(params->queue_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP device spec requires at least one queue");
  }
  if (IREE_UNLIKELY(params->queue_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HIP queue count %" PRIhsz
                            " exceeds uint32_t range",
                            params->queue_count);
  }
  if (IREE_UNLIKELY(!params->device_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP device spec allocator is NULL");
  }
  return iree_ok_status();
}

static iree_hal_physical_device_affinity_t
iree_hal_hip_device_spec_all_physical_device_affinity(
    iree_host_size_t physical_device_count) {
  return physical_device_count == IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT
             ? UINT64_MAX
             : ((1ull << physical_device_count) - 1ull);
}

static iree_status_t iree_hal_hip_device_spec_populate_identity(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_physical_device_spec_t* physical_devices = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      builder->host_allocator, params->physical_device_count,
      sizeof(*physical_devices), (void**)&physical_devices));
  memset(physical_devices, 0,
         params->physical_device_count * sizeof(*physical_devices));

  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_hal_hip_device_spec_physical_device_params_t* physical_device =
        &params->physical_devices[i];
    iree_hal_physical_device_spec_t* physical_spec = &physical_devices[i];
    physical_spec->identity.display_name = physical_device->display_name;
    physical_spec->identity.backend_path = physical_device->backend_path;
    if (iree_all_bits_set(physical_device->flags,
                          IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID)) {
      physical_spec->identity.flags |=
          IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID;
      memcpy(physical_spec->identity.uuid.bytes, physical_device->uuid.bytes,
             sizeof(physical_spec->identity.uuid.bytes));
    }
    if (iree_all_bits_set(
            physical_device->flags,
            IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS)) {
      physical_spec->identity.flags |=
          IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS;
      physical_spec->identity.pci = physical_device->pci;
    }
    physical_spec->physical_ordinal = physical_device->physical_ordinal;
    physical_spec->partition_count = 1;
    physical_spec->physical_device_affinity = 1ull << i;
  }

  iree_hal_device_identity_spec_t identity = {
      .logical_device_id = params->logical_device_id,
      .display_name = params->display_name,
      .driver_id = IREE_SV("hip"),
      .backend_id = IREE_SV("hip"),
      .device_path = params->physical_devices[0].backend_path,
      .logical_ordinal = params->physical_devices[0].physical_ordinal,
      .physical_device_count = params->physical_device_count,
      .physical_devices = physical_devices,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(builder, &identity);
  iree_allocator_free(builder->host_allocator, physical_devices);
  return status;
}

static iree_string_view_t iree_hal_hip_device_spec_memory_heap_name(
    const iree_hal_allocator_memory_heap_t* heap) {
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    return IREE_SV("host");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_SV("device-managed");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    return IREE_SV("device");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_SV("host-visible");
  }
  return IREE_SV("memory");
}

static iree_hal_memory_access_t iree_hal_hip_device_spec_memory_access(
    const iree_hal_allocator_memory_heap_t* heap) {
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL) ||
      iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_HAL_MEMORY_ACCESS_ALL;
  }
  return IREE_HAL_MEMORY_ACCESS_NONE;
}

static uint32_t iree_hal_hip_device_spec_memory_type_mask(
    iree_host_size_t memory_type_count,
    const iree_hal_memory_type_spec_t* memory_types,
    iree_hal_memory_type_t required_type) {
  uint32_t memory_type_mask = 0;
  for (iree_host_size_t i = 0; i < memory_type_count; ++i) {
    if (!iree_all_bits_set(memory_types[i].memory_type, required_type)) {
      continue;
    }
    if (i >= 32) return UINT32_MAX;
    memory_type_mask |= 1u << i;
  }
  return memory_type_mask;
}

static iree_status_t iree_hal_hip_device_spec_query_memory_heaps(
    const iree_hal_hip_device_spec_params_t* params,
    iree_allocator_t host_allocator,
    iree_hal_allocator_memory_heap_t** out_allocator_heaps,
    iree_host_size_t* out_heap_count) {
  *out_allocator_heaps = NULL;
  *out_heap_count = 0;

  iree_host_size_t heap_count = 0;
  iree_status_t status = iree_hal_allocator_query_memory_heaps(
      params->device_allocator, 0, NULL, &heap_count);
  if (iree_status_is_out_of_range(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }
  if (!iree_status_is_ok(status) || heap_count == 0) return status;

  iree_hal_allocator_memory_heap_t* allocator_heaps = NULL;
  status = iree_allocator_malloc_array(host_allocator, heap_count,
                                       sizeof(*allocator_heaps),
                                       (void**)&allocator_heaps);
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_query_memory_heaps(
        params->device_allocator, heap_count, allocator_heaps, &heap_count);
  }
  if (iree_status_is_ok(status)) {
    *out_allocator_heaps = allocator_heaps;
    *out_heap_count = heap_count;
  } else {
    iree_allocator_free(host_allocator, allocator_heaps);
  }
  return status;
}

static iree_status_t iree_hal_hip_device_spec_populate_memory(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_allocator_memory_heap_t* allocator_heaps = NULL;
  iree_host_size_t heap_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hip_device_spec_query_memory_heaps(
      params, builder->host_allocator, &allocator_heaps, &heap_count));
  if (heap_count == 0) return iree_ok_status();

  iree_hal_memory_heap_spec_t* heaps = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      builder->host_allocator, heap_count, sizeof(*heaps), (void**)&heaps);
  iree_hal_memory_type_spec_t* memory_types = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(builder->host_allocator, heap_count,
                                         sizeof(*memory_types),
                                         (void**)&memory_types);
  }

  iree_hal_external_buffer_handle_spec_t external_buffer_handles[2] = {{0}};
  iree_host_size_t external_buffer_handle_count = 0;
  if (iree_status_is_ok(status)) {
    memset(heaps, 0, heap_count * sizeof(*heaps));
    memset(memory_types, 0, heap_count * sizeof(*memory_types));
    for (iree_host_size_t i = 0; i < heap_count; ++i) {
      const iree_hal_allocator_memory_heap_t* allocator_heap =
          &allocator_heaps[i];
      heaps[i] = (iree_hal_memory_heap_spec_t){
          .name = iree_hal_hip_device_spec_memory_heap_name(allocator_heap),
          .allocation_granularity = 1,
          .allocation_alignment = allocator_heap->min_alignment,
          .maximum_allocation_size = allocator_heap->max_allocation_size,
          .physical_device_affinity =
              iree_hal_hip_device_spec_all_physical_device_affinity(
                  params->physical_device_count),
          .flags = IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN,
      };
      memory_types[i] = (iree_hal_memory_type_spec_t){
          .heap_index = (uint32_t)i,
          .memory_type = allocator_heap->type,
          .allowed_buffer_usage = allocator_heap->allowed_usage,
          .allowed_memory_access =
              iree_hal_hip_device_spec_memory_access(allocator_heap),
          .minimum_alignment = allocator_heap->min_alignment,
          .optimal_transfer_granularity = 1,
          .flags = IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
      };
    }

    const uint32_t host_memory_type_mask =
        iree_hal_hip_device_spec_memory_type_mask(
            heap_count, memory_types, IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
    if (host_memory_type_mask != 0) {
      external_buffer_handles[external_buffer_handle_count++] =
          (iree_hal_external_buffer_handle_spec_t){
              .handle_type_mask = IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE,
              .direction_flags = IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
              .allowed_buffer_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                      IREE_HAL_BUFFER_USAGE_DISPATCH |
                                      IREE_HAL_BUFFER_USAGE_MAPPING,
              .allowed_memory_access = IREE_HAL_MEMORY_ACCESS_ALL,
              .compatible_memory_type_mask = host_memory_type_mask,
              .flags = IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_BORROWED,
          };
    }

    const uint32_t device_memory_type_mask =
        iree_hal_hip_device_spec_memory_type_mask(
            heap_count, memory_types, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
    if (device_memory_type_mask != 0) {
      external_buffer_handles[external_buffer_handle_count++] =
          (iree_hal_external_buffer_handle_spec_t){
              .handle_type_mask = IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE,
              .direction_flags =
                  IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
                  IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
              .allowed_buffer_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                      IREE_HAL_BUFFER_USAGE_DISPATCH,
              .allowed_memory_access = IREE_HAL_MEMORY_ACCESS_NONE,
              .compatible_memory_type_mask = device_memory_type_mask,
              .flags = IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_BORROWED,
          };
    }

    iree_hal_device_memory_spec_t memory = {
        .heap_count = heap_count,
        .heaps = heaps,
        .memory_type_count = heap_count,
        .memory_types = memory_types,
        .external_buffer_handle_count = external_buffer_handle_count,
        .external_buffer_handles =
            external_buffer_handle_count ? external_buffer_handles : NULL,
        .flags = IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_memory(builder, &memory);
  }

  iree_allocator_free(builder->host_allocator, memory_types);
  iree_allocator_free(builder->host_allocator, heaps);
  iree_allocator_free(builder->host_allocator, allocator_heaps);
  return status;
}

static iree_status_t iree_hal_hip_device_spec_populate_queues(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_queue_family_spec_t* families = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      builder->host_allocator, params->physical_device_count, sizeof(*families),
      (void**)&families));
  memset(families, 0, params->physical_device_count * sizeof(*families));

  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_hal_hip_device_spec_physical_device_params_t* physical_device =
        &params->physical_devices[i];
    const uint64_t timestamp_frequency_hz =
        physical_device->facts.clocks.clock_instruction_frequency_hz;
    iree_hal_queue_family_role_flags_t role_flags =
        IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
        IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER;
    if (timestamp_frequency_hz != 0) {
      role_flags |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING;
    }
    families[i] = (iree_hal_queue_family_spec_t){
        .name = physical_device->display_name,
        .queue_count = (uint32_t)params->queue_count,
        .priority_count = 1,
        .timestamp_valid_bits = timestamp_frequency_hz ? 32 : 0,
        .timestamp_frequency_hz = timestamp_frequency_hz,
        .physical_device_affinity = 1ull << i,
        .queue_affinity = (iree_hal_queue_affinity_t)1 << i,
        .role_flags = role_flags,
        .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
    };
  }

  iree_hal_external_timepoint_handle_spec_t external_timepoint_handles[1] = {
      {
          .handle_type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT,
          .direction_flags = IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
                             IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          .compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
                           IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
          .flags = IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE,
      },
  };
  iree_hal_device_queue_spec_t queues = {
      .family_count = params->physical_device_count,
      .families = families,
      .external_timepoint_handle_count =
          IREE_ARRAYSIZE(external_timepoint_handles),
      .external_timepoint_handles = external_timepoint_handles,
      .flags = IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };
  iree_status_t status =
      iree_hal_device_spec_builder_set_queues(builder, &queues);

  iree_allocator_free(builder->host_allocator, families);
  return status;
}

static uint32_t iree_hal_hip_device_spec_minimum_u32(uint32_t lhs,
                                                     uint32_t rhs) {
  return lhs == 0 ? rhs : iree_min(lhs, rhs);
}

static uint64_t iree_hal_hip_device_spec_minimum_u64(uint64_t lhs,
                                                     uint64_t rhs) {
  return lhs == 0 ? rhs : iree_min(lhs, rhs);
}

static uint64_t iree_hal_hip_device_spec_subgroup_size_mask(
    uint32_t subgroup_size) {
  return subgroup_size < 64 ? (1ull << subgroup_size) : 0;
}

static iree_status_t iree_hal_hip_device_spec_aggregate(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_hip_device_facts_t* out_aggregate) {
  iree_hal_hip_device_facts_t aggregate = params->physical_devices[0].facts;
  aggregate.execution_unit_count = 0;

  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_hal_hip_device_facts_t* source =
        &params->physical_devices[i].facts;
    if (IREE_UNLIKELY(UINT32_MAX - aggregate.execution_unit_count <
                      source->execution_unit_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HIP compute unit count overflow");
    }
    aggregate.execution_unit_count += source->execution_unit_count;
    aggregate.subgroup_size = iree_hal_hip_device_spec_minimum_u32(
        aggregate.subgroup_size, source->subgroup_size);
    aggregate.launch.maximum_workgroup_invocations =
        iree_hal_hip_device_spec_minimum_u32(
            aggregate.launch.maximum_workgroup_invocations,
            source->launch.maximum_workgroup_invocations);
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(source->launch.maximum_workgroup_size); ++j) {
      aggregate.launch.maximum_workgroup_size[j] =
          iree_hal_hip_device_spec_minimum_u32(
              aggregate.launch.maximum_workgroup_size[j],
              source->launch.maximum_workgroup_size[j]);
      aggregate.launch.maximum_workgroup_count[j] =
          iree_hal_hip_device_spec_minimum_u32(
              aggregate.launch.maximum_workgroup_count[j],
              source->launch.maximum_workgroup_count[j]);
    }
    aggregate.launch.maximum_workgroups_per_execution_unit =
        iree_hal_hip_device_spec_minimum_u32(
            aggregate.launch.maximum_workgroups_per_execution_unit,
            source->launch.maximum_workgroups_per_execution_unit);
    aggregate.launch.maximum_invocations_per_execution_unit =
        iree_hal_hip_device_spec_minimum_u32(
            aggregate.launch.maximum_invocations_per_execution_unit,
            source->launch.maximum_invocations_per_execution_unit);
    aggregate.launch.maximum_workgroup_register_count =
        iree_hal_hip_device_spec_minimum_u32(
            aggregate.launch.maximum_workgroup_register_count,
            source->launch.maximum_workgroup_register_count);
    aggregate.launch.maximum_local_memory_size =
        iree_hal_hip_device_spec_minimum_u64(
            aggregate.launch.maximum_local_memory_size,
            source->launch.maximum_local_memory_size);
    aggregate.launch.maximum_workgroup_local_memory_size =
        iree_hal_hip_device_spec_minimum_u64(
            aggregate.launch.maximum_workgroup_local_memory_size,
            source->launch.maximum_workgroup_local_memory_size);
    aggregate.clocks.clock_instruction_frequency_hz =
        iree_hal_hip_device_spec_minimum_u64(
            aggregate.clocks.clock_instruction_frequency_hz,
            source->clocks.clock_instruction_frequency_hz);
  }
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_spec_populate_dispatch(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_hip_device_facts_t aggregate;
  IREE_RETURN_IF_ERROR(iree_hal_hip_device_spec_aggregate(params, &aggregate));
  iree_hal_device_dispatch_spec_t dispatch = {
      .launch.maximum_workgroup_invocations =
          aggregate.launch.maximum_workgroup_invocations,
      .launch.maximum_workgroup_size =
          {
              aggregate.launch.maximum_workgroup_size[0],
              aggregate.launch.maximum_workgroup_size[1],
              aggregate.launch.maximum_workgroup_size[2],
          },
      .launch.maximum_workgroup_count =
          {
              aggregate.launch.maximum_workgroup_count[0],
              aggregate.launch.maximum_workgroup_count[1],
              aggregate.launch.maximum_workgroup_count[2],
          },
      .subgroup.default_size = aggregate.subgroup_size,
      .subgroup.minimum_size = aggregate.subgroup_size,
      .subgroup.maximum_size = aggregate.subgroup_size,
      .subgroup.supported_size_mask =
          iree_hal_hip_device_spec_subgroup_size_mask(aggregate.subgroup_size),
      .execution.unit_count = aggregate.execution_unit_count,
      .execution.group_count = (uint32_t)params->physical_device_count,
      .execution.maximum_resident_workgroup_count =
          aggregate.launch.maximum_workgroups_per_execution_unit,
      .execution.maximum_resident_invocation_count =
          aggregate.launch.maximum_invocations_per_execution_unit,
      .execution.maximum_workgroup_register_count =
          aggregate.launch.maximum_workgroup_register_count,
      .execution.maximum_local_memory_size =
          aggregate.launch.maximum_local_memory_size,
      .execution.maximum_workgroup_local_memory_size =
          aggregate.launch.maximum_workgroup_local_memory_size,
      .addressing.pointer_size_bits = 64,
      .addressing.address_space_bits = 64,
      .flags = IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_set_dispatch(builder, &dispatch);
}

static iree_status_t iree_hal_hip_device_spec_populate_timing(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_hip_device_facts_t aggregate;
  IREE_RETURN_IF_ERROR(iree_hal_hip_device_spec_aggregate(params, &aggregate));
  iree_hal_device_timing_spec_flags_t flags =
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_NONE;
  if (aggregate.clocks.clock_instruction_frequency_hz != 0) {
    flags |= IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS |
             IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DISPATCH_EVENTS |
             IREE_HAL_DEVICE_TIMING_SPEC_FLAG_PROFILING_PERTURBS_EXECUTION;
  }
  iree_hal_device_timing_spec_t timing = {
      .timestamp_valid_bits =
          aggregate.clocks.clock_instruction_frequency_hz ? 32 : 0,
      .timestamp_frequency_hz = aggregate.clocks.clock_instruction_frequency_hz,
      .flags = flags,
  };
  return iree_hal_device_spec_builder_set_timing(builder, &timing);
}

static iree_status_t iree_hal_hip_device_spec_populate_executables(
    const iree_hal_hip_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_string_view_t gcn_arch_name = iree_make_cstring_view(
        params->physical_devices[i].facts.architecture.gcn_arch_name);
    if (iree_string_view_is_empty(gcn_arch_name)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "HIP physical device %" PRIhsz
                              " did not report a GCN architecture",
                              i);
    }
    iree_hal_amdgpu_target_identity_t exact_identity;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_artifact_key(
                             gcn_arch_name, &exact_identity),
                         "parsing HIP physical device %" PRIhsz
                         " target ID '%.*s'",
                         i, (int)gcn_arch_name.size, gcn_arch_name.data);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_target_identity_resolve_physical_target(
            params->physical_devices[i].facts.architecture.asic_revision,
            &exact_identity),
        "qualifying HIP physical device %" PRIhsz " ASIC revision", i);
    const iree_hal_physical_device_affinity_t target_affinity = 1ull << i;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_spec_builder_add_executable_targets(
            builder, &exact_identity, target_affinity));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_hip_device_spec_create(
    const iree_hal_hip_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_hip_device_spec_verify_params(params));

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_hip_device_spec_populate_identity(params, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_spec_populate_memory(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_spec_populate_queues(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_spec_populate_dispatch(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_spec_populate_timing(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_spec_populate_executables(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}
