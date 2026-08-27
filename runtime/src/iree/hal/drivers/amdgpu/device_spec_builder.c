// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec_builder.h"

#include <string.h>

#include "iree/hal/executable/amdgpu/executable_target.h"

static iree_status_t iree_hal_amdgpu_device_spec_verify_params(
    const iree_hal_amdgpu_device_spec_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(!params->physical_device_count ||
                    !params->physical_devices)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU device spec requires initialized physical devices");
  }
  if (IREE_UNLIKELY(params->physical_device_count >
                    IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU device spec physical device count %" PRIhsz
                            " exceeds physical-device affinity capacity %d",
                            params->physical_device_count,
                            IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT);
  }
  if (IREE_UNLIKELY(!params->device_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU device spec allocator is NULL");
  }
  const iree_host_size_t queue_count = params->physical_devices[0].queue_count;
  iree_host_size_t total_queue_count = 0;
  if (IREE_UNLIKELY(queue_count == 0 ||
                    !iree_host_size_checked_mul(params->physical_device_count,
                                                queue_count,
                                                &total_queue_count) ||
                    total_queue_count > IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU logical queue count must be in [1, %" PRIhsz
                            "] (physical_devices=%" PRIhsz
                            ", queues_per_device=%" PRIhsz ")",
                            (iree_host_size_t)IREE_HAL_MAX_QUEUES,
                            params->physical_device_count, queue_count);
  }
  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    if (IREE_UNLIKELY(params->physical_devices[i].queue_count != queue_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU physical devices must expose a uniform queue count; device "
          "0 reports %" PRIhsz " but device %" PRIhsz " reports %u",
          queue_count, i, params->physical_devices[i].queue_count);
    }
    if (IREE_UNLIKELY(params->physical_devices[i].timestamp_frequency_hz ==
                      0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU physical device %" PRIhsz
                              " has no timestamp frequency; timestamp deltas "
                              "cannot be converted to a duration",
                              i);
    }
  }

  // The device-scope timing spec carries one tick rate for the whole logical
  // device, so physical devices ticking at different rates have no correct
  // value to publish there. The params come from an exported entry point, so
  // this check is input validation rather than an assertable invariant.
  for (iree_host_size_t i = 1; i < params->physical_device_count; ++i) {
    if (IREE_UNLIKELY(params->physical_devices[i].timestamp_frequency_hz !=
                      params->physical_devices[0].timestamp_frequency_hz)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU physical devices have no common timestamp frequency: "
          "device 0 reports %" PRIu64 " hz but device %" PRIhsz
          " reports %" PRIu64 " hz",
          params->physical_devices[0].timestamp_frequency_hz, i,
          params->physical_devices[i].timestamp_frequency_hz);
    }
  }
  return iree_ok_status();
}

static iree_hal_physical_device_affinity_t
iree_hal_amdgpu_device_spec_all_physical_device_affinity(
    iree_host_size_t physical_device_count) {
  return physical_device_count == IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT
             ? UINT64_MAX
             : ((1ull << physical_device_count) - 1ull);
}

static iree_hal_queue_affinity_t
iree_hal_amdgpu_device_spec_queue_affinity_for_physical_device(
    iree_host_size_t physical_device_ordinal,
    iree_host_size_t queue_count_per_physical_device) {
  const iree_host_size_t first_queue_ordinal =
      physical_device_ordinal * queue_count_per_physical_device;
  if (queue_count_per_physical_device == IREE_HAL_MAX_QUEUES) {
    return IREE_HAL_QUEUE_AFFINITY_ANY;
  }
  return (
      (((iree_hal_queue_affinity_t)1 << queue_count_per_physical_device) - 1)
      << first_queue_ordinal);
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_identity(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_physical_device_spec_t* physical_devices = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      builder->host_allocator, params->physical_device_count,
      sizeof(*physical_devices), (void**)&physical_devices));
  memset(physical_devices, 0,
         params->physical_device_count * sizeof(*physical_devices));

  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_hal_amdgpu_device_spec_physical_device_params_t*
        physical_device = &params->physical_devices[i];
    iree_hal_physical_device_spec_t* physical_spec = &physical_devices[i];
    physical_spec->identity.display_name = physical_device->identity.processor;
    physical_spec->identity.backend_path = physical_device->identity.processor;
    if (iree_all_bits_set(
            physical_device->flags,
            IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID)) {
      physical_spec->identity.flags |=
          IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID;
      memcpy(physical_spec->identity.uuid.bytes, physical_device->uuid.bytes,
             sizeof(physical_spec->identity.uuid.bytes));
    }
    if (iree_all_bits_set(
            physical_device->flags,
            IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS)) {
      physical_spec->identity.flags |=
          IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS;
      physical_spec->identity.pci = physical_device->pci;
    }
    physical_spec->identity.flags |=
        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE;
    physical_spec->identity.numa = physical_device->numa;
    physical_spec->physical_ordinal = physical_device->physical_ordinal;
    physical_spec->partition_count = 1;
    physical_spec->physical_device_affinity = 1ull << i;
  }

  iree_hal_device_identity_spec_t identity = {
      .logical_device_id = params->logical_device_id,
      .display_name = params->display_name,
      .driver_id = IREE_SV("amdgpu"),
      .backend_id = IREE_SV("hsa"),
      .physical_device_count = params->physical_device_count,
      .physical_devices = physical_devices,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(builder, &identity);
  iree_allocator_free(builder->host_allocator, physical_devices);
  return status;
}

static iree_string_view_t iree_hal_amdgpu_device_spec_memory_heap_name(
    const iree_hal_allocator_memory_heap_t* heap) {
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    return IREE_SV("host");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_SV("device-fine");
  }
  return IREE_SV("device");
}

static iree_hal_memory_access_t iree_hal_amdgpu_device_spec_memory_access(
    const iree_hal_allocator_memory_heap_t* heap) {
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL) ||
      iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_HAL_MEMORY_ACCESS_ALL;
  }
  return IREE_HAL_MEMORY_ACCESS_NONE;
}

static bool iree_hal_amdgpu_device_spec_memory_heap_is_device_local(
    const iree_hal_allocator_memory_heap_t* heap) {
  return iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_memory(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_host_size_t heap_count = 0;
  iree_status_t status = iree_hal_allocator_query_memory_heaps(
      params->device_allocator, 0, NULL, &heap_count);
  if (!iree_status_is_out_of_range(status)) return status;
  iree_status_free(status);
  status = iree_ok_status();

  iree_hal_allocator_memory_heap_t* allocator_heaps = NULL;
  iree_hal_memory_heap_spec_t* heaps = NULL;
  iree_hal_memory_type_spec_t* memory_types = NULL;
  if (heap_count != 0) {
    status = iree_allocator_malloc_array(builder->host_allocator, heap_count,
                                         sizeof(*allocator_heaps),
                                         (void**)&allocator_heaps);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(builder->host_allocator, heap_count,
                                           sizeof(*heaps), (void**)&heaps);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(builder->host_allocator, heap_count,
                                           sizeof(*memory_types),
                                           (void**)&memory_types);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_query_memory_heaps(
        params->device_allocator, heap_count, allocator_heaps, &heap_count);
  }
  if (iree_status_is_ok(status)) {
    memset(heaps, 0, heap_count * sizeof(*heaps));
    memset(memory_types, 0, heap_count * sizeof(*memory_types));
    bool device_memory_capacity_attached = false;
    for (iree_host_size_t i = 0; i < heap_count; ++i) {
      const iree_hal_allocator_memory_heap_t* allocator_heap =
          &allocator_heaps[i];
      iree_hal_memory_heap_spec_flags_t heap_flags =
          IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN;
      uint64_t capacity_bytes = 0;
      if (!device_memory_capacity_attached &&
          params->device_memory_capacity_bytes != 0 &&
          iree_hal_amdgpu_device_spec_memory_heap_is_device_local(
              allocator_heap)) {
        capacity_bytes = params->device_memory_capacity_bytes;
        heap_flags = IREE_HAL_MEMORY_HEAP_SPEC_FLAG_NONE;
        device_memory_capacity_attached = true;
      }
      heaps[i] = (iree_hal_memory_heap_spec_t){
          .name = iree_hal_amdgpu_device_spec_memory_heap_name(allocator_heap),
          .capacity_bytes = capacity_bytes,
          .allocation_granularity = 1,
          .allocation_alignment = allocator_heap->min_alignment,
          .maximum_allocation_size = allocator_heap->max_allocation_size,
          .physical_device_affinity =
              iree_hal_amdgpu_device_spec_all_physical_device_affinity(
                  params->physical_device_count),
          .flags = heap_flags,
      };
      memory_types[i] = (iree_hal_memory_type_spec_t){
          .heap_index = (uint32_t)i,
          .memory_type = allocator_heap->type,
          .allowed_buffer_usage = allocator_heap->allowed_usage,
          .allowed_memory_access =
              iree_hal_amdgpu_device_spec_memory_access(allocator_heap),
          .minimum_alignment = allocator_heap->min_alignment,
          .optimal_transfer_granularity = 1,
          .atomic_operations = allocator_heap->atomic_operations,
          .flags = IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
      };
    }

    uint32_t external_device_memory_type_mask = 0;
    for (iree_host_size_t i = 0; i < heap_count; ++i) {
      if (!iree_all_bits_set(memory_types[i].memory_type,
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
        continue;
      }
      if (i >= 32) {
        external_device_memory_type_mask = UINT32_MAX;
        break;
      }
      external_device_memory_type_mask |= 1u << i;
    }
    iree_hal_external_buffer_handle_spec_t external_buffer_handle = {0};
    iree_host_size_t external_buffer_handle_count = 0;
    if (external_device_memory_type_mask != 0 &&
        iree_all_bits_set(params->flags,
                          IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_DMABUF)) {
      external_buffer_handle = (iree_hal_external_buffer_handle_spec_t){
          .handle_type_mask = IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF,
          .direction_flags = IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
                             IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          .allowed_buffer_usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
          .allowed_memory_access = IREE_HAL_MEMORY_ACCESS_NONE,
          .compatible_memory_type_mask = external_device_memory_type_mask,
          .flags = IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS |
                   IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_OWNING,
      };
      external_buffer_handle_count = 1;
    }
    iree_hal_device_memory_spec_t memory = {
        .heap_count = heap_count,
        .heaps = heaps,
        .memory_type_count = heap_count,
        .memory_types = memory_types,
        .external_buffer_handle_count = external_buffer_handle_count,
        .external_buffer_handles =
            external_buffer_handle_count ? &external_buffer_handle : NULL,
        .flags = IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_memory(builder, &memory);
  }

  iree_allocator_free(builder->host_allocator, memory_types);
  iree_allocator_free(builder->host_allocator, heaps);
  iree_allocator_free(builder->host_allocator, allocator_heaps);
  return status;
}

static iree_hal_atomic_capabilities_t
iree_hal_amdgpu_device_spec_atomic_capabilities(
    iree_hal_atomic_operation_flags_t operations) {
  const iree_hal_atomic_operation_capabilities_t operation_capabilities = {
      .device_scope_32 = operations,
      .device_scope_64 = operations,
      .system_scope_32 = operations,
      .system_scope_64 = operations,
  };
  const iree_hal_atomic_wait_condition_flags_t wait_conditions =
      iree_any_bit_set(operations, IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT)
          ? IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL
          : IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE;
  const iree_hal_atomic_wait_condition_capabilities_t
      wait_condition_capabilities = {
          .device_scope_32 = wait_conditions,
          .device_scope_64 = wait_conditions,
          .system_scope_32 = wait_conditions,
          .system_scope_64 = wait_conditions,
      };
  return (iree_hal_atomic_capabilities_t){
      .operations = operation_capabilities,
      .wait_conditions = wait_condition_capabilities,
  };
}

static iree_hal_atomic_capabilities_t
iree_hal_amdgpu_device_spec_zero_compute_atomic_capabilities(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  iree_hal_atomic_operation_flags_t operations =
      IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  if (iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_wait(
          capabilities)) {
    operations |= IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  }
  if (iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_store(
          capabilities)) {
    operations |= IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  }
  return iree_hal_amdgpu_device_spec_atomic_capabilities(operations);
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_queues(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_queue_family_spec_t* families = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      builder->host_allocator, params->physical_device_count, sizeof(*families),
      (void**)&families));
  memset(families, 0, params->physical_device_count * sizeof(*families));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < params->physical_device_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amdgpu_device_spec_physical_device_params_t*
        physical_device = &params->physical_devices[i];
    families[i] = (iree_hal_queue_family_spec_t){
        .name = physical_device->identity.processor,
        .queue_count = physical_device->queue_count,
        .priority_count = 1,
        .timestamp_valid_bits = 64,
        .timestamp_frequency_hz = physical_device->timestamp_frequency_hz,
        .physical_device_affinity = 1ull << i,
        .queue_affinity =
            iree_hal_amdgpu_device_spec_queue_affinity_for_physical_device(
                i, physical_device->queue_count),
        .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
                      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING |
                      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC,
        .atomic_capabilities = iree_hal_amdgpu_device_spec_atomic_capabilities(
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL),
        .zero_compute_atomic_capabilities =
            iree_hal_amdgpu_device_spec_zero_compute_atomic_capabilities(
                physical_device->vendor_packet_capabilities),
        .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
    };
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_queue_spec_t queues = {
        .family_count = params->physical_device_count,
        .families = families,
        .flags = IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_queues(builder, &queues);
  }

  iree_allocator_free(builder->host_allocator, families);
  return status;
}

static void iree_hal_amdgpu_device_spec_query_wavefront_support(
    const iree_hal_amdgpu_device_spec_physical_device_params_t* physical_device,
    iree_hal_amdgpu_wavefront_size_support_t* out_support) {
  if (iree_hal_amdgpu_target_identity_lookup_wavefront_size_support(
          &physical_device->identity, out_support)) {
    return;
  }

  out_support->default_size = physical_device->wavefront_size;
  out_support->explicit_supported_sizes =
      IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE;
}

static uint32_t iree_hal_amdgpu_device_spec_minimum_subgroup_size(
    iree_hal_amdgpu_wavefront_size_flags_t supported_sizes) {
  if (iree_any_bit_set(supported_sizes,
                       IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32)) {
    return 32;
  }
  if (iree_any_bit_set(supported_sizes,
                       IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64)) {
    return 64;
  }
  return 0;
}

static uint32_t iree_hal_amdgpu_device_spec_maximum_subgroup_size(
    iree_hal_amdgpu_wavefront_size_flags_t supported_sizes) {
  if (iree_any_bit_set(supported_sizes,
                       IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64)) {
    return 64;
  }
  if (iree_any_bit_set(supported_sizes,
                       IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32)) {
    return 32;
  }
  return 0;
}

static uint64_t iree_hal_amdgpu_device_spec_subgroup_size_mask(
    iree_hal_amdgpu_wavefront_size_flags_t supported_sizes) {
  uint64_t mask = 0;
  if (iree_any_bit_set(supported_sizes,
                       IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32)) {
    mask |= 1ull << 32;
  }
  return mask;
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_dispatch(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  uint32_t default_wavefront_size = params->physical_devices[0].wavefront_size;
  iree_hal_amdgpu_wavefront_size_flags_t supported_wavefront_sizes =
      IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE;
  uint32_t maximum_workgroup_local_memory_size =
      params->physical_devices[0].maximum_workgroup_local_memory_size;
  uint32_t maximum_resident_subgroup_count =
      params->physical_devices[0].maximum_waves_per_compute_unit;
  uint32_t maximum_resident_invocation_count =
      params->physical_devices[0].maximum_waves_per_compute_unit *
      params->physical_devices[0].wavefront_size;
  uint32_t compute_unit_count = 0;
  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_hal_amdgpu_device_spec_physical_device_params_t*
        physical_device = &params->physical_devices[i];
    iree_hal_amdgpu_wavefront_size_support_t physical_wavefront_support;
    iree_hal_amdgpu_device_spec_query_wavefront_support(
        physical_device, &physical_wavefront_support);
    const iree_hal_amdgpu_wavefront_size_flags_t physical_supported_sizes =
        physical_wavefront_support.explicit_supported_sizes |
        iree_hal_amdgpu_wavefront_size_flag(
            physical_wavefront_support.default_size);
    default_wavefront_size = iree_min(default_wavefront_size,
                                      physical_wavefront_support.default_size);
    supported_wavefront_sizes =
        i == 0 ? physical_supported_sizes
               : supported_wavefront_sizes & physical_supported_sizes;
    maximum_workgroup_local_memory_size =
        iree_min(maximum_workgroup_local_memory_size,
                 physical_device->maximum_workgroup_local_memory_size);
    maximum_resident_subgroup_count =
        iree_min(maximum_resident_subgroup_count,
                 physical_device->maximum_waves_per_compute_unit);
    maximum_resident_invocation_count =
        iree_min(maximum_resident_invocation_count,
                 physical_device->maximum_waves_per_compute_unit *
                     physical_device->wavefront_size);
    if (IREE_UNLIKELY(UINT32_MAX - compute_unit_count <
                      physical_device->compute_unit_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU compute unit count overflow");
    }
    compute_unit_count += physical_device->compute_unit_count;
  }

  if (IREE_UNLIKELY(supported_wavefront_sizes ==
                    IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical devices have no common wavefront size");
  }
  if (!iree_any_bit_set(
          supported_wavefront_sizes,
          iree_hal_amdgpu_wavefront_size_flag(default_wavefront_size))) {
    default_wavefront_size = iree_hal_amdgpu_device_spec_minimum_subgroup_size(
        supported_wavefront_sizes);
  }
  iree_hal_device_dispatch_spec_t dispatch = {
      .subgroup.default_size = default_wavefront_size,
      .subgroup.minimum_size =
          iree_hal_amdgpu_device_spec_minimum_subgroup_size(
              supported_wavefront_sizes),
      .subgroup.maximum_size =
          iree_hal_amdgpu_device_spec_maximum_subgroup_size(
              supported_wavefront_sizes),
      .subgroup.supported_size_mask =
          iree_hal_amdgpu_device_spec_subgroup_size_mask(
              supported_wavefront_sizes),
      .execution.unit_count = compute_unit_count,
      .execution.group_count = (uint32_t)params->physical_device_count,
      .execution.maximum_resident_invocation_count =
          maximum_resident_invocation_count,
      .execution.maximum_resident_subgroup_count =
          maximum_resident_subgroup_count,
      .execution.maximum_workgroup_local_memory_size =
          maximum_workgroup_local_memory_size,
      .execution.maximum_workgroup_local_memory_size_optin =
          maximum_workgroup_local_memory_size,
      .addressing.pointer_size_bits = 64,
      .addressing.address_space_bits = 64,
      .flags = IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_set_dispatch(builder, &dispatch);
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_timing(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  // Every GPU agent has a wallclock at a known rate independent of its gfx
  // arch, so every logical device this driver builds describes a domain.
  iree_hal_device_timing_spec_flags_t flags =
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS |
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HOST_CORRELATION |
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HARDWARE_COUNTERS |
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_TRACE_CAPTURE |
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_PROFILING_PERTURBS_EXECUTION;

  // verify_params established that every physical device reports the same
  // nonzero rate, so device 0's rate is the rate of every agent here.
  iree_hal_device_timing_spec_t timing = {
      // The agent wallclock is a 64-bit counter.
      .timestamp_valid_bits = 64,
      .timestamp_frequency_hz =
          params->physical_devices[0].timestamp_frequency_hz,
      .flags = flags,
  };
  return iree_hal_device_spec_builder_set_timing(builder, &timing);
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_executables(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  for (iree_host_size_t i = 0; i < params->physical_device_count; ++i) {
    const iree_hal_amdgpu_target_identity_t* exact_identity =
        &params->physical_devices[i].identity;
    const iree_hal_physical_device_affinity_t target_affinity = 1ull << i;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_spec_builder_add_executable_targets(
            builder, exact_identity, target_affinity));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_device_spec_populate_sanitizer(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  return iree_hal_device_spec_builder_set_sanitizer(builder,
                                                    &params->sanitizer);
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_device_spec_create(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_spec_verify_params(params));

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_amdgpu_device_spec_populate_identity(params, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_spec_populate_memory(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_spec_populate_queues(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_spec_populate_dispatch(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_spec_populate_timing(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_spec_populate_executables(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_spec_populate_sanitizer(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}
