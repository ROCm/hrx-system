// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_spec_builder.h"

#include <string.h>

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_builder_add_facet(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_vulkan_device_spec_t* spec,
    iree_host_size_t cooperative_matrix_property_count,
    const iree_hal_vulkan_cooperative_matrix_property_t*
        cooperative_matrix_properties) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(spec);

  iree_host_size_t payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      cooperative_matrix_property_count, &payload_size));
  uint8_t* payload_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      builder->host_allocator, payload_size, (void**)&payload_storage));

  iree_status_t status = iree_hal_vulkan_device_spec_encode(
      spec, cooperative_matrix_property_count, cooperative_matrix_properties,
      iree_make_byte_span(payload_storage, payload_size));
  if (iree_status_is_ok(status)) {
    iree_hal_device_spec_facet_t facet = {
        .schema_id =
            iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID),
        .schema_version = IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION,
        .payload = iree_make_const_byte_span(payload_storage, payload_size),
    };
    status = iree_hal_device_spec_builder_add_facet(builder, &facet);
  }

  iree_allocator_free(builder->host_allocator, payload_storage);
  return status;
}

static iree_status_t iree_hal_vulkan_device_spec_verify_params(
    const iree_hal_vulkan_device_spec_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(!params->physical_device)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec physical device is NULL");
  }
  if (IREE_UNLIKELY(!params->device_plan)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec plan is NULL");
  }
  if (IREE_UNLIKELY(!params->device_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec allocator is NULL");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_spec_populate_identity(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  const iree_hal_vulkan_physical_device_snapshot_t* physical_device =
      params->physical_device;
  const VkPhysicalDeviceProperties* properties =
      &physical_device->properties2.properties;

  iree_hal_physical_device_spec_t physical_spec = {
      .identity =
          {
              .display_name = iree_make_cstring_view(properties->deviceName),
              .backend_path = iree_make_cstring_view(properties->deviceName),
              .vendor_id = properties->vendorID,
              .device_id = properties->deviceID,
              .flags = IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID,
          },
      .physical_ordinal = physical_device->ordinal,
      .partition_count = 1,
      .physical_device_affinity = 1ull,
  };
  memcpy(physical_spec.identity.uuid.bytes,
         physical_device->id_properties.deviceUUID,
         sizeof(physical_spec.identity.uuid.bytes));

  iree_hal_device_identity_spec_t identity = {
      .logical_device_id = params->logical_device_id,
      .display_name = params->display_name,
      .driver_id = IREE_SV("vulkan"),
      .backend_id = IREE_SV("vulkan"),
      .device_path = physical_spec.identity.backend_path,
      .vendor_id = properties->vendorID,
      .device_id = properties->deviceID,
      .logical_ordinal = physical_device->ordinal,
      .physical_device_count = 1,
      .physical_devices = &physical_spec,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_set_identity(builder, &identity);
}

static iree_string_view_t iree_hal_vulkan_device_spec_memory_heap_name(
    const iree_hal_allocator_memory_heap_t* heap) {
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    return IREE_SV("host");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_SV("device-host-visible");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    return IREE_SV("device");
  }
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_SV("host-visible");
  }
  return IREE_SV("memory");
}

static iree_hal_memory_access_t iree_hal_vulkan_device_spec_memory_access(
    const iree_hal_allocator_memory_heap_t* heap) {
  if (iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL) ||
      iree_all_bits_set(heap->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return IREE_HAL_MEMORY_ACCESS_ALL;
  }
  return IREE_HAL_MEMORY_ACCESS_NONE;
}

static iree_status_t iree_hal_vulkan_device_spec_query_memory_heaps(
    const iree_hal_vulkan_device_spec_params_t* params,
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

static uint32_t iree_hal_vulkan_device_spec_memory_type_mask(
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

static iree_status_t iree_hal_vulkan_device_spec_populate_memory(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_allocator_memory_heap_t* allocator_heaps = NULL;
  iree_host_size_t heap_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_query_memory_heaps(
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

  if (iree_status_is_ok(status)) {
    memset(heaps, 0, heap_count * sizeof(*heaps));
    memset(memory_types, 0, heap_count * sizeof(*memory_types));
    for (iree_host_size_t i = 0; i < heap_count; ++i) {
      const iree_hal_allocator_memory_heap_t* allocator_heap =
          &allocator_heaps[i];
      heaps[i] = (iree_hal_memory_heap_spec_t){
          .name = iree_hal_vulkan_device_spec_memory_heap_name(allocator_heap),
          .allocation_granularity = 1,
          .allocation_alignment = allocator_heap->min_alignment,
          .maximum_allocation_size = allocator_heap->max_allocation_size,
          .physical_device_affinity = 1ull,
          .flags = IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN,
      };
      memory_types[i] = (iree_hal_memory_type_spec_t){
          .heap_index = (uint32_t)i,
          .memory_type = allocator_heap->type,
          .allowed_buffer_usage = allocator_heap->allowed_usage,
          .allowed_memory_access =
              iree_hal_vulkan_device_spec_memory_access(allocator_heap),
          .minimum_alignment = allocator_heap->min_alignment,
          .optimal_transfer_granularity = 1,
          .flags = IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
      };
    }

    iree_hal_device_memory_spec_t memory = {
        .heap_count = heap_count,
        .heaps = heaps,
        .memory_type_count = heap_count,
        .memory_types = memory_types,
        .flags = IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_memory(builder, &memory);
  }

  iree_allocator_free(builder->host_allocator, memory_types);
  iree_allocator_free(builder->host_allocator, heaps);
  iree_allocator_free(builder->host_allocator, allocator_heaps);
  return status;
}

static iree_status_t iree_hal_vulkan_device_spec_populate_virtual_memory(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  if (!iree_hal_allocator_supports_virtual_memory(params->device_allocator)) {
    return iree_ok_status();
  }

  iree_hal_allocator_memory_heap_t* allocator_heaps = NULL;
  iree_host_size_t heap_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_query_memory_heaps(
      params, builder->host_allocator, &allocator_heaps, &heap_count));

  iree_hal_memory_type_spec_t* memory_types = NULL;
  iree_status_t status = iree_ok_status();
  if (heap_count != 0) {
    status = iree_allocator_malloc_array(builder->host_allocator, heap_count,
                                         sizeof(*memory_types),
                                         (void**)&memory_types);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < heap_count; ++i) {
      memory_types[i] = (iree_hal_memory_type_spec_t){
          .memory_type = allocator_heaps[i].type,
      };
    }
  }

  const uint32_t compatible_memory_type_mask =
      iree_status_is_ok(status)
          ? iree_hal_vulkan_device_spec_memory_type_mask(
                heap_count, memory_types, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
          : 0;
  iree_hal_buffer_params_t buffer_params = {
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
              IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  if (iree_status_is_ok(status) && compatible_memory_type_mask != 0) {
    status = iree_hal_allocator_virtual_memory_query_granularity(
        params->device_allocator, buffer_params, &minimum_page_size,
        &recommended_page_size);
  }
  if (iree_status_is_ok(status) && compatible_memory_type_mask != 0) {
    const VkPhysicalDeviceLimits* limits =
        &params->physical_device->properties2.properties.limits;
    iree_hal_virtual_memory_class_spec_t memory_class = {
        .compatible_memory_type_mask = compatible_memory_type_mask,
        .allowed_buffer_usage = buffer_params.usage,
        .allowed_memory_access = buffer_params.access,
        .minimum_page_size = minimum_page_size,
        .recommended_page_size = recommended_page_size,
        .maximum_reservation_size = limits->maxStorageBufferRange,
        .maximum_physical_allocation_size =
            params->physical_device->properties11.maxMemoryAllocationSize,
        .operation_flags =
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RESERVE |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RELEASE |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_ALLOCATE |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_FREE |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_MAP |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_UNMAP |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PROTECT |
            IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_ADVISE,
        .protection_flags = IREE_HAL_MEMORY_PROTECTION_READ_WRITE,
        .advice_flags = IREE_HAL_MEMORY_ADVICE_NORMAL,
        .flags = IREE_HAL_VIRTUAL_MEMORY_CLASS_SPEC_FLAG_NONE,
    };
    iree_hal_device_virtual_memory_spec_t virtual_memory = {
        .class_count = 1,
        .classes = &memory_class,
        .flags = IREE_HAL_DEVICE_VIRTUAL_MEMORY_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_virtual_memory(builder,
                                                             &virtual_memory);
  }

  iree_allocator_free(builder->host_allocator, memory_types);
  iree_allocator_free(builder->host_allocator, allocator_heaps);
  return status;
}

static uint64_t iree_hal_vulkan_device_spec_timestamp_frequency_hz(
    float timestamp_period_ns) {
  if (timestamp_period_ns <= 0.0f) return 0;
  const double frequency_hz = 1000000000.0 / (double)timestamp_period_ns;
  if (frequency_hz >= (double)UINT64_MAX) return UINT64_MAX;
  return (uint64_t)(frequency_hz + 0.5);
}

static uint32_t iree_hal_vulkan_device_spec_queue_timestamp_valid_bits(
    const iree_hal_vulkan_queue_assignment_t* queue_assignment) {
  uint32_t timestamp_valid_bits =
      queue_assignment->compute.timestamp_valid_bits;
  if (queue_assignment->transfer.timestamp_valid_bits != 0) {
    timestamp_valid_bits =
        timestamp_valid_bits == 0
            ? queue_assignment->transfer.timestamp_valid_bits
            : iree_min(timestamp_valid_bits,
                       queue_assignment->transfer.timestamp_valid_bits);
  }
  if (queue_assignment->sparse_binding.family_index !=
          IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID &&
      queue_assignment->sparse_binding.timestamp_valid_bits != 0) {
    timestamp_valid_bits =
        timestamp_valid_bits == 0
            ? queue_assignment->sparse_binding.timestamp_valid_bits
            : iree_min(timestamp_valid_bits,
                       queue_assignment->sparse_binding.timestamp_valid_bits);
  }
  return timestamp_valid_bits;
}

static iree_status_t iree_hal_vulkan_device_spec_populate_queues(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  const iree_hal_vulkan_queue_assignment_t* queue_assignment =
      &params->device_plan->queue_assignment;
  const uint32_t timestamp_valid_bits =
      iree_hal_vulkan_device_spec_queue_timestamp_valid_bits(queue_assignment);
  iree_hal_queue_family_role_flags_t role_flags =
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER;
  if (timestamp_valid_bits != 0) {
    role_flags |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING;
  }
  iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("default"),
      .queue_count = (uint32_t)queue_assignment->queue_count,
      .priority_count = 1,
      .timestamp_valid_bits = timestamp_valid_bits,
      .timestamp_frequency_hz =
          iree_hal_vulkan_device_spec_timestamp_frequency_hz(
              params->physical_device->properties2.properties.limits
                  .timestampPeriod),
      .physical_device_affinity = 1ull,
      .role_flags = role_flags,
      .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
  };
  iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &queue_family,
      .flags = IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_set_queues(builder, &queues);
}

static uint64_t iree_hal_vulkan_device_spec_subgroup_size_mask(
    uint32_t minimum_subgroup_size, uint32_t maximum_subgroup_size) {
  if (minimum_subgroup_size == 0 || maximum_subgroup_size == 0 ||
      minimum_subgroup_size > maximum_subgroup_size) {
    return 0;
  }
  uint64_t subgroup_size_mask = 0;
  uint32_t subgroup_size = minimum_subgroup_size;
  while (subgroup_size < 64 && subgroup_size <= maximum_subgroup_size) {
    subgroup_size_mask |= 1ull << subgroup_size;
    if (subgroup_size > UINT32_MAX / 2) break;
    subgroup_size *= 2;
  }
  return subgroup_size_mask;
}

static iree_status_t iree_hal_vulkan_device_spec_populate_dispatch(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  const VkPhysicalDeviceLimits* limits =
      &params->physical_device->properties2.properties.limits;
  const bool has_subgroup_size_control =
      iree_all_bits_set(params->device_plan->enabled_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL);
  const uint32_t subgroup_size =
      params->physical_device->subgroup_properties.subgroupSize;
  const uint32_t subgroup_size_minimum =
      has_subgroup_size_control
          ? params->physical_device->subgroup_size_control_properties
                .minSubgroupSize
          : subgroup_size;
  const uint32_t subgroup_size_maximum =
      has_subgroup_size_control
          ? params->physical_device->subgroup_size_control_properties
                .maxSubgroupSize
          : subgroup_size;
  iree_hal_device_dispatch_spec_t dispatch = {
      .launch.maximum_workgroup_invocations =
          limits->maxComputeWorkGroupInvocations,
      .launch.maximum_workgroup_size =
          {
              limits->maxComputeWorkGroupSize[0],
              limits->maxComputeWorkGroupSize[1],
              limits->maxComputeWorkGroupSize[2],
          },
      .launch.maximum_workgroup_count =
          {
              limits->maxComputeWorkGroupCount[0],
              limits->maxComputeWorkGroupCount[1],
              limits->maxComputeWorkGroupCount[2],
          },
      .subgroup.default_size = subgroup_size,
      .subgroup.minimum_size = subgroup_size_minimum,
      .subgroup.maximum_size = subgroup_size_maximum,
      .subgroup.supported_size_mask =
          iree_hal_vulkan_device_spec_subgroup_size_mask(subgroup_size_minimum,
                                                         subgroup_size_maximum),
      .execution.unit_count = 1,
      .execution.group_count = 1,
      .execution.maximum_workgroup_local_memory_size =
          limits->maxComputeSharedMemorySize,
      .addressing.pointer_size_bits = 64,
      .addressing.address_space_bits = 64,
      .addressing.minimum_buffer_device_address_alignment =
          iree_all_bits_set(
              params->device_plan->enabled_features,
              IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)
              ? limits->minStorageBufferOffsetAlignment
              : 0,
      .flags = IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_set_dispatch(builder, &dispatch);
}

static iree_status_t iree_hal_vulkan_device_spec_populate_timing(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_device_timing_spec_flags_t flags =
      IREE_HAL_DEVICE_TIMING_SPEC_FLAG_PROFILING_PERTURBS_EXECUTION;
  const uint32_t timestamp_valid_bits =
      iree_hal_vulkan_device_spec_queue_timestamp_valid_bits(
          &params->device_plan->queue_assignment);
  if (timestamp_valid_bits != 0) {
    flags |= IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS |
             IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DISPATCH_EVENTS;
  }
  if (iree_all_bits_set(
          params->device_plan->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS) &&
      iree_all_bits_set(
          params->physical_device->calibrated_timestamp_time_domains,
          IREE_HAL_VULKAN_TIME_DOMAIN_DEVICE)) {
    flags |= IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HOST_CORRELATION;
  }
  if (iree_all_bits_set(params->device_plan->request_flags,
                        IREE_HAL_VULKAN_REQUEST_FLAG_TRACING)) {
    flags |= IREE_HAL_DEVICE_TIMING_SPEC_FLAG_TRACE_CAPTURE;
  }
  iree_hal_device_timing_spec_t timing = {
      .timestamp_valid_bits = timestamp_valid_bits,
      .timestamp_frequency_hz =
          iree_hal_vulkan_device_spec_timestamp_frequency_hz(
              params->physical_device->properties2.properties.limits
                  .timestampPeriod),
      .flags = flags,
  };
  return iree_hal_device_spec_builder_set_timing(builder, &timing);
}

static iree_status_t iree_hal_vulkan_device_spec_populate_executables(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  if (!iree_all_bits_set(
          params->device_plan->enabled_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)) {
    return iree_ok_status();
  }
  iree_hal_executable_target_t executable_target = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      .priority = 50,
      .physical_device_affinity = 1ull,
      .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_add_executable_target(builder,
                                                            &executable_target);
}

static iree_status_t iree_hal_vulkan_device_spec_populate_facet(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  const VkPhysicalDeviceProperties* physical_device_properties =
      &params->physical_device->properties2.properties;
  iree_hal_vulkan_device_spec_t vulkan_spec = {
      .api_version = physical_device_properties->apiVersion,
      .driver_version = physical_device_properties->driverVersion,
      .physical_device_type = physical_device_properties->deviceType,
      .enabled_features = params->device_plan->enabled_features,
      .flags = IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
  };

  iree_host_size_t property_count = 0;
  const VkCooperativeMatrixPropertiesKHR* source_properties = NULL;
  if (iree_all_bits_set(params->device_plan->enabled_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX)) {
    property_count = params->physical_device->cooperative_matrix_property_count;
    source_properties =
        params->physical_device->cooperative_matrix_property_rows;
    if (IREE_UNLIKELY(property_count != 0 && !source_properties)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Vulkan cooperative matrix support has %" PRIhsz
                              " property rows but no retained row storage",
                              property_count);
    }
  }

  iree_hal_vulkan_cooperative_matrix_property_t* cooperative_matrix_properties =
      NULL;
  iree_status_t status = iree_ok_status();
  if (property_count != 0) {
    status =
        iree_allocator_malloc_array(builder->host_allocator, property_count,
                                    sizeof(*cooperative_matrix_properties),
                                    (void**)&cooperative_matrix_properties);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < property_count; ++i) {
      cooperative_matrix_properties[i] =
          (iree_hal_vulkan_cooperative_matrix_property_t){
              .m_size = source_properties[i].MSize,
              .n_size = source_properties[i].NSize,
              .k_size = source_properties[i].KSize,
              .a_type = source_properties[i].AType,
              .b_type = source_properties[i].BType,
              .c_type = source_properties[i].CType,
              .result_type = source_properties[i].ResultType,
              .saturating_accumulation =
                  source_properties[i].saturatingAccumulation,
              .scope = source_properties[i].scope,
          };
    }
    status = iree_hal_vulkan_device_spec_builder_add_facet(
        builder, &vulkan_spec, property_count, cooperative_matrix_properties);
  }
  iree_allocator_free(builder->host_allocator, cooperative_matrix_properties);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_create(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_verify_params(params));

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_vulkan_device_spec_populate_identity(params, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_spec_populate_memory(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_device_spec_populate_virtual_memory(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_spec_populate_queues(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_spec_populate_dispatch(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_spec_populate_timing(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_spec_populate_executables(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_spec_populate_facet(params, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}
