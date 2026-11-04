// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/memory.h"

#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/memory_profile.h"
#include "libamdf/src/platform/host_memory.h"

typedef struct amdf_gpu_dxg_memory_plan_t {
  // Page-rounded physical allocation length.
  uint64_t byte_length;
  // Guaranteed base alignment across every exposed address space.
  uint64_t alignment;
  // Private physical allocation domain.
  amdf_dxg_bridge_gpu_allocation_domain_t allocation_domain;
  // Private physical allocation behavior.
  amdf_dxg_bridge_gpu_allocation_flags_t allocation_flags;
  // Complete achieved public memory properties.
  amdf_memory_flags_t achieved_flags;
} amdf_gpu_dxg_memory_plan_t;

typedef enum amdf_gpu_dxg_host_storage_kind_e {
  // No host storage is owned; host_pointer may borrow registered pages.
  AMDF_GPU_DXG_HOST_STORAGE_NONE = 0,
  // Ordinary aligned platform host memory is owned.
  AMDF_GPU_DXG_HOST_STORAGE_MEMORY = 1,
  // Remappable platform host backing and its persistent view are owned.
  AMDF_GPU_DXG_HOST_STORAGE_BACKING = 2,
} amdf_gpu_dxg_host_storage_kind_t;

struct amdf_gpu_dxg_memory_t {
  // Stable device attachment copied from the live provider device.
  amdf_gpu_dxg_memory_device_t device;
  // KMT resource grouping the native allocations, or zero when ungrouped.
  uint32_t resource_handle;
  // Independent native reference retained for exact-format re-export.
  uintptr_t external_reference;
  // Capacity of trailing allocation_handles storage.
  uint32_t allocation_capacity;
  // Number of live native handles in allocation_handles.
  uint32_t allocation_count;
  // Kind of owned host storage represented by host_pointer and host_backing.
  amdf_gpu_dxg_host_storage_kind_t host_storage_kind;
  // Maximum byte length represented by each native handle except the tail.
  uint64_t maximum_native_allocation_byte_length;
  // Native source and persistent view for remappable host backing.
  amdf_platform_host_backing_t host_backing;
  // Host-visible allocation base, borrowed or owned according to storage kind.
  void* host_pointer;
  // Aggregate physical allocation length in bytes.
  uint64_t byte_length;
  // First usable byte of the GPU virtual-address reservation.
  uint64_t device_address;
  // Native base retained for releasing the GPU virtual-address reservation.
  uint64_t device_reservation_base;
  // Native GPU virtual-address reservation length.
  uint64_t device_reservation_byte_length;
  // Achieved public memory properties.
  amdf_memory_flags_t flags;
  // Exact GPU page-table access granted to every native allocation.
  amdf_memory_access_t device_access;
  // Immutable host visibility established by the selected native profile.
  amdf_memory_host_description_t host_description;
  // Native allocation handles in increasing byte-offset order.
  uint32_t allocation_handles[];
};

static void amdf_gpu_dxg_memory_plan(
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_dxg_memory_plan_t* out_plan) {
  amdf_gpu_dxg_memory_plan_t plan = {0};
  plan.achieved_flags = profile->guaranteed_flags | create_info->required_flags;
  plan.alignment = create_info->minimum_alignment;
  if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
    plan.allocation_domain =
        AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST;
    if (plan.alignment < AMDF_GPU_DXG_PAGE_SIZE) {
      plan.alignment = AMDF_GPU_DXG_PAGE_SIZE;
    }
  } else {
    plan.allocation_domain = profile->memory_class == AMDF_MEMORY_CLASS_LOCAL
                                 ? AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
                                 : AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
    if (plan.alignment < AMDF_GPU_DXG_RESERVATION_GRANULARITY) {
      plan.alignment = AMDF_GPU_DXG_RESERVATION_GRANULARITY;
    }
  }

  plan.byte_length = (create_info->byte_length + AMDF_GPU_DXG_PAGE_SIZE - 1) &
                     ~(AMDF_GPU_DXG_PAGE_SIZE - 1);
  if ((create_info->required_flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
    plan.allocation_flags |= AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN;
  }
  if ((create_info->required_flags & AMDF_MEMORY_FLAG_QUEUE_STORAGE) != 0) {
    plan.allocation_flags |= AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE;
  }
  *out_plan = plan;
}

static amdf_status_t amdf_gpu_dxg_memory_allocate_state(
    const amdf_gpu_dxg_memory_device_t* device, uint32_t allocation_capacity,
    amdf_gpu_dxg_memory_t** out_memory) {
  const size_t handle_byte_length =
      (size_t)allocation_capacity * sizeof(uint32_t);
  amdf_gpu_dxg_memory_t* memory = NULL;
  const amdf_status_t status = amdf_calloc_with_trailing(
      device->host_allocator,
      offsetof(amdf_gpu_dxg_memory_t, allocation_handles), handle_byte_length,
      amdf_alignof(amdf_gpu_dxg_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->device = *device;
  memory->allocation_capacity = allocation_capacity;
  memory->host_backing.handle = UINTPTR_MAX;
  *out_memory = memory;
  return AMDF_STATUS_OK;
}

static void amdf_gpu_dxg_memory_capture_host_description(
    const amdf_memory_native_profile_t* profile,
    amdf_gpu_dxg_memory_t* memory) {
  if (profile->visibility.describe_host != NULL) {
    memory->host_description = profile->visibility.describe_host(
        profile->visibility.data, AMDF_EXTERNAL_MEMORY_TYPE_NONE, memory->flags,
        profile->host_mapping.properties);
  }
}

static amdf_status_t amdf_gpu_dxg_memory_allocate_host_storage(
    const amdf_gpu_dxg_memory_plan_t* plan, amdf_gpu_dxg_memory_t* memory) {
  if ((plan->achieved_flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) != 0) {
    memory->host_storage_kind = AMDF_GPU_DXG_HOST_STORAGE_BACKING;
    const amdf_status_t status = amdf_platform_host_backing_prepare(
        plan->byte_length, plan->alignment, &memory->host_backing);
    memory->host_pointer = memory->host_backing.pointer;
    return status;
  }
  memory->host_storage_kind = AMDF_GPU_DXG_HOST_STORAGE_MEMORY;
  return amdf_platform_host_memory_allocate(plan->byte_length, plan->alignment,
                                            &memory->host_pointer);
}

static amdf_status_t amdf_gpu_dxg_memory_reserve_device_address(
    uint64_t byte_length, uint64_t alignment, uint64_t maximum_address,
    amdf_gpu_dxg_memory_t* memory) {
  const uint64_t usable_reservation_byte_length =
      (byte_length + AMDF_GPU_DXG_RESERVATION_GRANULARITY - 1) &
      ~(AMDF_GPU_DXG_RESERVATION_GRANULARITY - 1);
  uint64_t reservation_byte_length = usable_reservation_byte_length;
  if (alignment > AMDF_GPU_DXG_RESERVATION_GRANULARITY) {
    reservation_byte_length += alignment - AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  }

  uint64_t reservation_base = 0;
  amdf_status_t status = memory->device.native_api->reserve_address(
      memory->device.native_context, memory->device.adapter_handle, 0, 0,
      reservation_byte_length, &reservation_base);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->device_reservation_base = reservation_base;
  memory->device_reservation_byte_length = reservation_byte_length;
  const uint64_t aligned_address =
      (reservation_base + alignment - 1) & ~(alignment - 1);
  if (aligned_address < reservation_base ||
      aligned_address - reservation_base >
          reservation_byte_length - usable_reservation_byte_length ||
      aligned_address > maximum_address ||
      usable_reservation_byte_length - 1 > maximum_address - aligned_address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  memory->device_address = aligned_address;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_dxg_memory_create_allocations(
    const amdf_gpu_dxg_memory_plan_t* plan, amdf_gpu_dxg_memory_t* memory) {
  const amdf_dxg_bridge_gpu_allocation_create_info_t create_info = {
      .structure_size = sizeof(create_info),
      .device_handle = memory->device.device_handle,
      .domain = plan->allocation_domain,
      .flags = plan->allocation_flags,
      .byte_length = plan->byte_length,
      .placement_device_address =
          plan->allocation_domain == AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
              ? memory->device_address
              : 0,
      .host_pointer =
          plan->allocation_domain == AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
              ? NULL
              : memory->host_pointer,
  };
  uint32_t created_allocation_count = 0;
  const amdf_status_t status = amdf_gpu_dxg_adapter_create_allocations(
      memory->device.bridge_adapter, &create_info, memory->allocation_capacity,
      memory->allocation_handles, &memory->resource_handle,
      &created_allocation_count);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (created_allocation_count > memory->allocation_capacity) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  // Capture every usable native handle before rejecting malformed provider
  // output. This object owns rollback and the dependent backing.
  for (uint32_t i = 0; i < created_allocation_count; ++i) {
    const uint32_t handle = memory->allocation_handles[i];
    if (handle != 0) {
      memory->allocation_handles[memory->allocation_count++] = handle;
    }
  }
  if (memory->allocation_count != created_allocation_count ||
      created_allocation_count != memory->allocation_capacity) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_dxg_memory_map_device_address(
    amdf_gpu_dxg_memory_t* memory) {
  uint64_t remaining_byte_length = memory->byte_length;
  uint64_t mapped_byte_length = 0;
  uint64_t last_paging_fence = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0;
       amdf_status_is_ok(status) && i < memory->allocation_count; ++i) {
    const uint64_t chunk_byte_length =
        remaining_byte_length < memory->maximum_native_allocation_byte_length
            ? remaining_byte_length
            : memory->maximum_native_allocation_byte_length;
    uint64_t mapped_address = 0;
    uint64_t paging_fence = 0;
    status = memory->device.native_api->map_allocation(
        memory->device.native_context, memory->device.paging_queue_handle,
        memory->device_address + mapped_byte_length,
        memory->allocation_handles[i], 0, chunk_byte_length,
        memory->device_access, &mapped_address, &paging_fence);
    if (amdf_status_is_ok(status) &&
        mapped_address != memory->device_address + mapped_byte_length) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    if (amdf_status_is_ok(status)) {
      mapped_byte_length += chunk_byte_length;
      remaining_byte_length -= chunk_byte_length;
      if (paging_fence > last_paging_fence) {
        last_paging_fence = paging_fence;
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    status = memory->device.native_api->wait_for_paging(
        memory->device.native_context, memory->device.device_handle,
        memory->device.paging_synchronization_object_handle,
        memory->device.paging_progress_fence, last_paging_fence);
  }
  return status;
}

static amdf_status_t amdf_gpu_dxg_memory_make_resident(
    amdf_gpu_dxg_memory_t* memory) {
  uint32_t resident_allocation_count = 0;
  uint64_t paging_fence = 0;
  amdf_status_t status = memory->device.native_api->make_resident(
      memory->device.native_context, memory->device.paging_queue_handle,
      memory->allocation_count, memory->allocation_handles,
      &resident_allocation_count, &paging_fence);
  if (amdf_status_is_ok(status)) {
    status = memory->device.native_api->wait_for_paging(
        memory->device.native_context, memory->device.device_handle,
        memory->device.paging_synchronization_object_handle,
        memory->device.paging_progress_fence, paging_fence);
  }
  if (amdf_status_is_ok(status) &&
      resident_allocation_count != memory->allocation_count) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return status;
}

static amdf_status_t amdf_gpu_dxg_memory_prepare_device_address(
    uint64_t alignment, uint64_t maximum_address,
    amdf_gpu_dxg_memory_t* memory) {
  amdf_status_t status = amdf_gpu_dxg_memory_reserve_device_address(
      memory->byte_length, alignment, maximum_address, memory);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_map_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_make_resident(memory);
  }
  return status;
}

static amdf_status_t amdf_gpu_dxg_memory_destroy_allocations(
    amdf_gpu_dxg_memory_t* memory) {
  if (memory->resource_handle == 0 && memory->allocation_count == 0) {
    return AMDF_STATUS_OK;
  }
  const amdf_gpu_dxg_memory_release_result_t result =
      memory->device.native_api->destroy_allocations(
          memory->device.native_context, memory->device.device_handle,
          memory->resource_handle, memory->allocation_count,
          memory->allocation_handles, memory->host_pointer != NULL);
  if (result.ownership == AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED) {
    memory->resource_handle = 0;
    memory->allocation_count = 0;
  }
  return result.status;
}

static amdf_status_t amdf_gpu_dxg_memory_free_device_address(
    amdf_gpu_dxg_memory_t* memory) {
  if (memory->device_reservation_base == 0) {
    return AMDF_STATUS_OK;
  }
  const amdf_status_t status = memory->device.native_api->free_address(
      memory->device.native_context, memory->device.adapter_handle,
      memory->device_reservation_base, memory->device_reservation_byte_length);
  if (amdf_status_is_ok(status)) {
    memory->device_address = 0;
    memory->device_reservation_base = 0;
    memory->device_reservation_byte_length = 0;
  }
  return status;
}

static amdf_status_t amdf_gpu_dxg_memory_release_external_reference(
    amdf_gpu_dxg_memory_t* memory) {
  if (memory->external_reference == 0) {
    return AMDF_STATUS_OK;
  }
  const amdf_status_t status =
      memory->device.native_api->release_external_reference(
          memory->external_reference);
  if (amdf_status_is_ok(status)) {
    memory->external_reference = 0;
  }
  return status;
}

static amdf_status_t amdf_gpu_dxg_memory_free_host_storage(
    amdf_gpu_dxg_memory_t* memory) {
  if (memory->host_storage_kind == AMDF_GPU_DXG_HOST_STORAGE_NONE) {
    return AMDF_STATUS_OK;
  }
  if (memory->host_storage_kind == AMDF_GPU_DXG_HOST_STORAGE_BACKING) {
    const amdf_status_t status = amdf_platform_host_backing_destroy(
        &memory->host_backing, memory->byte_length);
    memory->host_pointer = memory->host_backing.pointer;
    if (memory->host_backing.handle == UINTPTR_MAX &&
        memory->host_backing.pointer == NULL) {
      memory->host_storage_kind = AMDF_GPU_DXG_HOST_STORAGE_NONE;
    }
    return status;
  }
  if (memory->host_pointer == NULL) {
    memory->host_storage_kind = AMDF_GPU_DXG_HOST_STORAGE_NONE;
    return AMDF_STATUS_OK;
  }
  const amdf_status_t status =
      amdf_platform_host_memory_free(memory->host_pointer, memory->byte_length);
  if (amdf_status_is_ok(status)) {
    memory->host_pointer = NULL;
    memory->host_storage_kind = AMDF_GPU_DXG_HOST_STORAGE_NONE;
  }
  return status;
}

static amdf_status_t amdf_gpu_dxg_memory_release_native(
    amdf_gpu_dxg_memory_t* memory) {
  // Allocation destruction removes mappings and residency. The separately
  // reserved VA, external reference, and host backing follow in that order.
  amdf_status_t status = amdf_gpu_dxg_memory_destroy_allocations(memory);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_free_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_release_external_reference(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_free_host_storage(memory);
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_memory_prepare(
    const amdf_gpu_dxg_memory_device_t* device,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_dxg_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  amdf_gpu_dxg_memory_plan_t plan = {0};
  amdf_gpu_dxg_memory_plan(profile, create_info, &plan);

  uint32_t allocation_count = 0;
  uint64_t maximum_native_allocation_byte_length = 0;
  amdf_status_t status = amdf_gpu_dxg_adapter_query_allocation_layout(
      device->bridge_adapter, plan.byte_length, &allocation_count,
      &maximum_native_allocation_byte_length);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (allocation_count == 0 || maximum_native_allocation_byte_length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  amdf_gpu_dxg_memory_t* memory = NULL;
  status =
      amdf_gpu_dxg_memory_allocate_state(device, allocation_count, &memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *memory_state = memory;
  memory->maximum_native_allocation_byte_length =
      maximum_native_allocation_byte_length;
  memory->byte_length = plan.byte_length;
  memory->flags = plan.achieved_flags;
  memory->device_access = create_info->device_access;
  amdf_gpu_dxg_memory_capture_host_description(profile, memory);

  if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
    memory->host_pointer = create_info->registered_host_pointer;
  } else if (profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM) {
    status = amdf_gpu_dxg_memory_allocate_host_storage(&plan, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_reserve_device_address(
        plan.byte_length, plan.alignment,
        profile->device_address.maximum_address, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_create_allocations(&plan, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_map_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_make_resident(memory);
  }

  if (amdf_status_is_ok(status)) {
    amdf_gpu_umd_memory_result_t result = {
        .flags = plan.achieved_flags,
        .byte_length = plan.byte_length,
        .alignment = plan.alignment,
        .native_allocation_byte_length = plan.byte_length,
        .native_allocation_granularity = AMDF_GPU_DXG_PAGE_SIZE,
        .device_address = memory->device_address,
    };
    if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
      result.physical_backing_id.words[0] =
          (uint64_t)(uintptr_t)memory->host_pointer;
      result.physical_backing_id.words[1] = memory->byte_length;
    } else {
      result.physical_backing_id.words[0] =
          ((uint64_t)memory->device.device_handle << 32) |
          (memory->resource_handle != 0 ? memory->resource_handle
                                        : memory->allocation_handles[0]);
      result.physical_backing_id.words[1] =
          ((uint64_t)memory->allocation_count << 32) |
          memory->allocation_handles[memory->allocation_count - 1];
    }
    *out_result = result;
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_memory_prepare_import(
    const amdf_gpu_dxg_memory_device_t* device,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_dxg_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  if (device->native_api->retain_external_reference == NULL ||
      device->native_api->release_external_reference == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_gpu_dxg_memory_t* memory = NULL;
  amdf_status_t status = amdf_gpu_dxg_memory_allocate_state(device, 1, &memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *memory_state = memory;
  memory->flags = profile->guaranteed_flags | import_info->required_flags;
  memory->device_access = import_info->device_access;
  amdf_gpu_dxg_memory_capture_host_description(profile, memory);

  status = device->native_api->retain_external_reference(
      (uintptr_t)external_memory->payload.native_handle,
      &memory->external_reference);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_dxg_bridge_gpu_buffer_import_info_t native_import = {
      .device_handle = device->device_handle,
      .adapter_luid = device->adapter_luid,
      .shared_handle = (void*)memory->external_reference,
  };
  uint64_t buffer_byte_length = 0;
  status = amdf_gpu_dxg_adapter_prepare_buffer_import(
      device->bridge_adapter, &native_import, &memory->resource_handle,
      &memory->allocation_handles[0], &memory->byte_length,
      &buffer_byte_length);
  memory->allocation_count = memory->allocation_handles[0] != 0 ? 1 : 0;
  memory->maximum_native_allocation_byte_length = memory->byte_length;
  if (amdf_status_is_ok(status) &&
      (external_memory->source_byte_offset > buffer_byte_length ||
       external_memory->byte_length >
           buffer_byte_length - external_memory->source_byte_offset ||
       memory->byte_length > profile->import.maximum_byte_length)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint64_t alignment =
      import_info->minimum_alignment > AMDF_GPU_DXG_RESERVATION_GRANULARITY
          ? import_info->minimum_alignment
          : AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_memory_prepare_device_address(
        alignment, profile->device_address.maximum_address, memory);
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t source_offset = external_memory->source_byte_offset;
    const uint64_t offset_alignment =
        source_offset == 0 ? alignment
                           : source_offset & (UINT64_C(0) - source_offset);
    *out_result = (amdf_gpu_umd_memory_result_t){
        .flags = memory->flags,
        .source_byte_offset = source_offset,
        .byte_length = external_memory->byte_length,
        .alignment =
            offset_alignment < alignment ? offset_alignment : alignment,
        .native_allocation_byte_length = memory->byte_length,
        .native_allocation_granularity = AMDF_GPU_DXG_PAGE_SIZE,
        .device_address = memory->device_address + source_offset,
    };
  }
  return status;
}

static void AMDF_CALL amdf_gpu_dxg_memory_release_export(
    void* user_data, amdf_external_memory_type_t type,
    amdf_external_memory_payload_t payload) {
  (void)type;
  const amdf_gpu_dxg_memory_native_api_t* api = user_data;
  const amdf_status_t status =
      api->release_external_reference((uintptr_t)payload.native_handle);
  amdf_assert(amdf_status_is_ok(status) &&
              "owned external memory reference must remain valid");
  (void)status;
}

amdf_status_t amdf_gpu_dxg_memory_export(
    amdf_gpu_dxg_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)export_info;
  if (memory->device.native_api->retain_external_reference == NULL ||
      memory->device.native_api->release_external_reference == NULL ||
      memory->external_reference == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  uintptr_t exported_reference = 0;
  const amdf_status_t status =
      memory->device.native_api->retain_external_reference(
          memory->external_reference, &exported_reference);
  if (amdf_status_is_ok(status)) {
    *out_value = (amdf_external_memory_t){
        .payload.native_handle = (void*)exported_reference,
        .release = amdf_gpu_dxg_memory_release_export,
        .release_user_data = (void*)memory->device.native_api,
    };
  }
  return status;
}

uintptr_t amdf_gpu_dxg_memory_host_mapping_source(
    amdf_gpu_dxg_memory_t* memory) {
  return memory->host_backing.handle;
}

uint64_t amdf_gpu_dxg_memory_remaining_backing_byte_length(
    const amdf_gpu_dxg_memory_t* memory) {
  const bool owns_host_storage =
      memory->host_storage_kind == AMDF_GPU_DXG_HOST_STORAGE_MEMORY
          ? memory->host_pointer != NULL
          : memory->host_storage_kind == AMDF_GPU_DXG_HOST_STORAGE_BACKING &&
                (memory->host_backing.handle != UINTPTR_MAX ||
                 memory->host_backing.pointer != NULL);
  return memory->resource_handle != 0 || memory->allocation_count != 0 ||
                 memory->external_reference != 0 || owns_host_storage
             ? memory->byte_length
             : 0;
}

amdf_status_t amdf_gpu_dxg_memory_destroy(amdf_gpu_dxg_memory_t* memory) {
  const amdf_status_t status = amdf_gpu_dxg_memory_release_native(memory);
  if (amdf_status_is_ok(status)) {
    amdf_free(memory->device.host_allocator, memory);
  }
  return status;
}

void amdf_gpu_dxg_memory_abandon(amdf_gpu_dxg_memory_t* memory) {
  amdf_free(memory->device.host_allocator, memory);
}

amdf_status_t amdf_gpu_dxg_memory_map(
    amdf_gpu_dxg_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_gpu_dxg_memory_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result) {
  *out_result = (amdf_gpu_umd_host_mapping_result_t){
      .flags = capabilities->supported_access,
      .pointer = (uint8_t*)memory->host_pointer + map_info->byte_offset,
      .byte_length = map_info->byte_length,
      .visibility = memory->host_description,
  };
  *out_mapping = memory;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_dxg_memory_cache_control(
    amdf_gpu_dxg_memory_t* memory, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length) {
  amdf_status_t status = amdf_platform_host_memory_cache_control(
      (uint8_t*)memory->host_pointer + memory_byte_offset, byte_length,
      memory->host_description.cache_line_size);
  if (!amdf_status_is_ok(status) || byte_length == 0 ||
      operation != AMDF_HOST_CACHE_OPERATION_FLUSH ||
      (memory->flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
    return status;
  }

  uint64_t allocation_byte_offset = memory_byte_offset;
  uint64_t remaining_byte_length = byte_length;
  while (remaining_byte_length != 0) {
    const amdf_gpu_dxg_memory_span_t span =
        amdf_gpu_dxg_memory_span(memory, allocation_byte_offset);
    const uint64_t native_byte_length = remaining_byte_length < span.byte_length
                                            ? remaining_byte_length
                                            : span.byte_length;
    status = memory->device.native_api->invalidate_allocation(
        memory->device.native_context, memory->device.device_handle,
        span.allocation_handle, span.byte_offset, native_byte_length);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    allocation_byte_offset += native_byte_length;
    remaining_byte_length -= native_byte_length;
  }
  return AMDF_STATUS_OK;
}

amdf_gpu_dxg_memory_span_t amdf_gpu_dxg_memory_span(
    const amdf_gpu_dxg_memory_t* memory, uint64_t byte_offset) {
  const uint64_t index =
      byte_offset / memory->maximum_native_allocation_byte_length;
  const uint64_t offset =
      byte_offset % memory->maximum_native_allocation_byte_length;
  const uint64_t remaining = memory->byte_length - byte_offset;
  const uint64_t available =
      memory->maximum_native_allocation_byte_length - offset;
  return (amdf_gpu_dxg_memory_span_t){
      .allocation_handle = memory->allocation_handles[index],
      .byte_offset = offset,
      .byte_length = remaining < available ? remaining : available,
  };
}
