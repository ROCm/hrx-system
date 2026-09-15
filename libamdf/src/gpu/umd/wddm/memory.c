// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/gpu/umd/wddm/memory_profile.h"
#include "libamdf/src/platform/windows/host_cache.h"

typedef struct amdf_windows_gpu_memory_plan_t {
  // Page-rounded physical allocation length.
  uint64_t byte_length;
  // Guaranteed base alignment across every exposed address space.
  uint64_t alignment;
  // Private WKMI physical allocation domain.
  amdf_wkmi_bridge_gpu_allocation_domain_t allocation_domain;
  // Private WKMI physical allocation behavior.
  amdf_wkmi_bridge_gpu_allocation_flags_t allocation_flags;
  // Complete achieved public memory properties.
  amdf_memory_flags_t achieved_flags;
} amdf_windows_gpu_memory_plan_t;

struct amdf_gpu_umd_memory_t {
  // Device borrowed while this attachment remains live.
  amdf_gpu_umd_device_t* device;
  // KMT resource grouping the native allocations, or zero when ungrouped.
  D3DKMT_HANDLE resource;
  // Capacity of trailing `allocation_handles` storage.
  uint32_t allocation_capacity;
  // Number of live native handles in `allocation_handles`.
  uint32_t allocation_count;
  // Maximum byte length represented by each native handle except the tail.
  uint64_t maximum_native_allocation_byte_length;
  // First byte of the host reservation owned by system memory.
  void* host_reservation;
  // Host-visible allocation base, or `NULL` for local memory.
  void* host_pointer;
  // Aggregate physical allocation length in bytes.
  uint64_t byte_length;
  // First usable byte of the GPU virtual-address reservation.
  uint64_t device_address;
  // Native base retained for releasing the GPU virtual-address reservation.
  uint64_t device_reservation_base;
  // Native GPU virtual-address reservation length.
  uint64_t device_reservation_byte_length;
  // Prefix length with valid GPU page-table mappings.
  uint64_t mapped_byte_length;
  // Unexpected driver-selected mapping retained until rollback succeeds.
  struct {
    // Driver-selected base address.
    uint64_t device_address;
    // Byte length awaiting rollback.
    uint64_t mapped_byte_length;
    // Paging fence that publishes the mapping before rollback.
    uint64_t paging_fence_value;
  } rollback;
  // Accepted page-table removal retained across wait failures.
  struct {
    // Paging fence that makes the removal visible.
    uint64_t paging_fence_value;
    // Nonzero while the accepted removal remains to be observed.
    uint32_t active;
  } pending_unmap;
  // Nonzero while the native allocations are resident.
  uint32_t is_resident;
  // Achieved public memory properties.
  amdf_memory_flags_t flags;
  // Exact GPU page-table access granted to every native allocation.
  amdf_memory_access_t device_access;
  // Native allocation handles in increasing byte-offset order.
  D3DKMT_HANDLE allocation_handles[];
};

struct amdf_gpu_umd_host_mapping_t {
  // Memory borrowed by the generic host-mapping parent.
  amdf_gpu_umd_memory_t* memory;
  // First byte exposed by this mapping.
  void* pointer;
  // Exposed byte length.
  uint64_t byte_length;
};

static bool amdf_windows_gpu_align_up(uint64_t value, uint64_t alignment,
                                      uint64_t* out_value) {
  const uint64_t mask = alignment - 1;
  if (value > UINT64_MAX - mask) {
    return false;
  }
  *out_value = (value + mask) & ~mask;
  return true;
}

static void amdf_windows_gpu_memory_plan(
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_windows_gpu_memory_plan_t* out_plan) {
  amdf_windows_gpu_memory_plan_t plan = {0};
  plan.achieved_flags = profile->guaranteed_flags | create_info->required_flags;
  plan.alignment = create_info->minimum_alignment;
  if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
    plan.allocation_domain =
        AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST;
    if (plan.alignment < AMDF_WINDOWS_GPU_PAGE_SIZE) {
      plan.alignment = AMDF_WINDOWS_GPU_PAGE_SIZE;
    }
  } else {
    plan.allocation_domain =
        profile->memory_class == AMDF_MEMORY_CLASS_LOCAL
            ? AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
            : AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
    if (plan.alignment < AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY) {
      plan.alignment = AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
    }
  }

  plan.byte_length =
      (create_info->byte_length + AMDF_WINDOWS_GPU_PAGE_SIZE - 1) &
      ~(AMDF_WINDOWS_GPU_PAGE_SIZE - 1);
  if ((create_info->required_flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
    plan.allocation_flags |= AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN;
  }
  if ((create_info->required_flags & AMDF_MEMORY_FLAG_QUEUE_STORAGE) != 0) {
    plan.allocation_flags |= AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE;
  }
  *out_plan = plan;
}

static amdf_status_t amdf_windows_gpu_memory_allocate_host_storage(
    const amdf_windows_gpu_memory_plan_t* plan, amdf_gpu_umd_memory_t* memory) {
  uint64_t reservation_byte_length = plan->byte_length;
  if (plan->alignment > AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY) {
    const uint64_t alignment_slack =
        plan->alignment - AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
    reservation_byte_length += alignment_slack;
  }

  memory->host_reservation = VirtualAlloc(NULL, (SIZE_T)reservation_byte_length,
                                          MEM_RESERVE, PAGE_READWRITE);
  if (memory->host_reservation == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  uint64_t aligned_pointer = 0;
  if (!amdf_windows_gpu_align_up((uint64_t)(uintptr_t)memory->host_reservation,
                                 plan->alignment, &aligned_pointer)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  memory->host_pointer =
      VirtualAlloc((void*)(uintptr_t)aligned_pointer, (SIZE_T)plan->byte_length,
                   MEM_COMMIT, PAGE_READWRITE);
  if (memory->host_pointer == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_gpu_memory_reserve_device_address(
    const amdf_windows_gpu_memory_plan_t* plan, uint64_t maximum_address,
    amdf_gpu_umd_memory_t* memory) {
  const uint64_t usable_reservation_byte_length =
      (plan->byte_length + AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY - 1) &
      ~(AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY - 1);
  uint64_t reservation_byte_length = usable_reservation_byte_length;
  if (plan->alignment > AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY) {
    const uint64_t alignment_slack =
        plan->alignment - AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
    reservation_byte_length += alignment_slack;
  }

  D3DDDI_RESERVEGPUVIRTUALADDRESS reserve = {0};
  reserve.hAdapter = memory->device->adapter;
  reserve.Size = reservation_byte_length;
  const amdf_status_t status = amdf_kmt_make_status(
      memory->device->kmt->reserve_gpu_virtual_address(&reserve));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->device_reservation_base = reserve.VirtualAddress;
  memory->device_reservation_byte_length = reservation_byte_length;
  if (!amdf_windows_gpu_align_up(reserve.VirtualAddress, plan->alignment,
                                 &memory->device_address) ||
      memory->device_address < reserve.VirtualAddress ||
      memory->device_address - reserve.VirtualAddress >
          reservation_byte_length - usable_reservation_byte_length ||
      memory->device_address > maximum_address ||
      usable_reservation_byte_length - 1 >
          maximum_address - memory->device_address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_gpu_memory_create_allocations(
    const amdf_windows_gpu_memory_plan_t* plan, amdf_gpu_umd_memory_t* memory) {
  amdf_wkmi_bridge_gpu_allocation_create_info_t create_info = {0};
  create_info.structure_size = sizeof(create_info);
  create_info.device_handle = memory->device->device;
  create_info.domain = plan->allocation_domain;
  create_info.flags = plan->allocation_flags;
  create_info.byte_length = plan->byte_length;
  if (plan->allocation_domain == AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL) {
    create_info.placement_device_address = memory->device_address;
  } else {
    create_info.host_pointer = memory->host_pointer;
  }
  uint32_t created_allocation_count = 0;
  const amdf_status_t status = amdf_gpu_wddm_wkmi_adapter_create_allocations(
      &memory->device->wkmi_adapter, &create_info, memory->allocation_capacity,
      memory->allocation_handles, &memory->resource, &created_allocation_count);
  if (!amdf_status_is_ok(status)) return status;
  // Capture every usable native handle before rejecting malformed driver
  // output. This memory object owns both rollback and the dependent backing.
  for (uint32_t i = 0; i < created_allocation_count; ++i) {
    const D3DKMT_HANDLE handle = memory->allocation_handles[i];
    if (handle != 0) {
      memory->allocation_handles[memory->allocation_count++] = handle;
    }
  }
  if (memory->allocation_count != created_allocation_count) {
    return amdf_kmt_make_status(STATUS_INVALID_HANDLE);
  }
  if (created_allocation_count != memory->allocation_capacity) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_unmap_range(
    amdf_gpu_umd_memory_t* memory, uint64_t device_address,
    uint64_t byte_length) {
  if (memory->pending_unmap.active == 0) {
    D3DDDI_MAPGPUVIRTUALADDRESS unmap = {0};
    unmap.hPagingQueue = memory->device->paging_queue;
    unmap.BaseAddress = device_address;
    unmap.SizeInPages = byte_length / AMDF_WINDOWS_GPU_PAGE_SIZE;
    unmap.Protection.NoAccess = 1;
    const NTSTATUS native_status =
        memory->device->kmt->map_gpu_virtual_address(&unmap);
    if (!amdf_kmt_status_is_success_or_pending(native_status)) {
      return amdf_kmt_make_status(native_status);
    }
    memory->pending_unmap.paging_fence_value = unmap.PagingFenceValue;
    memory->pending_unmap.active = 1;
  }
  const amdf_status_t status = amdf_kmt_wait_for_paging(
      memory->device->kmt, memory->device->device,
      memory->device->paging_sync_object, memory->device->paging_fence,
      memory->pending_unmap.paging_fence_value);
  if (amdf_status_is_ok(status)) {
    memory->pending_unmap.paging_fence_value = 0;
    memory->pending_unmap.active = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_rollback_unexpected_mapping(
    amdf_gpu_umd_memory_t* memory) {
  if (memory->rollback.mapped_byte_length == 0) {
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = amdf_kmt_wait_for_paging(
      memory->device->kmt, memory->device->device,
      memory->device->paging_sync_object, memory->device->paging_fence,
      memory->rollback.paging_fence_value);
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_unmap_range(
        memory, memory->rollback.device_address,
        memory->rollback.mapped_byte_length);
  }
  if (amdf_status_is_ok(status)) {
    memory->rollback.device_address = 0;
    memory->rollback.mapped_byte_length = 0;
    memory->rollback.paging_fence_value = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_map_device_address(
    amdf_gpu_umd_memory_t* memory) {
  uint64_t remaining_byte_length = memory->byte_length;
  uint64_t last_paging_fence_value = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0;
       amdf_status_is_ok(status) && i < memory->allocation_count; ++i) {
    const uint64_t chunk_byte_length =
        remaining_byte_length < memory->maximum_native_allocation_byte_length
            ? remaining_byte_length
            : memory->maximum_native_allocation_byte_length;
    D3DDDI_MAPGPUVIRTUALADDRESS map = {0};
    map.hPagingQueue = memory->device->paging_queue;
    map.BaseAddress = memory->device_address + memory->mapped_byte_length;
    map.hAllocation = memory->allocation_handles[i];
    map.SizeInPages = chunk_byte_length / AMDF_WINDOWS_GPU_PAGE_SIZE;
    map.Protection.Write =
        (memory->device_access & AMDF_MEMORY_ACCESS_WRITE) != 0;
    map.Protection.Execute =
        (memory->device_access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
    const NTSTATUS native_status =
        memory->device->kmt->map_gpu_virtual_address(&map);
    if (!amdf_kmt_status_is_success_or_pending(native_status)) {
      status = amdf_kmt_make_status(native_status);
    } else if (map.VirtualAddress != map.BaseAddress) {
      memory->rollback.device_address = map.VirtualAddress;
      memory->rollback.mapped_byte_length = chunk_byte_length;
      memory->rollback.paging_fence_value = map.PagingFenceValue;
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    } else {
      memory->mapped_byte_length += chunk_byte_length;
      if (map.PagingFenceValue > last_paging_fence_value) {
        last_paging_fence_value = map.PagingFenceValue;
      }
      remaining_byte_length -= chunk_byte_length;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_wait_for_paging(
        memory->device->kmt, memory->device->device,
        memory->device->paging_sync_object, memory->device->paging_fence,
        last_paging_fence_value);
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_make_resident(
    amdf_gpu_umd_memory_t* memory) {
  D3DDDI_MAKERESIDENT make_resident = {0};
  make_resident.hPagingQueue = memory->device->paging_queue;
  make_resident.NumAllocations = memory->allocation_count;
  make_resident.AllocationList = memory->allocation_handles;
  make_resident.Flags.CantTrimFurther = 1;
  const NTSTATUS native_status =
      memory->device->kmt->make_resident(&make_resident);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  memory->is_resident = 1;
  const amdf_status_t status = amdf_kmt_wait_for_paging(
      memory->device->kmt, memory->device->device,
      memory->device->paging_sync_object, memory->device->paging_fence,
      make_resident.PagingFenceValue);
  if (amdf_status_is_ok(status) &&
      make_resident.NumAllocations != memory->allocation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_unmap_device_address(
    amdf_gpu_umd_memory_t* memory) {
  if (memory->mapped_byte_length == 0) {
    return AMDF_STATUS_OK;
  }
  const amdf_status_t status = amdf_windows_gpu_memory_unmap_range(
      memory, memory->device_address, memory->mapped_byte_length);
  if (amdf_status_is_ok(status)) {
    memory->mapped_byte_length = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_evict(
    amdf_gpu_umd_memory_t* memory) {
  if (memory->is_resident == 0) {
    return AMDF_STATUS_OK;
  }
  D3DKMT_EVICT evict = {0};
  evict.hDevice = memory->device->device;
  evict.NumAllocations = memory->allocation_count;
  evict.AllocationList = memory->allocation_handles;
  const amdf_status_t status =
      amdf_kmt_make_status(memory->device->kmt->evict(&evict));
  if (amdf_status_is_ok(status)) {
    memory->is_resident = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_destroy_allocations(
    amdf_gpu_umd_memory_t* memory) {
  if (memory->resource == 0 && memory->allocation_count == 0) {
    return AMDF_STATUS_OK;
  }
  D3DKMT_DESTROYALLOCATION2 destroy = {0};
  destroy.hDevice = memory->device->device;
  destroy.hResource = memory->resource;
  if (memory->resource == 0) {
    destroy.phAllocationList = memory->allocation_handles;
    destroy.AllocationCount = memory->allocation_count;
  }
  destroy.Flags.AssumeNotInUse = 1;
  const amdf_status_t status =
      amdf_kmt_make_status(memory->device->kmt->destroy_allocation(&destroy));
  if (amdf_status_is_ok(status)) {
    memory->resource = 0;
    memory->allocation_count = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_free_device_address(
    amdf_gpu_umd_memory_t* memory) {
  if (memory->device_reservation_base == 0) {
    return AMDF_STATUS_OK;
  }
  D3DKMT_FREEGPUVIRTUALADDRESS free_address = {0};
  free_address.hAdapter = memory->device->adapter;
  free_address.BaseAddress = memory->device_reservation_base;
  free_address.Size = memory->device_reservation_byte_length;
  const amdf_status_t status = amdf_kmt_make_status(
      memory->device->kmt->free_gpu_virtual_address(&free_address));
  if (amdf_status_is_ok(status)) {
    memory->device_address = 0;
    memory->device_reservation_base = 0;
    memory->device_reservation_byte_length = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_gpu_memory_free_host_storage(
    amdf_gpu_umd_memory_t* memory) {
  if (memory->host_reservation == NULL) {
    return AMDF_STATUS_OK;
  }
  if (!VirtualFree(memory->host_reservation, 0, MEM_RELEASE)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  memory->host_reservation = NULL;
  memory->host_pointer = NULL;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_gpu_memory_release_native(
    amdf_gpu_umd_memory_t* memory) {
  amdf_status_t status =
      amdf_windows_gpu_memory_rollback_unexpected_mapping(memory);
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_unmap_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_evict(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_destroy_allocations(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_free_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_free_host_storage(memory);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (memory_profile_ordinal > 2 ||
      !amdf_kmt_api_supports_gpu_memory(device->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_gpu_wddm_query_memory_profile(
      &device->memory_capabilities, memory_profile_ordinal, out_profile);
}

amdf_status_t amdf_gpu_umd_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  (void)device;
  (void)profile;
  (void)import_info;
  (void)external_memory;
  (void)memory_state;
  (void)out_result;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)memory;
  (void)export_info;
  (void)out_value;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_memory_describe_site(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  (void)memory;
  (void)query;
  (void)out_description;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_memory_prepare(
    amdf_gpu_umd_device_t* device, uint32_t peer_count,
    amdf_gpu_umd_device_t* const* peer_devices,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  // This provider selects independent native preparations, not allocation
  // groups. Its selected profiles therefore always supply an empty peer set.
  (void)peer_count;
  (void)peer_devices;
  amdf_windows_gpu_memory_plan_t plan = {0};
  amdf_windows_gpu_memory_plan(profile, create_info, &plan);

  uint32_t allocation_count = 0;
  uint64_t maximum_native_allocation_byte_length = 0;
  amdf_status_t status = amdf_gpu_wddm_wkmi_adapter_query_allocation_layout(
      &device->wkmi_adapter, plan.byte_length, &allocation_count,
      &maximum_native_allocation_byte_length);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (allocation_count == 0 || maximum_native_allocation_byte_length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  const size_t handle_bytes = (size_t)allocation_count * sizeof(D3DKMT_HANDLE);
  amdf_gpu_umd_memory_t* memory = NULL;
  status = amdf_calloc_with_trailing(
      device->host_allocator,
      offsetof(amdf_gpu_umd_memory_t, allocation_handles), handle_bytes,
      amdf_alignof(amdf_gpu_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->device = device;
  *memory_state = memory;
  memory->allocation_capacity = allocation_count;
  memory->maximum_native_allocation_byte_length =
      maximum_native_allocation_byte_length;
  memory->byte_length = plan.byte_length;
  memory->flags = plan.achieved_flags;
  memory->device_access = create_info->device_access;
  if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
    memory->host_pointer = create_info->registered_host_pointer;
  } else if (profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM) {
    status = amdf_windows_gpu_memory_allocate_host_storage(&plan, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_reserve_device_address(
        &plan, profile->device_address.maximum_address, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_create_allocations(&plan, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_map_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_make_resident(memory);
  }

  if (amdf_status_is_ok(status)) {
    amdf_gpu_umd_memory_result_t result = {0};
    result.flags = plan.achieved_flags;
    result.source_byte_offset = 0;
    result.byte_length = plan.byte_length;
    result.alignment = plan.alignment;
    result.native_allocation_byte_length = plan.byte_length;
    result.native_allocation_granularity = AMDF_WINDOWS_GPU_PAGE_SIZE;
    if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
      result.physical_backing_id.words[0] =
          (uint64_t)(uintptr_t)memory->host_pointer;
      result.physical_backing_id.words[1] = memory->byte_length;
    } else {
      result.physical_backing_id.words[0] =
          ((uint64_t)memory->device->device << 32) |
          (memory->resource != 0 ? memory->resource
                                 : memory->allocation_handles[0]);
      result.physical_backing_id.words[1] =
          ((uint64_t)memory->allocation_count << 32) |
          memory->allocation_handles[memory->allocation_count - 1];
    }
    result.device_address = memory->device_address;
    *out_result = result;
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_destroy(amdf_gpu_umd_memory_t* memory) {
  const amdf_status_t status = amdf_windows_gpu_memory_release_native(memory);
  if (amdf_status_is_ok(status)) {
    amdf_free(memory->device->host_allocator, memory);
  }
  return status;
}

void amdf_gpu_umd_memory_abandon(amdf_gpu_umd_memory_t* memory) {
  amdf_free(memory->device->host_allocator, memory);
}

amdf_status_t amdf_gpu_umd_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result) {
  amdf_gpu_umd_host_mapping_t* mapping = NULL;
  amdf_status_t status =
      amdf_calloc(memory->device->host_allocator, sizeof(*mapping),
                  amdf_alignof(amdf_gpu_umd_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  mapping->memory = memory;
  mapping->pointer = (uint8_t*)memory->host_pointer + map_info->byte_offset;
  mapping->byte_length = map_info->byte_length;

  amdf_gpu_umd_host_mapping_result_t result = {0};
  result.flags = capabilities->supported_access;
  result.pointer = mapping->pointer;
  result.byte_length = mapping->byte_length;
  result.cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  result.cache_line_size = amdf_windows_host_cache_line_size();
  const bool coherent = (memory->flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
  result.flush = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = coherent ? AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT
                           : AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API,
      .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
      .host_instruction = coherent ? AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH
                                   : AMDF_HOST_CACHE_INSTRUCTION_NONE,
      .host_fence_after = coherent ? AMDF_HOST_CACHE_FENCE_X86_MFENCE
                                   : AMDF_HOST_CACHE_FENCE_NONE,
      .range_granularity = result.cache_line_size,
  };
  result.invalidate = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = result.cache_line_size,
  };
  *out_result = result;
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length) {
  amdf_gpu_umd_memory_t* memory = mapping->memory;
  amdf_status_t status = amdf_windows_host_cache_control(
      operation, (uint8_t*)memory->host_pointer + memory_byte_offset,
      byte_length);
  if (!amdf_status_is_ok(status) || byte_length == 0 ||
      operation != AMDF_HOST_CACHE_OPERATION_FLUSH ||
      (memory->flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
    return status;
  }

  uint64_t allocation_byte_offset = memory_byte_offset;
  uint64_t remaining_byte_length = byte_length;
  while (remaining_byte_length != 0) {
    const uint64_t allocation_index =
        allocation_byte_offset / memory->maximum_native_allocation_byte_length;
    const uint64_t native_byte_offset =
        allocation_byte_offset % memory->maximum_native_allocation_byte_length;
    const uint64_t native_available_byte_length =
        memory->maximum_native_allocation_byte_length - native_byte_offset;
    const uint64_t native_byte_length =
        remaining_byte_length < native_available_byte_length
            ? remaining_byte_length
            : native_available_byte_length;
    D3DKMT_INVALIDATECACHE invalidate = {0};
    invalidate.hDevice = memory->device->device;
    invalidate.hAllocation = memory->allocation_handles[allocation_index];
    invalidate.Offset = native_byte_offset;
    invalidate.Length = native_byte_length;
    status = amdf_kmt_make_status(
        memory->device->kmt->invalidate_cache(&invalidate));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    allocation_byte_offset += native_byte_length;
    remaining_byte_length -= native_byte_length;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_host_mapping_destroy(
    amdf_gpu_umd_host_mapping_t* mapping) {
  amdf_free(mapping->memory->device->host_allocator, mapping);
  return AMDF_STATUS_OK;
}
