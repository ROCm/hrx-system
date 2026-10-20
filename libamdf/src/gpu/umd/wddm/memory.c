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
#include "libamdf/src/gpu/umd/wddm/memory.h"
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
  // Independent same-access NT reference used for exact-format re-export.
  HANDLE shared_handle;
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
  // Achieved public memory properties.
  amdf_memory_flags_t flags;
  // Exact GPU page-table access granted to every native allocation.
  amdf_memory_access_t device_access;
  // Native allocation handles in increasing byte-offset order.
  D3DKMT_HANDLE allocation_handles[];
};

uintptr_t amdf_gpu_umd_memory_host_mapping_source(
    amdf_gpu_umd_memory_t* memory) {
  (void)memory;
  amdf_assert(false && "WDDM does not advertise HOST_MAPPING_SOURCE");
  return 0;
}

amdf_gpu_wddm_memory_span_t amdf_gpu_wddm_memory_span(
    const amdf_gpu_umd_memory_t* memory, uint64_t byte_offset) {
  const uint64_t index =
      byte_offset / memory->maximum_native_allocation_byte_length;
  const uint64_t offset =
      byte_offset % memory->maximum_native_allocation_byte_length;
  const uint64_t remaining = memory->byte_length - byte_offset;
  const uint64_t available =
      memory->maximum_native_allocation_byte_length - offset;
  return (amdf_gpu_wddm_memory_span_t){
      .allocation = memory->allocation_handles[index],
      .byte_offset = offset,
      .byte_length = remaining < available ? remaining : available,
  };
}

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
  DWORD allocation_type = MEM_RESERVE | MEM_COMMIT;
  if (plan->alignment > AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY) {
    const uint64_t alignment_slack =
        plan->alignment - AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
    reservation_byte_length += alignment_slack;
    // Stronger alignment needs a selected subrange, not committed padding.
    allocation_type = MEM_RESERVE;
  }

  memory->host_reservation = VirtualAlloc(NULL, (SIZE_T)reservation_byte_length,
                                          allocation_type, PAGE_READWRITE);
  if (memory->host_reservation == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  if ((allocation_type & MEM_COMMIT) != 0) {
    memory->host_pointer = memory->host_reservation;
    return AMDF_STATUS_OK;
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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

static amdf_status_t amdf_windows_gpu_memory_map_device_address(
    amdf_gpu_umd_memory_t* memory) {
  uint64_t remaining_byte_length = memory->byte_length;
  uint64_t mapped_byte_length = 0;
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
    map.BaseAddress = memory->device_address + mapped_byte_length;
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
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    } else {
      mapped_byte_length += chunk_byte_length;
      if (map.PagingFenceValue > last_paging_fence_value) {
        last_paging_fence_value = map.PagingFenceValue;
      }
      remaining_byte_length -= chunk_byte_length;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_device_status_wait_for_paging(
        &memory->device->status, memory->device->kmt, memory->device->device,
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
  const amdf_status_t status = amdf_kmt_device_status_wait_for_paging(
      &memory->device->status, memory->device->kmt, memory->device->device,
      memory->device->paging_sync_object, memory->device->paging_fence,
      make_resident.PagingFenceValue);
  if (amdf_status_is_ok(status) &&
      make_resident.NumAllocations != memory->allocation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
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
  // Complete OS reclamation before successful release returns host backing for
  // reuse. The caller has already retired execution.
  destroy.Flags.SynchronousDestroy = memory->host_pointer != NULL;
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

uint64_t amdf_gpu_umd_memory_remaining_backing_byte_length(
    const amdf_gpu_umd_memory_t* memory) {
  // A reserved but uncommitted host interval and empty GPU VA retain no
  // payload. A partially released allocation group conservatively retains its
  // full extent, including native alignment and its short final allocation.
  return memory->resource != 0 || memory->allocation_count != 0 ||
                 memory->shared_handle != NULL ||
                 (memory->host_reservation != NULL &&
                  memory->host_pointer != NULL)
             ? memory->byte_length
             : 0;
}

static amdf_status_t amdf_windows_gpu_memory_release_native(
    amdf_gpu_umd_memory_t* memory) {
  // Allocation destruction removes its mappings and residency, including
  // accepted preparation work. The separately reserved VA remains ours until
  // FreeGpuVirtualAddress; host backing is released only after reclamation.
  amdf_status_t status = amdf_windows_gpu_memory_destroy_allocations(memory);
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_free_device_address(memory);
  }
  if (amdf_status_is_ok(status) && memory->shared_handle != NULL) {
    if (CloseHandle(memory->shared_handle)) {
      memory->shared_handle = NULL;
    } else {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_free_host_storage(memory);
  }
  return status;
}

static amdf_memory_host_description_t amdf_gpu_umd_memory_describe_host(
    const void* data, amdf_external_memory_type_t external_memory_type,
    amdf_memory_flags_t flags) {
  (void)external_memory_type;
  (void)data;
  amdf_memory_host_description_t result = {0};
  result.cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  result.cache_line_size = amdf_windows_host_cache_line_size();
  const bool coherent = (flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
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
  return result;
}

amdf_status_t amdf_gpu_umd_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (memory_profile_ordinal > 3 ||
      !amdf_kmt_api_supports_gpu_memory(device->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_status_t status = amdf_gpu_wddm_query_memory_profile(
      &device->memory_capabilities, memory_profile_ordinal, out_profile);
  if (amdf_status_is_ok(status)) {
    out_profile->visibility.describe_host = amdf_gpu_umd_memory_describe_host;
    out_profile->visibility.data = device;
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  amdf_gpu_umd_memory_t* memory = NULL;
  amdf_status_t status = amdf_calloc_with_trailing(
      device->host_allocator,
      offsetof(amdf_gpu_umd_memory_t, allocation_handles),
      sizeof(D3DKMT_HANDLE), amdf_alignof(amdf_gpu_umd_memory_t),
      (void**)&memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->device = device;
  memory->allocation_capacity = 1;
  memory->flags = profile->guaranteed_flags | import_info->required_flags;
  memory->device_access = import_info->device_access;
  *memory_state = memory;
  if (!DuplicateHandle(GetCurrentProcess(),
                       external_memory->payload.native_handle,
                       GetCurrentProcess(), &memory->shared_handle, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  const amdf_wkmi_bridge_gpu_buffer_import_info_t native_import = {
      .device_handle = device->device,
      .adapter_luid = device->adapter_luid,
      .shared_handle = memory->shared_handle,
  };
  uint64_t buffer_byte_length = 0;
  status = amdf_gpu_wddm_wkmi_adapter_prepare_buffer_import(
      &device->wkmi_adapter, &native_import, &memory->resource,
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
      import_info->minimum_alignment > AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY
          ? import_info->minimum_alignment
          : AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
  const amdf_windows_gpu_memory_plan_t plan = {
      .byte_length = memory->byte_length,
      .alignment = alignment,
  };
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_reserve_device_address(
        &plan, profile->device_address.maximum_address, memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_map_device_address(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_gpu_memory_make_resident(memory);
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t source_offset = external_memory->source_byte_offset;
    const uint64_t offset_alignment =
        source_offset == 0 ? alignment
                           : source_offset & (UINT64_C(0) - source_offset);
    const amdf_gpu_umd_memory_result_t result = {
        .flags = memory->flags,
        .source_byte_offset = source_offset,
        .byte_length = external_memory->byte_length,
        .alignment =
            offset_alignment < alignment ? offset_alignment : alignment,
        .native_allocation_byte_length = memory->byte_length,
        .native_allocation_granularity = AMDF_WINDOWS_GPU_PAGE_SIZE,
        .device_address = memory->device_address + source_offset,
    };
    *out_result = result;
  }
  return status;
}

static void AMDF_CALL amdf_windows_gpu_memory_release_handle(
    void* user_data, amdf_external_memory_type_t type,
    amdf_external_memory_payload_t payload) {
  (void)user_data;
  (void)type;
  const BOOL closed = CloseHandle(payload.native_handle);
  amdf_assert(closed && "owned external memory handle must remain valid");
  (void)closed;
}

amdf_status_t amdf_gpu_umd_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)export_info;
  HANDLE exported = NULL;
  if (!DuplicateHandle(GetCurrentProcess(), memory->shared_handle,
                       GetCurrentProcess(), &exported, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  *out_value = (amdf_external_memory_t){
      .payload.native_handle = exported,
      .release = amdf_windows_gpu_memory_release_handle,
  };
  return AMDF_STATUS_OK;
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
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
  amdf_gpu_umd_host_mapping_result_t result = {0};
  result.flags = capabilities->supported_access;
  result.pointer = (uint8_t*)memory->host_pointer + map_info->byte_offset;
  result.byte_length = map_info->byte_length;
  result.visibility =
      amdf_gpu_umd_memory_describe_host(memory->device, 0, memory->flags);
  *out_result = result;
  // The public view owns the borrow; memory already owns the persistent map.
  *out_mapping = (amdf_gpu_umd_host_mapping_t*)memory;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length) {
  amdf_gpu_umd_memory_t* memory = (amdf_gpu_umd_memory_t*)mapping;
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

void amdf_gpu_umd_host_mapping_destroy(amdf_gpu_umd_host_mapping_t* mapping) {
  (void)mapping;
}
