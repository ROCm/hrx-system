// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include <stdint.h>

#include "libamdf/src/gpu/umd/dxg/memory.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/gpu/umd/wddm/memory.h"
#include "libamdf/src/gpu/umd/wddm/memory_profile.h"

static amdf_status_t amdf_gpu_wddm_memory_reserve_address(
    void* context, uint32_t adapter_handle, uint64_t base_address,
    uint64_t minimum_address, uint64_t byte_length, uint64_t* out_address) {
  amdf_gpu_umd_device_t* device = context;
  D3DDDI_RESERVEGPUVIRTUALADDRESS reserve = {0};
  reserve.hAdapter = adapter_handle;
  reserve.BaseAddress = base_address;
  reserve.MinimumAddress = minimum_address;
  reserve.Size = byte_length;
  const amdf_status_t status =
      amdf_kmt_make_status(device->kmt->reserve_gpu_virtual_address(&reserve));
  if (amdf_status_is_ok(status)) {
    *out_address = reserve.VirtualAddress;
  }
  return status;
}

static amdf_status_t amdf_gpu_wddm_memory_map_allocation(
    void* context, uint32_t paging_queue_handle, uint64_t address,
    uint32_t allocation_handle, uint64_t allocation_byte_offset,
    uint64_t byte_length, amdf_memory_access_t access, uint64_t* out_address,
    uint64_t* out_paging_fence) {
  amdf_gpu_umd_device_t* device = context;
  D3DDDI_MAPGPUVIRTUALADDRESS map = {0};
  map.hPagingQueue = paging_queue_handle;
  map.BaseAddress = address;
  map.hAllocation = allocation_handle;
  map.OffsetInPages = allocation_byte_offset / AMDF_GPU_DXG_PAGE_SIZE;
  map.SizeInPages = byte_length / AMDF_GPU_DXG_PAGE_SIZE;
  map.Protection.Write = (access & AMDF_MEMORY_ACCESS_WRITE) != 0;
  map.Protection.Execute = (access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
  map.Protection.NoAccess = allocation_handle == 0;
  const NTSTATUS native_status = device->kmt->map_gpu_virtual_address(&map);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  *out_address = map.VirtualAddress;
  *out_paging_fence = map.PagingFenceValue;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_wddm_memory_make_resident(
    void* context, uint32_t paging_queue_handle, uint32_t allocation_count,
    const uint32_t* allocation_handles, uint32_t* out_resident_allocation_count,
    uint64_t* out_paging_fence) {
  amdf_gpu_umd_device_t* device = context;
  D3DDDI_MAKERESIDENT make_resident = {0};
  make_resident.hPagingQueue = paging_queue_handle;
  make_resident.NumAllocations = allocation_count;
  make_resident.AllocationList = allocation_handles;
  make_resident.Flags.CantTrimFurther = 1;
  const NTSTATUS native_status = device->kmt->make_resident(&make_resident);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  *out_resident_allocation_count = make_resident.NumAllocations;
  *out_paging_fence = make_resident.PagingFenceValue;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_wddm_memory_wait_for_paging(
    void* context, uint32_t device_handle,
    uint32_t synchronization_object_handle,
    const volatile uint64_t* progress_fence, uint64_t target_value) {
  amdf_gpu_umd_device_t* device = context;
  return amdf_kmt_device_status_wait_for_paging(
      &device->status, device->kmt, device_handle,
      synchronization_object_handle, progress_fence, target_value);
}

static amdf_gpu_dxg_memory_release_result_t
amdf_gpu_wddm_memory_destroy_allocations(void* context, uint32_t device_handle,
                                         uint32_t resource_handle,
                                         uint32_t allocation_count,
                                         const uint32_t* allocation_handles,
                                         bool synchronous) {
  amdf_gpu_umd_device_t* device = context;
  D3DKMT_DESTROYALLOCATION2 destroy = {0};
  destroy.hDevice = device_handle;
  destroy.hResource = resource_handle;
  if (resource_handle == 0) {
    destroy.phAllocationList = allocation_handles;
    destroy.AllocationCount = allocation_count;
  }
  destroy.Flags.AssumeNotInUse = 1;
  destroy.Flags.SynchronousDestroy = synchronous;
  const amdf_status_t status =
      amdf_kmt_make_status(device->kmt->destroy_allocation(&destroy));
  return (amdf_gpu_dxg_memory_release_result_t){
      .status = status,
      .ownership = amdf_status_is_ok(status)
                       ? AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED
                       : AMDF_GPU_DXG_MEMORY_OWNERSHIP_RETAINED,
  };
}

static amdf_status_t amdf_gpu_wddm_memory_free_address(void* context,
                                                       uint32_t adapter_handle,
                                                       uint64_t address,
                                                       uint64_t byte_length) {
  amdf_gpu_umd_device_t* device = context;
  D3DKMT_FREEGPUVIRTUALADDRESS free_address = {0};
  free_address.hAdapter = adapter_handle;
  free_address.BaseAddress = address;
  free_address.Size = byte_length;
  return amdf_kmt_make_status(
      device->kmt->free_gpu_virtual_address(&free_address));
}

static amdf_status_t amdf_gpu_wddm_memory_invalidate_allocation(
    void* context, uint32_t device_handle, uint32_t allocation_handle,
    uint64_t byte_offset, uint64_t byte_length) {
  amdf_gpu_umd_device_t* device = context;
  D3DKMT_INVALIDATECACHE invalidate = {0};
  invalidate.hDevice = device_handle;
  invalidate.hAllocation = allocation_handle;
  invalidate.Offset = byte_offset;
  invalidate.Length = byte_length;
  return amdf_kmt_make_status(device->kmt->invalidate_cache(&invalidate));
}

static amdf_status_t amdf_gpu_wddm_memory_retain_external_reference(
    uintptr_t reference, uintptr_t* out_reference) {
  HANDLE retained = NULL;
  if (!DuplicateHandle(GetCurrentProcess(), (HANDLE)reference,
                       GetCurrentProcess(), &retained, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  *out_reference = (uintptr_t)retained;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_wddm_memory_release_external_reference(
    uintptr_t reference) {
  return CloseHandle((HANDLE)reference)
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
}

static const amdf_gpu_dxg_memory_native_api_t amdf_gpu_wddm_memory_native_api =
    {
        .reserve_address = amdf_gpu_wddm_memory_reserve_address,
        .map_allocation = amdf_gpu_wddm_memory_map_allocation,
        .make_resident = amdf_gpu_wddm_memory_make_resident,
        .wait_for_paging = amdf_gpu_wddm_memory_wait_for_paging,
        .destroy_allocations = amdf_gpu_wddm_memory_destroy_allocations,
        .free_address = amdf_gpu_wddm_memory_free_address,
        .invalidate_allocation = amdf_gpu_wddm_memory_invalidate_allocation,
        .retain_external_reference =
            amdf_gpu_wddm_memory_retain_external_reference,
        .release_external_reference =
            amdf_gpu_wddm_memory_release_external_reference,
};

amdf_gpu_dxg_memory_device_t amdf_gpu_wddm_memory_device(
    amdf_gpu_umd_device_t* device) {
  return (amdf_gpu_dxg_memory_device_t){
      .host_allocator = device->host_allocator,
      .native_api = &amdf_gpu_wddm_memory_native_api,
      .native_context = device,
      .bridge_adapter = &device->bridge_adapter,
      .adapter_handle = device->adapter,
      .adapter_luid = device->adapter_luid,
      .device_handle = device->device,
      .paging_queue_handle = device->paging_queue,
      .paging_synchronization_object_handle = device->paging_sync_object,
      .paging_progress_fence = device->paging_fence,
  };
}

uintptr_t amdf_gpu_umd_memory_host_mapping_source(
    amdf_gpu_umd_memory_t* memory) {
  return amdf_gpu_dxg_memory_host_mapping_source(
      (amdf_gpu_dxg_memory_t*)memory);
}

amdf_status_t amdf_gpu_umd_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (memory_profile_ordinal > 3 ||
      !amdf_kmt_api_supports_gpu_memory(device->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_gpu_dxg_query_memory_profile(&device->memory_capabilities,
                                           memory_profile_ordinal, out_profile);
}

amdf_status_t amdf_gpu_umd_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  const amdf_gpu_dxg_memory_device_t memory_device =
      amdf_gpu_wddm_memory_device(device);
  return amdf_gpu_dxg_memory_prepare_import(
      &memory_device, profile, import_info, external_memory,
      (amdf_gpu_dxg_memory_t**)memory_state, out_result);
}

amdf_status_t amdf_gpu_umd_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  return amdf_gpu_dxg_memory_export((amdf_gpu_dxg_memory_t*)memory, export_info,
                                    out_value);
}

amdf_status_t amdf_gpu_umd_memory_prepare(
    amdf_gpu_umd_device_t* device, uint32_t peer_count,
    amdf_gpu_umd_device_t* const* peer_devices,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  // DXG selects independent native preparations, not allocation groups.
  (void)peer_count;
  (void)peer_devices;
  const amdf_gpu_dxg_memory_device_t memory_device =
      amdf_gpu_wddm_memory_device(device);
  return amdf_gpu_dxg_memory_prepare(&memory_device, profile, create_info,
                                     (amdf_gpu_dxg_memory_t**)memory_state,
                                     out_result);
}

amdf_status_t amdf_gpu_umd_memory_destroy(amdf_gpu_umd_memory_t* memory) {
  return amdf_gpu_dxg_memory_destroy((amdf_gpu_dxg_memory_t*)memory);
}

uint64_t amdf_gpu_umd_memory_remaining_backing_byte_length(
    const amdf_gpu_umd_memory_t* memory) {
  return amdf_gpu_dxg_memory_remaining_backing_byte_length(
      (const amdf_gpu_dxg_memory_t*)memory);
}

void amdf_gpu_umd_memory_abandon(amdf_gpu_umd_memory_t* memory) {
  amdf_gpu_dxg_memory_abandon((amdf_gpu_dxg_memory_t*)memory);
}

amdf_status_t amdf_gpu_umd_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result) {
  return amdf_gpu_dxg_memory_map((amdf_gpu_dxg_memory_t*)memory, capabilities,
                                 map_info, (amdf_gpu_dxg_memory_t**)out_mapping,
                                 out_result);
}

amdf_status_t amdf_gpu_umd_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length) {
  return amdf_gpu_dxg_memory_cache_control((amdf_gpu_dxg_memory_t*)mapping,
                                           operation, memory_byte_offset,
                                           byte_length);
}

void amdf_gpu_umd_host_mapping_destroy(amdf_gpu_umd_host_mapping_t* mapping) {
  (void)mapping;
}
