// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/memory.h"

#include <stdint.h>

#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"
#include "libamdf/src/gpu/umd/dxg/memory.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

enum {
  AMDF_GPU_LINUX_DXG_STATUS_PENDING = 0x00000103u,
};

static bool amdf_gpu_linux_dxg_memory_status_is_success_or_pending(
    NTSTATUS status) {
  const uint32_t code = (uint32_t)status;
  return code == 0 || code == AMDF_GPU_LINUX_DXG_STATUS_PENDING;
}

static amdf_status_t amdf_gpu_linux_dxg_memory_reserve_address(
    void* context, uint32_t adapter_handle, uint64_t base_address,
    uint64_t minimum_address, uint64_t byte_length, uint64_t* out_address) {
  amdf_gpu_umd_device_t* device = context;
  D3DDDI_RESERVEGPUVIRTUALADDRESS reserve = {
      .hAdapter = adapter_handle,
      .BaseAddress = base_address,
      .MinimumAddress = minimum_address,
      .Size = byte_length,
  };
  const amdf_status_t status = amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_reserve_gpu_virtual_address(device->logical_device.kmt,
                                                     &reserve));
  if (amdf_status_is_ok(status)) {
    *out_address = reserve.VirtualAddress;
  }
  return status;
}

static amdf_status_t amdf_gpu_linux_dxg_memory_map_allocation(
    void* context, uint32_t paging_queue_handle, uint64_t address,
    uint32_t allocation_handle, uint64_t allocation_byte_offset,
    uint64_t byte_length, amdf_memory_access_t access, uint64_t* out_address,
    uint64_t* out_paging_fence) {
  amdf_gpu_umd_device_t* device = context;
  D3DDDI_MAPGPUVIRTUALADDRESS map = {
      .hPagingQueue = paging_queue_handle,
      .BaseAddress = address,
      .hAllocation = allocation_handle,
      .OffsetInPages = allocation_byte_offset / AMDF_GPU_DXG_PAGE_SIZE,
      .SizeInPages = byte_length / AMDF_GPU_DXG_PAGE_SIZE,
  };
  map.Protection.Write = (access & AMDF_MEMORY_ACCESS_WRITE) != 0;
  map.Protection.Execute = (access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
  map.Protection.NoAccess = allocation_handle == 0;
  const NTSTATUS native_status = amdf_linux_dxg_kmt_map_gpu_virtual_address(
      device->logical_device.kmt, &map);
  if (!amdf_gpu_linux_dxg_memory_status_is_success_or_pending(native_status)) {
    return amdf_linux_dxg_kmt_make_status(native_status);
  }
  *out_address = map.VirtualAddress;
  *out_paging_fence = map.PagingFenceValue;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_linux_dxg_memory_make_resident(
    void* context, uint32_t paging_queue_handle, uint32_t allocation_count,
    const uint32_t* allocation_handles, uint32_t* out_resident_allocation_count,
    uint64_t* out_paging_fence) {
  amdf_gpu_umd_device_t* device = context;
  D3DDDI_MAKERESIDENT make_resident = {
      .hPagingQueue = paging_queue_handle,
      .NumAllocations = allocation_count,
      .AllocationList = allocation_handles,
  };
  make_resident.Flags.CantTrimFurther = 1;
  const NTSTATUS native_status = amdf_linux_dxg_kmt_make_resident(
      device->logical_device.kmt, &make_resident);
  if (!amdf_gpu_linux_dxg_memory_status_is_success_or_pending(native_status)) {
    return amdf_linux_dxg_kmt_make_status(native_status);
  }
  *out_resident_allocation_count = make_resident.NumAllocations;
  *out_paging_fence = make_resident.PagingFenceValue;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_linux_dxg_memory_wait_for_paging(
    void* context, uint32_t device_handle,
    uint32_t synchronization_object_handle,
    const volatile uint64_t* progress_fence, uint64_t target_value) {
  amdf_gpu_umd_device_t* device = context;
  return amdf_gpu_linux_dxg_device_status_wait_for_paging(
      &device->status, device->logical_device.kmt, device_handle,
      synchronization_object_handle, progress_fence, target_value);
}

static amdf_gpu_dxg_memory_release_result_t
amdf_gpu_linux_dxg_memory_destroy_allocations(
    void* context, uint32_t device_handle, uint32_t resource_handle,
    uint32_t allocation_count, const uint32_t* allocation_handles,
    bool synchronous) {
  amdf_gpu_umd_device_t* device = context;
  D3DKMT_DESTROYALLOCATION2 destroy = {
      .hDevice = device_handle,
      .hResource = resource_handle,
  };
  if (resource_handle == 0) {
    destroy.phAllocationList = allocation_handles;
    destroy.AllocationCount = allocation_count;
  }
  destroy.Flags.AssumeNotInUse = 1;
  destroy.Flags.SynchronousDestroy = synchronous;
  const amdf_status_t status =
      amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_destroy_allocation(
          device->logical_device.kmt, &destroy));
  // The Linux DXG kernel removes validated handles and destroys guest objects
  // before its best-effort host notification. A trusted live request therefore
  // consumes the allocations on every native result.
  return (amdf_gpu_dxg_memory_release_result_t){
      .status = status,
      .ownership = AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED,
  };
}

static amdf_status_t amdf_gpu_linux_dxg_memory_free_address(
    void* context, uint32_t adapter_handle, uint64_t address,
    uint64_t byte_length) {
  amdf_gpu_umd_device_t* device = context;
  D3DKMT_FREEGPUVIRTUALADDRESS free_address = {
      .hAdapter = adapter_handle,
      .BaseAddress = address,
      .Size = byte_length,
  };
  return amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_free_gpu_virtual_address(device->logical_device.kmt,
                                                  &free_address));
}

static amdf_status_t amdf_gpu_linux_dxg_memory_invalidate_allocation(
    void* context, uint32_t device_handle, uint32_t allocation_handle,
    uint64_t byte_offset, uint64_t byte_length) {
  amdf_gpu_umd_device_t* device = context;
  D3DKMT_INVALIDATECACHE invalidate = {
      .hDevice = device_handle,
      .hAllocation = allocation_handle,
      .Offset = byte_offset,
      .Length = byte_length,
  };
  return amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_invalidate_cache(
      device->logical_device.kmt, &invalidate));
}

static const amdf_gpu_dxg_memory_native_api_t
    amdf_gpu_linux_dxg_memory_native_api = {
        .reserve_address = amdf_gpu_linux_dxg_memory_reserve_address,
        .map_allocation = amdf_gpu_linux_dxg_memory_map_allocation,
        .make_resident = amdf_gpu_linux_dxg_memory_make_resident,
        .wait_for_paging = amdf_gpu_linux_dxg_memory_wait_for_paging,
        .destroy_allocations = amdf_gpu_linux_dxg_memory_destroy_allocations,
        .free_address = amdf_gpu_linux_dxg_memory_free_address,
        .invalidate_allocation =
            amdf_gpu_linux_dxg_memory_invalidate_allocation,
};

amdf_gpu_dxg_memory_device_t amdf_gpu_linux_dxg_memory_device(
    amdf_gpu_umd_device_t* device) {
  return (amdf_gpu_dxg_memory_device_t){
      .host_allocator = device->host_allocator,
      .native_api = &amdf_gpu_linux_dxg_memory_native_api,
      .native_context = device,
      .provider_adapter = &device->logical_device.provider_adapter,
      .adapter_handle = device->adapter,
      .adapter_luid = device->adapter_luid,
      .device_handle = device->logical_device.native,
      .paging_queue_handle = device->paging_queue.native,
      .paging_synchronization_object_handle =
          device->paging_queue.synchronization_object,
      .paging_progress_fence = device->paging_queue.progress_fence,
  };
}

amdf_status_t amdf_gpu_linux_dxg_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  return amdf_gpu_dxg_query_memory_profile(&device->memory_capabilities,
                                           memory_profile_ordinal, out_profile);
}

amdf_status_t amdf_gpu_linux_dxg_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  const amdf_gpu_dxg_memory_device_t memory_device =
      amdf_gpu_linux_dxg_memory_device(device);
  return amdf_gpu_dxg_memory_prepare_import(
      &memory_device, profile, import_info, external_memory,
      (amdf_gpu_dxg_memory_t**)memory_state, out_result);
}

amdf_status_t amdf_gpu_linux_dxg_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  return amdf_gpu_dxg_memory_export((amdf_gpu_dxg_memory_t*)memory, export_info,
                                    out_value);
}

amdf_status_t amdf_gpu_linux_dxg_memory_prepare(
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
      amdf_gpu_linux_dxg_memory_device(device);
  return amdf_gpu_dxg_memory_prepare(&memory_device, profile, create_info,
                                     (amdf_gpu_dxg_memory_t**)memory_state,
                                     out_result);
}

uintptr_t amdf_gpu_linux_dxg_memory_host_mapping_source(
    amdf_gpu_umd_memory_t* memory) {
  return amdf_gpu_dxg_memory_host_mapping_source(
      (amdf_gpu_dxg_memory_t*)memory);
}

amdf_status_t amdf_gpu_linux_dxg_memory_destroy(amdf_gpu_umd_memory_t* memory) {
  return amdf_gpu_dxg_memory_destroy((amdf_gpu_dxg_memory_t*)memory);
}

uint64_t amdf_gpu_linux_dxg_memory_remaining_backing_byte_length(
    const amdf_gpu_umd_memory_t* memory) {
  return amdf_gpu_dxg_memory_remaining_backing_byte_length(
      (const amdf_gpu_dxg_memory_t*)memory);
}

void amdf_gpu_linux_dxg_memory_abandon(amdf_gpu_umd_memory_t* memory) {
  amdf_gpu_dxg_memory_abandon((amdf_gpu_dxg_memory_t*)memory);
}

amdf_status_t amdf_gpu_linux_dxg_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result) {
  return amdf_gpu_dxg_memory_map((amdf_gpu_dxg_memory_t*)memory, capabilities,
                                 map_info, (amdf_gpu_dxg_memory_t**)out_mapping,
                                 out_result);
}

amdf_status_t amdf_gpu_linux_dxg_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length) {
  return amdf_gpu_dxg_memory_cache_control((amdf_gpu_dxg_memory_t*)mapping,
                                           operation, memory_byte_offset,
                                           byte_length);
}

void amdf_gpu_linux_dxg_host_mapping_destroy(
    amdf_gpu_umd_host_mapping_t* mapping) {
  (void)mapping;
}
