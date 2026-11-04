// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory_reservation.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/gpu/umd/wddm/kernel_queue.h"
#include "libamdf/src/gpu/umd/wddm/memory.h"

struct amdf_gpu_umd_memory_reservation_t {
  // Allocator for this owner and its immutable consumer array.
  amdf_allocator_t host_allocator;
  // Base of the complete native reservation, including alignment slack.
  uint64_t reservation_address;
  // Native extent required by FreeGpuVirtualAddress.
  uint64_t reservation_byte_length;
  // Aligned application-visible base in the native reservation.
  uint64_t address;
  // Application-visible extent, shared by the explicitly admitted consumers.
  uint64_t byte_length;
  // Device of the latest mutation. Its paging queue orders host-completed
  // batches and detach. The caller retires queued updates before detach.
  amdf_gpu_umd_device_t* publisher;
  // Live devices sharing these native page tables, borrowed through release.
  amdf_gpu_umd_device_t* devices[];
};

bool amdf_gpu_umd_devices_share_mapping_domain(const amdf_gpu_umd_device_t* a,
                                               const amdf_gpu_umd_device_t* b) {
  return a->adapter_luid == b->adapter_luid &&
         a->physical_adapter_index == b->physical_adapter_index;
}

static amdf_memory_access_t amdf_gpu_wddm_required_mapping_access(
    const amdf_gpu_umd_device_t* device) {
  // MapGpuVirtualAddress cannot deny reads. Hardware lacking individual PTE
  // protections cannot promise read-only or non-executable mappings either.
  return AMDF_MEMORY_ACCESS_READ |
         (device->memory_capabilities.read_only_memory_supported
              ? 0
              : AMDF_MEMORY_ACCESS_WRITE) |
         (device->memory_capabilities.no_execute_memory_supported
              ? 0
              : AMDF_MEMORY_ACCESS_EXECUTE);
}

amdf_status_t amdf_gpu_umd_query_memory_reservation_capabilities(
    amdf_gpu_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  if (address_domain_ordinal != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint32_t bits = device->memory_capabilities.virtual_address_bit_count;
  const uint64_t maximum_address =
      bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
  const uint64_t maximum_length =
      (maximum_address / 2) & ~(AMDF_GPU_DXG_RESERVATION_GRANULARITY - 1);
  uint64_t maximum_alignment = 1;
  while (maximum_alignment <= maximum_length / 2) {
    maximum_alignment <<= 1;
  }
  *out_capabilities = (amdf_memory_reservation_capabilities_t){
      .supported_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                          AMDF_MEMORY_ACCESS_EXECUTE,
      .required_access = amdf_gpu_wddm_required_mapping_access(device),
      .supported_flags = AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS,
      .maximum_byte_length = maximum_length,
      .reservation_granularity = AMDF_GPU_DXG_RESERVATION_GRANULARITY,
      .maximum_alignment = maximum_alignment,
      .mapping_granularity = AMDF_GPU_DXG_PAGE_SIZE,
      .minimum_address = AMDF_GPU_DXG_RESERVATION_GRANULARITY,
      .maximum_address = maximum_address,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_umd_device_t* const* devices, uint64_t byte_length,
    uint64_t minimum_alignment, uint64_t common_address,
    amdf_gpu_umd_memory_reservation_t** out_reservation) {
  // Separate logical KMT devices on one physical adapter share process page
  // tables. A cross-adapter group additionally needs coordinated address
  // reservation and physical-adapter selection, not independent device VAs.
  for (uint32_t i = 1; i < device_count; ++i) {
    if (!amdf_gpu_umd_devices_share_mapping_domain(devices[0], devices[i])) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  amdf_gpu_umd_memory_reservation_t* reservation = NULL;
  amdf_status_t status = amdf_calloc_with_trailing(
      host_allocator, sizeof(*reservation),
      device_count * sizeof(reservation->devices[0]),
      amdf_alignof(amdf_gpu_umd_memory_reservation_t), (void**)&reservation);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  reservation->host_allocator = host_allocator;
  reservation->byte_length = byte_length;
  reservation->publisher = devices[0];
  for (uint32_t i = 0; i < device_count; ++i) {
    reservation->devices[i] = devices[i];
  }
  const uint64_t alignment =
      minimum_alignment > AMDF_GPU_DXG_RESERVATION_GRANULARITY
          ? minimum_alignment
          : AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  D3DDDI_RESERVEGPUVIRTUALADDRESS native = {0};
  native.hAdapter = devices[0]->adapter;
  native.BaseAddress = common_address;
  native.Size = common_address != 0 ? byte_length
                                    : byte_length + alignment -
                                          AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  native.MinimumAddress = AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  status = amdf_kmt_make_status(
      devices[0]->kmt->reserve_gpu_virtual_address(&native));
  if (amdf_status_is_ok(status)) {
    reservation->reservation_address = native.VirtualAddress;
    reservation->reservation_byte_length = native.Size;
    reservation->address =
        (native.VirtualAddress + alignment - 1) & ~(alignment - 1);
    *out_reservation = reservation;
  } else {
    amdf_free(host_allocator, reservation);
  }
  return status;
}

void amdf_gpu_umd_memory_reservation_query_access(
    const amdf_gpu_umd_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access) {
  *out_address = reservation->address;
  *out_mapping_access_ordinal = 0;
  *out_mapping_granularity = AMDF_GPU_DXG_PAGE_SIZE;
  *out_required_access = amdf_gpu_wddm_required_mapping_access(
      reservation->devices[access_ordinal]);
}

static amdf_status_t amdf_gpu_wddm_reservation_map(
    amdf_gpu_umd_device_t* device, uint64_t address, uint64_t byte_length,
    D3DKMT_HANDLE allocation, uint64_t source_byte_offset,
    amdf_memory_access_t access, uint64_t* last_paging_fence) {
  D3DDDI_MAPGPUVIRTUALADDRESS map = {0};
  map.hPagingQueue = device->paging_queue;
  map.BaseAddress = address;
  map.hAllocation = allocation;
  map.OffsetInPages = source_byte_offset / AMDF_GPU_DXG_PAGE_SIZE;
  map.SizeInPages = byte_length / AMDF_GPU_DXG_PAGE_SIZE;
  map.Protection.Write = (access & AMDF_MEMORY_ACCESS_WRITE) != 0;
  map.Protection.Execute = (access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
  map.Protection.NoAccess = allocation == 0;
  const NTSTATUS native_status = device->kmt->map_gpu_virtual_address(&map);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  if (map.VirtualAddress != address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (map.PagingFenceValue > *last_paging_fence) {
    *last_paging_fence = map.PagingFenceValue;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_memory_reservation_update(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started) {
  // Source handles belong to the selected logical device even though the page
  // tables are shared. Use that device's paging queue, not the first alias's.
  amdf_gpu_umd_device_t* device = reservation->devices[access_ordinal];
  reservation->publisher = device;
  uint64_t fence = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  *mutation_started = true;
  for (uint32_t i = 0; i < update_count && amdf_status_is_ok(status); ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    uint64_t completed = 0;
    while (completed < entry->byte_length && amdf_status_is_ok(status)) {
      amdf_gpu_wddm_memory_span_t source = {.byte_length = entry->byte_length};
      if (entry->memory != NULL) {
        source = amdf_gpu_wddm_memory_span(
            entry->memory, entry->source_byte_offset + completed);
      }
      const uint64_t remaining = entry->byte_length - completed;
      const uint64_t length =
          remaining < source.byte_length ? remaining : source.byte_length;
      status = amdf_gpu_wddm_reservation_map(
          device, reservation->address + entry->target_byte_offset + completed,
          length, source.allocation, source.byte_offset, entry->access, &fence);
      completed += length;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_device_status_wait_for_paging(
        &device->status, device->kmt, device->device,
        device->paging_sync_object, device->paging_fence, fence);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_reservation_detach(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal) {
  // All admitted aliases share this native group. Clear on the latest writer's
  // paging queue even when called through another alias: a failed host update
  // may have an accepted prefix still pending there.
  (void)access_ordinal;
  amdf_gpu_umd_device_t* device = reservation->publisher;
  uint64_t fence = 0;
  amdf_status_t status = amdf_gpu_wddm_reservation_map(
      device, reservation->address, reservation->byte_length, 0, 0, 0, &fence);
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_device_status_wait_for_paging(
        &device->status, device->kmt, device->device,
        device->paging_sync_object, device->paging_fence, fence);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_reservation_destroy(
    amdf_gpu_umd_memory_reservation_t* reservation) {
  amdf_gpu_umd_device_t* device = reservation->publisher;
  amdf_status_t status = amdf_gpu_umd_memory_reservation_detach(reservation, 0);
  if (amdf_status_is_ok(status)) {
    D3DKMT_FREEGPUVIRTUALADDRESS release = {0};
    release.hAdapter = device->adapter;
    release.BaseAddress = reservation->reservation_address;
    release.Size = reservation->reservation_byte_length;
    status =
        amdf_kmt_make_status(device->kmt->free_gpu_virtual_address(&release));
  }
  amdf_free(reservation->host_allocator, reservation);
  return status;
}

amdf_status_t amdf_gpu_umd_memory_reservation_update_on_queue(
    amdf_gpu_umd_memory_reservation_t* reservation,
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    uint64_t submission, bool* mutation_started, bool* accepted) {
  uint64_t operation_count = 0;
  for (uint32_t i = 0; i < update_count; ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    uint64_t position = 0;
    do {
      const uint64_t span =
          entry->memory == NULL
              ? entry->byte_length
              : amdf_gpu_wddm_memory_span(entry->memory,
                                          entry->source_byte_offset + position)
                    .byte_length;
      const uint64_t remaining = entry->byte_length - position;
      position += span < remaining ? span : remaining;
      ++operation_count;
    } while (position < entry->byte_length);
  }
  if (operation_count > UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations = NULL;
  amdf_status_t status = amdf_calloc_array(
      reservation->host_allocator, operation_count, sizeof(*operations),
      amdf_alignof(D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION),
      (void**)&operations);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t ordinal = 0;
  for (uint32_t i = 0; i < update_count; ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    uint64_t position = 0;
    do {
      D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operation =
          &operations[ordinal++];
      const uint64_t address =
          reservation->address + entry->target_byte_offset + position;
      uint64_t length = entry->byte_length - position;
      if (entry->memory == NULL) {
        operation->OperationType = D3DDDI_UPDATEGPUVIRTUALADDRESS_UNMAP;
        operation->Unmap.BaseAddress = address;
        operation->Unmap.SizeInBytes = length;
        operation->Unmap.Protection.NoAccess = 1;
      } else {
        const amdf_gpu_wddm_memory_span_t source = amdf_gpu_wddm_memory_span(
            entry->memory, entry->source_byte_offset + position);
        if (length > source.byte_length) {
          length = source.byte_length;
        }
        operation->OperationType = D3DDDI_UPDATEGPUVIRTUALADDRESS_MAP_PROTECT;
        operation->MapProtect.BaseAddress = address;
        operation->MapProtect.SizeInBytes = length;
        operation->MapProtect.hAllocation = source.allocation;
        operation->MapProtect.AllocationOffsetInBytes = source.byte_offset;
        operation->MapProtect.Protection.Write =
            (entry->access & AMDF_MEMORY_ACCESS_WRITE) != 0;
        operation->MapProtect.Protection.Execute =
            (entry->access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
      }
      position += length;
    } while (position < entry->byte_length);
  }
  reservation->publisher = reservation->devices[access_ordinal];
  status = amdf_gpu_wddm_kernel_queue_update_memory(
      queue, (uint32_t)operation_count, operations, submission,
      mutation_started, accepted);
  amdf_free(reservation->host_allocator, operations);
  return status;
}
