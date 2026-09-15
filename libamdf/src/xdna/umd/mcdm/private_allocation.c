// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/private_allocation.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/platform/windows/host_cache.h"

#define AMDF_WINDOWS_KMT_PAGE_SIZE UINT64_C(4096)

// Native allocation-private wire record.
typedef struct amdf_windows_xdna_private_allocation_wire_t {
  // Unresolved field fixed at zero.
  uint64_t reserved_0000;
  // Logical byte length requested by the caller.
  uint64_t requested_byte_length;
  // Page-aligned physical allocation length.
  uint64_t allocation_byte_length;
  // Additional private allocation selector.
  uint32_t selector;
  // Private allocation role discriminator.
  uint32_t type;
  // Driver placement policy.
  uint32_t policy;
  // Unresolved field fixed at zero.
  uint32_t reserved_0024;
  // Driver command-aperture and allocation flags.
  uint32_t xcl_flags;
  // Unresolved field fixed at zero.
  uint32_t reserved_002c;
  // Context-qualified firmware base returned for an instruction aperture.
  uint64_t firmware_address;
} amdf_windows_xdna_private_allocation_wire_t;

_Static_assert(sizeof(amdf_windows_xdna_private_allocation_wire_t) == 56,
               "XDNA private allocation record must match the installed ABI");
_Static_assert(offsetof(amdf_windows_xdna_private_allocation_wire_t,
                        firmware_address) == 0x30,
               "XDNA firmware address must match the native reply offset");

static amdf_status_t amdf_windows_xdna_private_allocation_wait_for_paging(
    amdf_windows_xdna_private_allocation_t* allocation) {
  return amdf_kmt_wait_for_paging(
      allocation->device->kmt, allocation->device->device,
      allocation->device->paging_sync_object, allocation->device->paging_fence,
      allocation->pending_paging_fence);
}

static void amdf_windows_xdna_private_allocation_clear_pending_paging(
    amdf_windows_xdna_private_allocation_t* allocation) {
  allocation->pending_paging_fence = 0;
  allocation->pending_device_address = 0;
  allocation->paging_operation_pending = 0;
  allocation->paging_operation_valid = 0;
}

static amdf_status_t amdf_windows_xdna_private_allocation_map(
    amdf_windows_xdna_private_allocation_t* allocation) {
  if (allocation->paging_operation_pending == 0) {
    D3DDDI_MAPGPUVIRTUALADDRESS map = {0};
    map.hPagingQueue = allocation->device->paging_queue;
    map.hAllocation = allocation->allocation;
    map.SizeInPages = allocation->descriptor.allocation_byte_length /
                      AMDF_WINDOWS_KMT_PAGE_SIZE;
    map.Protection.Write = 1;
    const NTSTATUS native_status =
        allocation->device->kmt->map_gpu_virtual_address(&map);
    if (!amdf_kmt_status_is_success_or_pending(native_status)) {
      return amdf_kmt_make_status(native_status);
    }
    allocation->pending_paging_fence = map.PagingFenceValue;
    allocation->pending_device_address = map.VirtualAddress;
    allocation->paging_operation_pending = 1;
    allocation->paging_operation_valid = map.VirtualAddress != 0;
  }
  if (allocation->paging_operation_valid == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const amdf_status_t status =
      amdf_windows_xdna_private_allocation_wait_for_paging(allocation);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  allocation->device_address = allocation->pending_device_address;
  amdf_windows_xdna_private_allocation_clear_pending_paging(allocation);
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_xdna_private_allocation_make_resident(
    amdf_windows_xdna_private_allocation_t* allocation) {
  if (allocation->paging_operation_pending == 0) {
    const D3DKMT_HANDLE allocation_handle = allocation->allocation;
    D3DDDI_MAKERESIDENT make_resident = {0};
    make_resident.hPagingQueue = allocation->device->paging_queue;
    make_resident.NumAllocations = 1;
    make_resident.AllocationList = &allocation_handle;
    const NTSTATUS native_status =
        allocation->device->kmt->make_resident(&make_resident);
    if (!amdf_kmt_status_is_success_or_pending(native_status)) {
      return amdf_kmt_make_status(native_status);
    }
    allocation->pending_paging_fence = make_resident.PagingFenceValue;
    allocation->paging_operation_pending = 1;
    allocation->paging_operation_valid = make_resident.NumAllocations == 1;
    if (allocation->paging_operation_valid == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  const amdf_status_t status =
      amdf_windows_xdna_private_allocation_wait_for_paging(allocation);
  if (amdf_status_is_ok(status)) {
    amdf_windows_xdna_private_allocation_clear_pending_paging(allocation);
  }
  return status;
}

void amdf_windows_xdna_private_allocation_initialize(
    amdf_xdna_umd_device_t* device,
    const amdf_windows_xdna_private_allocation_descriptor_t* descriptor,
    amdf_windows_xdna_private_allocation_t* allocation) {
  memset(allocation, 0, sizeof(*allocation));
  allocation->device = device;
  allocation->descriptor = *descriptor;
}

amdf_status_t amdf_windows_xdna_private_allocation_realize(
    amdf_windows_xdna_private_allocation_t* allocation) {
  const amdf_windows_xdna_private_allocation_descriptor_t* descriptor =
      &allocation->descriptor;

  amdf_windows_xdna_private_allocation_wire_t private_data = {0};
  private_data.requested_byte_length = descriptor->requested_byte_length;
  private_data.allocation_byte_length = descriptor->allocation_byte_length;
  private_data.selector = descriptor->selector;
  private_data.type = descriptor->type;
  private_data.policy = descriptor->policy;
  private_data.xcl_flags = descriptor->xcl_flags;

  amdf_status_t status = AMDF_STATUS_OK;
  if (allocation->realization_phase == 0) {
    D3DDDI_ALLOCATIONINFO2 allocation_info = {0};
    allocation_info.pPrivateDriverData = &private_data;
    allocation_info.PrivateDriverDataSize = sizeof(private_data);
    D3DKMT_CREATEALLOCATION create = {0};
    create.hDevice = allocation->device->device;
    create.NumAllocations = 1;
    create.pAllocationInfo2 = &allocation_info;
    if ((descriptor->flags &
         AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_SHARED_RESOURCE) != 0) {
      create.Flags.CreateResource = 1;
      create.Flags.CreateShared = 1;
    }
    status = amdf_kmt_make_status(
        allocation->device->kmt->create_allocation(&create));
    if (amdf_status_is_ok(status)) {
      allocation->resource = create.hResource;
      allocation->allocation = allocation_info.hAllocation;
      allocation->firmware_address = private_data.firmware_address;
      allocation->realization_phase = 1;
      if (allocation->allocation == 0) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      }
    }
  }
  if (amdf_status_is_ok(status) && allocation->realization_phase != 0 &&
      allocation->allocation == 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (amdf_status_is_ok(status) &&
      (descriptor->flags &
       AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS) != 0 &&
      allocation->realization_phase == 1) {
    status = amdf_windows_xdna_private_allocation_map(allocation);
    if (amdf_status_is_ok(status)) {
      allocation->realization_phase = 2;
    }
  }
  if (amdf_status_is_ok(status) &&
      (descriptor->flags &
       AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS) != 0 &&
      allocation->realization_phase == 2) {
    status = amdf_windows_xdna_private_allocation_make_resident(allocation);
    if (amdf_status_is_ok(status)) {
      allocation->realization_phase = 3;
    }
  }
  if (amdf_status_is_ok(status) &&
      (descriptor->flags &
       AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS) == 0 &&
      allocation->realization_phase == 1) {
    allocation->realization_phase = 3;
  }
  return status;
}

amdf_status_t amdf_windows_xdna_private_allocation_lock(
    amdf_windows_xdna_private_allocation_t* allocation) {
  if (allocation == NULL || allocation->allocation == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (allocation->host_lock_owned != 0) {
    return allocation->host_pointer != NULL
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  D3DKMT_LOCK2 lock = {0};
  lock.hDevice = allocation->device->device;
  lock.hAllocation = allocation->allocation;
  const amdf_status_t status =
      amdf_kmt_make_status(allocation->device->kmt->lock(&lock));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  allocation->host_pointer = lock.pData;
  allocation->host_lock_owned = 1;
  return lock.pData != NULL ? AMDF_STATUS_OK
                            : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
}

amdf_status_t amdf_windows_xdna_private_allocation_publish(
    amdf_windows_xdna_private_allocation_t* allocation, uint64_t byte_offset,
    uint64_t byte_length) {
  if (allocation == NULL || allocation->host_pointer == NULL ||
      byte_offset > allocation->descriptor.allocation_byte_length ||
      byte_length >
          allocation->descriptor.allocation_byte_length - byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_windows_host_cache_control(
      AMDF_HOST_CACHE_OPERATION_FLUSH,
      (uint8_t*)allocation->host_pointer + byte_offset, byte_length);
  if (!amdf_status_is_ok(status) || byte_length == 0) {
    return status;
  }
  D3DKMT_INVALIDATECACHE invalidate = {0};
  invalidate.hDevice = allocation->device->device;
  invalidate.hAllocation = allocation->allocation;
  invalidate.Offset = byte_offset;
  invalidate.Length = byte_length;
  return amdf_kmt_make_status(
      allocation->device->kmt->invalidate_cache(&invalidate));
}

amdf_status_t amdf_windows_xdna_private_allocation_destroy(
    amdf_windows_xdna_private_allocation_t* allocation) {
  if (allocation == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (allocation->paging_operation_pending != 0) {
    const amdf_status_t status =
        amdf_windows_xdna_private_allocation_wait_for_paging(allocation);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    amdf_windows_xdna_private_allocation_clear_pending_paging(allocation);
  }
  if (allocation->host_lock_owned != 0) {
    D3DKMT_UNLOCK2 unlock = {0};
    unlock.hDevice = allocation->device->device;
    unlock.hAllocation = allocation->allocation;
    const amdf_status_t status =
        amdf_kmt_make_status(allocation->device->kmt->unlock(&unlock));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    allocation->host_pointer = NULL;
    allocation->host_lock_owned = 0;
  }
  if (allocation->resource != 0 || allocation->allocation != 0) {
    D3DKMT_DESTROYALLOCATION2 destroy = {0};
    destroy.hDevice = allocation->device->device;
    destroy.hResource = allocation->resource;
    if (allocation->resource == 0) {
      destroy.phAllocationList = &allocation->allocation;
      destroy.AllocationCount = 1;
    }
    destroy.Flags.AssumeNotInUse = 1;
    const amdf_status_t status = amdf_kmt_make_status(
        allocation->device->kmt->destroy_allocation(&destroy));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    allocation->resource = 0;
    allocation->allocation = 0;
  }
  allocation->device_address = 0;
  allocation->firmware_address = 0;
  memset(&allocation->descriptor, 0, sizeof(allocation->descriptor));
  allocation->realization_phase = 0;
  allocation->device = NULL;
  return AMDF_STATUS_OK;
}
