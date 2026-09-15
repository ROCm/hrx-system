// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/allocation.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_allocator.h"
#include "wkmi.h"

namespace amdf::wkmi_bridge {
namespace {

constexpr uint64_t kGpuPageSize = 4ull * 1024;
constexpr uint64_t kMaximumNativeAllocationByteLength =
    2ull * 1024 * 1024 * 1024;

amdf_wkmi_bridge_result_t QueryAllocationCount(uint64_t byte_length,
                                               uint32_t* out_allocation_count) {
  if (byte_length == 0 || (byte_length & (kGpuPageSize - 1)) != 0) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  const uint64_t allocation_count =
      (byte_length - 1) / kMaximumNativeAllocationByteLength + 1;
  if (allocation_count > std::numeric_limits<uint32_t>::max()) {
    return AMDF_WKMI_BRIDGE_RESULT_OUT_OF_RANGE;
  }
  *out_allocation_count = static_cast<uint32_t>(allocation_count);
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

Wkmi::AllocDomain ToWkmiAllocationDomain(
    amdf_wkmi_bridge_gpu_allocation_domain_t domain) {
  switch (domain) {
    case AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL:
      return Wkmi::kLocal;
    case AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST:
      return Wkmi::kUserMemory;
    default:
      return Wkmi::kSystem;
  }
}

uint32_t ToWkmiAllocationFlags(amdf_wkmi_bridge_gpu_allocation_flags_t flags) {
  uint32_t wkmi_flags = 0;
  if ((flags & AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN) != 0) {
    wkmi_flags |= Wkmi::kFineGrain;
  }
  if ((flags & AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE) != 0) {
    wkmi_flags |= Wkmi::kQueueObject;
  }
  return wkmi_flags;
}

amdf_wkmi_bridge_result_t GpuAllocationCreateImpl(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_allocation_create_info_t& create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  uint32_t allocation_count = 0;
  amdf_wkmi_bridge_result_t result =
      QueryAllocationCount(create_info.byte_length, &allocation_count);
  if (result != AMDF_WKMI_BRIDGE_RESULT_SUCCESS) {
    return result;
  }
  if (allocation_handle_capacity < allocation_count) {
    *out_allocation_count = allocation_count;
    return AMDF_WKMI_BRIDGE_RESULT_BUFFER_TOO_SMALL;
  }

  int driver_private_size = 0;
  int allocation_private_size = 0;
  Wkmi::GetAllocPrivDataSize(&driver_private_size, &allocation_private_size);
  if (driver_private_size <= 0 || allocation_private_size <= 0) {
    return AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH;
  }
  if (static_cast<uint64_t>(allocation_private_size) >
      std::numeric_limits<size_t>::max() / allocation_count) {
    return AMDF_WKMI_BRIDGE_RESULT_OUT_OF_RANGE;
  }

  const size_t allocation_private_byte_length =
      static_cast<size_t>(allocation_private_size) * allocation_count;
  const size_t allocation_info_byte_length =
      static_cast<size_t>(allocation_count) * sizeof(D3DDDI_ALLOCATIONINFO2);
  HostBuffer driver_private;
  HostBuffer allocation_private;
  HostBuffer allocation_infos;
  if (!driver_private.Allocate(adapter->host_allocator,
                               static_cast<size_t>(driver_private_size)) ||
      !allocation_private.Allocate(adapter->host_allocator,
                                   allocation_private_byte_length) ||
      !allocation_infos.Allocate(adapter->host_allocator,
                                 allocation_info_byte_length,
                                 amdf_alignof(D3DDDI_ALLOCATIONINFO2))) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  auto* allocation_private_bytes =
      static_cast<uint8_t*>(allocation_private.data());
  auto* allocation_info_records =
      static_cast<D3DDDI_ALLOCATIONINFO2*>(allocation_infos.data());
  Wkmi::FillinAllocPrivDrvData(driver_private.data(), allocation_private_size);

  uint64_t remaining_byte_length = create_info.byte_length;
  uint64_t byte_offset = 0;
  for (uint32_t i = 0; i < allocation_count; ++i) {
    const uint64_t chunk_byte_length =
        std::min(remaining_byte_length, kMaximumNativeAllocationByteLength);
    void* private_data = allocation_private_bytes +
                         static_cast<size_t>(allocation_private_size) * i;
    const uint64_t placement_device_address =
        create_info.domain == AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
            ? create_info.placement_device_address + byte_offset
            : 0;
    Wkmi::SetAllocationInfo(private_data, chunk_byte_length,
                            ToWkmiAllocationDomain(create_info.domain),
                            placement_device_address,
                            ToWkmiAllocationFlags(create_info.flags),
                            Wkmi::KCOMPUTE0, adapter->device_info);

    D3DDDI_ALLOCATIONINFO2& allocation_info = allocation_info_records[i];
    if (create_info.host_pointer != nullptr) {
      allocation_info.pSystemMem =
          static_cast<uint8_t*>(create_info.host_pointer) + byte_offset;
    }
    allocation_info.pPrivateDriverData = private_data;
    allocation_info.PrivateDriverDataSize = allocation_private_size;
    allocation_info.VidPnSourceId = D3DDDI_ID_UNINITIALIZED;
    remaining_byte_length -= chunk_byte_length;
    byte_offset += chunk_byte_length;
  }

  D3DKMT_CREATEALLOCATION create = {};
  create.hDevice = static_cast<D3DKMT_HANDLE>(create_info.device_handle);
  create.pPrivateDriverData = driver_private.data();
  create.PrivateDriverDataSize = driver_private_size;
  create.NumAllocations = allocation_count;
  create.pAllocationInfo2 = allocation_info_records;
  const NTSTATUS native_status = D3DKMTCreateAllocation2(&create);
  if (native_status != STATUS_SUCCESS) {
    *out_native_status = static_cast<uint32_t>(native_status);
    return AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE;
  }
  // The backing owner receives every native result before validating it. This
  // bridge cannot hide a native allocation whose cleanup still needs backing.
  for (uint32_t i = 0; i < allocation_count; ++i) {
    out_allocation_handles[i] = allocation_info_records[i].hAllocation;
  }
  *out_resource_handle = create.hResource;
  *out_allocation_count = allocation_count;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

}  // namespace

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuAllocationQueryLayout(
    amdf_wkmi_bridge_gpu_adapter_t* adapter, uint64_t byte_length,
    uint32_t* out_allocation_count,
    uint64_t* out_maximum_allocation_byte_length) noexcept {
  if (adapter == nullptr || out_allocation_count == nullptr ||
      out_maximum_allocation_byte_length == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  uint32_t allocation_count = 0;
  const amdf_wkmi_bridge_result_t result =
      QueryAllocationCount(byte_length, &allocation_count);
  if (result == AMDF_WKMI_BRIDGE_RESULT_SUCCESS) {
    *out_allocation_count = allocation_count;
    *out_maximum_allocation_byte_length = kMaximumNativeAllocationByteLength;
  }
  return result;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuAllocationCreate(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || create_info == nullptr ||
      create_info->structure_size < sizeof(*create_info) ||
      create_info->device_handle == 0 ||
      (allocation_handle_capacity != 0 && out_allocation_handles == nullptr) ||
      out_resource_handle == nullptr || out_allocation_count == nullptr ||
      out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const amdf_wkmi_bridge_gpu_allocation_flags_t known_flags =
      AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN |
      AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE;
  const bool is_local =
      create_info->domain == AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL;
  const bool has_host_pointer = create_info->host_pointer != nullptr;
  const uintptr_t host_address =
      reinterpret_cast<uintptr_t>(create_info->host_pointer);
  if (create_info->domain < AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM ||
      create_info->domain >
          AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST ||
      (create_info->flags & ~known_flags) != 0 ||
      (is_local != (create_info->placement_device_address != 0)) ||
      (is_local && has_host_pointer) || (!is_local && !has_host_pointer) ||
      (!is_local && create_info->byte_length >
                        std::numeric_limits<uintptr_t>::max() - host_address) ||
      (is_local &&
       create_info->byte_length > std::numeric_limits<uint64_t>::max() -
                                      create_info->placement_device_address)) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }

  try {
    return GpuAllocationCreateImpl(adapter, *create_info,
                                   allocation_handle_capacity,
                                   out_allocation_handles, out_resource_handle,
                                   out_allocation_count, out_native_status);
  } catch (const std::bad_alloc&) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  } catch (...) {
    return AMDF_WKMI_BRIDGE_RESULT_INTERNAL;
  }
}

}  // namespace amdf::wkmi_bridge
