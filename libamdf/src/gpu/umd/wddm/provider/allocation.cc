// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/provider/allocation.h"

#include <algorithm>
#include <array>
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

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/provider_allocator.h"
#include "libamdf/src/gpu/umd/wddm/provider/adapter_state.h"

namespace amdf::wddm_provider {

using amdf::dxg_provider::HostBuffer;
namespace {

constexpr uint64_t kGpuPageSize = 4ull * 1024;
constexpr uint64_t kMaximumNativeAllocationByteLength =
    2ull * 1024 * 1024 * 1024;

amdf_dxg_provider_result_t QueryAllocationCount(
    uint64_t byte_length, uint32_t* out_allocation_count) {
  if (byte_length == 0 || (byte_length & (kGpuPageSize - 1)) != 0) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  const uint64_t allocation_count =
      (byte_length - 1) / kMaximumNativeAllocationByteLength + 1;
  if (allocation_count > std::numeric_limits<uint32_t>::max()) {
    return AMDF_DXG_PROVIDER_RESULT_OUT_OF_RANGE;
  }
  *out_allocation_count = static_cast<uint32_t>(allocation_count);
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_private_allocation_domain_t ToPrivateAllocationDomain(
    amdf_dxg_provider_gpu_allocation_domain_t domain) {
  switch (domain) {
    case AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL:
      return AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL;
    case AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST:
      return AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST;
    default:
      return AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM;
  }
}

amdf_dxg_private_allocation_flags_t ToPrivateAllocationFlags(
    amdf_dxg_provider_gpu_allocation_flags_t flags) {
  amdf_dxg_private_allocation_flags_t private_flags = 0;
  if ((flags & AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_FINE_GRAIN) != 0) {
    private_flags |= AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN;
  }
  if ((flags & AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED) != 0) {
    private_flags |= AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED;
  }
  if ((flags & AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_QUEUE_STORAGE) != 0) {
    private_flags |= AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE;
  }
  return private_flags;
}

amdf_dxg_provider_result_t GpuAllocationCreateImpl(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    const amdf_dxg_provider_gpu_allocation_create_info_t& create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  uint32_t allocation_count = 0;
  amdf_dxg_provider_result_t result =
      QueryAllocationCount(create_info.byte_length, &allocation_count);
  if (result != AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
    return result;
  }
  if (allocation_handle_capacity < allocation_count) {
    *out_allocation_count = allocation_count;
    return AMDF_DXG_PROVIDER_RESULT_BUFFER_TOO_SMALL;
  }

  const size_t allocation_private_byte_length =
      static_cast<size_t>(allocation_count) *
      AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH;
  constexpr size_t kAllocationInfoAlignment =
      amdf_alignof(D3DDDI_ALLOCATIONINFO2);
  const size_t allocation_info_offset =
      (allocation_private_byte_length + kAllocationInfoAlignment - 1) &
      ~(kAllocationInfoAlignment - 1);
  const size_t allocation_state_byte_length =
      allocation_info_offset +
      static_cast<size_t>(allocation_count) * sizeof(D3DDDI_ALLOCATIONINFO2);
  HostBuffer allocation_state;
  if (!allocation_state.Allocate(adapter->host_allocator,
                                 allocation_state_byte_length,
                                 kAllocationInfoAlignment)) {
    return AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED;
  }
  auto* allocation_private_bytes =
      static_cast<uint8_t*>(allocation_state.data());
  auto* allocation_info_records = reinterpret_cast<D3DDDI_ALLOCATIONINFO2*>(
      allocation_private_bytes + allocation_info_offset);
  std::array<uint8_t, AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH>
      driver_private;
  amdf_dxg_private_initialize_allocation_driver_record(driver_private.data());

  uint64_t remaining_byte_length = create_info.byte_length;
  uint64_t byte_offset = 0;
  for (uint32_t i = 0; i < allocation_count; ++i) {
    const uint64_t chunk_byte_length =
        std::min(remaining_byte_length, kMaximumNativeAllocationByteLength);
    void* private_data = allocation_private_bytes +
                         AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH * i;
    const amdf_dxg_private_allocation_info_t private_info = {
        .byte_length = chunk_byte_length,
        .domain = ToPrivateAllocationDomain(create_info.domain),
        .flags = ToPrivateAllocationFlags(create_info.flags),
    };
    amdf_dxg_private_initialize_wddm_allocation_record(
        &private_info, &adapter->private_info, private_data);

    D3DDDI_ALLOCATIONINFO2& allocation_info = allocation_info_records[i];
    if (create_info.host_pointer != nullptr) {
      allocation_info.pSystemMem =
          static_cast<uint8_t*>(create_info.host_pointer) + byte_offset;
    }
    allocation_info.pPrivateDriverData = private_data;
    allocation_info.PrivateDriverDataSize =
        AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH;
    allocation_info.VidPnSourceId = D3DDDI_ID_UNINITIALIZED;
    remaining_byte_length -= chunk_byte_length;
    byte_offset += chunk_byte_length;
  }

  D3DKMT_CREATEALLOCATION create = {};
  create.hDevice = static_cast<D3DKMT_HANDLE>(create_info.device_handle);
  create.pPrivateDriverData = driver_private.data();
  create.PrivateDriverDataSize =
      AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH;
  create.NumAllocations = allocation_count;
  create.pAllocationInfo2 = allocation_info_records;
  const NTSTATUS native_status = D3DKMTCreateAllocation2(&create);
  if (native_status != STATUS_SUCCESS) {
    *out_native_status = static_cast<uint32_t>(native_status);
    return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  }
  // The backing owner receives every native result before validating it. This
  // provider cannot hide a native allocation whose cleanup still needs backing.
  for (uint32_t i = 0; i < allocation_count; ++i) {
    out_allocation_handles[i] = allocation_info_records[i].hAllocation;
  }
  *out_resource_handle = create.hResource;
  *out_allocation_count = allocation_count;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

}  // namespace

amdf_dxg_provider_result_t GpuAllocationQueryLayout(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint64_t byte_length,
    uint32_t* out_allocation_count,
    uint64_t* out_maximum_allocation_byte_length) noexcept {
  if (adapter == nullptr || out_allocation_count == nullptr ||
      out_maximum_allocation_byte_length == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  uint32_t allocation_count = 0;
  const amdf_dxg_provider_result_t result =
      QueryAllocationCount(byte_length, &allocation_count);
  if (result == AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
    *out_allocation_count = allocation_count;
    *out_maximum_allocation_byte_length = kMaximumNativeAllocationByteLength;
  }
  return result;
}

amdf_dxg_provider_result_t GpuAllocationCreate(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    const amdf_dxg_provider_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || create_info == nullptr ||
      create_info->device_handle == 0 ||
      (allocation_handle_capacity != 0 && out_allocation_handles == nullptr) ||
      out_resource_handle == nullptr || out_allocation_count == nullptr ||
      out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const amdf_dxg_provider_gpu_allocation_flags_t known_flags =
      AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_FINE_GRAIN |
      AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_QUEUE_STORAGE |
      AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED;
  const bool is_local =
      create_info->domain == AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL;
  const bool has_host_pointer = create_info->host_pointer != nullptr;
  const uintptr_t host_address =
      reinterpret_cast<uintptr_t>(create_info->host_pointer);
  if (create_info->domain < AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_SYSTEM ||
      create_info->domain >
          AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST ||
      (create_info->flags & ~known_flags) != 0 ||
      (is_local != (create_info->placement_device_address != 0)) ||
      (is_local && has_host_pointer) || (!is_local && !has_host_pointer) ||
      (!is_local && create_info->byte_length >
                        std::numeric_limits<uintptr_t>::max() - host_address) ||
      (is_local &&
       create_info->byte_length > std::numeric_limits<uint64_t>::max() -
                                      create_info->placement_device_address)) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }

  return GpuAllocationCreateImpl(
      adapter, *create_info, allocation_handle_capacity, out_allocation_handles,
      out_resource_handle, out_allocation_count, out_native_status);
}

}  // namespace amdf::wddm_provider
