// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/wsl/allocation.h"

#include <sys/ioctl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/provider_allocator.h"
#include "libamdf/src/gpu/umd/dxg/wsl/adapter_state.h"
#include "libamdf/src/platform/linux/dxg/kmt_types.h"

namespace amdf::wsl_provider {
namespace {

using amdf::dxg_provider::HostBuffer;

constexpr uint64_t kGpuPageSize = 4ull * 1024;
constexpr uint64_t kMaximumNativeAllocationByteLength =
    2ull * 1024 * 1024 * 1024;
constexpr uint32_t kCreateAllocationOperation = 0x06;

amdf_dxg_provider_result_t QueryAllocationCount(
    uint64_t byte_length, uint32_t* out_allocation_count) {
  if (byte_length == 0 || (byte_length & (kGpuPageSize - 1)) != 0) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  const uint64_t allocation_count =
      (byte_length - 1) / kMaximumNativeAllocationByteLength + 1;
  if (allocation_count >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
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
  // Queue storage remains in its ordinary physical domain. The native user
  // queue domain denotes a separately driver-owned persistent queue object.
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

  HostBuffer driver_private_data;
  HostBuffer allocation_private_data;
  HostBuffer allocation_infos;
  if (!driver_private_data.Allocate(
          adapter->host_allocator,
          AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH) ||
      !allocation_private_data.Allocate(
          adapter->host_allocator,
          static_cast<size_t>(allocation_count) *
              AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH) ||
      !allocation_infos.Allocate(adapter->host_allocator,
                                 static_cast<size_t>(allocation_count) *
                                     sizeof(D3DDDI_ALLOCATIONINFO2),
                                 amdf_alignof(D3DDDI_ALLOCATIONINFO2))) {
    return AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED;
  }
  amdf_dxg_private_initialize_allocation_driver_record(
      driver_private_data.data());
  auto* native_infos =
      static_cast<D3DDDI_ALLOCATIONINFO2*>(allocation_infos.data());

  uint64_t remaining_byte_length = create_info.byte_length;
  uint64_t byte_offset = 0;
  for (uint32_t i = 0; i < allocation_count; ++i) {
    const uint64_t chunk_byte_length =
        std::min(remaining_byte_length, kMaximumNativeAllocationByteLength);
    void* private_record =
        static_cast<uint8_t*>(allocation_private_data.data()) +
        static_cast<size_t>(i) * AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH;
    const amdf_dxg_private_allocation_info_t private_info = {
        .byte_length = chunk_byte_length,
        .domain = ToPrivateAllocationDomain(create_info.domain),
        .flags = ToPrivateAllocationFlags(create_info.flags),
    };
    amdf_dxg_private_initialize_wsl_allocation_record(
        &private_info, &adapter->private_info, private_record);

    D3DDDI_ALLOCATIONINFO2& native_info = native_infos[i];
    if (create_info.host_pointer != nullptr) {
      native_info.pSystemMem =
          static_cast<uint8_t*>(create_info.host_pointer) + byte_offset;
    }
    native_info.pPrivateDriverData = private_record;
    native_info.PrivateDriverDataSize =
        AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH;
    native_info.VidPnSourceId = D3DDDI_ID_UNINITIALIZED;
    remaining_byte_length -= chunk_byte_length;
    byte_offset += chunk_byte_length;
  }

  D3DKMT_CREATEALLOCATION create = {
      .hDevice = static_cast<D3DKMT_HANDLE>(create_info.device_handle),
      .pPrivateDriverData = driver_private_data.data(),
      .PrivateDriverDataSize =
          AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH,
      .NumAllocations = allocation_count,
      .pAllocationInfo2 = native_infos,
  };
  const NTSTATUS native_status =
      static_cast<NTSTATUS>(adapter->native_transport.invoke(
          adapter->native_transport.context,
          _IOWR('G', kCreateAllocationOperation, D3DKMT_CREATEALLOCATION),
          &create));
  if (native_status != 0) {
    *out_native_status = static_cast<uint32_t>(native_status);
    return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  }

  for (uint32_t i = 0; i < allocation_count; ++i) {
    out_allocation_handles[i] = native_infos[i].hAllocation;
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

}  // namespace amdf::wsl_provider
