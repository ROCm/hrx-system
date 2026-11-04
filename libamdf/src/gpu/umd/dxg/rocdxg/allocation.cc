// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/allocation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "libamdf/src/gpu/umd/dxg/bridge_allocator.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/native_allocation_flags.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/native_transport.h"
#include "shared/include/dxcore_loader.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace amdf::rocdxg_bridge {
namespace {

using amdf::dxg_bridge::HostBuffer;

constexpr uint64_t kGpuPageSize = 4ull * 1024;
constexpr uint64_t kMaximumNativeAllocationByteLength =
    2ull * 1024 * 1024 * 1024;

amdf_dxg_bridge_result_t QueryAllocationCount(uint64_t byte_length,
                                              uint32_t* out_allocation_count) {
  if (byte_length == 0 || (byte_length & (kGpuPageSize - 1)) != 0) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  const uint64_t allocation_count =
      (byte_length - 1) / kMaximumNativeAllocationByteLength + 1;
  if (allocation_count >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return AMDF_DXG_BRIDGE_RESULT_OUT_OF_RANGE;
  }
  *out_allocation_count = static_cast<uint32_t>(allocation_count);
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

thunk_proxy::AllocDomain ToNativeAllocationDomain(
    amdf_dxg_bridge_gpu_allocation_domain_t domain) {
  switch (domain) {
    case AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL:
      return thunk_proxy::kLocal;
    case AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST:
      return thunk_proxy::kUserMemory;
    default:
      return thunk_proxy::kSystem;
  }
}

amdf_dxg_bridge_result_t GpuAllocationCreateImpl(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_allocation_create_info_t& create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  uint32_t allocation_count = 0;
  amdf_dxg_bridge_result_t result =
      QueryAllocationCount(create_info.byte_length, &allocation_count);
  if (result != AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
    return result;
  }
  if (allocation_handle_capacity < allocation_count) {
    *out_allocation_count = allocation_count;
    return AMDF_DXG_BRIDGE_RESULT_BUFFER_TOO_SMALL;
  }

  thunk_proxy::AllocPrivData private_data =
      thunk_proxy::MakeAllocPrivData(static_cast<int>(allocation_count));
  if (private_data.drv_data_size <= 0 || private_data.per_alloc_size <= 0) {
    return AMDF_DXG_BRIDGE_RESULT_VERSION_MISMATCH;
  }
  HostBuffer allocation_infos;
  if (!allocation_infos.Allocate(adapter->host_allocator,
                                 static_cast<size_t>(allocation_count) *
                                     sizeof(D3DDDI_ALLOCATIONINFO2),
                                 amdf_alignof(D3DDDI_ALLOCATIONINFO2))) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  auto* native_infos =
      static_cast<D3DDDI_ALLOCATIONINFO2*>(allocation_infos.data());

  uint64_t remaining_byte_length = create_info.byte_length;
  uint64_t byte_offset = 0;
  for (uint32_t i = 0; i < allocation_count; ++i) {
    const uint64_t chunk_byte_length =
        std::min(remaining_byte_length, kMaximumNativeAllocationByteLength);
    void* allocation_private_data =
        private_data.alloc_data_at(static_cast<int>(i));
    const uint64_t placement_device_address =
        create_info.domain == AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
            ? create_info.placement_device_address + byte_offset
            : 0;
    thunk_proxy::SetAllocationInfo(
        allocation_private_data, chunk_byte_length,
        ToNativeAllocationDomain(create_info.domain), placement_device_address,
        EncodeNativeAllocationFlags(create_info.flags), thunk_proxy::KCOMPUTE0,
        *adapter->device);

    D3DDDI_ALLOCATIONINFO2& native_info = native_infos[i];
    if (create_info.host_pointer != nullptr) {
      native_info.pSystemMem =
          static_cast<uint8_t*>(create_info.host_pointer) + byte_offset;
    }
    native_info.pPrivateDriverData = allocation_private_data;
    native_info.PrivateDriverDataSize = private_data.per_alloc_size;
    native_info.VidPnSourceId = D3DDDI_ID_UNINITIALIZED;
    remaining_byte_length -= chunk_byte_length;
    byte_offset += chunk_byte_length;
  }

  D3DKMT_CREATEALLOCATION create = {
      .hDevice = static_cast<D3DKMT_HANDLE>(create_info.device_handle),
      .pPrivateDriverData = private_data.drv_data(),
      .PrivateDriverDataSize =
          static_cast<uint32_t>(private_data.drv_data_size),
      .NumAllocations = allocation_count,
      .pAllocationInfo2 = native_infos,
  };
  NativeTransportScope transport_scope(adapter->native_transport);
  const NTSTATUS native_status =
      wsl::thunk::dxcore::DxcoreLoader::Instance().pfn_D3DKMTCreateAllocation2(
          &create);
  if (native_status != 0) {
    *out_native_status = static_cast<uint32_t>(native_status);
    return AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE;
  }

  for (uint32_t i = 0; i < allocation_count; ++i) {
    out_allocation_handles[i] = native_infos[i].hAllocation;
  }
  *out_resource_handle = create.hResource;
  *out_allocation_count = allocation_count;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

}  // namespace

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuAllocationQueryLayout(
    amdf_dxg_bridge_gpu_adapter_t* adapter, uint64_t byte_length,
    uint32_t* out_allocation_count,
    uint64_t* out_maximum_allocation_byte_length) noexcept {
  if (adapter == nullptr || out_allocation_count == nullptr ||
      out_maximum_allocation_byte_length == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  uint32_t allocation_count = 0;
  const amdf_dxg_bridge_result_t result =
      QueryAllocationCount(byte_length, &allocation_count);
  if (result == AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
    *out_allocation_count = allocation_count;
    *out_maximum_allocation_byte_length = kMaximumNativeAllocationByteLength;
  }
  return result;
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuAllocationCreate(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || create_info == nullptr ||
      create_info->structure_size < sizeof(*create_info) ||
      create_info->device_handle == 0 ||
      (allocation_handle_capacity != 0 && out_allocation_handles == nullptr) ||
      out_resource_handle == nullptr || out_allocation_count == nullptr ||
      out_native_status == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const amdf_dxg_bridge_gpu_allocation_flags_t known_flags =
      AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN |
      AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE |
      AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED;
  const bool is_local =
      create_info->domain == AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL;
  const bool has_host_pointer = create_info->host_pointer != nullptr;
  const uintptr_t host_address =
      reinterpret_cast<uintptr_t>(create_info->host_pointer);
  if (create_info->domain < AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM ||
      create_info->domain >
          AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST ||
      (create_info->flags & ~known_flags) != 0 ||
      (is_local != (create_info->placement_device_address != 0)) ||
      (is_local && has_host_pointer) || (!is_local && !has_host_pointer) ||
      (!is_local && create_info->byte_length >
                        std::numeric_limits<uintptr_t>::max() - host_address) ||
      (is_local &&
       create_info->byte_length > std::numeric_limits<uint64_t>::max() -
                                      create_info->placement_device_address)) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }

  try {
    return GpuAllocationCreateImpl(adapter, *create_info,
                                   allocation_handle_capacity,
                                   out_allocation_handles, out_resource_handle,
                                   out_allocation_count, out_native_status);
  } catch (const std::bad_alloc&) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  } catch (...) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
}

}  // namespace amdf::rocdxg_bridge
