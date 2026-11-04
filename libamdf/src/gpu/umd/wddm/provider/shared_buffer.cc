// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/provider/shared_buffer.h"

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

#include <d3d12.h>
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <ntstatus.h>
#include <wrl/client.h>

#include <limits>

#include "libamdf/src/gpu/umd/dxg/bridge_allocator.h"
#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/wddm/provider/adapter_state.h"

namespace amdf::wddm_provider {

using amdf::dxg_bridge::HostBuffer;
namespace {

amdf_status_t HresultStatus(HRESULT result) {
  return SUCCEEDED(result) ? AMDF_STATUS_OK
                           : amdf_make_status(AMDF_STATUS_DOMAIN_HRESULT,
                                              static_cast<uint32_t>(result));
}

amdf_status_t QueryBuffer(uint64_t adapter_luid, HANDLE shared_handle,
                          uint64_t* out_byte_length) {
  Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
  HRESULT result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
  if (FAILED(result)) {
    return HresultStatus(result);
  }
  const LUID luid = {static_cast<DWORD>(adapter_luid),
                     static_cast<LONG>(adapter_luid >> 32)};
  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  result = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
  if (FAILED(result)) {
    return HresultStatus(result);
  }
  Microsoft::WRL::ComPtr<ID3D12Device> device;
  result = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                             IID_PPV_ARGS(&device));
  if (FAILED(result)) {
    return HresultStatus(result);
  }
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  result = device->OpenSharedHandle(shared_handle, IID_PPV_ARGS(&resource));
  if (FAILED(result)) {
    return HresultStatus(result);
  }
  const D3D12_RESOURCE_DESC description = resource->GetDesc();
  D3D12_HEAP_PROPERTIES heap = {};
  D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
  result = resource->GetHeapProperties(&heap, &flags);
  if (FAILED(result)) {
    return HresultStatus(result);
  }
  if (description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
      description.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR ||
      description.Width == 0 || heap.Type != D3D12_HEAP_TYPE_DEFAULT ||
      (flags & D3D12_HEAP_FLAG_SHARED) == 0 ||
      (flags & (D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                D3D12_HEAP_FLAG_HARDWARE_PROTECTED)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_byte_length = description.Width;
  return AMDF_STATUS_OK;
}

bool TryAppendRegion(size_t byte_length, size_t* inout_total_byte_length,
                     size_t* out_offset) {
  constexpr size_t kAlignment = amdf_max_align_t;
  const size_t total_byte_length = *inout_total_byte_length;
  if (total_byte_length >
      std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
    return false;
  }
  const size_t offset =
      (total_byte_length + kAlignment - 1) & ~(kAlignment - 1);
  if (byte_length > std::numeric_limits<size_t>::max() - offset) {
    return false;
  }
  *out_offset = offset;
  *inout_total_byte_length = offset + byte_length;
  return true;
}

void* RegionPointer(void* storage, size_t offset, size_t byte_length) {
  return byte_length == 0 ? nullptr : static_cast<uint8_t*>(storage) + offset;
}

amdf_status_t PrepareBuffer(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_buffer_import_info_t& import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length, uint64_t* out_buffer_byte_length) {
  uint64_t buffer_byte_length = 0;
  amdf_status_t status = QueryBuffer(
      import_info.adapter_luid, import_info.shared_handle, &buffer_byte_length);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query = {};
  query.hDevice = import_info.device_handle;
  query.hNtHandle = import_info.shared_handle;
  NTSTATUS native_status = D3DKMTQueryResourceInfoFromNtHandle(&query);
  if (native_status != STATUS_SUCCESS) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                            static_cast<uint32_t>(native_status));
  }
  if (query.NumAllocations != 1 || query.TotalPrivateDriverDataSize == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  constexpr size_t kProxyByteLength =
      AMDF_DXG_PRIVATE_PROXY_RESOURCE_RECORD_BYTE_LENGTH;
  if (kProxyByteLength >
      std::numeric_limits<size_t>::max() - query.TotalPrivateDriverDataSize) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  size_t scratch_byte_length = 0;
  size_t runtime_data_offset = 0;
  size_t driver_data_offset = 0;
  size_t resource_data_offset = 0;
  if (!TryAppendRegion(query.PrivateRuntimeDataSize, &scratch_byte_length,
                       &runtime_data_offset) ||
      !TryAppendRegion(query.TotalPrivateDriverDataSize + kProxyByteLength,
                       &scratch_byte_length, &driver_data_offset) ||
      !TryAppendRegion(query.ResourcePrivateDriverDataSize,
                       &scratch_byte_length, &resource_data_offset)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  HostBuffer scratch;
  if (!scratch.Allocate(adapter->host_allocator, scratch_byte_length)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  D3DDDI_OPENALLOCATIONINFO2 allocation = {};
  D3DKMT_OPENRESOURCEFROMNTHANDLE open = {};
  open.hDevice = import_info.device_handle;
  open.hNtHandle = import_info.shared_handle;
  open.NumAllocations = 1;
  open.pOpenAllocationInfo2 = &allocation;
  open.PrivateRuntimeDataSize = query.PrivateRuntimeDataSize;
  open.pPrivateRuntimeData = RegionPointer(scratch.data(), runtime_data_offset,
                                           query.PrivateRuntimeDataSize);
  open.TotalPrivateDriverDataBufferSize = query.TotalPrivateDriverDataSize;
  open.pTotalPrivateDriverDataBuffer =
      RegionPointer(scratch.data(), driver_data_offset,
                    query.TotalPrivateDriverDataSize + kProxyByteLength);
  open.ResourcePrivateDriverDataSize = query.ResourcePrivateDriverDataSize;
  open.pResourcePrivateDriverData =
      RegionPointer(scratch.data(), resource_data_offset,
                    query.ResourcePrivateDriverDataSize);
  native_status = D3DKMTOpenResourceFromNtHandle(&open);
  if (native_status != STATUS_SUCCESS) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                            static_cast<uint32_t>(native_status));
  }
  // Transfer native ownership before examining fallible driver metadata.
  *resource_handle = open.hResource;
  *allocation_handle = allocation.hAllocation;
  if (open.hResource == 0 || allocation.hAllocation == 0 ||
      allocation.pPrivateDriverData == nullptr ||
      allocation.PrivateDriverDataSize == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  amdf_dxg_private_allocation_metadata_t metadata = {};
  if (!amdf_dxg_private_decode_allocation_metadata(
          allocation.pPrivateDriverData, allocation.PrivateDriverDataSize,
          &metadata)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  // ADDR_SW_LINEAR and ADDR_SW_LINEAR_GENERAL are the addrlib byte layouts.
  if ((metadata.swizzle_mode != 0 && metadata.swizzle_mode != 32) ||
      metadata.tile_swizzle != 0 || metadata.compression_mode != 0 ||
      metadata.byte_length < buffer_byte_length ||
      (metadata.byte_length & 4095) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_native_byte_length = metadata.byte_length;
  *out_buffer_byte_length = buffer_byte_length;
  return AMDF_STATUS_OK;
}

}  // namespace

amdf_status_t AMDF_DXG_BRIDGE_CALL GpuBufferPrepareImport(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_buffer_import_info_t* import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length,
    uint64_t* out_buffer_byte_length) noexcept {
  return PrepareBuffer(adapter, *import_info, resource_handle,
                       allocation_handle, out_native_byte_length,
                       out_buffer_byte_length);
}

}  // namespace amdf::wddm_provider
