// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/shared_memory.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_allocator.h"

namespace amdf::wkmi_bridge {
namespace {

amdf_status_t HresultStatus(HRESULT result) {
  return SUCCEEDED(result) ? AMDF_STATUS_OK
                           : amdf_make_status(AMDF_STATUS_DOMAIN_HRESULT,
                                              static_cast<uint32_t>(result));
}

amdf_status_t QueryMemoryWithModules(
    HMODULE dxgi_module, HMODULE d3d12_module, uint64_t adapter_luid,
    HANDLE shared_handle, amdf_external_memory_type_t external_memory_type,
    uint64_t* out_byte_length) {
  using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
  const auto create_factory = reinterpret_cast<CreateFactory>(
      GetProcAddress(dxgi_module, "CreateDXGIFactory1"));
  if (create_factory == nullptr) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  const auto create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
      GetProcAddress(d3d12_module, "D3D12CreateDevice"));
  if (create_device == nullptr) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
  HRESULT result = create_factory(IID_PPV_ARGS(&factory));
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
  result = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                         IID_PPV_ARGS(&device));
  if (FAILED(result)) {
    return HresultStatus(result);
  }
  D3D12_HEAP_PROPERTIES properties = {};
  D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
  uint64_t byte_length = 0;
  if (external_memory_type == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP) {
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;
    result = device->OpenSharedHandle(shared_handle, IID_PPV_ARGS(&heap));
    if (FAILED(result)) {
      return HresultStatus(result);
    }
    const D3D12_HEAP_DESC description = heap->GetDesc();
    properties = description.Properties;
    flags = description.Flags;
    byte_length = description.SizeInBytes;
  } else {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    result = device->OpenSharedHandle(shared_handle, IID_PPV_ARGS(&resource));
    if (FAILED(result)) {
      return HresultStatus(result);
    }
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        description.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    result = resource->GetHeapProperties(&properties, &flags);
    if (FAILED(result)) {
      return HresultStatus(result);
    }
    byte_length = description.Width;
  }
  if (byte_length == 0 || properties.Type != D3D12_HEAP_TYPE_DEFAULT ||
      (flags & D3D12_HEAP_FLAG_SHARED) == 0 ||
      (flags & (D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                D3D12_HEAP_FLAG_HARDWARE_PROTECTED)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_byte_length = byte_length;
  return AMDF_STATUS_OK;
}

// Typed admission is cold and owns its temporary API modules and objects.
// Native execution retains only KMT memory state, never a D3D12 device.
amdf_status_t QueryMemory(uint64_t adapter_luid, HANDLE shared_handle,
                          amdf_external_memory_type_t external_memory_type,
                          uint64_t* out_byte_length) {
  HMODULE dxgi_module =
      LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (dxgi_module == nullptr) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  HMODULE d3d12_module =
      LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  uint64_t byte_length = 0;
  amdf_status_t status =
      d3d12_module == nullptr
          ? amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError())
          : QueryMemoryWithModules(dxgi_module, d3d12_module, adapter_luid,
                                   shared_handle, external_memory_type,
                                   &byte_length);
  if (d3d12_module != nullptr && !FreeLibrary(d3d12_module)) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  if (!FreeLibrary(dxgi_module)) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  if (amdf_status_is_ok(status)) {
    *out_byte_length = byte_length;
  }
  return status;
}

amdf_status_t PrepareMemory(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_memory_import_info_t& import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length, uint64_t* out_logical_byte_length) {
  uint64_t logical_byte_length = 0;
  amdf_status_t status =
      QueryMemory(import_info.adapter_luid, import_info.shared_handle,
                  import_info.external_memory_type, &logical_byte_length);
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
  const size_t proxy_byte_length = Wkmi::GetProxyResourceInfoSize();
  if (proxy_byte_length > SIZE_MAX - query.TotalPrivateDriverDataSize) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  HostBuffer runtime_data;
  HostBuffer driver_data;
  HostBuffer resource_data;
  if ((query.PrivateRuntimeDataSize != 0 &&
       !runtime_data.Allocate(adapter->host_allocator,
                              query.PrivateRuntimeDataSize)) ||
      !driver_data.Allocate(
          adapter->host_allocator,
          query.TotalPrivateDriverDataSize + proxy_byte_length) ||
      (query.ResourcePrivateDriverDataSize != 0 &&
       !resource_data.Allocate(adapter->host_allocator,
                               query.ResourcePrivateDriverDataSize))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  D3DDDI_OPENALLOCATIONINFO2 allocation = {};
  D3DKMT_OPENRESOURCEFROMNTHANDLE open = {};
  open.hDevice = import_info.device_handle;
  open.hNtHandle = import_info.shared_handle;
  open.NumAllocations = 1;
  open.pOpenAllocationInfo2 = &allocation;
  open.PrivateRuntimeDataSize = query.PrivateRuntimeDataSize;
  open.pPrivateRuntimeData = runtime_data.data();
  open.TotalPrivateDriverDataBufferSize = query.TotalPrivateDriverDataSize;
  open.pTotalPrivateDriverDataBuffer = driver_data.data();
  open.ResourcePrivateDriverDataSize = query.ResourcePrivateDriverDataSize;
  open.pResourcePrivateDriverData = resource_data.data();
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
  const uint64_t native_byte_length =
      Wkmi::GetMemoryAllocationSize(allocation.pPrivateDriverData);
  const Wkmi::SurfaceSwizzleInfo layout =
      Wkmi::GetSurfaceSwizzleInfo(allocation.pPrivateDriverData);
  // ADDR_SW_LINEAR and ADDR_SW_LINEAR_GENERAL are the addrlib byte layouts.
  if (!layout.valid ||
      (layout.swizzle_mode != 0 && layout.swizzle_mode != 32) ||
      layout.tile_swizzle != 0 || layout.compression_mode != 0 ||
      native_byte_length < logical_byte_length ||
      (native_byte_length & 4095) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_native_byte_length = native_byte_length;
  *out_logical_byte_length = logical_byte_length;
  return AMDF_STATUS_OK;
}

}  // namespace

amdf_status_t AMDF_WKMI_BRIDGE_CALL GpuMemoryPrepareImport(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_memory_import_info_t* import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length,
    uint64_t* out_logical_byte_length) noexcept {
  try {
    return PrepareMemory(adapter, *import_info, resource_handle,
                         allocation_handle, out_native_byte_length,
                         out_logical_byte_length);
  } catch (const std::bad_alloc&) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  } catch (...) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
}

}  // namespace amdf::wkmi_bridge
