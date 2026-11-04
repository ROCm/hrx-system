// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/ioctl.h>

#include "libamdf/src/gpu/umd/dxg/rocdxg/native_transport.h"
#include "shared/include/dxcore_loader.h"

namespace {

thread_local const amdf_dxg_bridge_native_transport_t* active_transport =
    nullptr;

NTSTATUS InvokeQueryAdapterInfo(void* arguments) {
  if (active_transport == nullptr) {
    return static_cast<NTSTATUS>(0xC0000184u);  // STATUS_INVALID_DEVICE_STATE
  }
  return static_cast<NTSTATUS>(active_transport->invoke(
      active_transport->context, _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO),
      arguments));
}

}  // namespace

namespace amdf::rocdxg_bridge {

NativeTransportScope::NativeTransportScope(
    const amdf_dxg_bridge_native_transport_t& transport) noexcept
    : previous_(active_transport) {
  active_transport = &transport;
}

NativeTransportScope::~NativeTransportScope() { active_transport = previous_; }

}  // namespace amdf::rocdxg_bridge

namespace wsl::thunk::dxcore {
namespace {

using LoaderDeleter = void (*)(DxcoreLoader* loader);

DxcoreLoader* active_loader = nullptr;
LoaderDeleter active_loader_deleter = nullptr;

struct DxcoreLoaderCleanup {
  ~DxcoreLoaderCleanup() {
    if (active_loader != nullptr) {
      active_loader_deleter(active_loader);
    }
  }
};

DxcoreLoaderCleanup loader_cleanup;

}  // namespace

DxcoreLoader::DxcoreLoader()
    : pfn_D3DKMTCreateAllocation2(nullptr),
      pfn_D3DKMTDestroyAllocation2(nullptr),
      pfn_D3DKMTMapGpuVirtualAddress(nullptr),
      pfn_D3DKMTReserveGpuVirtualAddress(nullptr),
      pfn_D3DKMTFreeGpuVirtualAddress(nullptr),
      pfn_D3DKMTCreateDevice(nullptr),
      pfn_D3DKMTDestroyDevice(nullptr),
      pfn_D3DKMTEnumAdapters2(nullptr),
      pfn_D3DKMTQueryAdapterInfo(InvokeQueryAdapterInfo),
      pfn_D3DKMTCreateContextVirtual(nullptr),
      pfn_D3DKMTDestroyContext(nullptr),
      pfn_D3DKMTSubmitCommand(nullptr),
      pfn_D3DKMTCreateSynchronizationObject2(nullptr),
      pfn_D3DKMTDestroySynchronizationObject(nullptr),
      pfn_D3DKMTQueryStatistics(nullptr),
      pfn_D3DKMTEscape(nullptr),
      pfn_D3DKMTLock2(nullptr),
      pfn_D3DKMTUnlock2(nullptr),
      pfn_D3DKMTCreatePagingQueue(nullptr),
      pfn_D3DKMTDestroyPagingQueue(nullptr),
      pfn_D3DKMTWaitForSynchronizationObjectFromGpu(nullptr),
      pfn_D3DKMTSignalSynchronizationObjectFromGpu(nullptr),
      pfn_D3DKMTWaitForSynchronizationObjectFromCpu(nullptr),
      pfn_D3DKMTQueryClockCalibration(nullptr),
      pfn_D3DKMTMakeResident(nullptr),
      pfn_D3DKMTEvict(nullptr),
      pfn_D3DKMTShareObjects(nullptr),
      pfn_D3DKMTQueryResourceInfoFromNtHandle(nullptr),
      pfn_D3DKMTOpenResourceFromNtHandle(nullptr),
      pfn_D3DKMTCreateHwQueue(nullptr),
      pfn_D3DKMTDestroyHwQueue(nullptr),
      pfn_D3DKMTSubmitCommandToHwQueue(nullptr),
      pfn_D3DKMTEnumProcesses(nullptr),
      pfn_D3DKMTQueryVideoMemoryInfo(nullptr),
      dxcore_handle_(this),
      init_flag_() {
  active_loader = this;
  active_loader_deleter = [](DxcoreLoader* loader) { delete loader; };
}

DxcoreLoader::~DxcoreLoader() {
  Shutdown();
  active_loader = nullptr;
  active_loader_deleter = nullptr;
}

bool DxcoreLoader::Initialize() { return IsLoaded(); }

void DxcoreLoader::Shutdown() { dxcore_handle_ = nullptr; }

bool DxcoreLoader::LoadDxcoreApis() { return true; }

}  // namespace wsl::thunk::dxcore
