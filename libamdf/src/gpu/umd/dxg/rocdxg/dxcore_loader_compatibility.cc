// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <exception>

#include "shared/include/dxcore_loader.h"

namespace wsl::thunk::dxcore {
namespace {

using LoaderDeleter = void (*)(DxcoreLoader* loader);

DxcoreLoader* active_loader = nullptr;
LoaderDeleter active_loader_deleter = nullptr;

template <typename Procedure>
bool LoadProcedure(void* module, const char* name, Procedure* out_procedure) {
  static_assert(sizeof(Procedure) == sizeof(void*));
  void* symbol = dlsym(module, name);
  std::memcpy(out_procedure, &symbol, sizeof(symbol));
  return symbol != nullptr;
}

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
      pfn_D3DKMTQueryAdapterInfo(nullptr),
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
      dxcore_handle_(nullptr),
      init_flag_() {
  active_loader = this;
  active_loader_deleter = [](DxcoreLoader* loader) { delete loader; };
}

DxcoreLoader::~DxcoreLoader() {
  Shutdown();
  active_loader = nullptr;
  active_loader_deleter = nullptr;
}

bool DxcoreLoader::Initialize() {
  std::call_once(init_flag_, [this]() {
    dxcore_handle_ = dlopen("libdxcore.so", RTLD_NOW | RTLD_LOCAL);
    if (dxcore_handle_ != nullptr && !LoadDxcoreApis()) {
      if (dlclose(dxcore_handle_) != 0) {
        std::terminate();
      }
      dxcore_handle_ = nullptr;
    }
  });
  return IsLoaded();
}

void DxcoreLoader::Shutdown() {
  if (dxcore_handle_ != nullptr) {
    if (dlclose(dxcore_handle_) != 0) {
      std::terminate();
    }
    dxcore_handle_ = nullptr;
  }
}

bool DxcoreLoader::LoadDxcoreApis() {
  if (dxcore_handle_ == nullptr) {
    return false;
  }

#define AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(name)           \
  if (!LoadProcedure(dxcore_handle_, #name, &pfn_##name)) { \
    return false;                                           \
  }

  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTCreateAllocation2)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTDestroyAllocation2)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTMapGpuVirtualAddress)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTReserveGpuVirtualAddress)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTFreeGpuVirtualAddress)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTCreateDevice)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTDestroyDevice)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTEnumAdapters2)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTQueryAdapterInfo)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTCreateContextVirtual)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTDestroyContext)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTSubmitCommand)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTCreateSynchronizationObject2)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTDestroySynchronizationObject)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTQueryStatistics)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTEscape)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTLock2)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTUnlock2)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTCreatePagingQueue)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTDestroyPagingQueue)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTWaitForSynchronizationObjectFromGpu)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTSignalSynchronizationObjectFromGpu)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTWaitForSynchronizationObjectFromCpu)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTQueryClockCalibration)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTMakeResident)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTEvict)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTShareObjects)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTQueryResourceInfoFromNtHandle)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTOpenResourceFromNtHandle)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTCreateHwQueue)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTDestroyHwQueue)
  AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE(D3DKMTSubmitCommandToHwQueue)

#undef AMDF_LOAD_REQUIRED_DXCORE_PROCEDURE

  LoadProcedure(dxcore_handle_, "D3DKMTEnumProcesses",
                &pfn_D3DKMTEnumProcesses);
  LoadProcedure(dxcore_handle_, "D3DKMTQueryVideoMemoryInfo",
                &pfn_D3DKMTQueryVideoMemoryInfo);
  return true;
}

}  // namespace wsl::thunk::dxcore
