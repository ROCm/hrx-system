// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

// The pinned WKMI archive retains ROCr's inline DxcoreLoader singleton but not
// its constructor. The archive allocates exactly 0x130 bytes for this private
// object: 38 procedure pointers with D3DKMTQueryAdapterInfo at slot 8. Keep
// module ownership out of the object because newer ROCr source declarations
// have a larger tail that is not present in the distributed archive.
namespace wsl::thunk::dxcore {

using Procedure = NTSTATUS(WINAPI*)(void* arguments);

class DxcoreLoader {
 public:
  // Procedure slots in the order fixed by the pinned ROCr declaration.
  Procedure procedures[38];

 private:
  friend struct DxcoreLoaderCleanup;

  DxcoreLoader();
  ~DxcoreLoader();
};

static_assert(sizeof(Procedure) == sizeof(void*),
              "WKMI procedure slots must have pointer width");
static_assert(sizeof(DxcoreLoader) == 0x130,
              "WKMI loader must match the pinned archive allocation");

static DxcoreLoader* active_loader = nullptr;

DxcoreLoader::DxcoreLoader() : procedures{} {
  procedures[8] = reinterpret_cast<Procedure>(&D3DKMTQueryAdapterInfo);
  active_loader = this;
}

DxcoreLoader::~DxcoreLoader() = default;

struct DxcoreLoaderCleanup {
  ~DxcoreLoaderCleanup() {
    delete active_loader;
    active_loader = nullptr;
  }
};

static DxcoreLoaderCleanup loader_cleanup;

}  // namespace wsl::thunk::dxcore
