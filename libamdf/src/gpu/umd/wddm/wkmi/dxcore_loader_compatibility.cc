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

#include <mutex>

// The pinned WKMI archive retains ROCr's inline DxcoreLoader singleton but not
// its constructor. Its private object contains 38 procedure pointers followed
// by a module handle and initialization flag, totaling 0x140 bytes. Only
// D3DKMTQueryAdapterInfo at slot 8 is used by the archive; the shim binds it
// directly and leaves the dynamic loader state unused.
namespace wsl::thunk::dxcore {

using Procedure = NTSTATUS(WINAPI*)(void* arguments);

class DxcoreLoader {
 public:
  // Procedure slots in the order fixed by the pinned ROCr declaration.
  Procedure procedures[38];
  // Native loader module slot; no dynamic module is owned by this shim.
  void* module_handle = nullptr;
  // Native loader initialization state, retained for the archive's object ABI.
  std::once_flag initialization;

 private:
  friend struct DxcoreLoaderCleanup;

  DxcoreLoader();
  ~DxcoreLoader();
};

static_assert(sizeof(Procedure) == sizeof(void*),
              "WKMI procedure slots must have pointer width");
static_assert(sizeof(DxcoreLoader) == 0x140,
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
