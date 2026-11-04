// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_D3D12_API_H_
#define AMDF_SRC_PLATFORM_WINDOWS_D3D12_API_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef HRESULT(WINAPI* amdf_create_dxgi_factory_1_fn_t)(REFIID riid,
                                                         void** factory);
typedef HRESULT(WINAPI* amdf_d3d12_create_device_fn_t)(
    IUnknown* adapter, D3D_FEATURE_LEVEL minimum_feature_level, REFIID riid,
    void** device);

// Explicitly loaded Direct3D entry points used for typed shared resources.
typedef struct amdf_d3d12_api_t {
  // System DXGI module owning `create_dxgi_factory_1`.
  HMODULE dxgi_module;
  // System Direct3D 12 module owning `create_device`.
  HMODULE d3d12_module;
  // Creates the factory used to select an adapter by LUID.
  amdf_create_dxgi_factory_1_fn_t create_dxgi_factory_1;
  // Creates the transient device used to inspect one shared resource.
  amdf_d3d12_create_device_fn_t create_device;
} amdf_d3d12_api_t;

// Loads the available system DXGI and Direct3D 12 modules and resolves their
// procedures. Missing modules or procedures leave buffer import unsupported.
void amdf_d3d12_api_initialize(amdf_d3d12_api_t* out_api);

// Unloads the system modules after every transient Direct3D object is gone.
// A failed unload preserves ownership of that module for retry.
amdf_status_t amdf_d3d12_api_deinitialize(amdf_d3d12_api_t* api);

// Returns true when typed D3D12 shared-buffer inspection is available.
bool amdf_d3d12_api_supports_buffer_import(const amdf_d3d12_api_t* api);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_WINDOWS_D3D12_API_H_
