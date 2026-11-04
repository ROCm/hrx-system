// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/d3d12_api.h"

bool amdf_d3d12_api_supports_buffer_import(const amdf_d3d12_api_t* api) {
  return api->create_dxgi_factory_1 != NULL && api->create_device != NULL;
}

void amdf_d3d12_api_initialize(amdf_d3d12_api_t* out_api) {
  amdf_d3d12_api_t api = {0};
  api.dxgi_module =
      LoadLibraryExW(L"dxgi.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (api.dxgi_module != NULL) {
    api.create_dxgi_factory_1 = (amdf_create_dxgi_factory_1_fn_t)GetProcAddress(
        api.dxgi_module, "CreateDXGIFactory1");
  }
  api.d3d12_module =
      LoadLibraryExW(L"d3d12.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (api.d3d12_module != NULL) {
    api.create_device = (amdf_d3d12_create_device_fn_t)GetProcAddress(
        api.d3d12_module, "D3D12CreateDevice");
  }
  *out_api = api;
}

amdf_status_t amdf_d3d12_api_deinitialize(amdf_d3d12_api_t* api) {
  amdf_status_t status = AMDF_STATUS_OK;
  if (api->d3d12_module != NULL) {
    if (FreeLibrary(api->d3d12_module)) {
      api->d3d12_module = NULL;
      api->create_device = NULL;
    } else {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  if (api->dxgi_module != NULL) {
    if (FreeLibrary(api->dxgi_module)) {
      api->dxgi_module = NULL;
      api->create_dxgi_factory_1 = NULL;
    } else if (amdf_status_is_ok(status)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  return status;
}
