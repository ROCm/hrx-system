// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_LOADER_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_LOADER_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <stdint.h>
#include <windows.h>

#include "amdf/amdf.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Loaded private WKMI bridge module.
typedef struct amdf_gpu_wddm_wkmi_loader_t {
  // Module owning every bridge procedure and returned API table.
  HMODULE module;
} amdf_gpu_wddm_wkmi_loader_t;

// Loads the bridge. Failure leaves |out_loader| unchanged.
amdf_status_t amdf_gpu_wddm_wkmi_loader_initialize(
    amdf_allocator_t host_allocator, amdf_gpu_wddm_wkmi_loader_t* out_loader);

// Negotiates the most recent API supported by a loaded bridge.
// Failure leaves |out_api| unchanged.
amdf_status_t amdf_gpu_wddm_wkmi_loader_query_api(
    const amdf_gpu_wddm_wkmi_loader_t* loader,
    const amdf_wkmi_bridge_api_t** out_api);

// Unloads the bridge after all calls through its API table have returned.
amdf_status_t amdf_gpu_wddm_wkmi_loader_deinitialize(
    amdf_gpu_wddm_wkmi_loader_t* loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_LOADER_H_
