// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_RESET_MONITOR_H_
#define AMDF_SRC_GPU_UMD_KFD_RESET_MONITOR_H_

#include <drm/amdgpu_drm.h>
#include <stdbool.h>
#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native DRM context operation used by the reset monitor.
typedef struct amdf_gpu_kfd_reset_monitor_native_api_t {
  // Opaque state passed to `context_ioctl`.
  void* user_data;
  // Performs one DRM context ioctl against `render_descriptor`.
  amdf_status_t (*context_ioctl)(void* user_data, int render_descriptor,
                                 union drm_amdgpu_ctx* context);
} amdf_gpu_kfd_reset_monitor_native_api_t;

// One device-epoch reset observer owned by a KFD device.
typedef struct amdf_gpu_kfd_reset_monitor_t {
  // Native operation table borrowed through deinitialization.
  const amdf_gpu_kfd_reset_monitor_native_api_t* native_api;
  // Device render file borrowed through deinitialization.
  int render_descriptor;
  // Owned AMDGPU context identity.
  uint32_t context_identifier;
  // Whether `context_identifier` still requires native release.
  bool context_owned;
  // Sticky indication that the physical reset counter changed.
  amdf_atomic_uint32_t reset_observed;
} amdf_gpu_kfd_reset_monitor_t;

// One coherent native reset observation.
typedef struct amdf_gpu_kfd_reset_state_t {
  // Whether any physical GPU reset occurred in this device epoch.
  bool reset_observed;
  // Whether the reset domain is currently held for recovery.
  bool reset_in_progress;
} amdf_gpu_kfd_reset_state_t;

// Returns the production DRM operation table.
const amdf_gpu_kfd_reset_monitor_native_api_t*
amdf_gpu_kfd_reset_monitor_default_native_api(void);

// Allocates the native observer context. Failure leaves `out_monitor`
// unchanged.
amdf_status_t amdf_gpu_kfd_reset_monitor_initialize(
    int render_descriptor,
    const amdf_gpu_kfd_reset_monitor_native_api_t* native_api,
    amdf_gpu_kfd_reset_monitor_t* out_monitor);

// Samples and latches reset state. Failure leaves `out_state` unchanged.
amdf_status_t amdf_gpu_kfd_reset_monitor_query(
    amdf_gpu_kfd_reset_monitor_t* monitor,
    amdf_gpu_kfd_reset_state_t* out_state);

// Releases the observer context. Native failure retains it for a later retry.
amdf_status_t amdf_gpu_kfd_reset_monitor_deinitialize(
    amdf_gpu_kfd_reset_monitor_t* monitor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KFD_RESET_MONITOR_H_
