// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_PERFMON_H_
#define IREE_HAL_DRIVER_AMD_GPU_PERFMON_H_

#include "amdf/api.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Host prerequisites of thread tracing and hardware counters on one physical
// device, held for a profiling session. This owner opens the endpoint's own
// DRM render node and a DRM context on it; it issues no command and touches no
// queue. Closing it returns the device to its previous power policy.
typedef struct iree_hal_amd_gpu_perfmon_t {
  // Owned render-node descriptor, or -1 when closed.
  int descriptor;
  // Owned DRM context carrying the power-state request.
  uint32_t context_id;
  // context_id names a live context.
  bool has_context;
  // The kernel accepted the stable power-state request on that context. The
  // request needs no privilege; it also enables the perfmon clock.
  bool stable_power_state;
  // Shader engines reported by the kernel driver.
  uint32_t shader_engine_count;
} iree_hal_amd_gpu_perfmon_t;

// Opens the endpoint's render node, reads the device geometry and requests a
// stable power state. Fails when the node cannot be opened or queried; a
// refused power-state request is not a failure and leaves stable_power_state
// false. On failure the owner is closed.
iree_status_t iree_hal_amd_gpu_perfmon_open(
    const amdf_endpoint_info_t* endpoint_info,
    iree_hal_amd_gpu_perfmon_t* out_perfmon);

// Drops the power-state request and closes the node. Accepts a zeroed or
// already closed owner.
void iree_hal_amd_gpu_perfmon_close(iree_hal_amd_gpu_perfmon_t* perfmon);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_PERFMON_H_
