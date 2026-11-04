// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_DEVICE_STATUS_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_DEVICE_STATUS_H_

#include "libamdf/src/atomics.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Observed terminal execution failure shared by one Linux DXG device's queues.
typedef struct amdf_gpu_linux_dxg_device_status_t {
  // First confirmed terminal failure; zero until one has been observed.
  amdf_atomic_uint64_t terminal_status;
} amdf_gpu_linux_dxg_device_status_t;

// Initializes an unpublished device's failure latch.
void amdf_gpu_linux_dxg_device_status_initialize(
    amdf_gpu_linux_dxg_device_status_t* device_status);

// Returns the observed terminal failure without a system call or polling.
amdf_status_t amdf_gpu_linux_dxg_device_status_query(
    const amdf_gpu_linux_dxg_device_status_t* device_status);

// Records a monitored-fence reset indication without a native query.
void amdf_gpu_linux_dxg_device_status_record_reset(
    amdf_gpu_linux_dxg_device_status_t* device_status);

// Samples an owned monitored fence whose normal values exclude UINT64_MAX.
static inline uint64_t amdf_gpu_linux_dxg_device_status_query_fence_progress(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    const volatile uint64_t* fence_pointer) {
  const uint64_t progress = __atomic_load_n(fence_pointer, __ATOMIC_ACQUIRE);
  if (progress == UINT64_MAX) {
    amdf_gpu_linux_dxg_device_status_record_reset(device_status);
    return 0;
  }
  return progress;
}

// Classifies a failed operation and records confirmed terminal device state.
// Ambiguous native errors issue one execution-state query on this cold path.
amdf_status_t amdf_gpu_linux_dxg_device_status_observe_error(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE device,
    amdf_status_t operation_status);

// Completes one paging operation and independently confirms device execution
// state before publishing its result. A zero point completed synchronously.
amdf_status_t amdf_gpu_linux_dxg_device_status_wait_for_paging(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE device,
    D3DKMT_HANDLE paging_synchronization_object,
    const volatile uint64_t* current_paging_fence, uint64_t target_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_DEVICE_STATUS_H_
