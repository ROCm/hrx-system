// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_PAGING_QUEUE_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_PAGING_QUEUE_H_

#include "libamdf/src/gpu/umd/dxg/linux/logical_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One Linux DXG paging queue and its monitored progress fence.
typedef struct amdf_gpu_linux_dxg_paging_queue_t {
  // KMT transport borrowed through queue consumption.
  const amdf_linux_dxg_kmt_t* kmt;
  // Native paging-queue handle owned by this object.
  D3DKMT_HANDLE native;
  // Native monitored-fence synchronization object owned by the queue.
  D3DKMT_HANDLE synchronization_object;
  // Read-only CPU mapping of the queue's monitored progress fence.
  const volatile uint64_t* progress_fence;
} amdf_gpu_linux_dxg_paging_queue_t;

// Creates the normal-priority paging queue for one logical device. Failure
// consumes partial native state and preserves the output.
amdf_status_t amdf_gpu_linux_dxg_paging_queue_initialize(
    const amdf_gpu_linux_dxg_logical_device_t* logical_device,
    uint32_t physical_adapter_index,
    amdf_gpu_linux_dxg_paging_queue_t* out_queue);

// Consumes a live paging queue on every result. Linux removes its guest handles
// before a later host notification can fail, so the returned status is terminal
// teardown evidence and never authorizes replaying the native handle.
amdf_status_t amdf_gpu_linux_dxg_paging_queue_consume(
    amdf_gpu_linux_dxg_paging_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_PAGING_QUEUE_H_
