// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_KERNEL_QUEUE_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_KERNEL_QUEUE_H_

#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Publishes prepared mapping descriptors and the subsequent queue dependency.
// Native acceptance sets accepted even if ordering then fails; that failure is
// terminal on the queue. The shared publisher owns capacity and point
// selection.
amdf_status_t amdf_gpu_linux_dxg_kernel_queue_update_memory(
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations, uint64_t submission,
    bool* mutation_started, bool* accepted);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_KERNEL_QUEUE_H_
