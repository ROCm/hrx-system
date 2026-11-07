// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_TRANSFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_TRANSFER_H_

#include "amdf/gpu.h"
#include "iree/hal/driver/amd/gpu/queue/dma_transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Captures SDMA packet encoding independently of staging and payload ownership.
iree_hal_amd_gpu_dma_transfer_encoding_t
iree_hal_amd_gpu_sdma_transfer_encoding(amdf_queue_format_features_t features);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_TRANSFER_H_
