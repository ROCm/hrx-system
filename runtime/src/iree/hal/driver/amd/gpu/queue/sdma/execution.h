// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_EXECUTION_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_EXECUTION_H_

#include "iree/hal/driver/amd/gpu/queue/sdma/operation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Takes the capture's initial reference and joins accepted-work accounting.
// No subsequent caller access to the capture is permitted.
void iree_hal_amd_gpu_sdma_execution_submit(
    iree_hal_amd_gpu_sdma_operation_t* operation);

// Transfers the unpublished suffix back to the same native publisher. Does not
// execute native work on allocation, dependency, or retirement callbacks.
void iree_hal_amd_gpu_sdma_execution_resume(
    iree_hal_amd_gpu_sdma_operation_t* operation);

// Fails an unpublished suffix while preserving every accepted native use.
// Ordinary preparation errors still join input retirement; device-wide failure
// instead joins the parent stop protocol. Consumes status.
void iree_hal_amd_gpu_sdma_execution_abandon(
    iree_hal_amd_gpu_sdma_operation_t* operation, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_EXECUTION_H_
