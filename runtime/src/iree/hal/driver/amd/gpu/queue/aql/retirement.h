// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_RETIREMENT_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_RETIREMENT_H_

#include "iree/hal/driver/amd/gpu/completion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Polls the queue's accepted native prefix under exclusive completion
// ownership. Returns payload and argument credit, publishes logical outcomes,
// and supplies the next native wait or host download to the completion owner.
// user_data is the queue registered by its completion-source factory.
bool iree_hal_amd_gpu_aql_retirement_poll(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_RETIREMENT_H_
