// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_ROCM_HOSTCALL_H_
#define IREE_EXPERIMENTAL_STREAMING_ROCM_HOSTCALL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;

// Calculates the power-of-two packet capacity required for one host queue.
iree_status_t iree_hal_streaming_rocm_hostcall_calculate_packet_count(
    const iree_hal_streaming_device_t* device, uint32_t* out_packet_count);

// Returns a context-owned ROCm hostcall service buffer device pointer.
// Synchronization: the first request initializes under the context lock.
iree_status_t iree_hal_streaming_context_rocm_hostcall_buffer(
    iree_hal_streaming_context_t* context, uint64_t* out_buffer_device_ptr);

// Shuts down any lazily-created ROCm hostcall service for |context|.
void iree_hal_streaming_context_deinitialize_rocm_hostcall_service(
    iree_hal_streaming_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_ROCM_HOSTCALL_H_
