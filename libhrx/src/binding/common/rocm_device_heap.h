// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_ROCM_DEVICE_HEAP_H_
#define IREE_EXPERIMENTAL_STREAMING_ROCM_DEVICE_HEAP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;

// Returns a device pointer to the per-context ROCm device malloc heap.
// Synchronization: the first request initializes under the context lock.
iree_status_t iree_hal_streaming_context_rocm_device_malloc_heap(
    iree_hal_streaming_context_t* context, uint64_t* out_heap_device_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_ROCM_DEVICE_HEAP_H_
