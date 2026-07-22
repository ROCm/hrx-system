// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_ROCM_DEVICE_RUNTIME_H_
#define IREE_EXPERIMENTAL_STREAMING_ROCM_DEVICE_RUNTIME_H_

#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-context storage supporting runtime services used by ROCm device code.
typedef struct iree_hal_streaming_rocm_device_runtime_t {
  // Device-local heap header buffer passed through hidden_heap_v1.
  iree_hal_buffer_t* malloc_heap_buffer;
  // Device-local initial slab storage referenced by the heap header.
  iree_hal_buffer_t* malloc_initial_slabs_buffer;
  // Non-zero after the immutable malloc heap and header are fully initialized.
  iree_atomic_uint64_t malloc_heap_device_ptr;
  // Lazily-created hostcall listener and queue-visible service buffer.
  struct iree_hal_streaming_rocm_hostcall_service_t* hostcall_service;
  // Non-zero after the hostcall listener and service buffer are ready.
  iree_atomic_uint64_t hostcall_buffer_device_ptr;
} iree_hal_streaming_rocm_device_runtime_t;

// Returns the ready device-runtime heap address, or 0 before initialization.
static inline uint64_t iree_hal_streaming_rocm_device_runtime_cached_heap(
    const iree_hal_streaming_rocm_device_runtime_t* runtime) {
  return (uint64_t)iree_atomic_load(&runtime->malloc_heap_device_ptr,
                                    iree_memory_order_acquire);
}

// Returns the ready hostcall service-buffer address, or 0 before creation.
static inline uint64_t
iree_hal_streaming_rocm_device_runtime_cached_hostcall_buffer(
    const iree_hal_streaming_rocm_device_runtime_t* runtime) {
  return (uint64_t)iree_atomic_load(&runtime->hostcall_buffer_device_ptr,
                                    iree_memory_order_acquire);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_ROCM_DEVICE_RUNTIME_H_
