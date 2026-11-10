// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_INIT_H_
#define LIBHRX_SRC_BINDING_COMMON_INIT_H_

#include "libhrx/src/libhrx/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
// Construction policy supplied by the compatibility frontend.
typedef struct iree_hal_streaming_init_options_t {
  // Runtime service configuration, including this frontend's event sink.
  hrx_runtime_options_t runtime;
  // Selected GPU devices and borrowed creation extensions.
  hrx_gpu_device_group_options_t devices;
} iree_hal_streaming_init_options_t;

// Initializes global state.
// Synchronization: none (one-time initialization).
iree_status_t iree_hal_streaming_init_global(
    const iree_hal_streaming_init_options_t* options,
    iree_allocator_t host_allocator);

// Cleans up global state and releases all resources.
// Synchronization: all contexts (synchronizes all active contexts).
void iree_hal_streaming_cleanup_global(void);

// Global context list management.
// Synchronization: none (thread-safe internal locking).
void iree_hal_streaming_register_context(iree_hal_streaming_context_t* context);
void iree_hal_streaming_unregister_context(
    iree_hal_streaming_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_INIT_H_
