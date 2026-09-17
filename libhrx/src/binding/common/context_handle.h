// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_CONTEXT_HANDLE_H_
#define IREE_HAL_STREAMING_CONTEXT_HANDLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_registry_t
    iree_hal_streaming_device_registry_t;

// Classifies whether a raw context handle is owned by an explicit context API
// or names the device-managed primary context. Transitional states serialize
// concurrent explicit destroy attempts.
typedef enum iree_hal_streaming_context_handle_state_e {
  IREE_HAL_STREAMING_CONTEXT_HANDLE_STATE_EXPLICIT_LIVE = 0,
  IREE_HAL_STREAMING_CONTEXT_HANDLE_STATE_EXPLICIT_DESTROYING,
  IREE_HAL_STREAMING_CONTEXT_HANDLE_STATE_EXPLICIT_DESTROYED,
  IREE_HAL_STREAMING_CONTEXT_HANDLE_STATE_PRIMARY,
} iree_hal_streaming_context_handle_state_t;

// Validates |handle| by pointer identity against the contexts currently in
// |device_registry| and exclusively reserves an explicit context's public owner
// for destruction. The returned context is retained and must be released by the
// caller. Primary handles and handles already being destroyed are rejected.
// This validates only the current registry incarnation: after a successful
// destroy, raw-handle use is outside the contract and allocator address reuse
// is not distinguishable from the address of a later context. Synchronization:
// thread-safe internal locking.
iree_status_t iree_hal_streaming_context_begin_handle_destroy(
    iree_hal_streaming_device_registry_t* device_registry,
    iree_hal_streaming_context_t* handle,
    iree_hal_streaming_context_t** out_context);

// Restores an explicitly-created handle reserved by begin_handle_destroy after
// a fallible teardown step aborts. Does not release the caller's reference.
// Synchronization: thread-safe internal locking.
void iree_hal_streaming_context_cancel_handle_destroy(
    iree_hal_streaming_context_t* context);

// Permanently invalidates an explicitly-created handle reserved by
// begin_handle_destroy. The caller then consumes the public owner and releases
// its retained reference separately.
// Synchronization: thread-safe internal locking.
void iree_hal_streaming_context_commit_handle_destroy(
    iree_hal_streaming_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_STREAMING_CONTEXT_HANDLE_H_
