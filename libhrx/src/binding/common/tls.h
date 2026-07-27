// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Binding-local thread-specific storage keys with optional thread-exit
// destructors.

#ifndef IREE_EXPERIMENTAL_STREAMING_TLS_H_
#define IREE_EXPERIMENTAL_STREAMING_TLS_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum process-global TLS keys that can be allocated through this binding
// API. The fixed capacity keeps key allocation bounded and lets get/set remain
// direct slot operations.
#define IREE_HAL_STREAMING_TLS_KEY_CAPACITY 128

// Invalid key value used for uninitialized key storage.
#define IREE_HAL_STREAMING_TLS_KEY_INVALID ((iree_hal_streaming_tls_key_t) - 1)

// Process-global TLS key. Keys are small indexes into an internal fixed table.
typedef iree_host_size_t iree_hal_streaming_tls_key_t;

// Invoked with a non-NULL thread-local value when a thread exits.
typedef void(IREE_API_PTR* iree_hal_streaming_tls_destructor_t)(void* value);

// Creates a process-global TLS key with an optional thread-exit |destructor|.
//
// Keys are usually created once and kept for process lifetime. Deleting a key
// is only safe after all threads that could read, write, or destruct values for
// the key are gone or externally synchronized. Deletion does not invoke
// destructors.
//
// Windows destructors run from DllMain thread-detach/process-detach handling
// and must obey the loader-lock restrictions that implies.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor);

// Deletes a key previously created with iree_hal_streaming_tls_key_create.
IREE_API_EXPORT void iree_hal_streaming_tls_key_delete(
    iree_hal_streaming_tls_key_t key);

// Returns the value associated with |key| on the current thread.
//
// Returns NULL for valid unset keys and invalid/deleted keys.
IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key);

// Sets the value associated with |key| on the current thread.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_TLS_H_
