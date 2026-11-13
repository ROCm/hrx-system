// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_FUNCTION_HANDLE_H_
#define HRX_BINDING_HIP_FUNCTION_HANDLE_H_

#include "common/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Identifies opaque module-function handles separately from registered host
// function addresses. The tag classifies a token; lookup establishes liveness.
#define IREE_HIP_FUNCTION_HANDLE_TAG_MASK 0x00FF000000000000ULL
#define IREE_HIP_FUNCTION_HANDLE_TAG_VALUE 0x00EE000000000000ULL

static inline bool iree_hip_function_handle_has_tag(const void* handle) {
  return ((uintptr_t)handle & IREE_HIP_FUNCTION_HANDLE_TAG_MASK) ==
         IREE_HIP_FUNCTION_HANDLE_TAG_VALUE;
}

// Returns the stable opaque public handle for |symbol|, creating it on the
// first query. The handle remains resolvable until |module| is retired.
iree_status_t iree_hip_function_handle_get_or_create(
    iree_hal_streaming_module_t* module, iree_hal_streaming_symbol_t* symbol,
    void** out_handle);

// Resolves a live public handle without dereferencing the caller-provided
// value. On success |out_module| is retained for |out_symbol|.
bool iree_hip_function_handle_lookup(const void* handle,
                                     iree_hal_streaming_symbol_t** out_symbol,
                                     iree_hal_streaming_module_t** out_module);

// Retires every public function handle owned by |module|. Concurrent lookups
// that already retained an entry may finish safely; later lookups fail.
void iree_hip_function_handle_retire_module(
    iree_hal_streaming_module_t* module);

// Reclaims empty lookup tables after runtime teardown. Handle identity remains
// monotonic across runtime incarnations.
void iree_hip_function_handle_trim(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_FUNCTION_HANDLE_H_
