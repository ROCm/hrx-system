// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass-local state shared across source functions lowered in one module.

#ifndef LOOM_CODEGEN_LOW_LOWER_MODULE_STATE_H_
#define LOOM_CODEGEN_LOW_LOWER_MODULE_STATE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_module_state_t loom_low_lower_module_state_t;

// Creates a module-scope target-state container allocated from |arena|.
//
// Callers pass the returned state through every source-to-Low function
// lowering in one module pass and invoke target module finalizers before
// releasing |arena|. Target-owned state stored here is pass-local scratch until
// a module finalizer materializes durable IR.
iree_status_t loom_low_lower_module_state_create(
    iree_arena_allocator_t* arena,
    loom_low_lower_module_state_t** out_module_state);

// Returns module-scope target state for |key|, allocating zeroed storage on
// first use.
//
// Keys must be target-owned static addresses. Reusing a key with a different
// |data_length| violates the target state contract. The returned storage
// remains valid until the arena passed to loom_low_lower_module_state_create is
// released.
iree_status_t loom_low_lower_module_state_get_or_allocate(
    loom_low_lower_module_state_t* module_state, const void* key,
    iree_host_size_t data_length, void** out_data);

// Allocates uninitialized pass-local module-state storage.
iree_status_t loom_low_lower_module_state_allocate(
    loom_low_lower_module_state_t* module_state, iree_host_size_t byte_length,
    void** out_ptr);

// Allocates an uninitialized pass-local module-state array.
iree_status_t loom_low_lower_module_state_allocate_array(
    loom_low_lower_module_state_t* module_state, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_MODULE_STATE_H_
