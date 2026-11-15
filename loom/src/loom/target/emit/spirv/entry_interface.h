// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Module-variable membership retained across SPIR-V function calls.

#ifndef LOOM_TARGET_EMIT_SPIRV_ENTRY_INTERFACE_H_
#define LOOM_TARGET_EMIT_SPIRV_ENTRY_INTERFACE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// One module variable shared by all transitive users during serial emission.
typedef struct loom_spirv_interface_variable_t {
  // Emitted OpVariable result ID.
  uint32_t id;
  // Last function that included this variable in its completed use list.
  uint32_t last_function_id;
} loom_spirv_interface_variable_t;

// Unique non-Input/Output variables in first-use order. Builtins have a fixed
// indexed representation of their own. Function IDs are unique and functions
// emit callee-before-caller, so membership needs no hash or repeated search.
typedef struct loom_spirv_entry_interface_t {
  // Arena-owned pointers to module-lifetime variable identities.
  loom_spirv_interface_variable_t** variables;
  // Number of initialized variable pointers.
  iree_host_size_t count;
  // Allocated capacity of the pointer array.
  iree_host_size_t capacity;
} loom_spirv_entry_interface_t;

// Adds a newly declared variable owned by this function.
iree_status_t loom_spirv_entry_interface_declare(
    loom_spirv_entry_interface_t* interface, uint32_t function_id,
    uint32_t variable_id, iree_arena_allocator_t* arena);

// Includes each callee variable once, including repeated and diamond calls.
iree_status_t loom_spirv_entry_interface_include(
    loom_spirv_entry_interface_t* interface, uint32_t function_id,
    const loom_spirv_entry_interface_t* callee, iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_ENTRY_INTERFACE_H_
