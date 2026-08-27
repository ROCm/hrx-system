// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned function versions.
//
// A function version has stable compiler identity while its current IR
// function may be replaced by lowering. The version is not stored in IR and
// does not use a function name, symbol ID, or operation pointer as identity.

#ifndef LOOM_IR_FUNCTION_VERSION_H_
#define LOOM_IR_FUNCTION_VERSION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_function_version_t loom_function_version_t;

// Static identity for one compiler-owned function-version representation.
typedef struct loom_function_version_type_t {
  // Stable diagnostic name for the representation.
  iree_string_view_t name;
} loom_function_version_type_t;

// Base embedded first in compiler-owned function-version representations.
struct loom_function_version_t {
  // Static representation identity used for checked casts.
  const loom_function_version_type_t* type;

  // Current live IR function implementing this version.
  loom_func_like_t function;
};

// Borrowed concrete function versions participating in one compilation.
typedef struct loom_function_version_list_t {
  // Stable version objects, or NULL when |count| is zero.
  loom_function_version_t* const* values;

  // Number of version objects in |values|.
  iree_host_size_t count;
} loom_function_version_list_t;

// Arena-backed owner for the concrete function versions in one compilation.
//
// Appending may replace the pointer vector exposed by |list|, but the list
// object and every version object retain stable addresses. Consumers borrow
// the list object and observe its current values at explicit pass/emission
// boundaries. The owner does not own the version objects themselves; callers
// normally allocate them from the same arena.
typedef struct loom_function_version_owner_t {
  // Current borrowed view over |storage|.
  loom_function_version_list_t list;

  // Arena used to grow |storage|.
  iree_arena_allocator_t* arena;

  // Writable pointer vector also exposed through |list.values|.
  loom_function_version_t** storage;

  // Number of pointer slots allocated in |storage|.
  iree_host_size_t capacity;
} loom_function_version_owner_t;

// Initializes an empty function-version owner allocating from |arena|.
void loom_function_version_owner_initialize(
    iree_arena_allocator_t* arena, loom_function_version_owner_t* out_owner);

// Ensures that |owner| can hold at least |minimum_capacity| versions.
//
// Existing version order is preserved. Superseded pointer vectors remain
// arena-owned and are reclaimed with the arena.
iree_status_t loom_function_version_owner_reserve(
    loom_function_version_owner_t* owner, iree_host_size_t minimum_capacity);

// Appends one stable concrete version handle to |owner|.
iree_status_t loom_function_version_owner_append(
    loom_function_version_owner_t* owner, loom_function_version_t* version);

// Returns the stable borrowed list view owned by |owner|.
static inline const loom_function_version_list_t*
loom_function_version_owner_list(const loom_function_version_owner_t* owner) {
  return owner != NULL ? &owner->list : NULL;
}

// Finds the version currently implemented by |function|, or returns NULL.
//
// This is a cold observation-boundary query for module algorithms reconciling
// a mutable symbol snapshot with concrete compiler products. Function pass
// frames bind the version once and carry it directly.
loom_function_version_t* loom_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function);

// Transfers |version| to the replacement |function|.
//
// Function-replacing transforms call this at the replacement boundary. The
// version identity and all attached compiler facts remain unchanged.
void loom_function_version_update(loom_function_version_t* version,
                                  loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_FUNCTION_VERSION_H_
