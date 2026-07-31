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
