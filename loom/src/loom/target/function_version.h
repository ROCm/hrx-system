// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target facts owned by one concrete compiler function version.

#ifndef LOOM_TARGET_FUNCTION_VERSION_H_
#define LOOM_TARGET_FUNCTION_VERSION_H_

#include "iree/base/api.h"
#include "loom/ir/function_version.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_function_version_t {
  // Generic compiler function-version base. Must remain the first field.
  loom_function_version_t base;

  // Authored target witness name used for diagnostics and reports. Empty when
  // the function had no authored target.
  iree_string_view_t authored_target_name;

  // Facts projected from the authored target witness, or NULL when absent.
  const loom_target_facts_t* authored_target_facts;

  // Exact invocation-refined facts used to compile this function version.
  const loom_target_facts_t* effective_target_facts;
} loom_target_function_version_t;

// Static identity for target-refined function versions.
extern const loom_function_version_type_t loom_target_function_version_type;

// Returns |version| as a target-refined version, or NULL for another type.
static inline loom_target_function_version_t* loom_target_function_version_cast(
    loom_function_version_t* version) {
  return version != NULL && version->type == &loom_target_function_version_type
             ? (loom_target_function_version_t*)version
             : NULL;
}

// Returns |version| as a target-refined version, or NULL for another type.
static inline const loom_target_function_version_t*
loom_target_function_version_const_cast(
    const loom_function_version_t* version) {
  return version != NULL && version->type == &loom_target_function_version_type
             ? (const loom_target_function_version_t*)version
             : NULL;
}

// Finds the target-refined version currently implemented by |function|.
loom_target_function_version_t* loom_target_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FUNCTION_VERSION_H_
