// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-function target specialization.
//
// Specialization is an invocation-local compiler action. Each request names
// one function version and supplies the structured target profile that should
// become its exact effective target. The binder writes that durable target
// identity onto the function and retains only supplemental profile facts in
// the returned context.

#ifndef LOOM_TARGET_SPECIALIZATION_H_
#define LOOM_TARGET_SPECIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;

// One function version to specialize to one structured target profile.
typedef struct loom_target_specialization_request_t {
  // Function symbol name, with an optional leading '@'.
  iree_string_view_t function_name;

  // Structured target profile borrowed for the specialization lifetime.
  const loom_target_profile_t* target_profile;
} loom_target_specialization_request_t;

// Borrowed list of per-function target specialization requests.
typedef struct loom_target_specialization_request_list_t {
  // Request rows, or NULL when |count| is zero.
  const loom_target_specialization_request_t* values;

  // Number of request rows in |values|.
  iree_host_size_t count;
} loom_target_specialization_request_list_t;

// Supplemental per-function target facts for one specialization invocation.
//
// Durable effective targets live on function ops. This context contains only
// profile facts that cannot be reconstructed from target records and must not
// be used as a fallback target source.
typedef struct loom_target_specialization_context_t {
  // Profile table indexed by stable module string ID.
  const loom_target_profile_t* const* profiles_by_function_name_id;

  // Number of entries in |profiles_by_function_name_id|.
  iree_host_size_t profile_capacity;
} loom_target_specialization_context_t;

// Result of binding a specialization request list.
typedef struct loom_target_specialization_result_t {
  // Supplemental profile context for successfully bound functions.
  loom_target_specialization_context_t context;

  // Number of source compatibility diagnostics emitted while validating the
  // complete request list.
  uint32_t error_count;
} loom_target_specialization_result_t;

// Resolves and binds every target specialization request.
//
// The input module must be verified and mutable for this compiler invocation.
// All function names, profiles, and authored target requirements are resolved
// before any function target attribute is changed. Source incompatibilities
// emit structured diagnostics and return OK with a nonzero |error_count|;
// malformed external requests and infrastructure failures return a status.
//
// Target profiles and |arena| storage must outlive every pass that consumes the
// returned context.
iree_status_t loom_target_specialize_functions(
    const loom_target_environment_t* environment, loom_module_t* module,
    loom_target_specialization_request_list_t requests,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_target_specialization_result_t* out_result);

// Returns the supplemental profile assigned to |function|, or NULL.
//
// This query never infers a target from another function or from the module.
// Callers resolve durable target identity from loom_func_like_target().
const loom_target_profile_t* loom_target_specialization_context_lookup(
    const loom_target_specialization_context_t* context,
    const loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_SPECIALIZATION_H_
