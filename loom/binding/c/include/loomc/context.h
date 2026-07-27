// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_CONTEXT_H_
#define LOOMC_CONTEXT_H_

#include "loomc/status.h"

/// @file
/// Reusable Loom API context.
///
/// A context owns process-local Loom language registration state used by
/// parsing, bytecode indexing, linking, and compilation. Embedders normally
/// create one context for a process or toolchain instance and share it with
/// prepared linkers, compilers, and link-index builders.
///
/// @par Example
/// Create a context once, then reuse it while building persistent indexes:
///
/// @code{.c}
/// loomc_context_t* context = NULL;
/// loomc_status_t status =
///     loomc_context_create(NULL, loomc_allocator_system(), &context);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// // Use context to create link-index builders, linkers, and compilers.
///
/// loomc_context_release(context);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable Loom API context.
///
/// @thread_safety
/// Contexts are immutable after creation and may be shared across threads.
/// Operations that need mutable source-location tables serialize that mutation
/// through their owning builder or invocation object rather than exposing it on
/// the context API.
typedef struct loomc_context_t loomc_context_t;

/// Context creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_context_options_t)`, and fill the requested fields.
typedef struct loomc_context_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for context options such as
  /// `loomc_context_target_options_t`.
  const void* next;
} loomc_context_options_t;

/// Creates a reusable Loom API context.
///
/// @param options Context options, or `NULL` for defaults.
/// @param allocator Host allocator used for context-owned storage.
/// @param out_context Receives one retained context on success.
/// @return OK when the context was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_context_release`.
///
/// @thread_safety
/// The returned context is immutable and may be shared across prepared tools
/// and worker threads.
LOOMC_API_EXPORT loomc_status_t loomc_context_create(
    const loomc_context_options_t* options, loomc_allocator_t allocator,
    loomc_context_t** out_context);

/// Retains `context` for another owner.
///
/// @param context Context to retain.
///
/// @thread_safety
/// Retain/release operations are safe to perform from multiple threads.
LOOMC_API_EXPORT void loomc_context_retain(loomc_context_t* context);

/// Releases `context` from one owner.
///
/// @param context Context to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe to perform from multiple threads. The
/// context is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_context_release(loomc_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_CONTEXT_H_
