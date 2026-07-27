// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_RESULT_H_
#define LOOMC_RESULT_H_

#include "loomc/artifact.h"
#include "loomc/diagnostic.h"

/// @file
/// Shared operation result container.
///
/// A result exists when an operation completed far enough to report
/// domain-specific state. API and infrastructure failures that prevent result
/// creation are returned as non-OK `loomc_status_t` values instead.
///
/// @par Example
/// Check status first, then inspect the result. Diagnostics are structured
/// data, not text that callers have to scrape:
///
/// @code{.c}
/// loomc_result_t* result = NULL;
/// // `run_loom_operation` stands in for an operation-specific API that returns
/// // OK and writes a retained result when domain diagnostics are available.
/// loomc_status_t status = run_loom_operation(&result);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
///      ++i) {
///   const loomc_diagnostic_t* diagnostic =
///       loomc_result_diagnostic_at(result, i);
///   // `handle_diagnostic` is application-owned reporting policy.
///   handle_diagnostic(diagnostic->severity, diagnostic->code,
///                     diagnostic->message, diagnostic->range);
/// }
///
/// if (loomc_result_succeeded(result)) {
///   for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result);
///        ++i) {
///     const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
///     // `handle_artifact` is application-owned loading or packaging policy.
///     handle_artifact(artifact->kind, artifact->format, artifact->contents);
///   }
/// }
///
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable operation result.
///
/// @thread_safety
/// Results are immutable after creation. Retained result handles may be shared
/// across threads.
typedef struct loomc_result_t loomc_result_t;

/// Terminal operation-domain state.
typedef enum loomc_result_state_e {
  /// Operation succeeded and requested outputs are available.
  LOOMC_RESULT_STATE_SUCCEEDED = 0,

  /// Operation completed with diagnostics and no infrastructure status failure.
  LOOMC_RESULT_STATE_FAILED = 1,

  /// Operation completed after observing cancellation.
  LOOMC_RESULT_STATE_CANCELLED = 2,
} loomc_result_state_t;

/// Retains `result` for another owner.
///
/// @param result Result to retain.
///
/// @thread_safety
/// Retain/release operations are safe to perform from multiple threads.
LOOMC_API_EXPORT void loomc_result_retain(loomc_result_t* result);

/// Releases `result` from one owner.
///
/// @param result Result to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe to perform from multiple threads. The
/// result is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_result_release(loomc_result_t* result);

/// Returns the result state.
///
/// @param result Result to inspect.
/// @return Terminal operation-domain state.
LOOMC_API_EXPORT loomc_result_state_t
loomc_result_state(const loomc_result_t* result);

/// Returns true when `result` succeeded.
///
/// @param result Result to inspect.
/// @return True when `loomc_result_state(result)` is
/// `LOOMC_RESULT_STATE_SUCCEEDED`.
LOOMC_API_EXPORT bool loomc_result_succeeded(const loomc_result_t* result);

/// Returns the number of diagnostics attached to `result`.
///
/// @param result Result to inspect.
/// @return Diagnostic count.
LOOMC_API_EXPORT loomc_host_size_t
loomc_result_diagnostic_count(const loomc_result_t* result);

/// Returns a diagnostic by index.
///
/// @param result Result to inspect.
/// @param index Zero-based diagnostic index.
/// @return Borrowed diagnostic view, or `NULL` when `index` is out of range.
///
/// @lifetime
/// The returned pointer remains valid until `result` is released.
LOOMC_API_EXPORT const loomc_diagnostic_t* loomc_result_diagnostic_at(
    const loomc_result_t* result, loomc_host_size_t index);

/// Returns the number of artifacts attached to `result`.
///
/// @param result Result to inspect.
/// @return Artifact count.
LOOMC_API_EXPORT loomc_host_size_t
loomc_result_artifact_count(const loomc_result_t* result);

/// Returns an artifact by index.
///
/// @param result Result to inspect.
/// @param index Zero-based artifact index.
/// @return Borrowed artifact view, or `NULL` when `index` is out of range.
///
/// @lifetime
/// The returned pointer remains valid until `result` is released.
LOOMC_API_EXPORT const loomc_artifact_t* loomc_result_artifact_at(
    const loomc_result_t* result, loomc_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_RESULT_H_
