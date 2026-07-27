// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_STATUS_H_
#define LOOMC_STATUS_H_

#include "loomc/base.h"

/// @file
/// Status codes and rich infrastructure error reporting.
///
/// `loomc_status_t` reports API misuse, allocation failure, cancellation
/// infrastructure, and other failures that prevent an operation result from
/// being produced. Source, linker, compiler, and configuration failures
/// that are part of a completed operation are reported through result state and
/// diagnostics instead of being hidden in status.
///
/// Status values are ABI-compatible with IREE status values when
/// `loomc/iree.h` is included, but core Loom C API headers do not expose IREE
/// types.
///
/// @par Example
/// Consume or free every non-OK status exactly once:
///
/// @code{.c}
/// // `do_work` stands in for a status-returning API or application helper.
/// loomc_status_t status = do_work();
/// if (!loomc_status_is_ok(status)) {
///   loomc_string_view_t message = loomc_status_message(status);
///   if (!loomc_string_view_is_empty(message)) {
///     fwrite(message.data, 1, message.size, stderr);
///   }
///   return (int)loomc_status_consume_code(status);
/// }
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Status mode feature bit for source file and line capture.
#define LOOMC_STATUS_FEATURE_SOURCE_LOCATION (1 << 0)

/// Status mode feature bit for annotated messages.
#define LOOMC_STATUS_FEATURE_ANNOTATIONS (1 << 1)

/// Status mode feature bit for stack trace capture.
#define LOOMC_STATUS_FEATURE_STACK_TRACE (1 << 2)

/// Rich status storage mode.
///
/// Mode `0` stores status codes only and performs no rich status allocation.
/// Mode `1` adds source locations. Mode `2` adds annotations. Mode `3` adds
/// stack traces. Release builds default to mode `2`; non-release builds default
/// to mode `3`.
#if !defined(LOOMC_STATUS_MODE)
#if defined(IREE_STATUS_MODE)
#define LOOMC_STATUS_MODE IREE_STATUS_MODE
#elif defined(NDEBUG)
#define LOOMC_STATUS_MODE 2
#else
#define LOOMC_STATUS_MODE 3
#endif
#endif

/// Feature bit mask derived from `LOOMC_STATUS_MODE`.
#if !defined(LOOMC_STATUS_FEATURES)
#if defined(LOOMC_STATUS_MODE) && LOOMC_STATUS_MODE == 1
#define LOOMC_STATUS_FEATURES (LOOMC_STATUS_FEATURE_SOURCE_LOCATION)
#elif defined(LOOMC_STATUS_MODE) && LOOMC_STATUS_MODE == 2
#define LOOMC_STATUS_FEATURES \
  (LOOMC_STATUS_FEATURE_SOURCE_LOCATION | LOOMC_STATUS_FEATURE_ANNOTATIONS)
#elif defined(LOOMC_STATUS_MODE) && LOOMC_STATUS_MODE == 3
#define LOOMC_STATUS_FEATURES                                                \
  (LOOMC_STATUS_FEATURE_SOURCE_LOCATION | LOOMC_STATUS_FEATURE_ANNOTATIONS | \
   LOOMC_STATUS_FEATURE_STACK_TRACE)
#else
#define LOOMC_STATUS_FEATURES 0
#endif
#endif

/// Stable status codes shared by all Loom status modes.
///
/// The numeric values are stable ABI and intentionally match the values used by
/// optional IREE adapters so statuses can be passed across that boundary
/// without conversion or information loss.
typedef enum loomc_status_code_e {
  /// Successful operation.
  LOOMC_STATUS_OK = 0,

  /// Operation was cancelled by the caller.
  LOOMC_STATUS_CANCELLED = 1,

  /// Unknown error or unmapped failure.
  LOOMC_STATUS_UNKNOWN = 2,

  /// The caller provided an invalid argument.
  LOOMC_STATUS_INVALID_ARGUMENT = 3,

  /// A deadline was exceeded before the call completed.
  LOOMC_STATUS_DEADLINE_EXCEEDED = 4,

  /// A referenced resource could not be found.
  LOOMC_STATUS_NOT_FOUND = 5,

  /// The resource the caller attempted to create already exists.
  LOOMC_STATUS_ALREADY_EXISTS = 6,

  /// The caller does not have permission for the requested operation.
  LOOMC_STATUS_PERMISSION_DENIED = 7,

  /// A required resource was exhausted.
  LOOMC_STATUS_RESOURCE_EXHAUSTED = 8,

  /// The system is not in a required state for the operation.
  LOOMC_STATUS_FAILED_PRECONDITION = 9,

  /// The operation was aborted by the system.
  LOOMC_STATUS_ABORTED = 10,

  /// The operation was attempted outside a valid range.
  LOOMC_STATUS_OUT_OF_RANGE = 11,

  /// The operation has not been implemented or is not supported.
  LOOMC_STATUS_UNIMPLEMENTED = 12,

  /// An internal invariant was violated.
  LOOMC_STATUS_INTERNAL = 13,

  /// The underlying system is currently unavailable.
  LOOMC_STATUS_UNAVAILABLE = 14,

  /// Unrecoverable data loss or corruption occurred.
  LOOMC_STATUS_DATA_LOSS = 15,

  /// The requested operation does not have proper authentication.
  LOOMC_STATUS_UNAUTHENTICATED = 16,

  /// The operation has been deferred and must be resumed later.
  LOOMC_STATUS_DEFERRED = 17,

  /// The program or environment is incompatible with the request.
  LOOMC_STATUS_INCOMPATIBLE = 18,

  /// Mask covering all status code bits in a `loomc_status_t` value.
  LOOMC_STATUS_CODE_MASK = 0x1Fu,
} loomc_status_code_t;

/// Source location captured when rich status mode supports it.
typedef struct loomc_status_source_location_t {
  /// Source file path, or empty when unavailable.
  loomc_string_view_t file;

  /// One-based source line, or zero when unavailable.
  uint32_t line;
} loomc_status_source_location_t;

/// Returns a status value for `code`.
///
/// @param code Status code to encode.
/// @return A status carrying `code`.
///
/// @ownership
/// Code-only statuses do not allocate storage and may be passed to
/// `loomc_status_free` safely.
static inline loomc_status_t loomc_status_from_code(loomc_status_code_t code) {
  return (loomc_status_t)((uintptr_t)code & LOOMC_STATUS_CODE_MASK);
}

/// Returns an OK status.
///
/// @return Status value whose code is `LOOMC_STATUS_OK`.
static inline loomc_status_t loomc_ok_status(void) {
  return loomc_status_from_code(LOOMC_STATUS_OK);
}

/// Returns the status code carried by `status`.
///
/// @param status Status to inspect.
/// @return Stable status code stored in `status`.
static inline loomc_status_code_t loomc_status_code(loomc_status_t status) {
  return (loomc_status_code_t)((uintptr_t)status & LOOMC_STATUS_CODE_MASK);
}

/// Returns true when `status` is OK.
///
/// @param status Status to inspect.
/// @return True when `status` is exactly OK.
static inline bool loomc_status_is_ok(loomc_status_t status) {
  return (uintptr_t)status == LOOMC_STATUS_OK;
}

/// Allocates a rich status when enabled by `LOOMC_STATUS_MODE`.
///
/// @param code Status code to report.
/// @param file Source file path associated with the failure.
/// @param line Source line associated with the failure.
/// @param message Primary human-readable message. The message bytes are copied
/// when rich status storage is enabled.
/// @return A status carrying `code` and any rich payload enabled by the current
/// status mode.
///
/// @ownership
/// Non-OK returned statuses must eventually be passed to `loomc_status_free`,
/// `loomc_status_consume_code`, or `loomc_status_join`.
LOOMC_API_EXPORT loomc_status_t
loomc_status_allocate(loomc_status_code_t code, const char* file, uint32_t line,
                      loomc_string_view_t message);

/// Allocates a status with the current source file and line.
///
/// @param code Status code to report.
/// @param message NUL-terminated human-readable message.
/// @return A status carrying `code` and `message`.
#define loomc_make_status(code, message)            \
  loomc_status_allocate((code), __FILE__, __LINE__, \
                        loomc_make_cstring_view(message))

/// Frees status storage, if any.
///
/// @param status Status to free. Passing OK or a code-only status is allowed.
LOOMC_API_EXPORT void loomc_status_free(loomc_status_t status);

/// Frees status storage and returns its code.
///
/// @param status Status to consume.
/// @return Code carried by `status`.
LOOMC_API_EXPORT loomc_status_code_t
loomc_status_consume_code(loomc_status_t status);

/// Joins two statuses without discarding cleanup failures.
///
/// @param base_status Existing status, often the primary operation status.
/// @param new_status New status, often produced by cleanup.
/// @return A status that preserves a non-OK status when either input failed.
///
/// @ownership
/// The returned status owns any retained rich payload. The caller must not use
/// `base_status` or `new_status` after passing them to this function.
LOOMC_API_EXPORT loomc_status_t loomc_status_join(loomc_status_t base_status,
                                                  loomc_status_t new_status);

/// Returns a stable C string naming `code`.
///
/// @param code Status code to name.
/// @return Static NUL-terminated string owned by the library.
LOOMC_API_EXPORT const char* loomc_status_code_string(loomc_status_code_t code);

/// Returns the primary message stored in `status`.
///
/// @param status Status to inspect.
/// @return Borrowed message view owned by `status`, or an empty view when no
/// message is available.
///
/// @lifetime
/// The returned view remains valid until `status` is freed or consumed.
LOOMC_API_EXPORT loomc_string_view_t
loomc_status_message(loomc_status_t status);

/// Returns the source location captured by `status`, when available.
///
/// @param status Status to inspect.
/// @return Source location payload, or empty fields when unavailable.
///
/// @lifetime
/// The returned file view remains valid until `status` is freed or consumed.
LOOMC_API_EXPORT loomc_status_source_location_t
loomc_status_source_location(loomc_status_t status);

/// Formats `status` into `buffer` using the two-pass pattern.
///
/// @param status Status to format.
/// @param buffer_capacity Number of bytes available in `buffer`.
/// @param buffer Output buffer, or `NULL` when querying the required length.
/// @param out_length Receives the number of bytes required or written,
/// excluding any trailing NUL byte.
/// @return True when `buffer` was large enough and formatting completed.
///
/// @lifetime
/// Formatted bytes are owned by the caller-provided buffer.
LOOMC_API_EXPORT bool loomc_status_format(loomc_status_t status,
                                          loomc_host_size_t buffer_capacity,
                                          char* buffer,
                                          loomc_host_size_t* out_length);

/// Returns immediately when an expression produces a non-OK status.
///
/// @param expr Status-returning expression to evaluate exactly once.
#define LOOMC_RETURN_IF_ERROR(expr)      \
  do {                                   \
    loomc_status_t status__ = (expr);    \
    if (!loomc_status_is_ok(status__)) { \
      return status__;                   \
    }                                    \
  } while (0)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_STATUS_H_
