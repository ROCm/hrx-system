// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_DIAGNOSTIC_H_
#define LOOMC_DIAGNOSTIC_H_

#include "loomc/source.h"

/// @file
/// Structured diagnostics produced by completed operations.
///
/// Diagnostics describe source, configuration, linker, compiler, and
/// lowering outcomes that belong to an operation result. They are typed data so
/// bindings can expose native diagnostic objects instead of parsing strings.

#ifdef __cplusplus
extern "C" {
#endif

/// Diagnostic severity.
typedef enum loomc_diagnostic_severity_e {
  /// Informational note.
  LOOMC_DIAGNOSTIC_SEVERITY_NOTE = 0,

  /// Warning that does not make the operation fail.
  LOOMC_DIAGNOSTIC_SEVERITY_WARNING = 1,

  /// Error that makes the operation fail.
  LOOMC_DIAGNOSTIC_SEVERITY_ERROR = 2,
} loomc_diagnostic_severity_t;

/// Source range attached to a diagnostic.
///
/// Byte offsets are zero-based and line/column values are one-based when
/// available. A zero line or column means the location was not computed or is
/// not meaningful for the source format.
///
/// @lifetime
/// The source pointer is retained by the owning result when present. The range
/// view remains valid until that result is released.
typedef struct loomc_source_range_t {
  /// Source handle that owns or identifies the bytes.
  const loomc_source_t* source;

  /// First byte in source contents covered by the range.
  loomc_host_size_t start;

  /// One-past-last byte in source contents covered by the range.
  loomc_host_size_t end;

  /// One-based starting line, or zero when unavailable.
  uint32_t start_line;

  /// One-based starting column, or zero when unavailable.
  uint32_t start_column;

  /// One-based ending line, or zero when unavailable.
  uint32_t end_line;

  /// One-based ending column, or zero when unavailable.
  uint32_t end_column;
} loomc_source_range_t;

/// Borrowed diagnostic view owned by a result object.
///
/// @lifetime
/// Diagnostic strings and source ranges are owned by the result that returned
/// this view. They remain valid until that result is released.
typedef struct loomc_diagnostic_t {
  /// Diagnostic severity.
  loomc_diagnostic_severity_t severity;

  /// Stable diagnostic code.
  loomc_string_view_t code;

  /// Human-readable rendered message.
  loomc_string_view_t message;

  /// Primary source range.
  loomc_source_range_t range;
} loomc_diagnostic_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_DIAGNOSTIC_H_
