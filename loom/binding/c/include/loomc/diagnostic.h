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
/// not meaningful for the source format. A source can retain its identifier
/// without contents; recorded byte offsets and line/column values remain valid
/// against a caller-owned copy of the identified source. Unknown byte offsets
/// are zero.
///
/// @lifetime
/// The source pointer is retained by the owning result when present. The range
/// view remains valid until that result is released.
typedef struct loomc_source_range_t {
  /// Source identity with optional contents, or NULL for an unknown source.
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

/// A labeled source location providing context for a primary diagnostic.
///
/// Examples include a callee's declaration or the previous use of a value.
/// The owning result retains the label and source identity. Source contents
/// are optional, as with the primary range.
typedef struct loomc_diagnostic_related_location_t {
  /// Human-readable relationship, such as "contract defined here".
  loomc_string_view_t label;

  /// Source range of the related location.
  loomc_source_range_t range;
} loomc_diagnostic_related_location_t;

/// A named diagnostic parameter rendered for structured consumers.
///
/// Parameter values use the same canonical human-readable spelling as the
/// diagnostic message. This keeps compiler-owned parameter variants and type
/// handles behind the LoomC boundary while allowing tools to match or report
/// structured values without parsing message text.
typedef struct loomc_diagnostic_parameter_t {
  /// Stable parameter name from the diagnostic definition.
  loomc_string_view_t name;

  /// Canonical human-readable parameter value.
  loomc_string_view_t value;
} loomc_diagnostic_parameter_t;

/// Borrowed diagnostic view owned by a result object.
///
/// @lifetime
/// Diagnostic strings, parameters, related locations, and source ranges are
/// owned by the result that returned this view. They remain valid until that
/// result is released.
typedef struct loomc_diagnostic_t {
  /// Diagnostic severity.
  loomc_diagnostic_severity_t severity;

  /// Stable diagnostic code.
  loomc_string_view_t code;

  /// Human-readable rendered message.
  loomc_string_view_t message;

  /// Canonical human-readable rendering of the complete diagnostic.
  ///
  /// This includes source excerpts, carets, fix hints, and related locations
  /// when available, and ends with a newline. It is intended for terminal and
  /// log presentation; tools requiring stable data should consume the
  /// structured fields instead. The view is empty when no canonical rendering
  /// is available.
  loomc_string_view_t formatted_text;

  /// Primary source range.
  loomc_source_range_t range;

  /// Related locations in producer order, or NULL when the count is zero.
  const loomc_diagnostic_related_location_t* related_locations;

  /// Number of entries in related_locations.
  loomc_host_size_t related_location_count;

  /// Number of additional resolved locations omitted by the producer's limit.
  loomc_host_size_t related_location_omitted_count;

  /// Named rendered parameters in diagnostic-definition order, or NULL when
  /// the count is zero.
  const loomc_diagnostic_parameter_t* parameters;

  /// Number of entries in parameters.
  loomc_host_size_t parameter_count;

  /// Canonical JSON object for the native compiler diagnostic, without a
  /// trailing newline. This preserves producer-specific fields, typed JSON
  /// parameters, source provenance, and fix hints that the common fields above
  /// do not express. It is rendered during the operation while compiler-owned
  /// values are available and remains valid until the result is released.
  /// Empty for diagnostics produced without a native compiler record.
  loomc_string_view_t formatted_json;
} loomc_diagnostic_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_DIAGNOSTIC_H_
