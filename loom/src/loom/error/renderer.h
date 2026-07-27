// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Message template renderer for structured diagnostics.
//
// Expands {param_name} placeholders in error definition message templates
// using typed runtime parameter values. The renderer is independent of
// the text format module — it lives in error/ so any sink (JSON, stderr,
// custom) can render messages without pulling in the parser/printer.
//
// Type parameters (LOOM_PARAM_TYPE) are rendered via an optional callback.
// When no type formatter is provided (formatter.fn == NULL), TYPE params
// render as "<type>" (useful for builds without the text format linked in).
// The text format module provides a formatter that calls
// loom_text_print_type().
//
// All output flows through a loom_output_stream_t — the same
// zero-allocation write-callback abstraction used by the text printer.
// Callers with an iree_string_builder_t can wrap it using
// loom_output_stream_for_builder().
//
// Usage:
//
//   iree_string_builder_t builder;
//   iree_string_builder_initialize(allocator, &builder);
//   loom_output_stream_t stream;
//   loom_output_stream_for_builder(&builder, &stream);
//   loom_type_formatter_t formatter = {loom_type_format_minimal, NULL};
//   const loom_error_def_t* error = ...;
//   loom_diagnostic_render_message(error, params, 4, formatter, &stream);
//   // builder now contains: "'lhs' type f32 does not match 'rhs' type i32"
//   iree_string_builder_deinitialize(&builder);

#ifndef LOOM_ERROR_RENDERER_H_
#define LOOM_ERROR_RENDERER_H_

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Callback for rendering a loom_type_t into an output stream.
// Implementations live in format modules (e.g., the text printer).
typedef iree_status_t (*loom_type_format_fn_t)(loom_type_t type,
                                               void* user_data,
                                               loom_output_stream_t* stream);

// A type formatter: callback + context. When .fn is NULL, TYPE params
// render as "<type>".
typedef struct loom_type_formatter_t {
  loom_type_format_fn_t fn;
  void* user_data;
} loom_type_formatter_t;

// Renders a diagnostic message by substituting params into the error
// def's message template. Literal text is copied verbatim. Placeholders
// ({param_name}) are looked up in the error def's param_defs array and
// rendered from the corresponding runtime param value.
//
// Param rendering per kind:
//   STRING → raw text
//   I64    → decimal integer (PRId64)
//   U32    → decimal unsigned (PRIu32)
//   U64    → decimal unsigned (PRIu64)
//   BOOL   → "true" / "false"
//   TYPE   → type_formatter callback, or "<type>" when .fn is NULL
//   STRING_LIST → bracketed comma-separated raw text values
//
// Returns iree_ok_status() on success.
iree_status_t loom_diagnostic_render_message(
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream);

// Renders the fix hint template using the same parameter substitution
// as loom_diagnostic_render_message. Returns iree_ok_status() if the
// error has no fix hint (appends nothing).
iree_status_t loom_diagnostic_render_fix_hint(
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream);

// A minimal type formatter that uses loom_scalar_type_name() for scalar
// types and emits the kind name for others (e.g., "tile<...>", "tensor<...>").
// Suitable for builds without the full text printer linked in.
iree_status_t loom_type_format_minimal(loom_type_t type, void* user_data,
                                       loom_output_stream_t* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_RENDERER_H_
