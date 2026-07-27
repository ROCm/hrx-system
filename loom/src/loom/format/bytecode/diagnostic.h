// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostics for the bytecode reader.
//
// The bytecode reader validates untrusted binary input. Malformed user
// bytecode should produce catalog-backed diagnostics tagged as
// LOOM_EMITTER_BYTECODE_READER, not ad hoc status strings. Infrastructure
// failures such as OOM can still return normal iree_status_t values.

#ifndef LOOM_FORMAT_BYTECODE_DIAGNOSTIC_H_
#define LOOM_FORMAT_BYTECODE_DIAGNOSTIC_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/error/error_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Diagnostic context shared by bytecode reader decode routines.
typedef struct loom_bytecode_reader_diagnostic_context_t {
  loom_diagnostic_sink_t sink;  // Destination for structured diagnostics.
  // Logical bytecode input name, such as a path or archive member name.
  iree_string_view_t filename;
} loom_bytecode_reader_diagnostic_context_t;

// Absolute byte range in the bytecode file.
typedef struct loom_bytecode_reader_byte_range_t {
  uint64_t offset;  // Byte offset from the beginning of the bytecode file.
  uint64_t length;  // Byte length of the offending payload.
} loom_bytecode_reader_byte_range_t;

// Builds an absolute byte range for diagnostics.
static inline loom_bytecode_reader_byte_range_t loom_bytecode_reader_byte_range(
    uint64_t offset, uint64_t length) {
  return (loom_bytecode_reader_byte_range_t){
      /*.offset=*/offset,
      /*.length=*/length,
  };
}

// Emits |error| with bytecode-reader provenance and the supplied absolute byte
// range. The returned status only reports sink/infrastructure failure;
// malformed bytecode itself is represented by the emitted diagnostic.
iree_status_t loom_bytecode_reader_emit_diagnostic(
    const loom_bytecode_reader_diagnostic_context_t* context,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, loom_bytecode_reader_byte_range_t byte_range);

// Emits ERR_BYTECODE_006 for a malformed field in a section table record.
iree_status_t loom_bytecode_reader_emit_invalid_record_field(
    const loom_bytecode_reader_diagnostic_context_t* context,
    iree_string_view_t section_name, iree_string_view_t table_name,
    uint64_t record_index, iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t failure_code);

// Emits ERR_BYTECODE_007 for an out-of-bounds or overlapping byte range.
iree_status_t loom_bytecode_reader_emit_invalid_range(
    const loom_bytecode_reader_diagnostic_context_t* context,
    iree_string_view_t range_name, uint64_t offset, uint64_t length,
    uint64_t container_length);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_DIAGNOSTIC_H_
