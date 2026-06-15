// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/diagnostic.h"

#include "loom/error/error_catalog.h"

static loom_source_range_t loom_bytecode_reader_make_source_range(
    const loom_bytecode_reader_diagnostic_context_t* context,
    loom_bytecode_reader_byte_range_t byte_range) {
  loom_source_range_t range = {0};
  range.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
  if (context) {
    range.filename = context->filename;
  }

  const uint64_t max_host_size = (uint64_t)(iree_host_size_t)-1;
  if (byte_range.offset <= max_host_size) {
    range.start = (iree_host_size_t)byte_range.offset;
    if (byte_range.length <= UINT64_MAX - byte_range.offset &&
        byte_range.offset + byte_range.length <= max_host_size) {
      uint64_t end = byte_range.offset + byte_range.length;
      range.end = (iree_host_size_t)end;
    } else {
      range.end = range.start;
    }
  }
  return range;
}

iree_status_t loom_bytecode_reader_emit_diagnostic(
    const loom_bytecode_reader_diagnostic_context_t* context,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count,
    loom_bytecode_reader_byte_range_t byte_range) {
  loom_source_range_t source_range =
      loom_bytecode_reader_make_source_range(context, byte_range);
  loom_diagnostic_t diagnostic = {
      .severity = error->severity,
      .error = error,
      .params = params,
      .param_count = param_count,
      .emitter = LOOM_EMITTER_BYTECODE_READER,
      .origin = source_range,
      .source_location = source_range,
  };
  return loom_diagnostic_emit(context ? &context->sink : NULL, &diagnostic);
}

iree_status_t loom_bytecode_reader_emit_invalid_record_field(
    const loom_bytecode_reader_diagnostic_context_t* context,
    iree_string_view_t section_name, iree_string_view_t table_name,
    uint64_t record_index, iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t failure_code) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(section_name), loom_param_string(table_name),
      loom_param_u64(record_index),    loom_param_string(field_name),
      loom_param_u64(offset),          loom_param_string(failure_code),
  };
  return loom_bytecode_reader_emit_diagnostic(
      context, LOOM_ERR_BYTECODE_006, params, IREE_ARRAYSIZE(params),
      loom_bytecode_reader_byte_range(offset, 1));
}

iree_status_t loom_bytecode_reader_emit_invalid_range(
    const loom_bytecode_reader_diagnostic_context_t* context,
    iree_string_view_t range_name, uint64_t offset, uint64_t length,
    uint64_t container_length) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(range_name),
      loom_param_u64(offset),
      loom_param_u64(length),
      loom_param_u64(container_length),
  };
  return loom_bytecode_reader_emit_diagnostic(
      context, LOOM_ERR_BYTECODE_007, params, IREE_ARRAYSIZE(params),
      loom_bytecode_reader_byte_range(offset, length));
}
