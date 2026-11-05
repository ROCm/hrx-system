// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/decoder.h"

#include "loom/error/error_catalog.h"

// Converts successful error-diagnostic delivery into the private unwind marker.
static iree_status_t loom_bytecode_reader_defer_emitted_error(
    iree_status_t sink_status) {
  if (IREE_UNLIKELY(iree_status_is_deferred(sink_status))) {
    iree_status_ignore(sink_status);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "bytecode diagnostic sinks must not return the reserved DEFERRED "
        "status");
  }
  if (IREE_UNLIKELY(!iree_status_is_ok(sink_status))) {
    return sink_status;
  }
  return iree_status_from_code(IREE_STATUS_DEFERRED);
}

iree_status_t loom_bytecode_reader_emit_error(
    loom_bytecode_reader_decoder_t* decoder, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    uint64_t offset, uint64_t length) {
  ++*decoder->error_count;
  return loom_bytecode_reader_defer_emitted_error(
      loom_bytecode_reader_emit_diagnostic(
          &decoder->diagnostic_context, error, params, param_count,
          loom_bytecode_reader_byte_range(offset, length)));
}

iree_status_t loom_bytecode_reader_emit_unexpected_end(
    loom_bytecode_reader_decoder_t* decoder, uint64_t offset, uint64_t needed,
    uint64_t available) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(offset),
      loom_param_u64(needed),
      loom_param_u64(available),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_003, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

iree_status_t loom_bytecode_reader_emit_invalid_varint(
    loom_bytecode_reader_decoder_t* decoder, uint64_t offset,
    iree_status_t decode_status) {
  const bool reached_end = iree_status_is_out_of_range(decode_status);
  iree_status_free(decode_status);
  const iree_string_view_t failure_code =
      reached_end ? IREE_SV("unterminated_varint")
                  : IREE_SV("noncanonical_or_uint64_overflow");
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(offset),
      loom_param_string(failure_code),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_008, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

iree_status_t loom_bytecode_reader_emit_invalid_field(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t section_name,
    iree_string_view_t table_name, uint64_t record_index,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t failure_code) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(section_name), loom_param_string(table_name),
      loom_param_u64(record_index),    loom_param_string(field_name),
      loom_param_u64(offset),          loom_param_string(failure_code),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_006, params,
                                         IREE_ARRAYSIZE(params), offset, 1);
}

iree_status_t loom_bytecode_reader_emit_range_error(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(range_name),
      loom_param_u64(offset),
      loom_param_u64(length),
      loom_param_u64(container_length),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_007, params,
                                         IREE_ARRAYSIZE(params), offset,
                                         length);
}

iree_status_t loom_bytecode_reader_emit_count_exceeds(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t table_name,
    uint64_t count, uint64_t limit, uint64_t offset) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(table_name),
      loom_param_u64(count),
      loom_param_u64(limit),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_009, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

iree_status_t loom_bytecode_reader_emit_table_ref(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t table_name,
    uint64_t reference_id, uint64_t table_count, uint64_t offset) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(table_name),
      loom_param_u64(reference_id),
      loom_param_u64(table_count),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_012, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

iree_status_t loom_bytecode_reader_emit_enum_value(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t field_name,
    uint64_t actual_value, uint64_t case_count, uint64_t offset) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u64(actual_value),
      loom_param_u64(case_count),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_011, params,
                                         IREE_ARRAYSIZE(params), offset, 1);
}
