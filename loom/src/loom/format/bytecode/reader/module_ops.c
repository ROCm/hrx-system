// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/module_ops.h"

#include "loom/error/error_catalog.h"

iree_status_t loom_bytecode_module_ops_summary_read(
    loom_bytecode_reader_decoder_t* decoder,
    iree_const_byte_span_t payload_bytes, uint64_t payload_absolute_offset,
    loom_bytecode_module_ops_summary_t* out_summary) {
  *out_summary = (loom_bytecode_module_ops_summary_t){0};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      payload_bytes.data, payload_bytes.data_length, payload_absolute_offset,
      IREE_SV("MODULE_OPS"), &cursor);
  uint64_t value_count = 0;
  uint64_t region_count = 0;
  uint64_t block_count = 0;
  uint64_t op_count = 0;
  uint64_t root_op_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &value_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &region_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &block_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &op_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &root_op_count));
  const iree_host_size_t payload_length = cursor.cursor.length;
  if (root_op_count == 0 || root_op_count > op_count ||
      value_count > UINT32_MAX || region_count > UINT32_MAX ||
      block_count > UINT32_MAX || op_count > UINT32_MAX ||
      root_op_count > UINT32_MAX || value_count > payload_length ||
      region_count > payload_length || block_count > payload_length ||
      op_count > payload_length || root_op_count > payload_length) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("MODULE_OPS")),
        loom_param_u64(payload_absolute_offset),
        loom_param_string(
            IREE_SV("module_operation_allocation_summary_is_invalid")),
    };
    return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_016,
                                           params, IREE_ARRAYSIZE(params),
                                           payload_absolute_offset, 0);
  }
  *out_summary = (loom_bytecode_module_ops_summary_t){
      .value_count = (uint32_t)value_count,
      .region_count = (uint32_t)region_count,
      .block_count = (uint32_t)block_count,
      .op_count = (uint32_t)op_count,
      .root_op_count = (uint32_t)root_op_count,
      .payload_offset = (uint8_t)cursor.cursor.position,
  };
  return iree_ok_status();
}
