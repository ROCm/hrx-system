// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/format.h"

#include "iree/base/alignment.h"

iree_status_t loom_cmd_program_format_calculate_layout(
    uint32_t buffer_ref_count, uint32_t entry_schema_count,
    uint32_t entry_schema_kind_count, uint32_t argument_data_length,
    uint32_t command_count, uint32_t parameter_root_count,
    uint32_t parameter_count, uint32_t parameter_key_length,
    loom_cmd_program_format_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  uint64_t offset = LOOM_CMD_PROGRAM_HEADER_SIZE;
  const uint64_t buffer_ref_offset = offset;
  uint64_t table_length = 0;
  if (!iree_checked_mul_u64(buffer_ref_count, LOOM_CMD_PROGRAM_BUFFER_REF_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command buffer-reference table is too large");
  }
  const uint64_t entry_schema_offset = offset;
  if (!iree_checked_mul_u64(entry_schema_count,
                            LOOM_CMD_PROGRAM_ENTRY_SCHEMA_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command entry-schema table is too large");
  }
  const uint64_t entry_schema_kind_offset = offset;
  if (!iree_checked_add_u64(offset, entry_schema_kind_count, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command entry-schema kind table is too large");
  }
  const uint64_t argument_data_offset = offset;
  if (!iree_checked_add_u64(offset, argument_data_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command argument payloads are too large");
  }
  const uint64_t command_offset = offset;
  if (!iree_checked_mul_u64(command_count, LOOM_CMD_PROGRAM_COMMAND_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command table is too large");
  }
  const uint64_t parameter_root_offset = offset;
  if (!iree_checked_mul_u64(parameter_root_count,
                            LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parameter-root table is too large");
  }
  const uint64_t parameter_offset = offset;
  if (!iree_checked_mul_u64(parameter_count, LOOM_CMD_PROGRAM_PARAMETER_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parameter table is too large");
  }
  const uint64_t parameter_key_offset = offset;
  if (!iree_checked_add_u64(offset, parameter_key_length, &offset) ||
      offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parameter keys are too large");
  }
  *out_layout = (loom_cmd_program_format_layout_t){
      .buffer_ref_offset = (uint32_t)buffer_ref_offset,
      .entry_schema_offset = (uint32_t)entry_schema_offset,
      .entry_schema_kind_offset = (uint32_t)entry_schema_kind_offset,
      .argument_data_offset = (uint32_t)argument_data_offset,
      .command_offset = (uint32_t)command_offset,
      .parameter_root_offset = (uint32_t)parameter_root_offset,
      .parameter_offset = (uint32_t)parameter_offset,
      .parameter_key_offset = (uint32_t)parameter_key_offset,
      .total_length = (uint32_t)offset,
  };
  return iree_ok_status();
}
