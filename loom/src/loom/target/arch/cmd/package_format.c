// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/package_format.h"

iree_status_t loom_cmd_program_package_format_calculate_layout(
    uint32_t export_count, uint32_t entry_count, uint32_t string_length,
    uint32_t payload_length,
    loom_cmd_program_package_format_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  uint64_t offset = LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE;
  const uint64_t export_offset = offset;
  uint64_t table_length = 0;
  if (!iree_checked_mul_u64(export_count, LOOM_CMD_PROGRAM_PACKAGE_EXPORT_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command package export table is too large");
  }
  const uint64_t entry_offset = offset;
  if (!iree_checked_mul_u64(entry_count, LOOM_CMD_PROGRAM_PACKAGE_ENTRY_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command package entry table is too large");
  }
  const uint64_t string_offset = offset;
  if (!iree_checked_add_u64(offset, string_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command package string table is too large");
  }
  const uint64_t payload_offset = offset;
  if (!iree_checked_add_u64(offset, payload_length, &offset) ||
      offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command package payloads are too large");
  }
  *out_layout = (loom_cmd_program_package_format_layout_t){
      .export_offset = (uint32_t)export_offset,
      .entry_offset = (uint32_t)entry_offset,
      .string_offset = (uint32_t)string_offset,
      .payload_offset = (uint32_t)payload_offset,
      .total_length = (uint32_t)offset,
  };
  return iree_ok_status();
}
