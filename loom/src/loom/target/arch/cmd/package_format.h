// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical little-endian encoding of a multi-root command package.
//
// The fixed header is followed without padding by the export table, flattened
// executable-entry table, string table, and payload storage. Strings appear in
// export order as one export name followed by its entry names. Payloads appear
// in export order as one command program per export. Name offsets are relative
// to the string table; payload offsets are relative to the artifact. Multi-byte
// fields may be unaligned and must be accessed with the IREE unaligned
// little-endian helpers. All offsets, lengths, and counts are 32-bit values.

#ifndef LOOM_TARGET_ARCH_CMD_PACKAGE_FORMAT_H_
#define LOOM_TARGET_ARCH_CMD_PACKAGE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Eight-byte artifact identity including the trailing NUL byte.
#define LOOM_CMD_PROGRAM_PACKAGE_FORMAT_MAGIC "LOOMCPK"

enum {
  LOOM_CMD_PROGRAM_PACKAGE_FORMAT_MAGIC_LENGTH = 8,
  LOOM_CMD_PROGRAM_PACKAGE_FORMAT_VERSION = 0,

  LOOM_CMD_PROGRAM_PACKAGE_HEADER_MAGIC_OFFSET = 0,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_VERSION_OFFSET = 8,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE_OFFSET = 10,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_TOTAL_LENGTH_OFFSET = 12,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_COUNT_OFFSET = 16,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_COUNT_OFFSET = 20,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_TABLE_OFFSET = 24,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_TABLE_OFFSET = 28,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_STRING_TABLE_OFFSET = 32,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_STRING_TABLE_LENGTH_OFFSET = 36,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_PAYLOAD_OFFSET = 40,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_PAYLOAD_LENGTH_OFFSET = 44,
  LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE = 48,

  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_OFFSET_OFFSET = 0,
  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_LENGTH_OFFSET = 4,
  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_FIRST_ENTRY_OFFSET = 8,
  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_ENTRY_COUNT_OFFSET = 12,
  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_OFFSET_OFFSET = 16,
  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_LENGTH_OFFSET = 20,
  LOOM_CMD_PROGRAM_PACKAGE_EXPORT_SIZE = 24,

  LOOM_CMD_PROGRAM_PACKAGE_ENTRY_EXECUTABLE_INDEX_OFFSET = 0,
  LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_OFFSET_OFFSET = 4,
  LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_LENGTH_OFFSET = 8,
  LOOM_CMD_PROGRAM_PACKAGE_ENTRY_SIZE = 12,
};

// Canonical byte offsets derived from package table and payload lengths.
typedef struct loom_cmd_program_package_format_layout_t {
  // First byte of the command export table.
  uint32_t export_offset;
  // First byte of the flattened executable-entry table.
  uint32_t entry_offset;
  // First byte of concatenated export and entry names.
  uint32_t string_offset;
  // First byte of concatenated command program payloads.
  uint32_t payload_offset;
  // Total canonical artifact byte length.
  uint32_t total_length;
} loom_cmd_program_package_format_layout_t;

// Calculates the canonical layout for one command package artifact.
iree_status_t loom_cmd_program_package_format_calculate_layout(
    uint32_t export_count, uint32_t entry_count, uint32_t string_length,
    uint32_t payload_length,
    loom_cmd_program_package_format_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_PACKAGE_FORMAT_H_
