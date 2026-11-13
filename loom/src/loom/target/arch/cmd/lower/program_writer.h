// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable command encoding from resolved semantic tables.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_WRITER_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_WRITER_H_

#include "iree/base/api.h"
#include "loom/runtime/command/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed semantic tables for one closed command program.
//
// The producer owns resource placement, entry/schema identities, argument
// payload encoding, and command order. All referenced rows, slices, and keys
// remain live through loom_cmd_program_write(). Each table count fits its
// encoded width; the writer checks the aggregate artifact length.
typedef struct loom_cmd_program_tables_t {
  // External resources required by the program.
  loom_cmd_program_requirements_t requirements;
  // Source launch-buffer placements in signature order.
  const loom_cmd_program_launch_binding_t* launch_bindings;
  // Number of entries in |launch_bindings|.
  uint32_t launch_binding_count;
  // Buffer ranges used by fills, copies, and indirect launch counts.
  const loom_cmd_program_buffer_ref_t* buffer_refs;
  // Number of entries in |buffer_refs|.
  uint32_t buffer_ref_count;
  // Executable-entry logical argument schemas.
  const loom_cmd_program_entry_schema_t* entry_schemas;
  // Number of entries in |entry_schemas|.
  uint32_t entry_schema_count;
  // Flattened one-byte logical argument kinds referenced by entry schemas.
  const uint8_t* entry_schema_kinds;
  // Number of bytes in |entry_schema_kinds|.
  uint32_t entry_schema_kind_count;
  // Tagless dispatch argument payloads in command traversal order.
  const uint8_t* argument_data;
  // Number of bytes in |argument_data|.
  uint32_t argument_data_length;
  // Ordered command rows, including execution barriers and wave flags.
  const loom_cmd_program_command_t* commands;
  // Number of entries in |commands|.
  uint32_t command_count;
  // Fixed parameter-buffer roots in canonical ascending root order.
  const loom_cmd_program_parameter_root_t* parameter_roots;
  // Number of entries in |parameter_roots|.
  uint32_t parameter_root_count;
  // Unique fixed-root/key parameters in first source occurrence order.
  const loom_cmd_program_parameter_t* parameters;
  // Number of entries in |parameters|.
  uint32_t parameter_count;
  // Exact sum of the key lengths in |parameters|, retained by placement.
  uint32_t parameter_key_length;
} loom_cmd_program_tables_t;

// Encodes trusted compiler-produced tables into one owned portable artifact.
//
// Encoding preserves all row identities and copies keys and argument bytes
// into the output. No input storage is retained. Format-size overflow and
// output allocation are the only fallible operations.
//
// |out_data| is empty on failure. The caller owns returned bytes and must free
// them with |host_allocator|.
iree_status_t loom_cmd_program_write(const loom_cmd_program_tables_t* tables,
                                     iree_allocator_t host_allocator,
                                     iree_byte_span_t* out_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_WRITER_H_
