// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable multi-root portable command program packages.

#ifndef LOOM_TARGET_ARCH_CMD_PACKAGE_H_
#define LOOM_TARGET_ARCH_CMD_PACKAGE_H_

#include "iree/base/api.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// One executable-entry association supplied while building a package.
typedef struct loom_cmd_program_package_source_entry_t {
  // Root-local executable index providing the entry.
  uint32_t executable_index;
  // Public entry name exported by the selected executable.
  iree_string_view_t name;
} loom_cmd_program_package_source_entry_t;

// One root supplied while building a package.
typedef struct loom_cmd_program_package_source_export_t {
  // Public command-program export name.
  iree_string_view_t name;
  // Validated portable command program to embed.
  const loom_cmd_program_t* program;
  // Root-local executable entry associations in command ABI order.
  const loom_cmd_program_package_source_entry_t* entries;
  // Number of entries in |entries|.
  iree_host_size_t entry_count;
} loom_cmd_program_package_source_export_t;

// One decoded command-program export.
typedef struct loom_cmd_program_package_export_t {
  // Public command-program export name borrowed from package storage.
  iree_string_view_t name;
  // Embedded portable command program bytes.
  iree_const_byte_span_t program_data;
  // First entry in the package-global entry table.
  uint32_t first_entry;
  // Number of root-local executable entry associations.
  uint32_t entry_count;
} loom_cmd_program_package_export_t;

// One decoded root-local executable entry association.
typedef struct loom_cmd_program_package_entry_t {
  // Root-local executable index providing the entry.
  uint32_t executable_index;
  // Public entry name exported by the selected executable.
  iree_string_view_t name;
} loom_cmd_program_package_entry_t;

// Validated zero-allocation view of a serialized command package.
//
// All storage is borrowed from |data| passed to
// loom_cmd_program_package_parse and must remain live while the view is used.
// Fields are read-only after parsing.
typedef struct loom_cmd_program_package_t {
  // Complete canonical package storage.
  iree_const_byte_span_t storage;
  // First byte of the fixed-size export records.
  const uint8_t* export_table;
  // Number of command-program exports.
  uint32_t export_count;
  // First byte of the flattened executable-entry records.
  const uint8_t* entry_table;
  // Number of executable-entry records across all exports.
  uint32_t entry_count;
  // Concatenated export and entry names.
  iree_const_byte_span_t strings;
  // Concatenated command program payloads.
  iree_const_byte_span_t payloads;
} loom_cmd_program_package_t;

// Builds one canonical multi-root command package.
//
// The builder validates the package-level associations around already
// validated command programs and performs exactly one allocation for the
// returned bytes. On success |out_data| owns that allocation and must be freed
// with |allocator| after |out_package| is no longer used. |out_package| borrows
// |out_data| without a validation pass.
iree_status_t loom_cmd_program_package_build(
    const loom_cmd_program_package_source_export_t* exports,
    iree_host_size_t export_count, iree_allocator_t allocator,
    iree_byte_span_t* out_data, loom_cmd_program_package_t* out_package);

// Parses and validates one complete multi-root command package.
//
// This is the untrusted byte boundary. Successful parsing guarantees that all
// table ranges, strings, payloads, nested command programs, and executable
// entry associations satisfy the canonical format.
// Parsing borrows |data| and performs no allocations.
iree_status_t loom_cmd_program_package_parse(
    iree_const_byte_span_t data, loom_cmd_program_package_t* out_package);

// Returns one validated command-program export.
loom_cmd_program_package_export_t loom_cmd_program_package_export_at(
    const loom_cmd_program_package_t* package, uint32_t export_index);

// Looks up a validated command-program export by public name.
//
// Returns false when no export has |name|.
bool loom_cmd_program_package_lookup_export(
    const loom_cmd_program_package_t* package, iree_string_view_t name,
    loom_cmd_program_package_export_t* out_export);

// Returns one validated root-local executable entry association.
loom_cmd_program_package_entry_t loom_cmd_program_package_export_entry_at(
    const loom_cmd_program_package_t* package,
    const loom_cmd_program_package_export_t* program_export,
    uint32_t entry_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_PACKAGE_H_
