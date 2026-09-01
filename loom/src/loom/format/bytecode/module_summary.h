// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Validated bytecode module allocation and table-count summary.

#ifndef LOOM_FORMAT_BYTECODE_MODULE_SUMMARY_H_
#define LOOM_FORMAT_BYTECODE_MODULE_SUMMARY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lightweight metadata decoded from one module header and its tables.
typedef struct loom_bytecode_module_metadata_summary_t {
  // Allocation-summary SSA value count.
  uint64_t value_count;
  // Allocation-summary region count.
  uint64_t region_count;
  // Allocation-summary block count.
  uint64_t block_count;
  // Allocation-summary operation count.
  uint64_t op_count;
  // STRINGS table entry count.
  uint64_t string_count;
  // SOURCES table entry count.
  uint64_t source_count;
  // TYPES table entry count.
  uint64_t type_count;
  // ENCODINGS instance table entry count.
  uint64_t encoding_count;
  // OPS table entry count.
  uint64_t op_name_count;
  // LOCATIONS table entry count, or zero when omitted.
  uint64_t location_count;
  // SYMBOLS table entry count.
  uint64_t symbol_count;
  // Dependency occurrence count across module and symbol rows.
  uint64_t dependency_count;
  // Abstract provider demand occurrence count across symbol rows.
  uint64_t template_demand_count;
} loom_bytecode_module_metadata_summary_t;

// Allocation summary decoded from the MODULE_OPS section prefix.
typedef struct loom_bytecode_module_ops_summary_t {
  // SSA values defined by root module operations and nested regions.
  uint32_t value_count;
  // Regions nested under root module operations.
  uint32_t region_count;
  // Blocks nested under root module operations.
  uint32_t block_count;
  // Root and nested operations in the section.
  uint32_t op_count;
  // Root non-symbol operations owned directly by the module body.
  uint32_t root_op_count;
  // Byte offset of the first root operation after the five bounded varints.
  uint8_t payload_offset;
} loom_bytecode_module_ops_summary_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_MODULE_SUMMARY_H_
