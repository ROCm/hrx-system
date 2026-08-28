// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode IR body serialization.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_BODY_H_
#define LOOM_FORMAT_BYTECODE_WRITER_BODY_H_

#include "loom/format/bytecode/writer/encoder.h"
#include "loom/format/bytecode/writer/numbering.h"
#include "loom/ir/module_record.h"

#ifdef __cplusplus
extern "C" {
#endif

// One independently bounded root-region payload in the IR section.
typedef struct loom_bytecode_ir_region_payload_t {
  // Byte offset of the payload from the IR section start.
  uint64_t offset;
  // Byte length of the payload.
  uint32_t length;
  // Declared region slot on the defining symbol operation.
  uint8_t region_index;
} loom_bytecode_ir_region_payload_t;

// Root-region payloads owned by one symbol in declared slot order.
typedef struct loom_bytecode_ir_region_list_t {
  // Arena-owned payload records.
  loom_bytecode_ir_region_payload_t* values;
  // Number of entries in |values|.
  uint8_t count;
} loom_bytecode_ir_region_list_t;

// Aggregate allocation counts for all serialized IR bodies in one module.
typedef struct loom_bytecode_body_counts_t {
  // Number of SSA values described by this allocation summary.
  uint64_t value_count;
  // Number of serialized regions described by this allocation summary.
  uint64_t region_count;
  // Number of serialized blocks described by this allocation summary.
  uint64_t block_count;
  // Number of serialized live operations described by this allocation summary.
  uint64_t op_count;
} loom_bytecode_body_counts_t;

// Counts the serialized body allocation totals for the module.
iree_status_t loom_bytecode_count_serialized_bodies(
    loom_bytecode_numbering_t* numbering, loom_bytecode_body_counts_t* counts);

// Counts the non-symbol module operation forest in wire order.
iree_status_t loom_bytecode_count_serialized_module_ops(
    const loom_module_t* module, const loom_module_record_plan_t* record_plan,
    loom_bytecode_body_counts_t* counts, iree_host_size_t* out_root_op_count);

// Appends one SSA value definition to a buffered metadata payload.
iree_status_t loom_bytecode_emit_value_def(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value);

// Streams the IR section and records each symbol's root-region ranges.
iree_status_t loom_bytecode_write_ir_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    loom_bytecode_ir_region_list_t* ir_regions);

// Streams the independently bounded non-symbol module operation forest.
iree_status_t loom_bytecode_write_module_ops_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    const loom_module_record_plan_t* record_plan,
    const loom_bytecode_body_counts_t* counts, iree_host_size_t root_op_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_BODY_H_
