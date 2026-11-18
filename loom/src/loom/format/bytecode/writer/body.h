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

#ifdef __cplusplus
extern "C" {
#endif

// Writes an operation's presentation prefix according to the writer policy.
iree_status_t loom_bytecode_write_op_trivia(
    loom_bytecode_page_writer_t* writer,
    const loom_bytecode_numbering_t* numbering, const loom_op_t* op);

// One independently bounded root-region payload in the IR section.
typedef struct loom_bytecode_ir_region_payload_t {
  // Byte offset of the payload from the IR section start.
  uint64_t offset;
  // Byte length of the payload.
  uint32_t length;
  // Declared region slot on the defining symbol operation.
  uint8_t region_index;
} loom_bytecode_ir_region_payload_t;

static_assert(sizeof(loom_bytecode_ir_region_payload_t) == 16,
              "root-region payload records must remain 16 bytes");

// Root-region payloads owned by one symbol in declared slot order.
typedef struct loom_bytecode_ir_region_list_t {
  // Arena-owned payload records.
  loom_bytecode_ir_region_payload_t* values;
  // Number of entries in |values|.
  uint8_t count;
} loom_bytecode_ir_region_list_t;

// Shift mapping a module symbol ID to its body-payload index page.
#define LOOM_BYTECODE_IR_REGION_PAGE_SHIFT 8u

// Number of module symbol rows represented by each body-payload index page.
#define LOOM_BYTECODE_IR_REGION_PAGE_CAPACITY \
  (1u << LOOM_BYTECODE_IR_REGION_PAGE_SHIFT)

// Direct payload-list rows for one page of module symbol IDs. Separate arrays
// avoid the pointer-alignment padding of loom_bytecode_ir_region_list_t while
// keeping both fields naturally aligned.
typedef struct loom_bytecode_ir_region_page_t {
  // Payload arrays indexed by the low bits of a module symbol ID.
  loom_bytecode_ir_region_payload_t*
      values[LOOM_BYTECODE_IR_REGION_PAGE_CAPACITY];
  // Payload counts indexed by the low bits of a module symbol ID.
  uint8_t counts[LOOM_BYTECODE_IR_REGION_PAGE_CAPACITY];
} loom_bytecode_ir_region_page_t;

static_assert(sizeof(loom_bytecode_ir_region_page_t) <= 3072,
              "body payload index pages must fit in arena blocks");
static_assert(((LOOM_SYMBOL_ID_INVALID + LOOM_BYTECODE_IR_REGION_PAGE_CAPACITY -
                1u) >>
               LOOM_BYTECODE_IR_REGION_PAGE_SHIFT) *
                      sizeof(loom_bytecode_ir_region_page_t*) <=
                  2048,
              "body payload page directories must fit in arena blocks");

// Invocation-owned direct index of serialized root-region payloads.
typedef struct loom_bytecode_ir_region_index_t {
  // Lazily allocated page directory indexed by the high symbol-ID bits.
  loom_bytecode_ir_region_page_t** pages;
  // Number of payloads across all indexed symbol rows.
  iree_host_size_t payload_count;
} loom_bytecode_ir_region_index_t;

// Returns the payload list for a valid module symbol ID. Symbols whose page or
// row was not produced return the canonical empty list by value.
static inline loom_bytecode_ir_region_list_t loom_bytecode_ir_region_index_list(
    const loom_bytecode_ir_region_index_t* index, loom_symbol_id_t symbol_id) {
  const loom_bytecode_ir_region_page_t* page =
      index->pages
          ? index->pages[symbol_id >> LOOM_BYTECODE_IR_REGION_PAGE_SHIFT]
          : NULL;
  if (!page) {
    return (loom_bytecode_ir_region_list_t){
        /*.values=*/NULL,
        /*.count=*/0,
    };
  }
  const iree_host_size_t page_offset =
      symbol_id & (LOOM_BYTECODE_IR_REGION_PAGE_CAPACITY - 1u);
  return (loom_bytecode_ir_region_list_t){
      /*.values=*/page->values[page_offset],
      /*.count=*/page->counts[page_offset],
  };
}

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

// Streams one SSA value definition.
iree_status_t loom_bytecode_write_value_def(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value);

// Streams the IR section and records each symbol's root-region ranges.
iree_status_t loom_bytecode_write_ir_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    loom_bytecode_ir_region_index_t* ir_region_index);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_BODY_H_
