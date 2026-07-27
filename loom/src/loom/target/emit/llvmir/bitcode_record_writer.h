// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM bitcode block and record writer.
//
// This layer owns the universal bitcode block/record envelope: subblock entry
// and exit, block-size patching, current abbreviation-code width, and
// unabbreviated record emission. LLVM IR-specific record codes and abbreviation
// definitions live above this layer.

#ifndef LOOM_TARGET_LLVMIR_BITCODE_RECORD_WRITER_H_
#define LOOM_TARGET_LLVMIR_BITCODE_RECORD_WRITER_H_

#include "loom/target/emit/llvmir/bitstream_writer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LLVMIR_BITCODE_MAX_BLOCK_DEPTH 64

typedef struct loom_llvmir_bitcode_block_frame_t {
  // Abbreviation ID width used by the parent block.
  uint32_t parent_abbrev_width;
  // Number of block-local application abbreviations in the parent block.
  uint32_t parent_application_abbrev_count;
  // Bit offset of the 32-bit block-size field to patch at block exit.
  uint64_t size_field_bit_offset;
  // Bit offset of the first bit in the block body.
  uint64_t body_start_bit_offset;
} loom_llvmir_bitcode_block_frame_t;

typedef struct loom_llvmir_bitcode_record_writer_t {
  // Physical bitstream writer used for all output.
  loom_llvmir_bitstream_writer_t* bitstream;
  // Current abbreviation ID width in bits.
  uint32_t abbrev_width;
  // Number of block-local application abbreviations defined in this block.
  uint32_t application_abbrev_count;
  // Number of active entries in |block_stack|.
  uint32_t block_depth;
  // Active nested block frames.
  loom_llvmir_bitcode_block_frame_t
      block_stack[LOOM_LLVMIR_BITCODE_MAX_BLOCK_DEPTH];
} loom_llvmir_bitcode_record_writer_t;

// Initializes |out_writer| with the top-level bitcode abbreviation width.
void loom_llvmir_bitcode_record_writer_initialize(
    loom_llvmir_bitstream_writer_t* bitstream,
    loom_llvmir_bitcode_record_writer_t* out_writer);

// Enters a subblock with |block_id| and |child_abbrev_width|.
iree_status_t loom_llvmir_bitcode_record_writer_enter_subblock(
    loom_llvmir_bitcode_record_writer_t* writer, uint32_t block_id,
    uint32_t child_abbrev_width);

// Exits the current subblock, aligns to 32 bits, and patches its word length.
iree_status_t loom_llvmir_bitcode_record_writer_exit_block(
    loom_llvmir_bitcode_record_writer_t* writer);

// Emits an unabbreviated record: code, operand count, then VBR6 operands.
iree_status_t loom_llvmir_bitcode_record_writer_write_unabbrev_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    const uint64_t* operands, iree_host_size_t operand_count);

// Defines a block-local abbreviation for records whose only payload is a blob.
// The returned abbreviation id can be used with
// loom_llvmir_bitcode_record_writer_write_blob_record().
iree_status_t loom_llvmir_bitcode_record_writer_define_blob_abbrev(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    uint32_t* out_abbrev_id);

// Emits a record using a blob abbreviation previously returned from
// loom_llvmir_bitcode_record_writer_define_blob_abbrev().
iree_status_t loom_llvmir_bitcode_record_writer_write_blob_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint32_t abbrev_id,
    iree_string_view_t value);

// Emits an unabbreviated record whose operands are bytes from |value| encoded
// as VBR6 integers. This is the canonical fallback for LLVM string records.
iree_status_t loom_llvmir_bitcode_record_writer_write_string_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    iree_string_view_t value);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_BITCODE_RECORD_WRITER_H_
