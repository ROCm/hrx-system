// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/bitcode_record_writer.h"

#include <string.h>

#include "loom/target/emit/llvmir/bitcode_format.h"

void loom_llvmir_bitcode_record_writer_initialize(
    loom_llvmir_bitstream_writer_t* bitstream,
    loom_llvmir_bitcode_record_writer_t* out_writer) {
  memset(out_writer, 0, sizeof(*out_writer));
  out_writer->bitstream = bitstream;
  out_writer->abbrev_width = 2;
}

iree_status_t loom_llvmir_bitcode_record_writer_enter_subblock(
    loom_llvmir_bitcode_record_writer_t* writer, uint32_t block_id,
    uint32_t child_abbrev_width) {
  if (child_abbrev_width == 0 || child_abbrev_width > 32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode child abbrev width is invalid");
  }
  if (writer->block_depth == LOOM_LLVMIR_BITCODE_MAX_BLOCK_DEPTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode block nesting is too deep");
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, LOOM_LLVMIR_BITCODE_ABBREV_ENTER_SUBBLOCK,
      writer->abbrev_width));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, block_id, LOOM_LLVMIR_BITCODE_BLOCK_ID_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, child_abbrev_width,
      LOOM_LLVMIR_BITCODE_CODE_LENGTH_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_align32(writer->bitstream));

  uint64_t size_field_bit_offset =
      loom_llvmir_bitstream_writer_bit_offset(writer->bitstream);
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, 0, LOOM_LLVMIR_BITCODE_BLOCK_SIZE_WIDTH));

  loom_llvmir_bitcode_block_frame_t* frame =
      &writer->block_stack[writer->block_depth++];
  frame->parent_abbrev_width = writer->abbrev_width;
  frame->parent_application_abbrev_count = writer->application_abbrev_count;
  frame->size_field_bit_offset = size_field_bit_offset;
  frame->body_start_bit_offset =
      loom_llvmir_bitstream_writer_bit_offset(writer->bitstream);
  writer->abbrev_width = child_abbrev_width;
  writer->application_abbrev_count = 0;
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitcode_record_writer_exit_block(
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (writer->block_depth == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "LLVM bitcode block stack is empty");
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, LOOM_LLVMIR_BITCODE_ABBREV_END_BLOCK,
      writer->abbrev_width));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_align32(writer->bitstream));

  loom_llvmir_bitcode_block_frame_t* frame =
      &writer->block_stack[writer->block_depth - 1];
  uint64_t body_end_bit_offset =
      loom_llvmir_bitstream_writer_bit_offset(writer->bitstream);
  uint64_t body_bit_count = body_end_bit_offset - frame->body_start_bit_offset;
  if ((body_bit_count & 31) != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "LLVM bitcode block body is not word-aligned");
  }
  uint64_t body_word_count = body_bit_count / 32;
  if (body_word_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode block body is too large");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_patch_u32(
      writer->bitstream, frame->size_field_bit_offset,
      (uint32_t)body_word_count));

  writer->abbrev_width = frame->parent_abbrev_width;
  writer->application_abbrev_count = frame->parent_application_abbrev_count;
  --writer->block_depth;
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitcode_record_writer_write_unabbrev_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    const uint64_t* operands, iree_host_size_t operand_count) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, LOOM_LLVMIR_BITCODE_ABBREV_UNABBREV_RECORD,
      writer->abbrev_width));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, code, LOOM_LLVMIR_BITCODE_RECORD_CODE_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, operand_count,
      LOOM_LLVMIR_BITCODE_RECORD_OPERAND_COUNT_WIDTH));
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
        writer->bitstream, operands[i],
        LOOM_LLVMIR_BITCODE_RECORD_OPERAND_WIDTH));
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitcode_record_writer_define_blob_abbrev(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    uint32_t* out_abbrev_id) {
  uint32_t abbrev_id = LOOM_LLVMIR_BITCODE_ABBREV_FIRST_APPLICATION +
                       writer->application_abbrev_count;
  if (writer->abbrev_width < 32 && (abbrev_id >> writer->abbrev_width) != 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode abbrev id exceeds block width");
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, LOOM_LLVMIR_BITCODE_ABBREV_DEFINE_ABBREV,
      writer->abbrev_width));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, 2, LOOM_LLVMIR_BITCODE_ABBREV_OPERAND_COUNT_WIDTH));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitstream_writer_write_bits(writer->bitstream, 1, 1));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, code, LOOM_LLVMIR_BITCODE_ABBREV_LITERAL_WIDTH));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitstream_writer_write_bits(writer->bitstream, 0, 1));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, LOOM_LLVMIR_BITCODE_ABBREV_ENCODING_BLOB,
      LOOM_LLVMIR_BITCODE_ABBREV_ENCODING_WIDTH));

  writer->application_abbrev_count += 1;
  *out_abbrev_id = abbrev_id;
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitcode_record_writer_write_blob_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint32_t abbrev_id,
    iree_string_view_t value) {
  uint32_t max_abbrev_id = LOOM_LLVMIR_BITCODE_ABBREV_FIRST_APPLICATION +
                           writer->application_abbrev_count;
  if (abbrev_id < LOOM_LLVMIR_BITCODE_ABBREV_FIRST_APPLICATION ||
      abbrev_id >= max_abbrev_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode blob abbrev is not defined");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, abbrev_id, writer->abbrev_width));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, value.size,
      LOOM_LLVMIR_BITCODE_RECORD_OPERAND_COUNT_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_align32(writer->bitstream));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bytes(
      writer->bitstream, value.data, value.size));
  return loom_llvmir_bitstream_writer_align32(writer->bitstream);
}

iree_status_t loom_llvmir_bitcode_record_writer_write_string_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_bits(
      writer->bitstream, LOOM_LLVMIR_BITCODE_ABBREV_UNABBREV_RECORD,
      writer->abbrev_width));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, code, LOOM_LLVMIR_BITCODE_RECORD_CODE_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
      writer->bitstream, value.size,
      LOOM_LLVMIR_BITCODE_RECORD_OPERAND_COUNT_WIDTH));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_write_vbr(
        writer->bitstream, (uint8_t)value.data[i],
        LOOM_LLVMIR_BITCODE_RECORD_OPERAND_WIDTH));
  }
  return iree_ok_status();
}
