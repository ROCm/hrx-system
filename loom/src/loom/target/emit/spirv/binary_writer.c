// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/binary_writer.h"

#include <inttypes.h>
#include <string.h>

void loom_spirv_binary_writer_initialize(
    iree_allocator_t allocator, loom_spirv_binary_writer_t* out_writer) {
  *out_writer = (loom_spirv_binary_writer_t){
      .allocator = allocator,
  };
}

void loom_spirv_binary_writer_deinitialize(loom_spirv_binary_writer_t* writer) {
  IREE_ASSERT_ARGUMENT(writer);
  iree_allocator_free(writer->allocator, writer->words);
  *writer = (loom_spirv_binary_writer_t){0};
}

void loom_spirv_binary_writer_steal_words(loom_spirv_binary_writer_t* writer,
                                          uint32_t** out_words,
                                          iree_host_size_t* out_word_count) {
  IREE_ASSERT_ARGUMENT(writer);
  IREE_ASSERT_ARGUMENT(out_words);
  IREE_ASSERT_ARGUMENT(out_word_count);
  *out_words = writer->words;
  *out_word_count = writer->word_count;
  writer->words = NULL;
  writer->word_count = 0;
  writer->word_capacity = 0;
}

static iree_status_t loom_spirv_binary_writer_reserve(
    loom_spirv_binary_writer_t* writer, iree_host_size_t additional_words) {
  iree_host_size_t minimum_word_capacity = 0;
  if (!iree_host_size_checked_add(writer->word_count, additional_words,
                                  &minimum_word_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SPIR-V binary word count overflow");
  }
  if (minimum_word_capacity <= writer->word_capacity) {
    return iree_ok_status();
  }

  iree_host_size_t new_word_capacity =
      writer->word_capacity ? writer->word_capacity * 2 : 128;
  if (new_word_capacity < writer->word_capacity ||
      new_word_capacity < minimum_word_capacity) {
    new_word_capacity = minimum_word_capacity;
  }
  iree_host_size_t new_byte_capacity = 0;
  if (!iree_host_size_checked_mul(new_word_capacity, sizeof(uint32_t),
                                  &new_byte_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SPIR-V binary byte count overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      writer->allocator, new_byte_capacity, (void**)&writer->words));
  writer->word_capacity = new_word_capacity;
  return iree_ok_status();
}

iree_status_t loom_spirv_binary_write_word(loom_spirv_binary_writer_t* writer,
                                           uint32_t word) {
  IREE_ASSERT_ARGUMENT(writer);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_writer_reserve(writer, 1));
  writer->words[writer->word_count++] = word;
  return iree_ok_status();
}

iree_status_t loom_spirv_binary_write_words(loom_spirv_binary_writer_t* writer,
                                            const uint32_t* words,
                                            iree_host_size_t word_count) {
  IREE_ASSERT_ARGUMENT(writer);
  if (word_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(words);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_writer_reserve(writer, word_count));
  memcpy(writer->words + writer->word_count, words,
         word_count * sizeof(uint32_t));
  writer->word_count += word_count;
  return iree_ok_status();
}

static iree_status_t loom_spirv_binary_write_instruction_header(
    loom_spirv_binary_writer_t* writer, uint16_t opcode,
    iree_host_size_t operand_word_count) {
  if (operand_word_count > UINT16_MAX - 1u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SPIR-V opcode %" PRIu16
                            " operand count exceeds instruction limit",
                            opcode);
  }
  const uint32_t instruction_word_count = (uint32_t)operand_word_count + 1u;
  const uint32_t header = (instruction_word_count << 16) | opcode;
  return loom_spirv_binary_write_word(writer, header);
}

iree_status_t loom_spirv_binary_write_instruction(
    loom_spirv_binary_writer_t* writer, uint16_t opcode,
    const uint32_t* operands, iree_host_size_t operand_count) {
  IREE_ASSERT_ARGUMENT(writer);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction_header(
      writer, opcode, operand_count));
  return loom_spirv_binary_write_words(writer, operands, operand_count);
}

static iree_status_t loom_spirv_binary_write_string_words(
    loom_spirv_binary_writer_t* writer, iree_string_view_t string,
    iree_host_size_t* out_word_count) {
  iree_host_size_t byte_count = 0;
  if (!iree_host_size_checked_add(string.size, 1, &byte_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SPIR-V literal string length overflow");
  }
  const iree_host_size_t word_count = iree_host_size_ceil_div(byte_count, 4);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_writer_reserve(writer, word_count));
  uint32_t* string_words = writer->words + writer->word_count;
  memset(string_words, 0, word_count * sizeof(uint32_t));
  memcpy(string_words, string.data, string.size);
  writer->word_count += word_count;
  *out_word_count = word_count;
  return iree_ok_status();
}

iree_status_t loom_spirv_binary_write_string_instruction(
    loom_spirv_binary_writer_t* writer, uint16_t opcode,
    const uint32_t* prefix_operands, iree_host_size_t prefix_operand_count,
    iree_string_view_t string, const uint32_t* suffix_operands,
    iree_host_size_t suffix_operand_count) {
  IREE_ASSERT_ARGUMENT(writer);
  iree_host_size_t string_byte_count = 0;
  if (!iree_host_size_checked_add(string.size, 1, &string_byte_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SPIR-V literal string length overflow");
  }
  const iree_host_size_t string_word_count =
      iree_host_size_ceil_div(string_byte_count, 4);
  iree_host_size_t operand_word_count = 0;
  if (!iree_host_size_checked_add(prefix_operand_count, string_word_count,
                                  &operand_word_count) ||
      !iree_host_size_checked_add(operand_word_count, suffix_operand_count,
                                  &operand_word_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SPIR-V instruction word count overflow");
  }

  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction_header(
      writer, opcode, operand_word_count));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_words(writer, prefix_operands,
                                                     prefix_operand_count));
  iree_host_size_t written_string_words = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_words(
      writer, string, &written_string_words));
  IREE_ASSERT_EQ(written_string_words, string_word_count);
  return loom_spirv_binary_write_words(writer, suffix_operands,
                                       suffix_operand_count);
}
