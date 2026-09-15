// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/transaction_interpreter.h"

#include <string.h>

static void amdf_xdna_transaction_interpreter_write_u32(uint8_t* bytes,
                                                        uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) bytes[i] = (uint8_t)(value >> (i * 8));
}

static void amdf_xdna_transaction_interpreter_write_u64(uint8_t* bytes,
                                                        uint64_t value) {
  for (uint32_t i = 0; i < 8; ++i) bytes[i] = (uint8_t)(value >> (i * 8));
}

void amdf_xdna_transaction_interpreter_packet_build(
    uint64_t instruction_address, uint32_t instruction_byte_length,
    amdf_xdna_transaction_interpreter_packet_t* out_packet) {
  memset(out_packet, 0, sizeof(*out_packet));
  // NEW state, START_CU opcode, CU packet type, and sixteen payload words.
  amdf_xdna_transaction_interpreter_write_u32(
      out_packet->bytes, AMDF_XDNA_TRANSACTION_INTERPRETER_HEADER_NEW);
  amdf_xdna_transaction_interpreter_write_u32(out_packet->bytes + 0x04, 1);
  // Transaction execution opcode for CU zero's interpreter.
  amdf_xdna_transaction_interpreter_write_u32(out_packet->bytes + 0x08, 3);
  amdf_xdna_transaction_interpreter_write_u64(out_packet->bytes + 0x10,
                                              instruction_address);
  amdf_xdna_transaction_interpreter_write_u32(
      out_packet->bytes + 0x18, instruction_byte_length / sizeof(uint32_t));
}
