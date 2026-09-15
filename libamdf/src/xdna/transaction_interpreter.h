// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_TRANSACTION_INTERPRETER_H_
#define AMDF_SRC_XDNA_TRANSACTION_INTERPRETER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fixed version 1 transaction-interpreter calling convention.
enum {
  AMDF_XDNA_TRANSACTION_INTERPRETER_PACKET_SIZE = 68,
  AMDF_XDNA_TRANSACTION_INTERPRETER_HEADER_NEW = 0x30010001,
  AMDF_XDNA_TRANSACTION_INTERPRETER_STATE_COMPLETED = 4,
};

// Firmware command shared by native operating-system admission envelopes.
typedef struct amdf_xdna_transaction_interpreter_packet_t {
  // Little-endian packet bytes; the native owner supplies storage alignment.
  uint8_t bytes[AMDF_XDNA_TRANSACTION_INTERPRETER_PACKET_SIZE];
} amdf_xdna_transaction_interpreter_packet_t;

// Frames a validated instruction range using the legacy interpreter envelope.
// Instruction byte length is a nonzero multiple of four. The unused argument
// area is zero. No allocation, validation, or native operation occurs here.
void amdf_xdna_transaction_interpreter_packet_build(
    uint64_t instruction_address, uint32_t instruction_byte_length,
    amdf_xdna_transaction_interpreter_packet_t* out_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_XDNA_TRANSACTION_INTERPRETER_H_
