// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/elf_packet.h"

#include <string.h>

_Static_assert(sizeof(amdf_xdna_elf_packet_t) == 48,
               "ERT ELF instruction descriptor must match the native ABI");

void amdf_xdna_elf_admission_write(uint8_t device_generation, uint8_t row_count,
                                   uint8_t memory_row_count,
                                   uint64_t bootstrap_pdi_address,
                                   uint32_t bootstrap_pdi_length,
                                   void* target) {
  uint8_t* bytes = target;
  memset(bytes, 0, AMDF_XDNA_ELF_ADMISSION_BYTE_LENGTH);
  bytes[1] = 1;  // Transaction 0.1.
  bytes[2] = device_generation;
  bytes[3] = row_count;
  bytes[4] = 1;
  bytes[5] = memory_row_count;
  bytes[12] = AMDF_XDNA_ELF_ADMISSION_BYTE_LENGTH;
  size_t offset = 16;
  if (bootstrap_pdi_length) {
    bytes[offset] = 8;  // XAIE_IO_LOAD_PDI.
    memcpy(bytes + offset + 4, &bootstrap_pdi_length,
           sizeof(bootstrap_pdi_length));
    memcpy(bytes + offset + 8, &bootstrap_pdi_address,
           sizeof(bootstrap_pdi_address));
    ++bytes[8];  // Transaction operation count.
    offset += 16;
  }
  for (; offset < AMDF_XDNA_ELF_ADMISSION_BYTE_LENGTH; offset += 4) {
    bytes[offset] = 5;  // XAIE_IO_NOOP.
    ++bytes[8];
  }
}

void amdf_xdna_elf_packet_build(uint64_t instruction_address,
                                uint32_t instruction_byte_length,
                                amdf_xdna_elf_packet_t* out_packet) {
  memset(out_packet, 0, sizeof(*out_packet));
  // NEW state, eleven payload words, opcode 22, and CU packet type.
  out_packet->words[0] =
      UINT32_C(0x3000B001) | (AMDF_XDNA_ELF_PACKET_OPCODE << 23);
  out_packet->words[1] = 1;
  memcpy(&out_packet->words[2], &instruction_address,
         sizeof(instruction_address));
  out_packet->words[8] = instruction_byte_length;
}
