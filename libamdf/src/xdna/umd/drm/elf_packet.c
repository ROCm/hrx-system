// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/drm/elf_packet.h"

#include <string.h>

_Static_assert(sizeof(amdf_linux_xdna_elf_packet_t) == 48,
               "ERT ELF instruction descriptor must match the native ABI");

void amdf_linux_xdna_elf_packet_build(
    uint64_t instruction_address, uint32_t instruction_byte_length,
    amdf_linux_xdna_elf_packet_t* out_packet) {
  memset(out_packet, 0, sizeof(*out_packet));
  // NEW state, eleven payload words, opcode 22, and CU packet type.
  out_packet->words[0] = UINT32_C(0x3B00B001);
  out_packet->words[1] = 1;
  memcpy(&out_packet->words[2], &instruction_address,
         sizeof(instruction_address));
  out_packet->words[8] = instruction_byte_length;
}
