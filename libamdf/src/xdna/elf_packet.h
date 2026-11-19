// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_ELF_PACKET_H_
#define AMDF_SRC_XDNA_ELF_PACKET_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Native ERT opcode for a firmware-addressed ELF transaction.
enum { AMDF_XDNA_ELF_PACKET_OPCODE = 22 };

enum { AMDF_XDNA_ELF_ADMISSION_BYTE_LENGTH = 64 };

// Writes a transaction of NOOPs to identify the native ELF transport. A
// nonempty bootstrap PDI adds a LOAD_PDI from that firmware address. The
// provider supplies its own inert PDI and retains it through retirement.
void amdf_xdna_elf_admission_write(uint8_t device_generation, uint8_t row_count,
                                   uint8_t memory_row_count,
                                   uint64_t bootstrap_pdi_address,
                                   uint32_t bootstrap_pdi_length, void* target);

// ERT_START_NPU_PREEMPT_ELF with one CU mask and no save/restore or properties.
typedef struct amdf_xdna_elf_packet_t {
  // Native little-endian transport words, including kernel-written state.
  uint32_t words[12];
} amdf_xdna_elf_packet_t;

// Frames a validated instruction range without accessing its contents. The
// native queue exclusively owns packet storage through checked retirement.
void amdf_xdna_elf_packet_build(uint64_t instruction_address,
                                uint32_t instruction_byte_length,
                                amdf_xdna_elf_packet_t* out_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_XDNA_ELF_PACKET_H_
