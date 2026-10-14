// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

#include <string.h>

// Type-3 counts exclude the header and the following DWORD.
static inline uint32_t iree_hal_amd_gpu_pm4_header(uint32_t opcode,
                                                   uint32_t word_count) {
  return UINT32_C(0xc0000000) | ((word_count - 2) << 16) | (opcode << 8);
}

uint32_t* iree_hal_amd_gpu_pm4_emit_registers(uint32_t* words,
                                              uint32_t first_register,
                                              uint32_t value_count,
                                              const uint32_t* values) {
  // ShaderCompute selects the ordinary compute form of SET_SH_REG.
  words[0] = iree_hal_amd_gpu_pm4_header(0x76, value_count + 2) | 2;
  words[1] = first_register - 0x2c00;
  memcpy(words + 2, values, value_count * sizeof(*values));
  return words + 2 + value_count;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_dispatch_direct(
    uint32_t* words, const uint32_t dimensions[3], uint32_t initiator) {
  words[0] = UINT32_C(0xc0031502);
  words[1] = dimensions[0];
  words[2] = dimensions[1];
  words[3] = dimensions[2];
  words[4] = initiator;
  return words + 5;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_dispatch_indirect(
    uint32_t* words, uint64_t dimensions_address, uint32_t initiator) {
  words[0] = UINT32_C(0xc0021602);
  words[1] = (uint32_t)dimensions_address;
  words[2] = (uint32_t)(dimensions_address >> 32);
  words[3] = initiator;
  return words + 4;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_compute_flush(uint32_t* words) {
  words[0] = UINT32_C(0xc0004600);
  words[1] = 7 | (4 << 8);  // CS_PARTIAL_FLUSH, partial-flush event index.
  return words + 2;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_acquire(uint32_t* words,
                                            uint32_t cache_control) {
  words[0] = UINT32_C(0xc0065800);
  words[1] = 0;
  // MEC's whole-address-space extent has an eight-bit high size field.
  words[2] = UINT32_MAX;
  words[3] = 0xff;
  words[4] = 0;
  words[5] = 0;
  words[6] = 0x0a;
  words[7] = cache_control;
  return words + 8;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_release32(uint32_t* words,
                                              uint32_t cache_control,
                                              uint64_t target_address,
                                              uint32_t value) {
  words[0] = UINT32_C(0xc0064900);
  // CACHE_FLUSH_AND_INV_TS_EVENT, end-of-pipe index, prepared release GCR.
  words[1] = 0x28 | (5 << 8) | (cache_control << 12);
  // Immediate DWORD, write confirmation without interrupt, TC/L2 destination.
  words[2] = (1 << 29) | (3 << 24) | (1 << 16);
  words[3] = (uint32_t)target_address;
  words[4] = (uint32_t)(target_address >> 32);
  words[5] = value;
  words[6] = 0;
  words[7] = 0;
  return words + 8;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_notify32(uint32_t* words,
                                             uint32_t cache_control,
                                             uint64_t target_address,
                                             uint32_t value,
                                             uint32_t interrupt_context) {
  words[0] = UINT32_C(0xc0064900);
  words[1] = 0x28 | (5 << 8) | (cache_control << 12);
  // Interrupt only after the immediate mailbox write has been confirmed.
  words[2] = (1 << 29) | (2 << 24) | (1 << 16);
  words[3] = (uint32_t)target_address;
  words[4] = (uint32_t)(target_address >> 32);
  words[5] = value;
  words[6] = 0;
  words[7] = interrupt_context;
  return words + 8;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_release64(uint32_t* words,
                                              uint32_t cache_control,
                                              uint64_t target_address,
                                              uint64_t value) {
  words[0] = UINT32_C(0xc0064900);
  words[1] = 0x28 | (5 << 8) | (cache_control << 12);
  // Immediate 64 bits, write confirmation without interrupt, TC/L2 destination.
  words[2] = (2 << 29) | (3 << 24) | (1 << 16);
  words[3] = (uint32_t)target_address;
  words[4] = (uint32_t)(target_address >> 32);
  words[5] = (uint32_t)value;
  words[6] = (uint32_t)(value >> 32);
  words[7] = 0;
  return words + 8;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_copy64(uint32_t* words,
                                           uint64_t source_address,
                                           uint64_t target_address) {
  words[0] = UINT32_C(0xc0044000);
  // TC/L2 source/destination, default policy, 64-bit count, write confirmation.
  words[1] = 2 | (2 << 8) | (1 << 16) | (1 << 20);
  words[2] = (uint32_t)source_address;
  words[3] = (uint32_t)(source_address >> 32);
  words[4] = (uint32_t)target_address;
  words[5] = (uint32_t)(target_address >> 32);
  return words + 6;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_copy32(uint32_t* words,
                                           uint64_t source_address,
                                           uint64_t target_address) {
  words[0] = UINT32_C(0xc0044000);
  words[1] = 2 | (2 << 8) | (1 << 20);
  words[2] = (uint32_t)source_address;
  words[3] = (uint32_t)(source_address >> 32);
  words[4] = (uint32_t)target_address;
  words[5] = (uint32_t)(target_address >> 32);
  return words + 6;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_write(uint32_t* words,
                                          uint64_t target_address,
                                          uint32_t value_count,
                                          const uint32_t* values) {
  words[0] = iree_hal_amd_gpu_pm4_header(0x37, value_count + 4);
  words[1] = (2 << 8) | (1 << 20);
  words[2] = (uint32_t)target_address;
  words[3] = (uint32_t)(target_address >> 32);
  memcpy(words + 4, values, value_count * sizeof(*values));
  return words + 4 + value_count;
}

uint32_t* iree_hal_amd_gpu_pm4_emit_nop(uint32_t* words, uint32_t word_count) {
  words[0] = iree_hal_amd_gpu_pm4_header(0x10, word_count);
  memset(words + 1, 0, (word_count - 1) * sizeof(*words));
  return words + word_count;
}
