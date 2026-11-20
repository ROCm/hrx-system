// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_STARTUP_RECORDS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_STARTUP_RECORDS_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/program.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

static inline loom_aie2p_program_record_t loom_aie2p_startup_write(
    uint32_t address, uint32_t value) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value.register_write32 = {address, value}};
}

static inline loom_aie2p_program_record_t loom_aie2p_startup_block(
    uint32_t address, const uint32_t* words, iree_host_size_t word_count) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
      .value.register_block_write32 = {address, words, word_count}};
}

static inline loom_aie2p_program_record_t loom_aie2p_startup_wait(
    uint32_t address, uint32_t mask, uint32_t value) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32,
      .value.register_mask_wait32 = {address, mask, value}};
}

// All temporary resources use engine, descriptor, and hardware semaphore zero.
static inline uint32_t loom_aie2p_startup_address(
    const loom_xdna_array_family_t* family, loom_xdna_tile_coordinate_t tile,
    loom_xdna_register_field_id_t field) {
  const uint16_t index = 0;
  return (uint32_t)loom_xdna_register_field_address(family, field, tile,
                                                    &index);
}

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_STARTUP_RECORDS_H_
