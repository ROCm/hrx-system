// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program.h"

// Record cardinalities and word extents are established by array programming.
uint32_t loom_aie2p_program_native_record_size(
    const loom_aie2p_program_record_t* record) {
  switch (record->type) {
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      return 24;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      return 28;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32:
      return 32;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
      return 16 +
             (uint32_t)record->value.register_block_write32.word_count * 4u;
    case LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
    case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      return 16;
    default:
      IREE_ASSERT_UNREACHABLE("native program record kind");
      return 0;
  }
}
