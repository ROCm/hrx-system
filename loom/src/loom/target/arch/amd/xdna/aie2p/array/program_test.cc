// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/program.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(Aie2pArrayProgramTest, EncodesTypedRecordsAndResolvesRelocation) {
  const uint32_t array_words[] = {0x01020304, 0x11223344, 0xA0B0C0D0};
  loom_aie2p_program_record_t array_records[4] = {};
  array_records[0].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32;
  array_records[0].value.register_write32 = {/*.address=*/0x1000,
                                             /*.value=*/0xCAFEBABE};
  array_records[1].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32;
  array_records[1].value.register_mask_write32 = {
      /*.address=*/0x2000,
      /*.mask=*/0x00FF0000,
      /*.value=*/0x005A0000,
  };
  array_records[2].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32;
  array_records[2].value.register_block_write32 = {
      /*.address=*/0x3000,
      /*.words=*/array_words,
      /*.word_count=*/IREE_ARRAYSIZE(array_words),
  };
  array_records[3].type = LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD;
  array_records[3].value.tile_program_load = {/*.tile_program_index=*/0};
  const uint32_t control_words[] = {0, 0, 0};
  loom_aie2p_program_record_t control_records[2] = {};
  control_records[0].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32;
  control_records[0].value.register_block_write32 = {
      /*.address=*/0x4000,
      /*.words=*/control_words,
      /*.word_count=*/IREE_ARRAYSIZE(control_words),
  };
  control_records[1].type = LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT;
  control_records[1].value.dma_task_wait = {
      /*.coordinate=*/{2, 0},
      /*.direction=*/LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/1,
      /*.column_count=*/3,
      /*.row_count=*/4,
  };
  const loom_aie2p_program_relocation_t relocation = {
      /*.target_record_index=*/0,
      /*.target_word_index=*/1,
      /*.binding_ordinal=*/5,
      /*.kind=*/LOOM_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS,
      /*.field_byte_width=*/8,
      /*.addend=*/64,
      /*.minimum_value=*/0,
      /*.maximum_value=*/UINT64_C(0x0000FFFFFFFFFFFC),
      /*.required_alignment=*/4,
  };
  const loom_aie2p_array_program_t program = {
      /*.array_records=*/array_records,
      /*.array_record_count=*/IREE_ARRAYSIZE(array_records),
      /*.control_records=*/control_records,
      /*.control_record_count=*/IREE_ARRAYSIZE(control_records),
      /*.relocations=*/&relocation,
      /*.relocation_count=*/1,
      /*.tile_program_count=*/1,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_encoded_array_program_t encoded = {};
  IREE_ASSERT_OK(loom_aie2p_array_program_encode(
      &program, /*first_tile_program_header_ordinal=*/7,
      /*control_program_header_ordinal=*/13, &arena, &encoded));

  ASSERT_EQ(encoded.array_payload.data_length, 108u);
  const uint8_t* array = encoded.array_payload.data;
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 0), LOOM_XDNA_ELF_ARRAY_MAGIC);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 12), 4u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 16), 108u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 24), 7u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 28), 1u);

  EXPECT_EQ(iree_unaligned_load_le_u16(array + 32),
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 36), 16u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 40), 0x1000u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 44), 0xCAFEBABEu);

  EXPECT_EQ(iree_unaligned_load_le_u16(array + 48),
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 52), 20u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 56), 0x2000u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 60), 0x00FF0000u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 64), 0x005A0000u);

  EXPECT_EQ(iree_unaligned_load_le_u16(array + 68),
            LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 72), 28u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 76), 0x3000u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 80), 3u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 84), array_words[0]);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 92), array_words[2]);

  EXPECT_EQ(iree_unaligned_load_le_u16(array + 96),
            LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 100), 12u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array + 104), 7u);

  ASSERT_EQ(encoded.control_payload.data_length, 68u);
  const uint8_t* control = encoded.control_payload.data;
  EXPECT_EQ(iree_unaligned_load_le_u32(control + 0),
            LOOM_XDNA_ELF_CONTROL_MAGIC);
  EXPECT_EQ(iree_unaligned_load_le_u32(control + 12), 2u);
  EXPECT_EQ(iree_unaligned_load_le_u32(control + 16), 68u);
  EXPECT_EQ(iree_unaligned_load_le_u16(control + 52),
            LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT);
  EXPECT_EQ(iree_unaligned_load_le_u32(control + 56), 16u);
  EXPECT_EQ(control[60], 2u);
  EXPECT_EQ(control[61], 0u);
  EXPECT_EQ(control[62], 0u);
  EXPECT_EQ(control[63], 1u);
  EXPECT_EQ(control[64], 3u);
  EXPECT_EQ(control[65], 4u);

  ASSERT_EQ(encoded.relocation_count, 1u);
  EXPECT_EQ(encoded.relocations[0].target_program_header_ordinal, 13u);
  EXPECT_EQ(encoded.relocations[0].target_byte_offset, 44u);
  EXPECT_EQ(encoded.relocations[0].binding_ordinal, 5u);
  EXPECT_EQ(encoded.relocations[0].field_byte_width, 8u);
  EXPECT_EQ(encoded.relocations[0].addend, 64);
  EXPECT_EQ(encoded.relocations[0].required_alignment, 4u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
