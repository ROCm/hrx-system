// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/tile_image.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

uint16_t LoadLeU16(const std::string& bytes, size_t offset) {
  return (uint16_t)(uint8_t)bytes[offset] |
         ((uint16_t)(uint8_t)bytes[offset + 1] << 8);
}

uint32_t LoadLeU32(const std::string& bytes, size_t offset) {
  return (uint32_t)(uint8_t)bytes[offset] |
         ((uint32_t)(uint8_t)bytes[offset + 1] << 8) |
         ((uint32_t)(uint8_t)bytes[offset + 2] << 16) |
         ((uint32_t)(uint8_t)bytes[offset + 3] << 24);
}

TEST(Aie2pTileImageTest, WritesExecutableElfFromLeafContribution) {
  const uint8_t code[16] = {
      0x7f, 0x20, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x19, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  const loom_native_section_contribution_t section = {
      /*.section_name=*/IREE_SV(".text.kernel"),
      /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
          LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
      /*.contribution_alignment=*/16,
      /*.entry_size=*/0,
      /*.link=*/0,
      /*.info=*/0,
      /*.contents=*/iree_make_const_byte_span(code, sizeof(code)),
  };
  const loom_native_object_symbol_t symbol = {
      /*.name=*/IREE_SV("kernel"),
      /*.section_contribution_index=*/0,
      /*.section_offset=*/4,
      /*.size=*/sizeof(code) - 4,
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  const loom_aie2p_leaf_contribution_t contribution = {
      /*.object=*/
      {
          /*.sections=*/&section,
          /*.section_count=*/1,
          /*.symbols=*/&symbol,
          /*.symbol_count=*/1,
      },
      /*.realization=*/
      {
          /*.target_identity=*/LOOM_AIE2P_LEAF_TARGET_IDENTITY,
          /*.abi_identity=*/LOOM_AIE2P_LEAF_ABI_IDENTITY,
          /*.entry_symbol_index=*/0,
          /*.elf_machine=*/LOOM_XDNA_ELF_MACHINE_AIE,
          /*.target_generation=*/LOOM_XDNA_TARGET_GENERATION_AIE2P,
          /*.elf_flags=*/LOOM_XDNA_ELF_AIE2P_FLAGS,
          /*.capability_flags=*/0,
          /*.code=*/{sizeof(code), 16},
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  StreamPtr stream_owner(stream, iree_io_stream_release);

  IREE_ASSERT_OK(loom_aie2p_tile_image_write(&contribution, stream, &arena));
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  ASSERT_GT(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ((uint8_t)bytes[4], 1u);
  EXPECT_EQ(LoadLeU16(bytes, 16), LOOM_NATIVE_ELF_FILE_TYPE_EXEC);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AIE);
  EXPECT_EQ(LoadLeU32(bytes, 24), 4u);
  EXPECT_EQ(LoadLeU32(bytes, 28), 52u);
  EXPECT_EQ(LoadLeU32(bytes, 36), 3u);
  EXPECT_EQ(LoadLeU16(bytes, 42), 32u);
  EXPECT_EQ(LoadLeU16(bytes, 44), 1u);

  constexpr size_t kProgramHeaderOffset = 52;
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 0),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD);
  const uint32_t code_offset = LoadLeU32(bytes, kProgramHeaderOffset + 4);
  EXPECT_EQ(code_offset % 16, 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 8), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 12), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 16), sizeof(code));
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 20), sizeof(code));
  EXPECT_EQ(
      LoadLeU32(bytes, kProgramHeaderOffset + 24),
      LOOM_NATIVE_ELF_PROGRAM_FLAG_READ | LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 28), 16u);
  ASSERT_LE((size_t)code_offset + sizeof(code), bytes.size());
  EXPECT_EQ(0, std::memcmp(bytes.data() + code_offset, code, sizeof(code)));

  const uint32_t section_header_offset = LoadLeU32(bytes, 32);
  const uint16_t section_header_count = LoadLeU16(bytes, 48);
  ASSERT_EQ(section_header_count, 5u);
  constexpr size_t kSectionHeaderSize = 40;
  const size_t string_table_section = section_header_offset + 2 * 40;
  const size_t symbol_table_section = section_header_offset + 3 * 40;
  EXPECT_EQ(LoadLeU32(bytes, string_table_section + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 24), 2u);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 28), 1u);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 36), 16u);

  const uint32_t symbol_table_offset =
      LoadLeU32(bytes, symbol_table_section + 16);
  const uint32_t symbol_table_size =
      LoadLeU32(bytes, symbol_table_section + 20);
  ASSERT_EQ(symbol_table_size, 32u);
  ASSERT_LE((size_t)symbol_table_offset + symbol_table_size, bytes.size());
  const size_t kernel_symbol = symbol_table_offset + 16;
  EXPECT_EQ(LoadLeU32(bytes, kernel_symbol + 0), 1u);
  EXPECT_EQ(LoadLeU32(bytes, kernel_symbol + 4), 4u);
  EXPECT_EQ(LoadLeU32(bytes, kernel_symbol + 8), sizeof(code) - 4);
  EXPECT_EQ((uint8_t)bytes[kernel_symbol + 12], 0x12u);
  EXPECT_EQ((uint8_t)bytes[kernel_symbol + 13], 0u);
  EXPECT_EQ(LoadLeU16(bytes, kernel_symbol + 14), 1u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
