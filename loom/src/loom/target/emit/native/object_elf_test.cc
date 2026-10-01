// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/object_elf.h"

#include <string>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(NativeObjectElfTest, PlacesStorageAndTranslatesSymbolSemantics) {
  const uint8_t first[] = {0x90, 0xc3};
  const uint8_t second[] = {0x48, 0x31, 0xc0, 0xc3};
  loom_native_section_contribution_t sections[3] = {};
  sections[0].section_name = IREE_SV(".text");
  sections[0].storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS;
  sections[0].access =
      LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_EXECUTE;
  sections[0].contribution_alignment = 16;
  sections[0].contents = iree_make_const_byte_span(first, sizeof(first));
  sections[1].section_name = IREE_SV(".bss");
  sections[1].storage = LOOM_NATIVE_SECTION_STORAGE_RESERVATION;
  sections[1].access =
      LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_WRITE;
  sections[1].contribution_alignment = 32;
  sections[1].reservation_length = 32;
  sections[2] = sections[0];
  sections[2].contents = iree_make_const_byte_span(second, sizeof(second));

  loom_native_object_symbol_t symbols[3] = {};
  symbols[0].name = IREE_SV("cache");
  symbols[0].section_contribution_index = 1;
  symbols[0].section_offset = 8;
  symbols[0].size = 16;
  symbols[0].binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL;
  symbols[0].kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA;
  symbols[0].visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN;
  symbols[1].name = IREE_SV("alpha");
  symbols[1].section_contribution_index = 0;
  symbols[1].size = sizeof(first);
  symbols[1].binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL;
  symbols[1].kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION;
  symbols[2].name = IREE_SV("beta");
  symbols[2].section_contribution_index = 2;
  symbols[2].size = sizeof(second);
  symbols[2].binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_WEAK;
  symbols[2].kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION;
  const loom_native_object_contribution_t object = {
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
      /*.symbols=*/symbols,
      /*.symbol_count=*/IREE_ARRAYSIZE(symbols),
  };
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  IREE_ASSERT_OK(loom_native_object_write_elf64le(
      &object, LOOM_NATIVE_ELF_MACHINE_X86_64, stream, &arena));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);

  // Read only after the compiler workspace has been destroyed. All tables and
  // payload bytes must have escaped into the artifact, including joined text.
  std::string bytes(static_cast<size_t>(iree_io_stream_length(stream)), '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  iree_io_stream_release(stream);
  auto u16 = [&](size_t offset) {
    return iree_unaligned_load_le_u16(bytes.data() + offset);
  };
  auto u32 = [&](size_t offset) {
    return iree_unaligned_load_le_u32(bytes.data() + offset);
  };
  auto u64 = [&](size_t offset) {
    return iree_unaligned_load_le_u64(bytes.data() + offset);
  };
  ASSERT_EQ(bytes.substr(0, 4), std::string("\177ELF", 4));
  EXPECT_EQ(u16(16), 1);   // ET_REL.
  EXPECT_EQ(u16(18), 62);  // EM_X86_64.
  ASSERT_EQ(u16(58), 64);  // Section header size.
  ASSERT_EQ(u16(60), 7);   // Null, text, BSS, symbols, strings, stack, names.
  const size_t headers = static_cast<size_t>(u64(40));
  const size_t text = static_cast<size_t>(u64(headers + 64 + 24));
  EXPECT_EQ(u64(headers + 64 + 32), 20u);
  EXPECT_EQ(bytes.substr(text, sizeof(first)),
            std::string(reinterpret_cast<const char*>(first), sizeof(first)));
  EXPECT_EQ(bytes.substr(text + 2, 14), std::string(14, '\0'));
  EXPECT_EQ(bytes.substr(text + 16, sizeof(second)),
            std::string(reinterpret_cast<const char*>(second), sizeof(second)));
  EXPECT_EQ(u32(headers + 2 * 64 + 4), 8u);  // SHT_NOBITS.
  EXPECT_EQ(u64(headers + 2 * 64 + 32), 32u);
  const size_t symbol_header = headers + 3 * 64;
  EXPECT_EQ(u32(symbol_header + 4), 2u);   // SHT_SYMTAB.
  EXPECT_EQ(u32(symbol_header + 40), 4u);  // Link to .strtab.
  EXPECT_EQ(u32(symbol_header + 44), 2u);  // First non-local symbol.
  EXPECT_EQ(u64(symbol_header + 56), 24u);
  const size_t symbol_table = static_cast<size_t>(u64(symbol_header + 24));
  EXPECT_EQ(bytes.substr(symbol_table, 24), std::string(24, '\0'));
  const size_t local = symbol_table + 24;
  EXPECT_EQ(static_cast<uint8_t>(bytes[local + 4]), 0x01);  // LOCAL/OBJECT.
  EXPECT_EQ(static_cast<uint8_t>(bytes[local + 5]), 2);     // HIDDEN.
  EXPECT_EQ(u16(local + 6), 2u);
  EXPECT_EQ(u64(local + 8), 8u);
  EXPECT_EQ(u64(local + 16), 16u);
  const size_t global = symbol_table + 48;
  EXPECT_EQ(static_cast<uint8_t>(bytes[global + 4]), 0x12);  // GLOBAL/FUNC.
  EXPECT_EQ(u16(global + 6), 1u);
  EXPECT_EQ(u64(global + 8), 0u);
  const size_t weak = symbol_table + 72;
  EXPECT_EQ(static_cast<uint8_t>(bytes[weak + 4]), 0x22);  // WEAK/FUNC.
  EXPECT_EQ(u16(weak + 6), 1u);
  EXPECT_EQ(u64(weak + 8), 16u);
  const size_t strings = static_cast<size_t>(u64(headers + 4 * 64 + 24));
  EXPECT_STREQ(bytes.c_str() + strings + u32(local), "cache");
  EXPECT_STREQ(bytes.c_str() + strings + u32(global), "alpha");
  EXPECT_STREQ(bytes.c_str() + strings + u32(weak), "beta");
  EXPECT_EQ(u64(headers + 5 * 64 + 8), 0u);  // Stack has no executable flag.
}

}  // namespace
