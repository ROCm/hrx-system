// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf.h"

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

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transient ELF writer storage.
  iree_arena_allocator_t arena_ = {0};
};

StreamPtr CreateStream() {
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
  return bytes;
}

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

uint64_t LoadLeU64(const std::string& bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)(uint8_t)bytes[offset + i] << (8 * i);
  }
  return value;
}

TEST(NativeElfTest, WritesAmdgpuNoteElfEnvelope) {
  const std::string note = std::string(
      "\x07\x00\x00\x00"
      "\x08\x00\x00\x00"
      "\x20\x00\x00\x00"
      "AMDGPU\0"
      "\0"
      "payload!",
      28);
  const loom_native_elf64le_section_t sections[] = {{
      /*.name=*/IREE_SV(".note"),
      /*.type=*/LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
      /*.flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      /*.address=*/0,
      /*.alignment=*/4,
      /*.entry_size=*/0,
      /*.link=*/0,
      /*.info=*/0,
      /*.contents=*/iree_make_const_byte_span(note.data(), note.size()),
  }};
  const loom_native_elf64le_segment_t segments[] = {{
      /*.type=*/LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE,
      /*.flags=*/LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      /*.file_offset=*/{},
      /*.file_size=*/{},
      /*.memory_size=*/{},
      /*.first_section=*/0,
      /*.section_count=*/1,
      /*.virtual_address=*/0,
      /*.physical_address=*/0,
      /*.alignment=*/4,
  }};
  const loom_native_elf64le_file_t file = {
      /*.type=*/LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      /*.machine=*/LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      /*.os_abi=*/LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA,
      /*.abi_version=*/LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V5,
      /*.flags=*/LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100,
      /*.entry=*/0,
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
      /*.segments=*/segments,
      /*.segment_count=*/IREE_ARRAYSIZE(segments),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_native_elf64le_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());
  const size_t note_offset = 120;
  const size_t string_table_offset = note_offset + note.size();
  const size_t string_table_size = 17;
  const size_t section_header_offset =
      (string_table_offset + string_table_size + 7) & ~(size_t)7;

  ASSERT_GE(bytes.size(), section_header_offset + 3 * 64);
  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ((uint8_t)bytes[4], 2u);
  EXPECT_EQ((uint8_t)bytes[5], 1u);
  EXPECT_EQ((uint8_t)bytes[6], 1u);
  EXPECT_EQ((uint8_t)bytes[7], LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA);
  EXPECT_EQ((uint8_t)bytes[8], LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V5);
  EXPECT_EQ(LoadLeU16(bytes, 16), LOOM_NATIVE_ELF_FILE_TYPE_DYN);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AMDGPU);
  EXPECT_EQ(LoadLeU32(bytes, 20), 1u);
  EXPECT_EQ(LoadLeU64(bytes, 24), 0u);
  EXPECT_EQ(LoadLeU64(bytes, 32), 64u);
  EXPECT_EQ(LoadLeU64(bytes, 40), section_header_offset);
  EXPECT_EQ(LoadLeU32(bytes, 48), LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100);
  EXPECT_EQ(LoadLeU16(bytes, 52), 64u);
  EXPECT_EQ(LoadLeU16(bytes, 54), 56u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 1u);
  EXPECT_EQ(LoadLeU16(bytes, 58), 64u);
  EXPECT_EQ(LoadLeU16(bytes, 60), 3u);
  EXPECT_EQ(LoadLeU16(bytes, 62), 2u);

  constexpr size_t kProgramHeaderOffset = 64;
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 0),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 4),
            LOOM_NATIVE_ELF_PROGRAM_FLAG_READ);
  EXPECT_EQ(LoadLeU64(bytes, kProgramHeaderOffset + 8), note_offset);
  EXPECT_EQ(LoadLeU64(bytes, kProgramHeaderOffset + 32), note.size());
  EXPECT_EQ(LoadLeU64(bytes, kProgramHeaderOffset + 40), note.size());
  EXPECT_EQ(LoadLeU64(bytes, kProgramHeaderOffset + 48), 4u);

  EXPECT_EQ(bytes.substr(note_offset, note.size()), note);
  EXPECT_EQ(bytes.substr(string_table_offset, string_table_size),
            std::string("\0.note\0.shstrtab\0", string_table_size));

  const size_t kSectionHeaderOffset = section_header_offset;
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ((uint8_t)bytes[kSectionHeaderOffset + i], 0u);
  }

  const size_t kNoteSectionHeaderOffset = kSectionHeaderOffset + 64;
  EXPECT_EQ(LoadLeU32(bytes, kNoteSectionHeaderOffset + 0), 1u);
  EXPECT_EQ(LoadLeU32(bytes, kNoteSectionHeaderOffset + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_NOTE);
  EXPECT_EQ(LoadLeU64(bytes, kNoteSectionHeaderOffset + 8),
            LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC);
  EXPECT_EQ(LoadLeU64(bytes, kNoteSectionHeaderOffset + 24), note_offset);
  EXPECT_EQ(LoadLeU64(bytes, kNoteSectionHeaderOffset + 32), note.size());
  EXPECT_EQ(LoadLeU64(bytes, kNoteSectionHeaderOffset + 48), 4u);

  const size_t kStringTableSectionHeaderOffset = kNoteSectionHeaderOffset + 64;
  EXPECT_EQ(LoadLeU32(bytes, kStringTableSectionHeaderOffset + 0), 7u);
  EXPECT_EQ(LoadLeU32(bytes, kStringTableSectionHeaderOffset + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB);
  EXPECT_EQ(LoadLeU64(bytes, kStringTableSectionHeaderOffset + 24),
            string_table_offset);
  EXPECT_EQ(LoadLeU64(bytes, kStringTableSectionHeaderOffset + 32),
            string_table_size);
  EXPECT_EQ(LoadLeU64(bytes, kStringTableSectionHeaderOffset + 48), 1u);
}

TEST(NativeElfTest, RejectsInvalidSectionAlignment) {
  const uint8_t contents[] = {0};
  const loom_native_elf64le_section_t sections[] = {{
      /*.name=*/IREE_SV(".bad"),
      /*.type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      /*.flags=*/{},
      /*.address=*/{},
      /*.alignment=*/3,
      /*.entry_size=*/{},
      /*.link=*/{},
      /*.info=*/{},
      /*.contents=*/iree_make_const_byte_span(contents, sizeof(contents)),
  }};
  const loom_native_elf64le_file_t file = {
      /*.type=*/LOOM_NATIVE_ELF_FILE_TYPE_REL,
      /*.machine=*/LOOM_NATIVE_ELF_MACHINE_X86_64,
      /*.os_abi=*/{},
      /*.abi_version=*/{},
      /*.flags=*/{},
      /*.entry=*/{},
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_native_elf64le_write_file(&file, stream.get(), arena.arena()));
}

TEST(NativeElfTest, RejectsInvalidSegmentRange) {
  const uint8_t contents[] = {0};
  const loom_native_elf64le_section_t sections[] = {{
      /*.name=*/IREE_SV(".note"),
      /*.type=*/LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
      /*.flags=*/{},
      /*.address=*/{},
      /*.alignment=*/4,
      /*.entry_size=*/{},
      /*.link=*/{},
      /*.info=*/{},
      /*.contents=*/iree_make_const_byte_span(contents, sizeof(contents)),
  }};
  const loom_native_elf64le_segment_t segments[] = {{
      /*.type=*/LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE,
      /*.flags=*/{},
      /*.file_offset=*/{},
      /*.file_size=*/{},
      /*.memory_size=*/{},
      /*.first_section=*/1,
      /*.section_count=*/1,
      /*.virtual_address=*/{},
      /*.physical_address=*/{},
      /*.alignment=*/4,
  }};
  const loom_native_elf64le_file_t file = {
      /*.type=*/LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      /*.machine=*/LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      /*.os_abi=*/{},
      /*.abi_version=*/{},
      /*.flags=*/{},
      /*.entry=*/{},
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
      /*.segments=*/segments,
      /*.segment_count=*/IREE_ARRAYSIZE(segments),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_native_elf64le_write_file(&file, stream.get(), arena.arena()));
}

}  // namespace
}  // namespace loom
