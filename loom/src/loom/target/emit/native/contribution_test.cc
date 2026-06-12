// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/contribution.h"

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
  // Arena receiving transient contribution assembly storage.
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

uint64_t LoadLeU64(const std::string& bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)(uint8_t)bytes[offset + i] << (8 * i);
  }
  return value;
}

TEST(NativeContributionTest, AssemblesAlignedSectionsAndWritesElf) {
  char text_section_name[] = ".text";
  char rodata_section_name[] = ".rodata";
  const uint8_t text0[] = {0x01, 0x02, 0x03};
  const uint8_t rodata0[] = {0xa0, 0xa1};
  const uint8_t text1[] = {0x10, 0x11};
  const loom_native_section_contribution_t contributions[] = {
      {
          /*.section_name=*/iree_make_string_view(
              text_section_name, sizeof(text_section_name) - 1u),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/4,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/iree_make_const_byte_span(text0, sizeof(text0)),
      },
      {
          /*.section_name=*/iree_make_string_view(
              rodata_section_name, sizeof(rodata_section_name) - 1u),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          /*.contribution_alignment=*/1,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/iree_make_const_byte_span(rodata0, sizeof(rodata0)),
      },
      {
          /*.section_name=*/iree_make_string_view(
              text_section_name, sizeof(text_section_name) - 1u),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/8,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/iree_make_const_byte_span(text1, sizeof(text1)),
      },
  };

  TestArena arena;
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_ASSERT_OK(loom_native_assemble_section_contributions(
      contributions, IREE_ARRAYSIZE(contributions), &assembly, arena.arena()));
  text_section_name[1] = 'X';
  rodata_section_name[1] = 'X';

  ASSERT_EQ(assembly.section_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(assembly.sections[0].name, IREE_SV(".text")));
  EXPECT_TRUE(
      iree_string_view_equal(assembly.sections[1].name, IREE_SV(".rodata")));
  ASSERT_EQ(assembly.contribution_layout_count, IREE_ARRAYSIZE(contributions));
  EXPECT_EQ(assembly.contribution_layouts[0].section_index, 0u);
  EXPECT_EQ(assembly.contribution_layouts[0].section_offset, 0u);
  EXPECT_EQ(assembly.contribution_layouts[1].section_index, 1u);
  EXPECT_EQ(assembly.contribution_layouts[1].section_offset, 0u);
  EXPECT_EQ(assembly.contribution_layouts[2].section_index, 0u);
  EXPECT_EQ(assembly.contribution_layouts[2].section_offset, 8u);

  ASSERT_EQ(assembly.sections[0].contents.data_length, 10u);
  EXPECT_EQ(std::string((const char*)assembly.sections[0].contents.data,
                        assembly.sections[0].contents.data_length),
            std::string("\x01\x02\x03\x00\x00\x00\x00\x00\x10\x11", 10));
  EXPECT_EQ(assembly.sections[0].alignment, 8u);
  ASSERT_EQ(assembly.sections[1].contents.data_length, 2u);
  EXPECT_EQ(std::string((const char*)assembly.sections[1].contents.data,
                        assembly.sections[1].contents.data_length),
            std::string("\xa0\xa1", 2));

  const loom_native_elf64le_file_t file = {
      /*.type=*/LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      /*.machine=*/LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      /*.os_abi=*/LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA,
      /*.abi_version=*/LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V5,
      /*.flags=*/LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100,
      /*.entry=*/{},
      /*.sections=*/assembly.sections,
      /*.section_count=*/assembly.section_count,
  };
  StreamPtr stream = CreateStream();
  IREE_ASSERT_OK(
      loom_native_elf64le_write_file(&file, stream.get(), arena.arena()));
  const std::string elf_bytes = StreamBytes(stream.get());
  ASSERT_EQ(LoadLeU16(elf_bytes, 60), 4u);
  const size_t section_header_offset = (size_t)LoadLeU64(elf_bytes, 40);
  ASSERT_LE(section_header_offset + 4 * 64, elf_bytes.size());
  const size_t text_section_header_offset = section_header_offset + 64;
  const size_t rodata_section_header_offset = text_section_header_offset + 64;
  const size_t text_offset =
      (size_t)LoadLeU64(elf_bytes, text_section_header_offset + 24);
  const size_t text_size =
      (size_t)LoadLeU64(elf_bytes, text_section_header_offset + 32);
  const size_t rodata_offset =
      (size_t)LoadLeU64(elf_bytes, rodata_section_header_offset + 24);
  const size_t rodata_size =
      (size_t)LoadLeU64(elf_bytes, rodata_section_header_offset + 32);
  ASSERT_LE(text_offset + text_size, elf_bytes.size());
  ASSERT_LE(rodata_offset + rodata_size, elf_bytes.size());
  EXPECT_EQ(elf_bytes.substr(text_offset, text_size),
            std::string("\x01\x02\x03\x00\x00\x00\x00\x00\x10\x11", 10));
  EXPECT_EQ(elf_bytes.substr(rodata_offset, rodata_size),
            std::string("\xa0\xa1", 2));
}

TEST(NativeContributionTest, RejectsConflictingSectionMetadata) {
  const uint8_t byte = 0;
  const loom_native_section_contribution_t contributions[] = {
      {
          /*.section_name=*/IREE_SV(".text"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          /*.contribution_alignment=*/4,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/iree_make_const_byte_span(&byte, sizeof(byte)),
      },
      {
          /*.section_name=*/IREE_SV(".text"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          /*.contribution_alignment=*/4,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/iree_make_const_byte_span(&byte, sizeof(byte)),
      },
  };

  TestArena arena;
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_native_assemble_section_contributions(
                            contributions, IREE_ARRAYSIZE(contributions),
                            &assembly, arena.arena()));
}

TEST(NativeContributionTest, RejectsInvalidAlignment) {
  const uint8_t byte = 0;
  const loom_native_section_contribution_t contribution = {
      /*.section_name=*/IREE_SV(".text"),
      /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      /*.section_flags=*/{},
      /*.contribution_alignment=*/3,
      /*.entry_size=*/{},
      /*.link=*/{},
      /*.info=*/{},
      /*.contents=*/iree_make_const_byte_span(&byte, sizeof(byte)),
  };

  TestArena arena;
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_native_assemble_section_contributions(
                            &contribution, 1, &assembly, arena.arena()));
}

}  // namespace
}  // namespace loom
