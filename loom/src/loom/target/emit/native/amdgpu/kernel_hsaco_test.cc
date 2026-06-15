// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/native/elf.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

struct Section {
  // Section table ordinal.
  size_t index;
  // Section name from .shstrtab.
  std::string name;
  // ELF section type.
  uint32_t type;
  // File offset of section contents.
  uint64_t offset;
  // Byte length of section contents.
  uint64_t size;
  // Linked section index.
  uint32_t link;
  // Section entry size for table-like sections.
  uint64_t entry_size;
};

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
  // Arena receiving transient HSACO writer storage.
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

std::string ReadNullTerminatedString(const std::string& bytes, size_t offset) {
  std::string value;
  while (offset < bytes.size() && bytes[offset] != '\0') {
    value.push_back(bytes[offset++]);
  }
  return value;
}

std::vector<Section> ReadSections(const std::string& bytes) {
  const size_t section_header_offset = (size_t)LoadLeU64(bytes, 40);
  const size_t section_count = LoadLeU16(bytes, 60);
  const size_t section_name_index = LoadLeU16(bytes, 62);
  EXPECT_LT(section_name_index, section_count);

  const size_t section_name_header =
      section_header_offset + section_name_index * 64;
  const size_t section_name_offset =
      (size_t)LoadLeU64(bytes, section_name_header + 24);
  const size_t section_name_size =
      (size_t)LoadLeU64(bytes, section_name_header + 32);
  EXPECT_LE(section_name_offset + section_name_size, bytes.size());

  std::vector<Section> sections;
  sections.reserve(section_count);
  for (size_t i = 0; i < section_count; ++i) {
    const size_t header_offset = section_header_offset + i * 64;
    const uint32_t name_offset = LoadLeU32(bytes, header_offset);
    EXPECT_LT(name_offset, section_name_size);
    sections.push_back({
        /*.index=*/i,
        /*.name=*/
        ReadNullTerminatedString(bytes, section_name_offset + name_offset),
        /*.type=*/LoadLeU32(bytes, header_offset + 4),
        /*.offset=*/LoadLeU64(bytes, header_offset + 24),
        /*.size=*/LoadLeU64(bytes, header_offset + 32),
        /*.link=*/LoadLeU32(bytes, header_offset + 40),
        /*.entry_size=*/LoadLeU64(bytes, header_offset + 56),
    });
  }
  return sections;
}

const Section& FindSection(const std::vector<Section>& sections,
                           const char* name) {
  for (const Section& section : sections) {
    if (section.name == name) {
      return section;
    }
  }
  ADD_FAILURE() << "section not found: " << name;
  return sections[0];
}

loom_amdgpu_metadata_kernel_t MinimalKernel(iree_string_view_t name,
                                            iree_string_view_t symbol) {
  return {
      /*.name=*/name,
      /*.descriptor_symbol=*/symbol,
      /*.kernarg_segment_size=*/0,
      /*.kernarg_segment_alignment=*/8,
      /*.wavefront_size=*/32,
      /*.group_segment_fixed_size=*/0,
      /*.private_segment_fixed_size=*/0,
      /*.sgpr_count=*/4,
      /*.vgpr_count=*/1,
      /*.max_flat_workgroup_size=*/64,
      /*.required_workgroup_size=*/{/*.x=*/64, /*.y=*/1, /*.z=*/1},
      /*.has_required_workgroup_size=*/true,
      /*.arguments=*/nullptr,
      /*.argument_count=*/0,
  };
}

loom_amdgpu_kernel_hsaco_contribution_t Contribution(
    iree_string_view_t name, iree_string_view_t descriptor_symbol,
    iree_const_byte_span_t text) {
  return {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1100"),
      /*.kernel=*/
      {
          /*.metadata=*/MinimalKernel(name, descriptor_symbol),
          /*.descriptor_options=*/{},
          /*.text=*/text,
      },
      /*.summary=*/
      {
          /*.instruction_count=*/1,
          /*.text_byte_count=*/text.data_length,
          /*.text_storage_byte_count=*/text.data_length,
      },
  };
}

TEST(AmdgpuKernelHsacoTest, JoinsMultipleContributionsIntoOneCodeObject) {
  const uint8_t s_endpgm_a[] = {0x00, 0x00, 0x81, 0xbf};
  const uint8_t s_endpgm_b[] = {0x00, 0x00, 0x81, 0xbf};
  const loom_amdgpu_kernel_hsaco_contribution_t contributions[] = {
      Contribution(IREE_SV("first_kernel"), IREE_SV("first_kernel.kd"),
                   iree_make_const_byte_span(s_endpgm_a, sizeof(s_endpgm_a))),
      Contribution(IREE_SV("second_kernel"), IREE_SV("second_kernel.kd"),
                   iree_make_const_byte_span(s_endpgm_b, sizeof(s_endpgm_b))),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(loom_amdgpu_write_kernel_hsaco_contributions(
      contributions, IREE_ARRAYSIZE(contributions), stream.get(),
      arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  ASSERT_GE(bytes.size(), 64u);
  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AMDGPU);
  EXPECT_EQ(LoadLeU32(bytes, 48), LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100);

  const std::vector<Section> sections = ReadSections(bytes);
  const Section& note = FindSection(sections, ".note");
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& rodata = FindSection(sections, ".rodata");
  const Section& text = FindSection(sections, ".text");
  EXPECT_EQ(note.type, LOOM_NATIVE_ELF_SECTION_TYPE_NOTE);
  EXPECT_EQ(dynsym.type, LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM);
  EXPECT_EQ(dynsym.link, dynstr.index);
  EXPECT_EQ(dynsym.entry_size, 24u);
  EXPECT_EQ(dynsym.size, 5u * 24u);
  EXPECT_GE(rodata.size, 2u * LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  EXPECT_GE(text.size, sizeof(s_endpgm_a) + sizeof(s_endpgm_b));

  const std::string dynstr_contents =
      bytes.substr((size_t)dynstr.offset, (size_t)dynstr.size);
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, dynsym.offset + 24u)),
            "first_kernel");
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, dynsym.offset + 48u)),
            "first_kernel.kd");
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, dynsym.offset + 72u)),
            "second_kernel");
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, dynsym.offset + 96u)),
            "second_kernel.kd");

  const uint64_t first_entry_address =
      LoadLeU64(bytes, dynsym.offset + 24u + 8);
  const uint64_t second_entry_address =
      LoadLeU64(bytes, dynsym.offset + 72u + 8);
  EXPECT_GT(second_entry_address, first_entry_address);

  const std::string note_contents =
      bytes.substr((size_t)note.offset, (size_t)note.size);
  EXPECT_NE(note_contents.find("first_kernel"), std::string::npos);
  EXPECT_NE(note_contents.find("second_kernel"), std::string::npos);
}

TEST(AmdgpuKernelHsacoTest, RejectsMismatchedContributionProcessor) {
  const uint8_t text[] = {0x00, 0x00, 0x81, 0xbf};
  loom_amdgpu_kernel_hsaco_contribution_t contributions[] = {
      Contribution(IREE_SV("first_kernel"), IREE_SV("first_kernel.kd"),
                   iree_make_const_byte_span(text, sizeof(text))),
      Contribution(IREE_SV("second_kernel"), IREE_SV("second_kernel.kd"),
                   iree_make_const_byte_span(text, sizeof(text))),
  };
  contributions[1].processor = IREE_SV("gfx1101");

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_write_kernel_hsaco_contributions(
                            contributions, IREE_ARRAYSIZE(contributions),
                            stream.get(), arena.arena()));
}

}  // namespace
}  // namespace loom
