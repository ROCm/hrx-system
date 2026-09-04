// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hsaco.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/amdhsa_target_id.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"
#include "loom/target/emit/native/elf.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

constexpr char kAsanConfigGlobalName[] = "iree_asan_config";
constexpr uint64_t kAsanConfigByteLength = 96;
constexpr char kFeedbackConfigGlobalName[] = "iree_feedback_config";
constexpr uint64_t kFeedbackConfigByteLength = 64;

struct Section {
  size_t index;
  std::string name;
  uint32_t type;
  uint64_t flags;
  uint64_t address;
  uint64_t offset;
  uint64_t size;
  uint32_t link;
  uint32_t info;
  uint64_t alignment;
  uint64_t entry_size;
};

struct Segment {
  // Program table ordinal.
  size_t index;
  // ELF program header type.
  uint32_t type;
  // ELF program header flags.
  uint32_t flags;
  // File offset of segment contents.
  uint64_t offset;
  // Runtime virtual address of the segment.
  uint64_t virtual_address;
  // Runtime physical address of the segment.
  uint64_t physical_address;
  // Byte length of the file image.
  uint64_t file_size;
  // Byte length of the memory image.
  uint64_t memory_size;
  // Program header alignment.
  uint64_t alignment;
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

int64_t LoadLeI64(const std::string& bytes, size_t offset) {
  return (int64_t)LoadLeU64(bytes, offset);
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
    const uint32_t name_offset = LoadLeU32(bytes, header_offset + 0);
    EXPECT_LT(name_offset, section_name_size);
    Section section = {
        /*.index=*/i,
        /*.name=*/
        ReadNullTerminatedString(bytes, section_name_offset + name_offset),
        /*.type=*/LoadLeU32(bytes, header_offset + 4),
        /*.flags=*/LoadLeU64(bytes, header_offset + 8),
        /*.address=*/LoadLeU64(bytes, header_offset + 16),
        /*.offset=*/LoadLeU64(bytes, header_offset + 24),
        /*.size=*/LoadLeU64(bytes, header_offset + 32),
        /*.link=*/LoadLeU32(bytes, header_offset + 40),
        /*.info=*/LoadLeU32(bytes, header_offset + 44),
        /*.alignment=*/LoadLeU64(bytes, header_offset + 48),
        /*.entry_size=*/LoadLeU64(bytes, header_offset + 56),
    };
    sections.push_back(section);
  }
  return sections;
}

std::vector<Segment> ReadSegments(const std::string& bytes) {
  const size_t program_header_offset = (size_t)LoadLeU64(bytes, 32);
  const size_t program_header_count = LoadLeU16(bytes, 56);

  std::vector<Segment> segments;
  segments.reserve(program_header_count);
  for (size_t i = 0; i < program_header_count; ++i) {
    const size_t offset = program_header_offset + i * 56;
    segments.push_back({
        /*.index=*/i,
        /*.type=*/LoadLeU32(bytes, offset + 0),
        /*.flags=*/LoadLeU32(bytes, offset + 4),
        /*.offset=*/LoadLeU64(bytes, offset + 8),
        /*.virtual_address=*/LoadLeU64(bytes, offset + 16),
        /*.physical_address=*/LoadLeU64(bytes, offset + 24),
        /*.file_size=*/LoadLeU64(bytes, offset + 32),
        /*.memory_size=*/LoadLeU64(bytes, offset + 40),
        /*.alignment=*/LoadLeU64(bytes, offset + 48),
    });
  }
  return segments;
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

uint64_t FindDynamicTag(const std::string& bytes, const Section& dynamic,
                        uint64_t tag) {
  EXPECT_EQ(dynamic.entry_size, 16u);
  for (uint64_t offset = 0; offset + 16 <= dynamic.size; offset += 16) {
    const uint64_t entry_tag =
        LoadLeU64(bytes, (size_t)(dynamic.offset + offset));
    const uint64_t entry_value =
        LoadLeU64(bytes, (size_t)(dynamic.offset + offset + 8));
    if (entry_tag == tag) {
      return entry_value;
    }
  }
  ADD_FAILURE() << "dynamic tag not found: " << tag;
  return 0;
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
      /*.workgroup_cluster_size=*/{},
      /*.has_workgroup_cluster_size=*/false,
      /*.target_extensions=*/{},
      /*.arguments=*/nullptr,
      /*.argument_count=*/0,
  };
}

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string CodeObjectTargetIdForIdentity(
    const loom_amdgpu_target_identity_t& identity) {
  TestArena arena;
  iree_string_view_t target_id = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_amdhsa_target_id_format(&identity, arena.arena(),
                                                    &target_id));
  return StringViewToString(target_id);
}

TEST(AmdgpuHsacoTest, WritesGfx1100CodeObjectEnvelope) {
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/
      MinimalKernel(IREE_SV("loom_kernel"), IREE_SV("loom_kernel.kd")),
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1100"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  ASSERT_GE(bytes.size(), 64u);
  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ((uint8_t)bytes[7], LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA);
  EXPECT_EQ((uint8_t)bytes[8], LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6);
  EXPECT_EQ(LoadLeU16(bytes, 16), LOOM_NATIVE_ELF_FILE_TYPE_DYN);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AMDGPU);
  EXPECT_EQ(LoadLeU32(bytes, 48), LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100);

  ASSERT_EQ(LoadLeU16(bytes, 56), 7u);
  const size_t program_header_offset = (size_t)LoadLeU64(bytes, 32);
  bool found_program_header_table = false;
  bool found_dynamic_program_header = false;
  bool found_stack_program_header = false;
  bool found_note_program_header = false;
  for (size_t i = 0; i < LoadLeU16(bytes, 56); ++i) {
    const size_t offset = program_header_offset + i * 56;
    const uint32_t type = LoadLeU32(bytes, offset);
    found_program_header_table |= type == LOOM_NATIVE_ELF_PROGRAM_TYPE_PHDR;
    found_dynamic_program_header |=
        type == LOOM_NATIVE_ELF_PROGRAM_TYPE_DYNAMIC;
    found_stack_program_header |=
        type == LOOM_NATIVE_ELF_PROGRAM_TYPE_GNU_STACK;
    found_note_program_header |= type == LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE;
  }
  EXPECT_TRUE(found_program_header_table);
  EXPECT_TRUE(found_dynamic_program_header);
  EXPECT_TRUE(found_stack_program_header);
  EXPECT_TRUE(found_note_program_header);

  const std::vector<Section> sections = ReadSections(bytes);
  const Section& note = FindSection(sections, ".note");
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& hash = FindSection(sections, ".hash");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& rodata = FindSection(sections, ".rodata");
  const Section& text = FindSection(sections, ".text");
  const Section& dynamic = FindSection(sections, ".dynamic");
  const Section& symtab = FindSection(sections, ".symtab");
  const Section& strtab = FindSection(sections, ".strtab");

  EXPECT_EQ(note.type, LOOM_NATIVE_ELF_SECTION_TYPE_NOTE);
  EXPECT_EQ(dynsym.type, LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM);
  EXPECT_EQ(hash.type, LOOM_NATIVE_ELF_SECTION_TYPE_HASH);
  EXPECT_EQ(dynstr.type, LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB);
  EXPECT_EQ(rodata.type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
  EXPECT_EQ(text.type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
  EXPECT_EQ(dynamic.type, LOOM_NATIVE_ELF_SECTION_TYPE_DYNAMIC);
  EXPECT_EQ(symtab.type, LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB);
  EXPECT_EQ(strtab.type, LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB);

  EXPECT_EQ(dynsym.link, dynstr.index);
  EXPECT_EQ(dynsym.info, 1u);
  EXPECT_EQ(dynsym.entry_size, 24u);
  EXPECT_EQ(hash.link, dynsym.index);
  EXPECT_EQ(dynamic.link, dynstr.index);
  EXPECT_EQ(symtab.link, strtab.index);
  EXPECT_EQ(symtab.info, 1u);
  EXPECT_EQ(rodata.size, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  EXPECT_EQ(text.size, sizeof(s_endpgm));
  EXPECT_EQ(bytes.substr((size_t)text.offset, sizeof(s_endpgm)),
            std::string((const char*)s_endpgm, sizeof(s_endpgm)));

  const size_t program_header_table = program_header_offset;
  EXPECT_EQ(LoadLeU32(bytes, program_header_table + 0),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_PHDR);
  EXPECT_EQ(LoadLeU64(bytes, program_header_table + 8), program_header_offset);
  EXPECT_EQ(LoadLeU64(bytes, program_header_table + 16), program_header_offset);
  EXPECT_EQ(LoadLeU64(bytes, program_header_table + 32), 7u * 56u);
  const size_t read_load = program_header_offset + 56u;
  EXPECT_EQ(LoadLeU32(bytes, read_load + 0), LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD);
  EXPECT_EQ(LoadLeU64(bytes, read_load + 8), 0u);
  EXPECT_EQ(LoadLeU64(bytes, read_load + 16), 0u);
  EXPECT_EQ(LoadLeU64(bytes, read_load + 32), rodata.offset + rodata.size);
  EXPECT_EQ(LoadLeU64(bytes, read_load + 48), 4096u);
  const size_t execute_load = program_header_offset + 2u * 56u;
  EXPECT_EQ(LoadLeU32(bytes, execute_load + 0),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD);
  EXPECT_EQ(LoadLeU64(bytes, execute_load + 8), text.offset);
  EXPECT_EQ(LoadLeU64(bytes, execute_load + 16), text.address);
  EXPECT_EQ(LoadLeU64(bytes, execute_load + 48), 4096u);
  EXPECT_EQ(text.offset & 4095u, text.address & 4095u);
  const size_t write_load = program_header_offset + 3u * 56u;
  EXPECT_EQ(LoadLeU32(bytes, write_load + 0),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD);
  EXPECT_EQ(LoadLeU64(bytes, write_load + 8), dynamic.offset);
  EXPECT_EQ(LoadLeU64(bytes, write_load + 16), dynamic.address);
  EXPECT_EQ(LoadLeU64(bytes, write_load + 48), 4096u);
  EXPECT_EQ(dynamic.offset & 4095u, dynamic.address & 4095u);

  const std::string dynstr_contents =
      bytes.substr((size_t)dynstr.offset, (size_t)dynstr.size);
  ASSERT_GE(dynstr_contents.size(), 1u);
  EXPECT_EQ(dynstr_contents[0], '\0');
  EXPECT_NE(dynstr_contents.find("loom_kernel"), std::string::npos);
  EXPECT_NE(dynstr_contents.find("loom_kernel.kd"), std::string::npos);

  ASSERT_EQ(dynsym.size, 3u * 24u);
  const size_t entry_symbol = (size_t)dynsym.offset + 24u;
  const size_t descriptor_symbol = entry_symbol + 24u;
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, entry_symbol + 0)),
            "loom_kernel");
  EXPECT_EQ((uint8_t)bytes[entry_symbol + 4], 0x12u);
  EXPECT_EQ((uint8_t)bytes[entry_symbol + 5], 3u);
  EXPECT_EQ(LoadLeU16(bytes, entry_symbol + 6), text.index);
  EXPECT_EQ(LoadLeU64(bytes, entry_symbol + 8), text.address);
  EXPECT_EQ(LoadLeU64(bytes, entry_symbol + 16), sizeof(s_endpgm));

  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, descriptor_symbol + 0)),
            "loom_kernel.kd");
  EXPECT_EQ((uint8_t)bytes[descriptor_symbol + 4], 0x11u);
  EXPECT_EQ((uint8_t)bytes[descriptor_symbol + 5], 3u);
  EXPECT_EQ(LoadLeU16(bytes, descriptor_symbol + 6), rodata.index);
  EXPECT_EQ(LoadLeU64(bytes, descriptor_symbol + 8), rodata.address);
  EXPECT_EQ(LoadLeU64(bytes, descriptor_symbol + 16),
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);

  constexpr uint64_t kDtHash = 4;
  constexpr uint64_t kDtStrtab = 5;
  constexpr uint64_t kDtSymtab = 6;
  constexpr uint64_t kDtStrsz = 10;
  constexpr uint64_t kDtSyment = 11;
  EXPECT_EQ(FindDynamicTag(bytes, dynamic, kDtHash), hash.address);
  EXPECT_EQ(FindDynamicTag(bytes, dynamic, kDtStrtab), dynstr.address);
  EXPECT_EQ(FindDynamicTag(bytes, dynamic, kDtSymtab), dynsym.address);
  EXPECT_EQ(FindDynamicTag(bytes, dynamic, kDtStrsz), dynstr.size);
  EXPECT_EQ(FindDynamicTag(bytes, dynamic, kDtSyment), 24u);

  const int64_t descriptor_entry_offset =
      LoadLeI64(bytes, (size_t)rodata.offset + 16u);
  EXPECT_EQ(descriptor_entry_offset, (int64_t)(text.address - rodata.address));

  const std::string note_contents =
      bytes.substr((size_t)note.offset, (size_t)note.size);
  EXPECT_NE(note_contents.find("amdhsa.target"), std::string::npos);
  EXPECT_NE(note_contents.find("amdgcn-amd-amdhsa--gfx1100"),
            std::string::npos);
  EXPECT_NE(note_contents.find("loom_kernel.kd"), std::string::npos);
}

TEST(AmdgpuHsacoTest, WritesWritableRuntimeDataSymbols) {
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/MinimalKernel(IREE_SV("loom_kernel"),
                                  IREE_SV("loom_kernel.kd")),
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
  };
  const loom_amdgpu_hsaco_data_symbol_t data_symbols[] = {
      {
          /*.name=*/IREE_SV(kAsanConfigGlobalName),
          /*.initial_contents=*/{},
          /*.byte_length=*/kAsanConfigByteLength,
          /*.alignment=*/8,
          /*.flags=*/LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
      },
      {
          /*.name=*/IREE_SV(kFeedbackConfigGlobalName),
          /*.initial_contents=*/{},
          /*.byte_length=*/kFeedbackConfigByteLength,
          /*.alignment=*/8,
          /*.flags=*/LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
      },
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1100"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
      /*.data_symbols=*/data_symbols,
      /*.data_symbol_count=*/IREE_ARRAYSIZE(data_symbols),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  const std::vector<Section> sections = ReadSections(bytes);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& data = FindSection(sections, ".data");
  const Section& dynamic = FindSection(sections, ".dynamic");

  EXPECT_EQ(data.type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
  EXPECT_EQ(data.flags, LOOM_NATIVE_ELF_SECTION_FLAG_WRITE |
                            LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC);
  EXPECT_EQ(data.alignment, 4096u);
  EXPECT_EQ(data.size, kAsanConfigByteLength + kFeedbackConfigByteLength);
  EXPECT_EQ(data.offset & 4095u, 0u);
  EXPECT_EQ(data.address & 4095u, 0u);
  ASSERT_LE(data.offset + data.size, bytes.size());
  for (uint64_t i = 0; i < data.size; ++i) {
    EXPECT_EQ(bytes[(size_t)(data.offset + i)], '\0');
  }

  const std::string dynstr_contents =
      bytes.substr((size_t)dynstr.offset, (size_t)dynstr.size);
  EXPECT_NE(dynstr_contents.find(kAsanConfigGlobalName), std::string::npos);
  EXPECT_NE(dynstr_contents.find(kFeedbackConfigGlobalName), std::string::npos);

  ASSERT_EQ(dynsym.size, 5u * 24u);
  const size_t asan_symbol = (size_t)dynsym.offset + 3u * 24u;
  const size_t feedback_symbol = asan_symbol + 24u;
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, asan_symbol + 0)),
            kAsanConfigGlobalName);
  EXPECT_EQ((uint8_t)bytes[asan_symbol + 4], 0x11u);
  EXPECT_EQ((uint8_t)bytes[asan_symbol + 5], 3u);
  EXPECT_EQ(LoadLeU16(bytes, asan_symbol + 6), data.index);
  EXPECT_EQ(LoadLeU64(bytes, asan_symbol + 8), data.address);
  EXPECT_EQ(LoadLeU64(bytes, asan_symbol + 16), kAsanConfigByteLength);

  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, feedback_symbol + 0)),
            kFeedbackConfigGlobalName);
  EXPECT_EQ((uint8_t)bytes[feedback_symbol + 4], 0x11u);
  EXPECT_EQ((uint8_t)bytes[feedback_symbol + 5], 3u);
  EXPECT_EQ(LoadLeU16(bytes, feedback_symbol + 6), data.index);
  EXPECT_EQ(LoadLeU64(bytes, feedback_symbol + 8),
            data.address + kAsanConfigByteLength);
  EXPECT_EQ(LoadLeU64(bytes, feedback_symbol + 16), kFeedbackConfigByteLength);

  const std::vector<Segment> segments = ReadSegments(bytes);
  ASSERT_GT(segments.size(), 4u);
  const Segment& write_load = segments[3];
  EXPECT_EQ(write_load.type, LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD);
  EXPECT_EQ(write_load.flags, LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
                                  LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE);
  EXPECT_EQ(write_load.offset, data.offset);
  EXPECT_EQ(write_load.virtual_address, data.address);
  EXPECT_EQ(write_load.physical_address, data.address);
  EXPECT_GE(write_load.file_size, dynamic.offset + dynamic.size - data.offset);
  EXPECT_EQ(write_load.memory_size, write_load.file_size);
  EXPECT_EQ(write_load.alignment, 4096u);
  EXPECT_EQ(write_load.offset & 4095u, write_load.virtual_address & 4095u);

  const Segment& dynamic_program_header = segments[4];
  EXPECT_EQ(dynamic_program_header.type, LOOM_NATIVE_ELF_PROGRAM_TYPE_DYNAMIC);
  EXPECT_EQ(dynamic_program_header.offset, dynamic.offset);
  EXPECT_EQ(dynamic_program_header.virtual_address, dynamic.address);
  EXPECT_EQ(dynamic_program_header.physical_address, dynamic.address);
  EXPECT_EQ(dynamic_program_header.file_size, dynamic.size);
}

TEST(AmdgpuHsacoTest, WritesAlignedReadOnlyDataSymbols) {
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  const uint8_t tag_bytes[] = {0x13, 0x37, 0x42, 0x5a};
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/MinimalKernel(IREE_SV("loom_kernel"),
                                  IREE_SV("loom_kernel.kd")),
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
  };
  const loom_amdgpu_hsaco_data_symbol_t data_symbol = {
      /*.name=*/IREE_SV("loom_const_tag"),
      /*.initial_contents=*/
      iree_make_const_byte_span(tag_bytes, sizeof(tag_bytes)),
      /*.byte_length=*/8,
      /*.alignment=*/128,
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1100"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
      /*.data_symbols=*/&data_symbol,
      /*.data_symbol_count=*/1,
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  const std::vector<Section> sections = ReadSections(bytes);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& rodata = FindSection(sections, ".rodata");
  EXPECT_EQ(rodata.alignment, 128u);
  EXPECT_EQ(rodata.offset & 127u, 0u);
  EXPECT_EQ(rodata.address & 127u, 0u);
  ASSERT_EQ(rodata.size, 128u + data_symbol.byte_length);
  ASSERT_LE(rodata.offset + rodata.size, bytes.size());
  EXPECT_EQ(bytes.substr((size_t)rodata.offset + 128u, sizeof(tag_bytes)),
            std::string((const char*)tag_bytes, sizeof(tag_bytes)));

  const std::string dynstr_contents =
      bytes.substr((size_t)dynstr.offset, (size_t)dynstr.size);
  ASSERT_EQ(dynsym.size, 4u * 24u);
  const size_t symbol = (size_t)dynsym.offset + 3u * 24u;
  EXPECT_EQ(
      ReadNullTerminatedString(dynstr_contents, LoadLeU32(bytes, symbol + 0)),
      "loom_const_tag");
  EXPECT_EQ((uint8_t)bytes[symbol + 4], 0x11u);
  EXPECT_EQ((uint8_t)bytes[symbol + 5], 3u);
  EXPECT_EQ(LoadLeU16(bytes, symbol + 6), rodata.index);
  EXPECT_EQ(LoadLeU64(bytes, symbol + 8), rodata.address + 128u);
  EXPECT_EQ(LoadLeU64(bytes, symbol + 16), data_symbol.byte_length);
}

TEST(AmdgpuHsacoTest, PatchesDataSymbolRel32TextFixups) {
  std::vector<uint8_t> text_bytes(48, 0xcc);
  const loom_amdgpu_hsaco_text_fixup_t text_fixups[] = {
      {
          /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
          /*.literal_byte_offset=*/4,
          /*.base_pc_byte_offset=*/0,
          /*.target_symbol=*/IREE_SV("loom_const_tag"),
          /*.target_symbol_byte_offset=*/4,
      },
      {
          /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_HI,
          /*.literal_byte_offset=*/12,
          /*.base_pc_byte_offset=*/0,
          /*.target_symbol=*/IREE_SV("loom_const_tag"),
          /*.target_symbol_byte_offset=*/4,
      },
      {
          /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
          /*.literal_byte_offset=*/28,
          /*.base_pc_byte_offset=*/16,
          /*.target_symbol=*/IREE_SV(kFeedbackConfigGlobalName),
          /*.target_symbol_byte_offset=*/16,
      },
      {
          /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_HI,
          /*.literal_byte_offset=*/36,
          /*.base_pc_byte_offset=*/16,
          /*.target_symbol=*/IREE_SV(kFeedbackConfigGlobalName),
          /*.target_symbol_byte_offset=*/16,
      },
  };
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/MinimalKernel(IREE_SV("loom_kernel"),
                                  IREE_SV("loom_kernel.kd")),
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(text_bytes.data(), text_bytes.size()),
      /*.text_fixups=*/text_fixups,
      /*.text_fixup_count=*/IREE_ARRAYSIZE(text_fixups),
  };
  const uint8_t tag_bytes[] = {0x13, 0x37, 0x42, 0x5a, 0x99, 0x88};
  const loom_amdgpu_hsaco_data_symbol_t data_symbols[] = {
      {
          /*.name=*/IREE_SV("loom_const_tag"),
          /*.initial_contents=*/
          iree_make_const_byte_span(tag_bytes, sizeof(tag_bytes)),
          /*.byte_length=*/16,
          /*.alignment=*/8,
      },
      {
          /*.name=*/IREE_SV(kFeedbackConfigGlobalName),
          /*.initial_contents=*/{},
          /*.byte_length=*/kFeedbackConfigByteLength,
          /*.alignment=*/8,
          /*.flags=*/LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
      },
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1100"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
      /*.data_symbols=*/data_symbols,
      /*.data_symbol_count=*/IREE_ARRAYSIZE(data_symbols),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  const std::vector<Section> sections = ReadSections(bytes);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& text = FindSection(sections, ".text");
  const std::string dynstr_contents =
      bytes.substr((size_t)dynstr.offset, (size_t)dynstr.size);

  ASSERT_EQ(dynsym.size, 5u * 24u);
  const size_t const_symbol = (size_t)dynsym.offset + 3u * 24u;
  const size_t feedback_symbol = const_symbol + 24u;
  ASSERT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, const_symbol + 0)),
            "loom_const_tag");
  ASSERT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(bytes, feedback_symbol + 0)),
            kFeedbackConfigGlobalName);

  const uint64_t const_target = LoadLeU64(bytes, const_symbol + 8) +
                                text_fixups[0].target_symbol_byte_offset;
  const uint64_t const_delta = const_target - text.address;
  EXPECT_EQ(LoadLeU32(bytes, (size_t)text.offset + 4), (uint32_t)const_delta);
  EXPECT_EQ(LoadLeU32(bytes, (size_t)text.offset + 12),
            (uint32_t)(const_delta >> 32));

  const uint64_t feedback_target = LoadLeU64(bytes, feedback_symbol + 8) +
                                   text_fixups[2].target_symbol_byte_offset;
  const uint64_t feedback_base =
      text.address + text_fixups[2].base_pc_byte_offset;
  const uint64_t feedback_delta = feedback_target - feedback_base;
  EXPECT_EQ(LoadLeU32(bytes, (size_t)text.offset + 28),
            (uint32_t)feedback_delta);
  EXPECT_EQ(LoadLeU32(bytes, (size_t)text.offset + 36),
            (uint32_t)(feedback_delta >> 32));

  for (size_t i = 0; i < text_bytes.size(); ++i) {
    const bool patched = (i >= 4 && i < 8) || (i >= 12 && i < 16) ||
                         (i >= 28 && i < 32) || (i >= 36 && i < 40);
    if (!patched) {
      EXPECT_EQ((uint8_t)bytes[(size_t)text.offset + i], 0xcc);
    }
  }
}

TEST(AmdgpuHsacoTest, RejectsInvalidTextFixups) {
  const uint8_t text_bytes[] = {0xcc, 0xcc, 0xcc, 0xcc};
  const loom_amdgpu_hsaco_data_symbol_t data_symbol = {
      /*.name=*/IREE_SV("loom_data"),
      /*.initial_contents=*/{},
      /*.byte_length=*/8,
      /*.alignment=*/8,
  };

  {
    const loom_amdgpu_hsaco_text_fixup_t text_fixup = {
        /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
        /*.literal_byte_offset=*/1,
        /*.base_pc_byte_offset=*/0,
        /*.target_symbol=*/IREE_SV("loom_data"),
    };
    const loom_amdgpu_hsaco_kernel_t kernel = {
        /*.metadata=*/MinimalKernel(IREE_SV("loom_kernel"),
                                    IREE_SV("loom_kernel.kd")),
        /*.descriptor_options=*/{},
        /*.text=*/iree_make_const_byte_span(text_bytes, sizeof(text_bytes)),
        /*.text_fixups=*/&text_fixup,
        /*.text_fixup_count=*/1,
    };
    const loom_amdgpu_hsaco_file_t file = {
        /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
        /*.processor=*/IREE_SV("gfx1100"),
        /*.kernels=*/&kernel,
        /*.kernel_count=*/1,
        /*.data_symbols=*/&data_symbol,
        /*.data_symbol_count=*/1,
    };

    StreamPtr stream = CreateStream();
    TestArena arena;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  }

  {
    const loom_amdgpu_hsaco_text_fixup_t text_fixup = {
        /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
        /*.literal_byte_offset=*/0,
        /*.base_pc_byte_offset=*/0,
        /*.target_symbol=*/IREE_SV("missing_data"),
    };
    const loom_amdgpu_hsaco_kernel_t kernel = {
        /*.metadata=*/MinimalKernel(IREE_SV("loom_kernel"),
                                    IREE_SV("loom_kernel.kd")),
        /*.descriptor_options=*/{},
        /*.text=*/iree_make_const_byte_span(text_bytes, sizeof(text_bytes)),
        /*.text_fixups=*/&text_fixup,
        /*.text_fixup_count=*/1,
    };
    const loom_amdgpu_hsaco_file_t file = {
        /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
        /*.processor=*/IREE_SV("gfx1100"),
        /*.kernels=*/&kernel,
        /*.kernel_count=*/1,
        /*.data_symbols=*/&data_symbol,
        /*.data_symbol_count=*/1,
    };

    StreamPtr stream = CreateStream();
    TestArena arena;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  }
}

TEST(AmdgpuHsacoTest, WritesEveryTargetCodeObjectFlags) {
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target);
    ASSERT_NE(processor, nullptr);
    ASSERT_TRUE(
        loom_amdgpu_processor_properties_support_hsaco(&processor->properties))
        << StringViewToString(target->name);

    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(target, &identity);
    loom_amdgpu_metadata_kernel_t metadata =
        MinimalKernel(IREE_SV("loom_kernel"), IREE_SV("loom_kernel.kd"));
    metadata.wavefront_size = processor->properties.wavefront.default_size;
    loom_amdgpu_target_profile_t target_profile = {};
    IREE_ASSERT_OK(
        loom_amdgpu_target_profile_initialize(&identity, &target_profile));
    metadata.target_extensions = target->kernel_metadata_extensions;
    const loom_amdgpu_hsaco_kernel_t kernel = {
        /*.metadata=*/metadata,
        /*.descriptor_options=*/{},
        /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
    };
    const std::string target_id = CodeObjectTargetIdForIdentity(identity);
    const loom_amdgpu_hsaco_file_t file = {
        /*.target=*/iree_make_string_view(target_id.data(), target_id.size()),
        /*.processor=*/processor->name,
        /*.kernels=*/&kernel,
        /*.kernel_count=*/1,
    };

    StreamPtr stream = CreateStream();
    TestArena arena;
    IREE_ASSERT_OK(
        loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()))
        << StringViewToString(target->name);
    const std::string bytes = StreamBytes(stream.get());

    ASSERT_GE(bytes.size(), 64u) << StringViewToString(processor->name);
    EXPECT_EQ((uint8_t)bytes[8], LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6)
        << StringViewToString(target->name);
    EXPECT_EQ(LoadLeU32(bytes, 48),
              processor->properties.elf.machine_flags |
                  processor->properties.elf.feature_flags |
                  (processor->properties.elf.generic_version
                   << LOOM_AMDGPU_ELF_GENERIC_VERSION_OFFSET_V6))
        << StringViewToString(target->name);

    const std::vector<Section> sections = ReadSections(bytes);
    const Section& note = FindSection(sections, ".note");
    const std::string note_contents =
        bytes.substr((size_t)note.offset, (size_t)note.size);
    EXPECT_NE(note_contents.find(target_id), std::string::npos)
        << StringViewToString(target->name);
  }
  EXPECT_GE(target_count, 9u);
}

TEST(AmdgpuHsacoTest, WritesNativeKernargLayoutsWithoutHalCompaction) {
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  loom_amdgpu_hsaco_kernel_t kernels[] = {
      {
          /*.metadata=*/MinimalKernel(IREE_SV("fill_x1"),
                                      IREE_SV("fill_x1.kd")),
          /*.descriptor_options=*/{},
          /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
      },
      {
          /*.metadata=*/MinimalKernel(IREE_SV("patch_dispatch"),
                                      IREE_SV("patch_dispatch.kd")),
          /*.descriptor_options=*/{},
          /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
      },
  };

  const loom_amdgpu_metadata_argument_t fill_arguments[] = {
      {
          /*.name=*/IREE_SV("target_ptr"),
          /*.offset=*/0,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
          /*.address_space=*/IREE_SV("global"),
          /*.access=*/{},
          /*.actual_access=*/IREE_SV("write_only"),
      },
      {
          /*.name=*/IREE_SV("element_length"),
          /*.offset=*/8,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
      {
          /*.name=*/IREE_SV("pattern"),
          /*.offset=*/16,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
  };
  const loom_amdgpu_metadata_argument_t patch_arguments[] = {
      {
          /*.name=*/IREE_SV("dispatch_state"),
          /*.offset=*/0,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
          /*.address_space=*/IREE_SV("global"),
          /*.access=*/{},
          /*.actual_access=*/IREE_SV("read_write"),
      },
      {
          /*.name=*/IREE_SV("opcode"),
          /*.offset=*/16,
          /*.size=*/2,
          /*.alignment=*/2,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
      {
          /*.name=*/IREE_SV("flags"),
          /*.offset=*/18,
          /*.size=*/2,
          /*.alignment=*/2,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
      {
          /*.name=*/IREE_SV("dispatch_id"),
          /*.offset=*/24,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
  };

  kernels[0].metadata.kernarg_segment_size = 24;
  kernels[0].metadata.kernarg_segment_alignment = 8;
  kernels[0].metadata.arguments = fill_arguments;
  kernels[0].metadata.argument_count = IREE_ARRAYSIZE(fill_arguments);
  kernels[0].descriptor_options.flags =
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X;
  kernels[1].metadata.kernarg_segment_size = 32;
  kernels[1].metadata.kernarg_segment_alignment = 8;
  kernels[1].metadata.arguments = patch_arguments;
  kernels[1].metadata.argument_count = IREE_ARRAYSIZE(patch_arguments);
  kernels[1].descriptor_options.flags =
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X;

  EXPECT_EQ(kernels[0].metadata.kernarg_segment_size, 24u);
  EXPECT_EQ(kernels[0].metadata.argument_count, 3u);
  EXPECT_EQ(kernels[0].metadata.arguments[1].offset, 8u);
  EXPECT_EQ(kernels[0].metadata.arguments[1].size, 8u);
  EXPECT_EQ(kernels[0].descriptor_options.flags,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X);
  EXPECT_EQ(kernels[1].metadata.kernarg_segment_size, 32u);
  EXPECT_EQ(kernels[1].metadata.argument_count, 4u);
  EXPECT_EQ(kernels[1].metadata.arguments[1].offset, 16u);
  EXPECT_EQ(kernels[1].metadata.arguments[1].size, 2u);

  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1100"),
      /*.kernels=*/kernels,
      /*.kernel_count=*/IREE_ARRAYSIZE(kernels),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());
  const std::vector<Section> sections = ReadSections(bytes);
  const Section& rodata = FindSection(sections, ".rodata");
  const Section& note = FindSection(sections, ".note");

  ASSERT_EQ(rodata.size, 2u * LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  EXPECT_EQ(LoadLeU32(bytes, (size_t)rodata.offset + 8u), 24u);
  EXPECT_EQ(LoadLeU32(bytes, (size_t)rodata.offset +
                                 LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH + 8u),
            32u);

  const std::string note_contents =
      bytes.substr((size_t)note.offset, (size_t)note.size);
  EXPECT_NE(note_contents.find("target_ptr"), std::string::npos);
  EXPECT_NE(note_contents.find("global_buffer"), std::string::npos);
  EXPECT_NE(note_contents.find("dispatch_id"), std::string::npos);
  EXPECT_NE(note_contents.find("by_value"), std::string::npos);
}

TEST(AmdgpuHsacoTest, WritesGfx942CodeObjectTargetFlags) {
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  loom_amdgpu_metadata_kernel_t metadata =
      MinimalKernel(IREE_SV("loom_kernel"), IREE_SV("loom_kernel.kd"));
  metadata.wavefront_size = 64;
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/metadata,
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx942"),
      /*.processor=*/IREE_SV("gfx942"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  ASSERT_GE(bytes.size(), 64u);
  EXPECT_EQ((uint8_t)bytes[7], LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA);
  EXPECT_EQ((uint8_t)bytes[8], LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6);
  EXPECT_EQ(LoadLeU32(bytes, 48), LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX942 |
                                      LOOM_AMDGPU_ELF_FEATURE_XNACK_ANY_V4 |
                                      LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ANY_V4);
}

TEST(AmdgpuHsacoTest, RejectsMismatchedProcessor) {
  const uint8_t text[] = {0x00, 0x00, 0x81, 0xbf};
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/
      MinimalKernel(IREE_SV("loom_kernel"), IREE_SV("loom_kernel.kd")),
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(text, sizeof(text)),
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.processor=*/IREE_SV("gfx1200"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
}

TEST(AmdgpuHsacoTest, WritesTargetFeatureSuffixCodeObjectFlags) {
  const uint8_t text[] = {0x00, 0x00, 0x81, 0xbf};
  loom_amdgpu_metadata_kernel_t metadata =
      MinimalKernel(IREE_SV("loom_kernel"), IREE_SV("loom_kernel.kd"));
  metadata.wavefront_size = 64;
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/metadata,
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(text, sizeof(text)),
  };
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
      /*.processor=*/IREE_SV("gfx942"),
      /*.kernels=*/&kernel,
      /*.kernel_count=*/1,
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  const std::string bytes = StreamBytes(stream.get());

  ASSERT_GE(bytes.size(), 64u);
  EXPECT_EQ(LoadLeU32(bytes, 48), LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX942 |
                                      LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ON_V4 |
                                      LOOM_AMDGPU_ELF_FEATURE_XNACK_OFF_V4);
}

}  // namespace
}  // namespace loom
