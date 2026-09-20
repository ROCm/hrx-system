// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf_object.h"

#include <cstdint>
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
  // Arena receiving transient object and ELF writer storage.
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

std::string LoadString(const std::string& bytes, size_t offset) {
  return bytes.substr(offset, bytes.find('\0', offset) - offset);
}

struct ElfSectionView {
  // ELF section-table index.
  uint16_t index;
  // Section name read through `.shstrtab`.
  std::string name;
  // ELF SHT_* section type.
  uint32_t type;
  // ELF SHF_* section flags.
  uint64_t flags;
  // Section payload byte offset in the file.
  uint64_t offset;
  // Logical section byte size.
  uint64_t size;
  // Section-type-specific linked section index.
  uint32_t link;
  // Section-type-specific auxiliary index or count.
  uint32_t info;
  // Section payload alignment.
  uint64_t alignment;
  // Fixed record size for table sections.
  uint64_t entry_size;
};

std::vector<ElfSectionView> ParseSections(const std::string& bytes) {
  const size_t section_table_offset = (size_t)LoadLeU64(bytes, 40);
  const uint16_t section_header_size = LoadLeU16(bytes, 58);
  const uint16_t section_count = LoadLeU16(bytes, 60);
  const uint16_t string_section_index = LoadLeU16(bytes, 62);
  EXPECT_EQ(section_header_size, 64u);
  EXPECT_LT(string_section_index, section_count);
  const size_t string_section_header_offset =
      section_table_offset + (size_t)string_section_index * section_header_size;
  const size_t string_table_offset =
      (size_t)LoadLeU64(bytes, string_section_header_offset + 24);

  std::vector<ElfSectionView> sections;
  sections.reserve(section_count);
  for (uint16_t i = 0; i < section_count; ++i) {
    const size_t header_offset =
        section_table_offset + (size_t)i * section_header_size;
    const uint32_t name_offset = LoadLeU32(bytes, header_offset + 0);
    sections.push_back(ElfSectionView{
        /*.index=*/i,
        /*.name=*/LoadString(bytes, string_table_offset + name_offset),
        /*.type=*/LoadLeU32(bytes, header_offset + 4),
        /*.flags=*/LoadLeU64(bytes, header_offset + 8),
        /*.offset=*/LoadLeU64(bytes, header_offset + 24),
        /*.size=*/LoadLeU64(bytes, header_offset + 32),
        /*.link=*/LoadLeU32(bytes, header_offset + 40),
        /*.info=*/LoadLeU32(bytes, header_offset + 44),
        /*.alignment=*/LoadLeU64(bytes, header_offset + 48),
        /*.entry_size=*/LoadLeU64(bytes, header_offset + 56),
    });
  }
  return sections;
}

const ElfSectionView& FindSection(const std::vector<ElfSectionView>& sections,
                                  const std::string& name) {
  for (const ElfSectionView& section : sections) {
    if (section.name == name) {
      return section;
    }
  }
  ADD_FAILURE() << "missing ELF section " << name;
  return sections.front();
}

uint32_t MapTestRelocation(const void*, uint32_t native_relocation_kind) {
  return native_relocation_kind == 7 ? 4u : 0u;
}

TEST(NativeElfObjectTest, WritesCoalescedSectionsSymbolsAndRelocations) {
  const uint8_t first_text[] = {0x11, 0x22, 0x33, 0x44};
  const uint8_t data[] = {0x55, 0x66};
  const uint8_t second_text[] = {0xe8, 0x00, 0x00, 0x00, 0x00};
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text"),
          /*.kind=*/LOOM_NATIVE_SECTION_KIND_BYTES,
          /*.flags=*/LOOM_NATIVE_SECTION_FLAG_ALLOCATED |
              LOOM_NATIVE_SECTION_FLAG_EXECUTABLE,
          /*.contribution_alignment=*/1,
          /*.contents=*/
          iree_make_const_byte_span(first_text, sizeof(first_text)),
      },
      {
          /*.section_name=*/IREE_SV(".data"),
          /*.kind=*/LOOM_NATIVE_SECTION_KIND_BYTES,
          /*.flags=*/LOOM_NATIVE_SECTION_FLAG_ALLOCATED |
              LOOM_NATIVE_SECTION_FLAG_WRITABLE,
          /*.contribution_alignment=*/4,
          /*.contents=*/iree_make_const_byte_span(data, sizeof(data)),
      },
      {
          /*.section_name=*/IREE_SV(".text"),
          /*.kind=*/LOOM_NATIVE_SECTION_KIND_BYTES,
          /*.flags=*/LOOM_NATIVE_SECTION_FLAG_ALLOCATED |
              LOOM_NATIVE_SECTION_FLAG_EXECUTABLE,
          /*.contribution_alignment=*/8,
          /*.contents=*/
          iree_make_const_byte_span(second_text, sizeof(second_text)),
      },
  };
  const loom_native_object_symbol_t symbols[] = {
      {
          /*.name=*/IREE_SV("caller"),
          /*.section_contribution_index=*/2,
          /*.section_offset=*/0,
          /*.size=*/sizeof(second_text),
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
          /*.definition=*/LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_SECTION,
      },
      {
          /*.name=*/IREE_SV("local_data"),
          /*.section_contribution_index=*/1,
          /*.section_offset=*/1,
          /*.size=*/1,
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA,
          /*.definition=*/LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_SECTION,
      },
      {
          /*.name=*/IREE_SV("callee"),
          /*.section_contribution_index=*/{},
          /*.section_offset=*/{},
          /*.size=*/{},
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
          /*.definition=*/LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_UNDEFINED,
      },
  };
  const loom_native_object_fixup_t fixups[] = {{
      /*.section_contribution_index=*/2,
      /*.section_offset=*/1,
      /*.relocation_kind=*/7,
      /*.target_symbol_index=*/2,
      /*.addend=*/-4,
  }};
  const loom_native_object_contribution_t object = {
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
      /*.symbols=*/symbols,
      /*.symbol_count=*/IREE_ARRAYSIZE(symbols),
      /*.fixups=*/fixups,
      /*.fixup_count=*/IREE_ARRAYSIZE(fixups),
  };
  loom_native_elf64le_relocatable_options_t options = {
      /*.machine=*/LOOM_NATIVE_ELF_MACHINE_X86_64,
      /*.os_abi=*/LOOM_NATIVE_ELF_OS_ABI_NONE,
      /*.abi_version=*/LOOM_NATIVE_ELF_ABI_VERSION_NONE,
      /*.flags=*/0,
      /*.relocation_mapper=*/{},
  };

  TestArena arena;
  StreamPtr rejected_stream = CreateStream();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_native_elf64le_write_relocatable_object(
          &object, &options, rejected_stream.get(), arena.arena()));
  EXPECT_EQ(iree_io_stream_length(rejected_stream.get()), 0);
  EXPECT_EQ(arena.arena()->used_allocation_size, 0u);

  options.relocation_mapper.map = MapTestRelocation;
  StreamPtr stream = CreateStream();
  IREE_ASSERT_OK(loom_native_elf64le_write_relocatable_object(
      &object, &options, stream.get(), arena.arena()));
  EXPECT_EQ(arena.arena()->used_allocation_size, 0u);
  const std::string bytes = StreamBytes(stream.get());

  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ((uint8_t)bytes[4], 2u);
  EXPECT_EQ((uint8_t)bytes[5], 1u);
  EXPECT_EQ(LoadLeU16(bytes, 16), LOOM_NATIVE_ELF_FILE_TYPE_REL);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_X86_64);

  const std::vector<ElfSectionView> elf_sections = ParseSections(bytes);
  ASSERT_EQ(elf_sections.size(), 7u);
  const ElfSectionView& text = FindSection(elf_sections, ".text");
  const ElfSectionView& data_section = FindSection(elf_sections, ".data");
  const ElfSectionView& rela_text = FindSection(elf_sections, ".rela.text");
  const ElfSectionView& symbol_table = FindSection(elf_sections, ".symtab");
  const ElfSectionView& string_table = FindSection(elf_sections, ".strtab");

  ASSERT_EQ(text.size, 13u);
  EXPECT_EQ(text.alignment, 8u);
  EXPECT_EQ(bytes.substr((size_t)text.offset, sizeof(first_text)),
            std::string((const char*)first_text, sizeof(first_text)));
  EXPECT_EQ(bytes.substr((size_t)text.offset + 4, 4), std::string(4, '\0'));
  EXPECT_EQ(bytes.substr((size_t)text.offset + 8, sizeof(second_text)),
            std::string((const char*)second_text, sizeof(second_text)));
  EXPECT_EQ(data_section.size, sizeof(data));

  EXPECT_EQ(symbol_table.type, LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB);
  EXPECT_EQ(symbol_table.link, string_table.index);
  EXPECT_EQ(symbol_table.info, 2u);
  EXPECT_EQ(symbol_table.entry_size, 24u);
  ASSERT_EQ(symbol_table.size, 4u * 24u);
  const size_t local_symbol = (size_t)symbol_table.offset + 24u;
  const size_t caller_symbol = local_symbol + 24u;
  const size_t callee_symbol = caller_symbol + 24u;
  EXPECT_EQ(LoadString(bytes, (size_t)string_table.offset +
                                  LoadLeU32(bytes, local_symbol)),
            "local_data");
  EXPECT_EQ((uint8_t)bytes[local_symbol + 4], 0x01u);
  EXPECT_EQ((uint8_t)bytes[local_symbol + 5], 0x02u);
  EXPECT_EQ(LoadLeU16(bytes, local_symbol + 6), data_section.index);
  EXPECT_EQ(LoadLeU64(bytes, local_symbol + 8), 1u);
  EXPECT_EQ(LoadString(bytes, (size_t)string_table.offset +
                                  LoadLeU32(bytes, caller_symbol)),
            "caller");
  EXPECT_EQ((uint8_t)bytes[caller_symbol + 4], 0x12u);
  EXPECT_EQ(LoadLeU16(bytes, caller_symbol + 6), text.index);
  EXPECT_EQ(LoadLeU64(bytes, caller_symbol + 8), 8u);
  EXPECT_EQ(LoadString(bytes, (size_t)string_table.offset +
                                  LoadLeU32(bytes, callee_symbol)),
            "callee");
  EXPECT_EQ((uint8_t)bytes[callee_symbol + 4], 0x12u);
  EXPECT_EQ(LoadLeU16(bytes, callee_symbol + 6), 0u);
  EXPECT_EQ(LoadLeU64(bytes, callee_symbol + 8), 0u);

  EXPECT_EQ(rela_text.type, LOOM_NATIVE_ELF_SECTION_TYPE_RELA);
  EXPECT_EQ(rela_text.flags, LOOM_NATIVE_ELF_SECTION_FLAG_INFO_LINK);
  EXPECT_EQ(rela_text.link, symbol_table.index);
  EXPECT_EQ(rela_text.info, text.index);
  EXPECT_EQ(rela_text.alignment, 8u);
  EXPECT_EQ(rela_text.entry_size, 24u);
  ASSERT_EQ(rela_text.size, 24u);
  EXPECT_EQ(LoadLeU64(bytes, (size_t)rela_text.offset + 0), 9u);
  EXPECT_EQ(LoadLeU64(bytes, (size_t)rela_text.offset + 8),
            (UINT64_C(3) << 32) | 4u);
  EXPECT_EQ(LoadLeU64(bytes, (size_t)rela_text.offset + 16), (uint64_t)-4);
}

}  // namespace
}  // namespace loom
