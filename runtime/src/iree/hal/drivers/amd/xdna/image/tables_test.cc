// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/tables.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::ImageBuilder;
using iree::hal::amd::xdna::testing::MakeProgramHeader;
using testing::HasSubstr;

struct ByteSequenceDeleter {
  void operator()(iree_byte_sequence_t* sequence) const {
    iree_byte_sequence_release(sequence);
  }
};

using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, ByteSequenceDeleter>;

struct DirectoryDeleter {
  void operator()(iree_hal_amd_xdna_image_directory_t* directory) const {
    iree_hal_amd_xdna_image_directory_destroy(directory);
  }
};

using DirectoryPtr =
    std::unique_ptr<iree_hal_amd_xdna_image_directory_t, DirectoryDeleter>;

struct TablesDeleter {
  void operator()(iree_hal_amd_xdna_image_tables_t* tables) const {
    iree_hal_amd_xdna_image_tables_destroy(tables);
  }
};

using TablesPtr =
    std::unique_ptr<iree_hal_amd_xdna_image_tables_t, TablesDeleter>;

enum : uint32_t {
  kNoteOffset = 216,
  kEntryOffset = 304,
  kBindingOffset = 376,
  kRelocationOffset = 512,
  kArrayOffset = 632,
};

static ByteSequencePtr make_owned_sequence(const std::vector<uint8_t>& bytes) {
  iree_byte_span_t storage = iree_byte_span_empty();
  storage.data_length = bytes.size();
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), storage.data_length, (void**)&storage.data));
  memcpy(storage.data, bytes.data(), bytes.size());
  iree_byte_sequence_t* sequence = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &storage, iree_allocator_system(), &sequence));
  return ByteSequencePtr(sequence);
}

static std::vector<uint8_t> encode_abi_note(
    iree_hal_amd_xdna_elf_capabilities_t required_capabilities) {
  const iree_hal_amd_xdna_elf_abi_note_t note = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_ABI_MINOR,
      /*.target_generation=*/IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P,
      /*.device_profile_revision=*/1,
      /*.device_profile_id=*/UINT64_C(0x535848414C4F0001),
      /*.firmware_abi_id=*/UINT64_C(0x4E5055320006000C),
      /*.policy_id=*/UINT64_C(0x413250504C414E01),
      /*.required_capabilities=*/required_capabilities,
      /*.context_origin_column=*/0,
      /*.context_origin_row=*/0,
      /*.context_column_count=*/8,
      /*.context_row_count=*/6,
      /*.coordinate_model=*/
      IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE,
  };
  std::vector<uint8_t> bytes(IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_abi_note(
      &note, iree_make_byte_span(bytes.data(), bytes.size())));
  return bytes;
}

static std::vector<uint8_t> encode_entry_table(
    const std::vector<iree_hal_amd_xdna_elf_entry_record_t>& entries,
    const std::vector<uint8_t>& names) {
  const uint32_t name_offset =
      IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
      (uint32_t)entries.size() * IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
      /*.record_count=*/(uint32_t)entries.size(),
      /*.byte_length=*/name_offset + (uint32_t)names.size(),
      /*.auxiliary_offset=*/name_offset,
  };
  std::vector<uint8_t> bytes(header.byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE)));
  for (iree_host_size_t i = 0; i < entries.size(); ++i) {
    IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_entry_record(
        &entries[i],
        iree_make_byte_span(bytes.data() +
                                IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                                i * IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
                            IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE)));
  }
  if (!names.empty()) {
    memcpy(bytes.data() + name_offset, names.data(), names.size());
  }
  return bytes;
}

static std::vector<uint8_t> encode_binding_table(
    const std::vector<iree_hal_amd_xdna_elf_binding_record_t>& bindings) {
  const uint32_t byte_length =
      IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
      (uint32_t)bindings.size() * IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_BINDING_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE,
      /*.record_count=*/(uint32_t)bindings.size(),
      /*.byte_length=*/byte_length,
      /*.auxiliary_offset=*/0,
  };
  std::vector<uint8_t> bytes(byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE)));
  for (iree_host_size_t i = 0; i < bindings.size(); ++i) {
    IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_binding_record(
        &bindings[i],
        iree_make_byte_span(bytes.data() +
                                IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                                i * IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE,
                            IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE)));
  }
  return bytes;
}

static std::vector<uint8_t> encode_relocation_table(
    const std::vector<iree_hal_amd_xdna_elf_relocation_record_t>& relocations) {
  const uint32_t byte_length = IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                               (uint32_t)relocations.size() *
                                   IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE,
      /*.record_count=*/(uint32_t)relocations.size(),
      /*.byte_length=*/byte_length,
      /*.auxiliary_offset=*/0,
  };
  std::vector<uint8_t> bytes(byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE)));
  for (iree_host_size_t i = 0; i < relocations.size(); ++i) {
    IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_relocation_record(
        &relocations[i],
        iree_make_byte_span(
            bytes.data() + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                i * IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE,
            IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE)));
  }
  return bytes;
}

static std::vector<uint8_t> make_valid_image() {
  const iree_hal_amd_xdna_elf_entry_record_t entry = {
      /*.export_ordinal=*/0,
      /*.name_offset=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
          IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
      /*.name_length=*/4,
      /*.array_program_header_ordinal=*/4,
      /*.control_program_header_ordinal=*/UINT32_MAX,
      /*.first_binding_ordinal=*/0,
      /*.binding_count=*/2,
      /*.flags=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT,
      /*.required_capabilities=*/
      IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS,
  };
  const iree_hal_amd_xdna_elf_binding_record_t buffer_binding = {
      /*.binding_ordinal=*/0,
      /*.entry_ordinal=*/0,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER,
      /*.address_space=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
      /*.access=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ |
          IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE,
      /*.usage=*/IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
      /*.minimum_byte_length=*/64,
      /*.minimum_alignment=*/16,
      /*.minimum_byte_offset=*/0,
      /*.maximum_byte_offset=*/256,
  };
  const iree_hal_amd_xdna_elf_binding_record_t scalar_binding = {
      /*.binding_ordinal=*/1,
      /*.entry_ordinal=*/0,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_SCALAR,
      /*.address_space=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_NONE,
      /*.access=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ,
      /*.usage=*/0,
      /*.minimum_byte_length=*/4,
      /*.minimum_alignment=*/4,
      /*.minimum_byte_offset=*/0,
      /*.maximum_byte_offset=*/0,
  };
  const iree_hal_amd_xdna_elf_relocation_record_t address_relocation = {
      /*.target_program_header_ordinal=*/4,
      /*.target_byte_offset=*/12,
      /*.binding_ordinal=*/0,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS,
      /*.field_byte_width=*/8,
      /*.flags=*/0,
      /*.addend=*/0,
      /*.minimum_value=*/0,
      /*.maximum_value=*/UINT64_MAX,
      /*.required_alignment=*/4,
  };
  const iree_hal_amd_xdna_elf_relocation_record_t scalar_relocation = {
      /*.target_program_header_ordinal=*/4,
      /*.target_byte_offset=*/20,
      /*.binding_ordinal=*/1,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_SCALAR_VALUE,
      /*.field_byte_width=*/4,
      /*.flags=*/0,
      /*.addend=*/0,
      /*.minimum_value=*/0,
      /*.maximum_value=*/UINT32_MAX,
      /*.required_alignment=*/4,
  };

  const std::vector<uint8_t> note =
      encode_abi_note(IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS);
  const std::vector<uint8_t> entries =
      encode_entry_table({entry}, {'m', 'a', 'i', 'n'});
  const std::vector<uint8_t> bindings =
      encode_binding_table({buffer_binding, scalar_binding});
  const std::vector<uint8_t> relocations =
      encode_relocation_table({address_relocation, scalar_relocation});
  auto entry_header = MakeProgramHeader(
      IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES, kEntryOffset, entries.size());
  entry_header.alignment = 8;
  auto binding_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS,
                        kBindingOffset, bindings.size());
  binding_header.alignment = 8;
  auto relocation_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS,
                        kRelocationOffset, relocations.size());
  relocation_header.alignment = 8;
  ImageBuilder builder;
  builder
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE,
                                    kNoteOffset, note.size()),
                  note)
      .AddProgram(entry_header, entries)
      .AddProgram(binding_header, bindings)
      .AddProgram(relocation_header, relocations)
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                                    kArrayOffset, 32));
  return builder.Build();
}

static std::vector<uint8_t> make_image_without_bindings_or_relocations() {
  const iree_hal_amd_xdna_elf_entry_record_t entry = {
      /*.export_ordinal=*/0,
      /*.name_offset=*/0,
      /*.name_length=*/0,
      /*.array_program_header_ordinal=*/3,
      /*.control_program_header_ordinal=*/UINT32_MAX,
      /*.first_binding_ordinal=*/0,
      /*.binding_count=*/0,
      /*.flags=*/0,
      /*.required_capabilities=*/0,
  };
  const std::vector<uint8_t> note = encode_abi_note(0);
  const std::vector<uint8_t> entries = encode_entry_table({entry}, {});
  const std::vector<uint8_t> bindings = encode_binding_table({});
  auto entry_header = MakeProgramHeader(
      IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES, kEntryOffset, entries.size());
  entry_header.alignment = 8;
  auto binding_header = MakeProgramHeader(
      IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS, 368, bindings.size());
  binding_header.alignment = 8;
  ImageBuilder builder;
  builder
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE,
                                    kNoteOffset, note.size()),
                  note)
      .AddProgram(entry_header, entries)
      .AddProgram(binding_header, bindings)
      .AddProgram(
          MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 392, 32));
  return builder.Build();
}

static DirectoryPtr open_directory(const std::vector<uint8_t>& bytes,
                                   ByteSequencePtr* out_sequence = nullptr) {
  ByteSequencePtr sequence = make_owned_sequence(bytes);
  iree_hal_amd_xdna_image_directory_t* directory = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_directory_create(
      sequence.get(), iree_allocator_system(), &directory));
  if (out_sequence != nullptr) *out_sequence = std::move(sequence);
  return DirectoryPtr(directory);
}

static TablesPtr open_tables(const std::vector<uint8_t>& bytes) {
  DirectoryPtr directory = open_directory(bytes);
  iree_hal_amd_xdna_image_tables_t* tables = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_tables_create(
      directory.get(), iree_allocator_system(), &tables));
  return TablesPtr(tables);
}

static Status open_tables_status(const std::vector<uint8_t>& bytes) {
  DirectoryPtr directory = open_directory(bytes);
  iree_hal_amd_xdna_image_tables_t* tables = nullptr;
  Status status(iree_hal_amd_xdna_image_tables_create(
      directory.get(), iree_allocator_system(), &tables));
  iree_hal_amd_xdna_image_tables_destroy(tables);
  return status;
}

static void expect_status(const Status& status, StatusCode expected_code,
                          const char* expected_message_substring) {
  EXPECT_EQ(status.code(), expected_code);
  EXPECT_THAT(status.ToString(), HasSubstr(expected_message_substring));
}

static uint8_t* program_header(std::vector<uint8_t>& bytes, uint32_t ordinal) {
  return bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE +
         ordinal * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE;
}

TEST(ImageTablesTest, DecodesAndOwnsGenericMetadata) {
  ByteSequencePtr sequence;
  DirectoryPtr directory = open_directory(make_valid_image(), &sequence);
  iree_hal_amd_xdna_image_tables_t* raw_tables = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_tables_create(
      directory.get(), iree_allocator_system(), &raw_tables));
  TablesPtr tables(raw_tables);
  directory.reset();
  sequence.reset();

  const auto* note = iree_hal_amd_xdna_image_tables_abi_note(tables.get());
  ASSERT_NE(note, nullptr);
  EXPECT_EQ(note->target_generation, IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P);
  EXPECT_EQ(note->context_column_count, 8);
  EXPECT_EQ(note->context_row_count, 6);

  ASSERT_EQ(iree_hal_amd_xdna_image_tables_entry_count(tables.get()), 1);
  const auto* entry = iree_hal_amd_xdna_image_tables_entry(tables.get(), 0);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->export_ordinal, 0);
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amd_xdna_image_tables_entry_name(tables.get(), 0),
      IREE_SV("main")));
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_entry(tables.get(), 1), nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_hal_amd_xdna_image_tables_entry_name(tables.get(), 1)));

  ASSERT_EQ(iree_hal_amd_xdna_image_tables_binding_count(tables.get()), 2);
  const auto* buffer = iree_hal_amd_xdna_image_tables_binding(tables.get(), 0);
  const auto* scalar = iree_hal_amd_xdna_image_tables_binding(tables.get(), 1);
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(buffer->kind, IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER);
  EXPECT_EQ(scalar->kind, IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_SCALAR);
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_binding(tables.get(), 2), nullptr);

  ASSERT_EQ(iree_hal_amd_xdna_image_tables_relocation_count(tables.get()), 2);
  const auto* relocation =
      iree_hal_amd_xdna_image_tables_relocation(tables.get(), 1);
  ASSERT_NE(relocation, nullptr);
  EXPECT_EQ(relocation->kind,
            IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_SCALAR_VALUE);
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_relocation(tables.get(), 2),
            nullptr);
}

TEST(ImageTablesTest, AcceptsNoBindingsOrRelocations) {
  TablesPtr tables = open_tables(make_image_without_bindings_or_relocations());
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_entry_count(tables.get()), 1);
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_hal_amd_xdna_image_tables_entry_name(tables.get(), 0)));
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_binding_count(tables.get()), 0);
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_relocation_count(tables.get()), 0);
}

TEST(ImageTablesTest, RejectsMissingDuplicateOrMalformedTablePrograms) {
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(program_header(bytes, 1), 0x6C58FFFF);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "missing a required metadata table");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(program_header(bytes, 4),
                                IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "more than one metadata program");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(program_header(bytes, 2) + 28, 4);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "requires alignment 8");
  }
}

TEST(ImageTablesTest, RejectsUnsupportedOrIncompleteAbiNotes) {
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + kNoteOffset + 26, 2);
    expect_status(open_tables_status(bytes), StatusCode::kUnimplemented,
                  "unsupported XDNA image ABI version");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u64(bytes.data() + kNoteOffset + 36, 0);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "identity is incomplete");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + kNoteOffset + 68, 1);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "not context-relative");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u64(bytes.data() + kNoteOffset + 60, UINT64_C(1)
                                                                     << 63);
    expect_status(open_tables_status(bytes), StatusCode::kUnimplemented,
                  "unknown capabilities");
  }
}

TEST(ImageTablesTest, RejectsMalformedFixedTableHeaders) {
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kEntryOffset, 0);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "fixed metadata table header is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + kEntryOffset + 4, 2);
    expect_status(open_tables_status(bytes), StatusCode::kUnimplemented,
                  "unsupported XDNA metadata table ABI version");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kEntryOffset + 20, 24);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "noncanonical name range");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kBindingOffset + 16, 24);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "fixed metadata table header is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kRelocationOffset + 12, 0);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "relocation table is empty");
  }
}

TEST(ImageTablesTest, RejectsNoncanonicalEntries) {
  const uint32_t entry_record_offset =
      kEntryOffset + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE;
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + entry_record_offset, 1);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "invalid identity, flags, or capabilities");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + entry_record_offset + 4,
                                entry_record_offset);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "names are not canonically packed");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    bytes[kEntryOffset + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
          IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE + 1] = 0;
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "names are not canonically packed");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + entry_record_offset + 24, 1);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "unowned bindings");
  }
}

TEST(ImageTablesTest, RejectsMalformedBindings) {
  const uint32_t first_binding_offset =
      kBindingOffset + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE;
  const uint32_t second_binding_offset =
      first_binding_offset + IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE;
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + first_binding_offset, 1);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "binding 0 is malformed");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + first_binding_offset + 16, 0);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "invalid resource contract");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + second_binding_offset + 8, 99);
    expect_status(open_tables_status(bytes), StatusCode::kUnimplemented,
                  "unknown resource kind 99");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u64(bytes.data() + first_binding_offset + 24,
                                UINT64_MAX);
    iree_unaligned_store_le_u64(bytes.data() + first_binding_offset + 40, 1);
    iree_unaligned_store_le_u64(bytes.data() + first_binding_offset + 48, 1);
    expect_status(open_tables_status(bytes), StatusCode::kOutOfRange,
                  "range overflows");
  }
}

TEST(ImageTablesTest, AcceptsBindingOffsetBoundsWithRepresentableRanges) {
  const uint32_t binding_offset =
      kBindingOffset + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE;
  for (uint64_t minimum_offset : {UINT64_C(0), UINT64_MAX - 64}) {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u64(bytes.data() + binding_offset + 40,
                                minimum_offset);
    iree_unaligned_store_le_u64(bytes.data() + binding_offset + 48, UINT64_MAX);
    DirectoryPtr directory = open_directory(bytes);
    iree_hal_amd_xdna_image_tables_t* raw_tables = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_tables_create(
        directory.get(), iree_allocator_system(), &raw_tables));
    TablesPtr tables(raw_tables);
    const auto* binding =
        iree_hal_amd_xdna_image_tables_binding(tables.get(), 0);
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->minimum_byte_length, 64u);
    EXPECT_EQ(binding->minimum_byte_offset, minimum_offset);
    EXPECT_EQ(binding->maximum_byte_offset, UINT64_MAX);
  }
}

TEST(ImageTablesTest, RejectsMalformedOrMisboundRelocations) {
  const uint32_t first_relocation_offset =
      kRelocationOffset + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE;
  const uint32_t second_relocation_offset =
      first_relocation_offset + IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE;
  {
    std::vector<uint8_t> bytes = make_valid_image();
    bytes[first_relocation_offset + 14] = 2;
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "invalid field contract");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(
        bytes.data() + first_relocation_offset + 12,
        IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_SCALAR_VALUE);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "requires a matching scalar binding");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + first_relocation_offset + 12,
                                99);
    expect_status(open_tables_status(bytes), StatusCode::kUnimplemented,
                  "unknown kind 99");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + second_relocation_offset + 4,
                                12);
    expect_status(open_tables_status(bytes), StatusCode::kInvalidArgument,
                  "overlap or are not canonically ordered");
  }
}

}  // namespace
