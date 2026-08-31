// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/elf_format.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(XdnaElfFormatTest, PacksTileDestinationWithoutTruncation) {
  const loom_xdna_elf_tile_destination_t destination = {
      /*.column=*/17,
      /*.row=*/3,
      /*.memory_space=*/LOOM_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
      /*.flags=*/0,
  };
  uint32_t physical_address = 0;
  IREE_ASSERT_OK(
      loom_xdna_elf_pack_tile_destination(&destination, &physical_address));
  EXPECT_EQ(physical_address, UINT32_C(0x00020311));

  const loom_xdna_elf_tile_destination_t decoded =
      loom_xdna_elf_unpack_tile_destination(physical_address);
  EXPECT_EQ(decoded.column, destination.column);
  EXPECT_EQ(decoded.row, destination.row);
  EXPECT_EQ(decoded.memory_space, destination.memory_space);
  EXPECT_EQ(decoded.flags, destination.flags);
}

TEST(XdnaElfFormatTest, RejectsUnknownTileDestinationState) {
  loom_xdna_elf_tile_destination_t destination = {
      /*.column=*/0,
      /*.row=*/0,
      /*.memory_space=*/
      static_cast<loom_xdna_elf_tile_memory_space_t>(0),
      /*.flags=*/0,
  };
  uint32_t physical_address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_xdna_elf_pack_tile_destination(&destination, &physical_address));
  destination.memory_space = LOOM_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM;
  destination.flags = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_xdna_elf_pack_tile_destination(&destination, &physical_address));
}

TEST(XdnaElfFormatTest, EncodesCanonicalAbiNote) {
  const loom_xdna_elf_abi_note_t note = {
      /*.abi_major=*/LOOM_XDNA_ELF_ABI_MAJOR,
      /*.abi_minor=*/LOOM_XDNA_ELF_ABI_MINOR,
      /*.target_generation=*/LOOM_XDNA_TARGET_GENERATION_AIE2P,
      /*.device_profile_revision=*/0x10,
      /*.device_profile_id=*/UINT64_C(0x1122334455667788),
      /*.firmware_abi_id=*/UINT64_C(0x8877665544332211),
      /*.policy_id=*/UINT64_C(0x0123456789ABCDEF),
      /*.required_capabilities=*/
      LOOM_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS,
      /*.partition_origin_column=*/1,
      /*.partition_origin_row=*/2,
      /*.partition_column_count=*/4,
      /*.partition_row_count=*/6,
      /*.coordinate_model=*/
      LOOM_XDNA_ELF_COORDINATE_MODEL_PARTITION_RELATIVE,
  };
  std::array<uint8_t, LOOM_XDNA_ELF_ABI_NOTE_SIZE> storage;
  IREE_ASSERT_OK(loom_xdna_elf_encode_abi_note(
      &note, iree_make_byte_span(storage.data(), storage.size())));

  EXPECT_EQ(iree_unaligned_load_le_u32(storage.data() + 0), 5u);
  EXPECT_EQ(iree_unaligned_load_le_u32(storage.data() + 4), 64u);
  EXPECT_EQ(iree_unaligned_load_le_u32(storage.data() + 8), 1u);
  EXPECT_EQ(storage[12], 'L');
  EXPECT_EQ(storage[16], 0u);
  EXPECT_EQ(storage[19], 0u);
  const uint8_t* description = storage.data() + 20;
  EXPECT_EQ(iree_unaligned_load_le_u32(description + 0),
            LOOM_XDNA_ELF_ABI_NOTE_MAGIC);
  EXPECT_EQ(iree_unaligned_load_le_u16(description + 4), 64u);
  EXPECT_EQ(iree_unaligned_load_le_u64(description + 16),
            note.device_profile_id);
  EXPECT_EQ(iree_unaligned_load_le_u64(description + 40),
            note.required_capabilities);
  EXPECT_EQ(iree_unaligned_load_le_u16(description + 52), 4u);
  EXPECT_EQ(iree_unaligned_load_le_u16(description + 54), 6u);
  EXPECT_EQ(description[56], LOOM_XDNA_ELF_COORDINATE_MODEL_PARTITION_RELATIVE);
  for (iree_host_size_t i = 57; i < 64; ++i) {
    EXPECT_EQ(description[i], 0u);
  }
}

TEST(XdnaElfFormatTest, EncodesFixedRecordsAtDeclaredWidths) {
  std::array<uint8_t, LOOM_XDNA_ELF_TABLE_HEADER_SIZE> table_storage;
  const loom_xdna_elf_table_header_t table = {
      /*.magic=*/LOOM_XDNA_ELF_ENTRY_TABLE_MAGIC,
      /*.abi_major=*/LOOM_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/LOOM_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/LOOM_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/LOOM_XDNA_ELF_ENTRY_RECORD_SIZE,
      /*.record_count=*/3,
      /*.byte_length=*/256,
      /*.auxiliary_offset=*/144,
  };
  IREE_ASSERT_OK(loom_xdna_elf_encode_table_header(
      &table, iree_make_byte_span(table_storage.data(), table_storage.size())));
  EXPECT_EQ(iree_unaligned_load_le_u32(table_storage.data() + 0),
            LOOM_XDNA_ELF_ENTRY_TABLE_MAGIC);
  EXPECT_EQ(iree_unaligned_load_le_u16(table_storage.data() + 10),
            LOOM_XDNA_ELF_ENTRY_RECORD_SIZE);
  EXPECT_EQ(iree_unaligned_load_le_u32(table_storage.data() + 20), 144u);

  std::array<uint8_t, LOOM_XDNA_ELF_ENTRY_RECORD_SIZE> entry_storage;
  const loom_xdna_elf_entry_record_t entry = {
      /*.export_ordinal=*/3,
      /*.name_offset=*/128,
      /*.name_length=*/7,
      /*.array_program_header_ordinal=*/5,
      /*.control_program_header_ordinal=*/6,
      /*.first_binding_ordinal=*/8,
      /*.binding_count=*/2,
      /*.flags=*/LOOM_XDNA_ELF_ENTRY_FLAG_DEFAULT,
      /*.required_capabilities=*/LOOM_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS,
  };
  IREE_ASSERT_OK(loom_xdna_elf_encode_entry_record(
      &entry, iree_make_byte_span(entry_storage.data(), entry_storage.size())));
  EXPECT_EQ(iree_unaligned_load_le_u32(entry_storage.data() + 0), 3u);
  EXPECT_EQ(iree_unaligned_load_le_u32(entry_storage.data() + 28), 1u);
  EXPECT_EQ(iree_unaligned_load_le_u64(entry_storage.data() + 32), 2u);

  std::array<uint8_t, LOOM_XDNA_ELF_BINDING_RECORD_SIZE> binding_storage;
  const loom_xdna_elf_binding_record_t binding = {
      /*.binding_ordinal=*/8,
      /*.entry_ordinal=*/3,
      /*.kind=*/LOOM_XDNA_ELF_BINDING_KIND_BUFFER,
      /*.address_space=*/LOOM_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
      /*.access=*/LOOM_XDNA_ELF_BINDING_ACCESS_READ,
      /*.usage=*/LOOM_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
      /*.minimum_byte_length=*/4096,
      /*.minimum_alignment=*/64,
      /*.minimum_byte_offset=*/0,
      /*.maximum_byte_offset=*/8192,
  };
  IREE_ASSERT_OK(loom_xdna_elf_encode_binding_record(
      &binding,
      iree_make_byte_span(binding_storage.data(), binding_storage.size())));
  EXPECT_EQ(iree_unaligned_load_le_u16(binding_storage.data() + 8), 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(binding_storage.data() + 20), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u64(binding_storage.data() + 32), 64u);

  std::array<uint8_t, LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE> relocation_storage;
  const loom_xdna_elf_relocation_record_t relocation = {
      /*.target_program_header_ordinal=*/4,
      /*.target_byte_offset=*/32,
      /*.binding_ordinal=*/8,
      /*.kind=*/LOOM_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS,
      /*.field_byte_width=*/8,
      /*.flags=*/0,
      /*.addend=*/-16,
      /*.minimum_value=*/0,
      /*.maximum_value=*/UINT64_MAX,
      /*.required_alignment=*/16,
  };
  IREE_ASSERT_OK(loom_xdna_elf_encode_relocation_record(
      &relocation, iree_make_byte_span(relocation_storage.data(),
                                       relocation_storage.size())));
  EXPECT_EQ(iree_unaligned_load_le_u32(relocation_storage.data() + 4), 32u);
  EXPECT_EQ(relocation_storage[14], 8u);
  EXPECT_EQ(static_cast<int64_t>(
                iree_unaligned_load_le_u64(relocation_storage.data() + 16)),
            -16);

  std::array<uint8_t, LOOM_XDNA_ELF_ARRAY_HEADER_SIZE> array_storage;
  const loom_xdna_elf_array_header_t array = {
      /*.abi_major=*/LOOM_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/LOOM_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/7,
      /*.byte_length=*/512,
      /*.flags=*/0,
      /*.first_tile_program_header_ordinal=*/4,
      /*.tile_program_header_count=*/9,
  };
  IREE_ASSERT_OK(loom_xdna_elf_encode_array_header(
      &array, iree_make_byte_span(array_storage.data(), array_storage.size())));
  EXPECT_EQ(iree_unaligned_load_le_u32(array_storage.data() + 0),
            LOOM_XDNA_ELF_ARRAY_MAGIC);
  EXPECT_EQ(iree_unaligned_load_le_u32(array_storage.data() + 16), 512u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array_storage.data() + 28), 9u);

  std::array<uint8_t, LOOM_XDNA_ELF_CONTROL_HEADER_SIZE> control_storage;
  const loom_xdna_elf_control_header_t control = {
      /*.abi_major=*/LOOM_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/LOOM_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/2,
      /*.byte_length=*/96,
      /*.flags=*/0,
  };
  IREE_ASSERT_OK(loom_xdna_elf_encode_control_header(
      &control,
      iree_make_byte_span(control_storage.data(), control_storage.size())));
  EXPECT_EQ(iree_unaligned_load_le_u32(control_storage.data() + 0),
            LOOM_XDNA_ELF_CONTROL_MAGIC);
  EXPECT_EQ(iree_unaligned_load_le_u16(control_storage.data() + 8),
            LOOM_XDNA_ELF_CONTROL_HEADER_SIZE);
  EXPECT_EQ(iree_unaligned_load_le_u32(control_storage.data() + 12), 2u);
}

TEST(XdnaElfFormatTest, RejectsIncompleteAbiIdentity) {
  loom_xdna_elf_abi_note_t note = {
      /*.abi_major=*/LOOM_XDNA_ELF_ABI_MAJOR,
      /*.abi_minor=*/LOOM_XDNA_ELF_ABI_MINOR,
      /*.target_generation=*/LOOM_XDNA_TARGET_GENERATION_AIE2P,
      /*.device_profile_revision=*/0x10,
      /*.device_profile_id=*/1,
      /*.firmware_abi_id=*/0,
      /*.policy_id=*/0,
      /*.required_capabilities=*/0,
      /*.partition_origin_column=*/0,
      /*.partition_origin_row=*/0,
      /*.partition_column_count=*/1,
      /*.partition_row_count=*/1,
      /*.coordinate_model=*/
      LOOM_XDNA_ELF_COORDINATE_MODEL_PARTITION_RELATIVE,
  };
  std::array<uint8_t, LOOM_XDNA_ELF_ABI_NOTE_SIZE> storage;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_xdna_elf_encode_abi_note(
          &note, iree_make_byte_span(storage.data(), storage.size())));

  note.firmware_abi_id = 1;
  note.partition_origin_column = UINT16_MAX;
  note.partition_column_count = 2;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_xdna_elf_encode_abi_note(
          &note, iree_make_byte_span(storage.data(), storage.size())));
}

TEST(XdnaElfFormatTest, RejectsNoncanonicalProgramRecord) {
  std::array<uint8_t, LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE> storage;
  loom_xdna_elf_program_record_header_t header = {
      /*.type=*/0,
      /*.flags=*/0,
      /*.byte_length=*/LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_xdna_elf_encode_program_record_header(
          &header, iree_make_byte_span(storage.data(), storage.size())));
  header.type = 1;
  header.byte_length = LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_xdna_elf_encode_program_record_header(
          &header, iree_make_byte_span(storage.data(), storage.size())));
}

}  // namespace
