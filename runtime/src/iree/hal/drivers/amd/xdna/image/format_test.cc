// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/format.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <size_t N>
static void ExpectBytes(const std::array<uint8_t, N>& actual,
                        std::initializer_list<uint8_t> expected) {
  ASSERT_EQ(expected.size(), actual.size());
  size_t i = 0;
  for (uint8_t value : expected) {
    EXPECT_EQ(value, actual[i]) << "byte offset " << i;
    ++i;
  }
}

TEST(XdnaImageFormatTest, PacksTileDestinationWithoutTruncation) {
  const iree_hal_amd_xdna_elf_tile_destination_t destination = {
      /*.column=*/17,
      /*.row=*/3,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
      /*.flags=*/0,
  };
  uint32_t physical_address = 0;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_pack_tile_destination(
      &destination, &physical_address));
  EXPECT_EQ(UINT32_C(0x00020311), physical_address);

  const iree_hal_amd_xdna_elf_tile_destination_t decoded =
      iree_hal_amd_xdna_elf_unpack_tile_destination(physical_address);
  EXPECT_EQ(destination.column, decoded.column);
  EXPECT_EQ(destination.row, decoded.row);
  EXPECT_EQ(destination.memory_space, decoded.memory_space);
  EXPECT_EQ(destination.flags, decoded.flags);
}

TEST(XdnaImageFormatTest, RejectsUnknownTileDestinationState) {
  iree_hal_amd_xdna_elf_tile_destination_t destination = {
      /*.column=*/0,
      /*.row=*/0,
      /*.memory_space=*/
      static_cast<iree_hal_amd_xdna_elf_tile_memory_space_t>(0),
      /*.flags=*/0,
  };
  uint32_t physical_address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_elf_pack_tile_destination(
                            &destination, &physical_address));
  destination.memory_space = IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM;
  destination.flags = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_elf_pack_tile_destination(
                            &destination, &physical_address));
}

TEST(XdnaImageFormatTest, AbiNoteMatchesWireAbiAndRoundTrips) {
  const iree_hal_amd_xdna_elf_abi_note_t note = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_ABI_MINOR,
      /*.target_generation=*/IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P,
      /*.device_profile_revision=*/UINT32_C(0x10203040),
      /*.device_profile_id=*/UINT64_C(0x1122334455667788),
      /*.firmware_abi_id=*/UINT64_C(0x8877665544332211),
      /*.policy_id=*/UINT64_C(0x0123456789ABCDEF),
      /*.required_capabilities=*/
      IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS |
          IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS,
      /*.context_origin_column=*/0,
      /*.context_origin_row=*/0,
      /*.context_column_count=*/4,
      /*.context_row_count=*/6,
      /*.coordinate_model=*/
      IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_abi_note(
      &note, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage,
              {
                  0x05, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,  // Envelope.
                  0x01, 0x00, 0x00, 0x00, 'L',  'O',  'O',  'M',  0x00, 0x00,
                  0x00, 0x00, 'X',  'D',  'N',  'A',  // Description.
                  0x40, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x40, 0x30,
                  0x20, 0x10, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
                  0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0xEF, 0xCD,
                  0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, 0x03, 0x00, 0x00, 0x00,
                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
                  0x06, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
              });

  iree_hal_amd_xdna_elf_abi_note_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_abi_note(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(note.abi_major, decoded.abi_major);
  EXPECT_EQ(note.abi_minor, decoded.abi_minor);
  EXPECT_EQ(note.target_generation, decoded.target_generation);
  EXPECT_EQ(note.device_profile_revision, decoded.device_profile_revision);
  EXPECT_EQ(note.device_profile_id, decoded.device_profile_id);
  EXPECT_EQ(note.firmware_abi_id, decoded.firmware_abi_id);
  EXPECT_EQ(note.policy_id, decoded.policy_id);
  EXPECT_EQ(note.required_capabilities, decoded.required_capabilities);
  EXPECT_EQ(note.context_origin_column, decoded.context_origin_column);
  EXPECT_EQ(note.context_origin_row, decoded.context_origin_row);
  EXPECT_EQ(note.context_column_count, decoded.context_column_count);
  EXPECT_EQ(note.context_row_count, decoded.context_row_count);
  EXPECT_EQ(note.coordinate_model, decoded.coordinate_model);
}

TEST(XdnaImageFormatTest, AbiNoteRequiresCompleteContextRelativeIdentity) {
  iree_hal_amd_xdna_elf_abi_note_t note = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_ABI_MINOR,
      /*.target_generation=*/IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P,
      /*.device_profile_revision=*/1,
      /*.device_profile_id=*/1,
      /*.firmware_abi_id=*/1,
      /*.policy_id=*/1,
      /*.required_capabilities=*/0,
      /*.context_origin_column=*/0,
      /*.context_origin_row=*/0,
      /*.context_column_count=*/1,
      /*.context_row_count=*/1,
      /*.coordinate_model=*/
      IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE> storage;

  note.policy_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_encode_abi_note(
          &note, iree_make_byte_span(storage.data(), storage.size())));
  note.policy_id = 1;
  note.context_origin_column = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_encode_abi_note(
          &note, iree_make_byte_span(storage.data(), storage.size())));
  note.context_origin_column = 0;
  note.required_capabilities = UINT64_C(1) << 63;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_encode_abi_note(
          &note, iree_make_byte_span(storage.data(), storage.size())));
}

TEST(XdnaImageFormatTest, AbiNoteDecoderRejectsNoncanonicalEnvelope) {
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE> storage = {};
  iree_unaligned_store_le_u32(storage.data() + 0,
                              IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE);
  iree_unaligned_store_le_u32(storage.data() + 4,
                              IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE);
  iree_unaligned_store_le_u32(storage.data() + 8,
                              IREE_HAL_AMD_XDNA_ELF_NOTE_TYPE_ABI);
  memcpy(storage.data() + 12, IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER,
         IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE);
  iree_unaligned_store_le_u32(storage.data() + 20,
                              IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_MAGIC);
  iree_unaligned_store_le_u16(storage.data() + 24,
                              IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE);
  storage[17] = 1;

  iree_hal_amd_xdna_elf_abi_note_t decoded = {
      /*.abi_major=*/UINT16_MAX,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_decode_abi_note(
          iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(0, decoded.abi_major);
}

TEST(XdnaImageFormatTest, FixedTableHeaderMatchesWireAbiAndRoundTrips) {
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
      /*.record_count=*/3,
      /*.byte_length=*/256,
      /*.auxiliary_offset=*/144,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage, {
                           0x58, 0x45, 0x4E, 0x54, 0x01, 0x00, 0x00, 0x00,
                           0x18, 0x00, 0x28, 0x00, 0x03, 0x00, 0x00, 0x00,
                           0x00, 0x01, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00,
                       });

  iree_hal_amd_xdna_elf_table_header_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_table_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(header.magic, decoded.magic);
  EXPECT_EQ(header.abi_major, decoded.abi_major);
  EXPECT_EQ(header.abi_minor, decoded.abi_minor);
  EXPECT_EQ(header.header_size, decoded.header_size);
  EXPECT_EQ(header.record_size, decoded.record_size);
  EXPECT_EQ(header.record_count, decoded.record_count);
  EXPECT_EQ(header.byte_length, decoded.byte_length);
  EXPECT_EQ(header.auxiliary_offset, decoded.auxiliary_offset);
}

TEST(XdnaImageFormatTest, EntryRecordMatchesWireAbiAndRoundTrips) {
  const iree_hal_amd_xdna_elf_entry_record_t record = {
      /*.export_ordinal=*/3,
      /*.name_offset=*/128,
      /*.name_length=*/7,
      /*.array_program_header_ordinal=*/5,
      /*.control_program_header_ordinal=*/6,
      /*.first_binding_ordinal=*/8,
      /*.binding_count=*/2,
      /*.flags=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT,
      /*.required_capabilities=*/
      IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_entry_record(
      &record, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage,
              {
                  0x03, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x07, 0x00,
                  0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
                  0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00,
                  0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
              });

  iree_hal_amd_xdna_elf_entry_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_entry_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(record.export_ordinal, decoded.export_ordinal);
  EXPECT_EQ(record.name_offset, decoded.name_offset);
  EXPECT_EQ(record.name_length, decoded.name_length);
  EXPECT_EQ(record.array_program_header_ordinal,
            decoded.array_program_header_ordinal);
  EXPECT_EQ(record.control_program_header_ordinal,
            decoded.control_program_header_ordinal);
  EXPECT_EQ(record.first_binding_ordinal, decoded.first_binding_ordinal);
  EXPECT_EQ(record.binding_count, decoded.binding_count);
  EXPECT_EQ(record.flags, decoded.flags);
  EXPECT_EQ(record.required_capabilities, decoded.required_capabilities);
}

TEST(XdnaImageFormatTest, BindingRecordMatchesWireAbiAndRoundTrips) {
  const iree_hal_amd_xdna_elf_binding_record_t record = {
      /*.binding_ordinal=*/8,
      /*.entry_ordinal=*/3,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER,
      /*.address_space=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
      /*.access=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ,
      /*.usage=*/IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
      /*.minimum_byte_length=*/UINT64_C(0x1122334455667788),
      /*.minimum_alignment=*/64,
      /*.minimum_byte_offset=*/16,
      /*.maximum_byte_offset=*/UINT64_C(0x8877665544332211),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_binding_record(
      &record, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage,
              {
                  0x08, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00,
                  0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                  0x00, 0x00, 0x00, 0x00, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33,
                  0x22, 0x11, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x22,
                  0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
              });

  iree_hal_amd_xdna_elf_binding_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_binding_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(record.binding_ordinal, decoded.binding_ordinal);
  EXPECT_EQ(record.entry_ordinal, decoded.entry_ordinal);
  EXPECT_EQ(record.kind, decoded.kind);
  EXPECT_EQ(record.address_space, decoded.address_space);
  EXPECT_EQ(record.access, decoded.access);
  EXPECT_EQ(record.usage, decoded.usage);
  EXPECT_EQ(record.minimum_byte_length, decoded.minimum_byte_length);
  EXPECT_EQ(record.minimum_alignment, decoded.minimum_alignment);
  EXPECT_EQ(record.minimum_byte_offset, decoded.minimum_byte_offset);
  EXPECT_EQ(record.maximum_byte_offset, decoded.maximum_byte_offset);

  storage[20] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_decode_binding_record(
          iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
}

TEST(XdnaImageFormatTest, RelocationRecordMatchesWireAbiAndRoundTrips) {
  const iree_hal_amd_xdna_elf_relocation_record_t record = {
      /*.target_program_header_ordinal=*/4,
      /*.target_byte_offset=*/32,
      /*.binding_ordinal=*/8,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS,
      /*.field_byte_width=*/8,
      /*.flags=*/0,
      /*.addend=*/-16,
      /*.minimum_value=*/0,
      /*.maximum_value=*/UINT64_MAX,
      /*.required_alignment=*/16,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_relocation_record(
      &record, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage,
              {
                  0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x08, 0x00,
                  0x00, 0x00, 0x01, 0x00, 0x08, 0x00, 0xF0, 0xFF, 0xFF, 0xFF,
                  0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                  0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
              });

  iree_hal_amd_xdna_elf_relocation_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_relocation_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(record.target_program_header_ordinal,
            decoded.target_program_header_ordinal);
  EXPECT_EQ(record.target_byte_offset, decoded.target_byte_offset);
  EXPECT_EQ(record.binding_ordinal, decoded.binding_ordinal);
  EXPECT_EQ(record.kind, decoded.kind);
  EXPECT_EQ(record.field_byte_width, decoded.field_byte_width);
  EXPECT_EQ(record.flags, decoded.flags);
  EXPECT_EQ(record.addend, decoded.addend);
  EXPECT_EQ(record.minimum_value, decoded.minimum_value);
  EXPECT_EQ(record.maximum_value, decoded.maximum_value);
  EXPECT_EQ(record.required_alignment, decoded.required_alignment);
}

TEST(XdnaImageFormatTest, ProgramHeadersMatchWireAbiAndRoundTrip) {
  const iree_hal_amd_xdna_elf_array_header_t array_header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/7,
      /*.byte_length=*/512,
      /*.flags=*/0,
      /*.first_tile_program_header_ordinal=*/4,
      /*.tile_program_header_count=*/9,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE> array_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_array_header(
      &array_header,
      iree_make_byte_span(array_storage.data(), array_storage.size())));
  ExpectBytes(array_storage, {
                                 0x58, 0x41, 0x52, 0x52, 0x01, 0x00, 0x00, 0x00,
                                 0x20, 0x00, 0x08, 0x00, 0x07, 0x00, 0x00, 0x00,
                                 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                 0x04, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
                             });
  iree_hal_amd_xdna_elf_array_header_t decoded_array;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_array_header(
      iree_make_const_byte_span(array_storage.data(), array_storage.size()),
      &decoded_array));
  EXPECT_EQ(array_header.abi_major, decoded_array.abi_major);
  EXPECT_EQ(array_header.abi_minor, decoded_array.abi_minor);
  EXPECT_EQ(array_header.record_count, decoded_array.record_count);
  EXPECT_EQ(array_header.byte_length, decoded_array.byte_length);
  EXPECT_EQ(array_header.flags, decoded_array.flags);
  EXPECT_EQ(array_header.first_tile_program_header_ordinal,
            decoded_array.first_tile_program_header_ordinal);
  EXPECT_EQ(array_header.tile_program_header_count,
            decoded_array.tile_program_header_count);

  const iree_hal_amd_xdna_elf_control_header_t control_header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/2,
      /*.byte_length=*/96,
      /*.flags=*/0,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE>
      control_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_control_header(
      &control_header,
      iree_make_byte_span(control_storage.data(), control_storage.size())));
  ExpectBytes(control_storage,
              {
                  0x58, 0x43, 0x54, 0x4C, 0x01, 0x00, 0x00, 0x00,
                  0x18, 0x00, 0x08, 0x00, 0x02, 0x00, 0x00, 0x00,
                  0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
              });
  iree_hal_amd_xdna_elf_control_header_t decoded_control;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_control_header(
      iree_make_const_byte_span(control_storage.data(), control_storage.size()),
      &decoded_control));
  EXPECT_EQ(control_header.abi_major, decoded_control.abi_major);
  EXPECT_EQ(control_header.abi_minor, decoded_control.abi_minor);
  EXPECT_EQ(control_header.record_count, decoded_control.record_count);
  EXPECT_EQ(control_header.byte_length, decoded_control.byte_length);
  EXPECT_EQ(control_header.flags, decoded_control.flags);
}

TEST(XdnaImageFormatTest, ProgramRecordHeaderValidatesFraming) {
  const iree_hal_amd_xdna_elf_program_record_header_t header = {
      /*.type=*/7,
      /*.flags=*/UINT16_C(0x1234),
      /*.byte_length=*/16,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_program_record_header(
      &header, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage, {0x07, 0x00, 0x34, 0x12, 0x10, 0x00, 0x00, 0x00});

  iree_hal_amd_xdna_elf_program_record_header_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_program_record_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(header.type, decoded.type);
  EXPECT_EQ(header.flags, decoded.flags);
  EXPECT_EQ(header.byte_length, decoded.byte_length);

  storage[4] = IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_decode_program_record_header(
          iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
}

TEST(XdnaImageFormatTest, CodecsRequireExactStorageWidths) {
  std::array<uint8_t, IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE> storage = {};
  iree_hal_amd_xdna_elf_table_header_t header;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_decode_table_header(
          iree_make_const_byte_span(storage.data(), storage.size() - 1),
          &header));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_elf_decode_table_header(
          iree_make_const_byte_span(storage.data(), storage.size() + 1),
          &header));
}

}  // namespace
