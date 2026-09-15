// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"

#include <cstring>
#include <iterator>

#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_builder.h"

namespace iree::hal::amd::xdna::testing {
namespace {

enum : uint32_t {
  kNoteOffset = 320,
  kEntryOffset = 408,
  kBindingOffset = 480,
  kRelocationOffset = 560,
  kTileOffset = 640,
  kArrayOffset = 672,
  kControlOffset = 716,
};

static std::vector<uint8_t> EncodeAbiNote(
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
      /*.context_column_count=*/3,
      /*.context_row_count=*/6,
      /*.coordinate_model=*/
      IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE,
  };
  std::vector<uint8_t> bytes(IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_abi_note(
      &note, iree_make_byte_span(bytes.data(), bytes.size())));
  return bytes;
}

static std::vector<uint8_t> EncodeEntryTable(
    const iree_hal_amd_xdna_elf_entry_record_t& entry) {
  constexpr uint32_t kNameLength = 4;
  constexpr uint32_t kNameOffset = IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                                   IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
      /*.record_count=*/1,
      /*.byte_length=*/kNameOffset + kNameLength,
      /*.auxiliary_offset=*/kNameOffset,
  };
  std::vector<uint8_t> bytes(header.byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE)));
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_entry_record(
      &entry, iree_make_byte_span(
                  bytes.data() + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
                  IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE)));
  memcpy(bytes.data() + kNameOffset, "main", kNameLength);
  return bytes;
}

static std::vector<uint8_t> EncodeBindingTable(
    const iree_hal_amd_xdna_elf_binding_record_t& binding) {
  constexpr uint32_t kByteLength = IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                                   IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_BINDING_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE,
      /*.record_count=*/1,
      /*.byte_length=*/kByteLength,
      /*.auxiliary_offset=*/0,
  };
  std::vector<uint8_t> bytes(kByteLength);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE)));
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_binding_record(
      &binding, iree_make_byte_span(
                    bytes.data() + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
                    IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE)));
  return bytes;
}

static std::vector<uint8_t> EncodeRelocationTable(
    const iree_hal_amd_xdna_elf_relocation_record_t& relocation) {
  constexpr uint32_t kByteLength = IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                                   IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_table_header_t header = {
      /*.magic=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_TABLE_MAGIC,
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR,
      /*.header_size=*/IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      /*.record_size=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE,
      /*.record_count=*/1,
      /*.byte_length=*/kByteLength,
      /*.auxiliary_offset=*/0,
  };
  std::vector<uint8_t> bytes(kByteLength);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_table_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE)));
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_relocation_record(
      &relocation, iree_make_byte_span(
                       bytes.data() + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
                       IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE)));
  return bytes;
}

static std::vector<uint8_t> EncodeArray(uint32_t tile_program_ordinal) {
  std::vector<uint8_t> record(
      IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE);
  const iree_hal_amd_xdna_aie2p_tile_program_load_t load = {
      /*.program_header_ordinal=*/tile_program_ordinal,
  };
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_encode_tile_program_load(
      &load, iree_make_byte_span(record.data(), record.size())));
  const uint32_t byte_length =
      IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE + (uint32_t)record.size();
  const iree_hal_amd_xdna_elf_array_header_t header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/1,
      /*.byte_length=*/byte_length,
      /*.flags=*/0,
      /*.first_tile_program_header_ordinal=*/
      kAie2pImageTileProgramHeaderOrdinal,
      /*.tile_program_header_count=*/1,
  };
  std::vector<uint8_t> bytes(byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_array_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE)));
  memcpy(bytes.data() + IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE, record.data(),
         record.size());
  return bytes;
}

static std::vector<uint8_t> EncodeControl(
    const Aie2pImageFixtureOptions& options) {
  std::vector<uint8_t> record;
  if (options.use_relocatable_block_write) {
    constexpr uint32_t kWords[] = {0, 0};
    record.resize(24);
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_encode_register_block_write32(
        options.control_register_address, std::size(kWords), kWords,
        iree_make_byte_span(record.data(), record.size())));
  } else {
    const iree_hal_amd_xdna_aie2p_register_write32_t write = {
        /*.address=*/UINT32_C(0x0001D000),
        /*.value=*/0,
    };
    record.resize(IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE);
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
        &write, iree_make_byte_span(record.data(), record.size())));
  }
  const uint32_t byte_length =
      IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE + (uint32_t)record.size();
  const iree_hal_amd_xdna_elf_control_header_t header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/1,
      /*.byte_length=*/byte_length,
      /*.flags=*/0,
  };
  std::vector<uint8_t> bytes(byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_control_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE)));
  memcpy(bytes.data() + IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE,
         record.data(), record.size());
  return bytes;
}

static iree_status_t ValidateConfigurationRegister(
    const void* user_data, iree_hal_amd_xdna_elf_program_type_t program_type,
    const iree_hal_amd_xdna_aie2p_program_record_t* record) {
  (void)program_type;
  (void)record;
  const auto* state = static_cast<const Aie2pImageTargetState*>(user_data);
  ++state->configuration_register_call_count;
  return iree_ok_status();
}

static iree_status_t ValidateDmaTaskWait(
    const void* user_data,
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait) {
  (void)user_data;
  (void)wait;
  return iree_ok_status();
}

static iree_status_t ResolveTileMemory(
    const void* user_data,
    const iree_hal_amd_xdna_elf_tile_destination_t* destination,
    uint32_t virtual_address, uint32_t byte_length,
    iree_hal_amd_xdna_image_tile_placement_t* out_placement) {
  const auto* state = static_cast<const Aie2pImageTargetState*>(user_data);
  *out_placement = {
      /*.program_header_ordinal=*/0,
      /*.owner_column=*/
      static_cast<uint16_t>(state->owner_column == UINT16_MAX
                                ? destination->column
                                : state->owner_column),
      /*.owner_row=*/destination->row,
      /*.memory_space=*/destination->memory_space,
      /*.owner_offset=*/virtual_address,
      /*.byte_length=*/byte_length,
      /*.available_capacity=*/4096,
  };
  return iree_ok_status();
}

}  // namespace

void ByteSequenceDeleter::operator()(iree_byte_sequence_t* sequence) const {
  iree_byte_sequence_release(sequence);
}

ByteSequencePtr MakeOwnedByteSequence(const std::vector<uint8_t>& bytes) {
  iree_byte_span_t storage = iree_byte_span_empty();
  storage.data_length = bytes.size();
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), storage.data_length, (void**)&storage.data));
  if (!bytes.empty()) {
    memcpy(storage.data, bytes.data(), bytes.size());
  }
  iree_byte_sequence_t* sequence = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &storage, iree_allocator_system(), &sequence));
  return ByteSequencePtr(sequence);
}

std::vector<uint8_t> BuildAie2pImage(const Aie2pImageFixtureOptions& options) {
  constexpr uint32_t kNameOffset = IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                                   IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE;
  const iree_hal_amd_xdna_elf_entry_record_t entry = {
      /*.export_ordinal=*/0,
      /*.name_offset=*/kNameOffset,
      /*.name_length=*/4,
      /*.array_program_header_ordinal=*/
      options.entry_array_program_header_ordinal,
      /*.control_program_header_ordinal=*/
      options.entry_control_program_header_ordinal,
      /*.first_binding_ordinal=*/0,
      /*.binding_count=*/1,
      /*.flags=*/IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT,
      /*.required_capabilities=*/options.required_capabilities,
  };
  const iree_hal_amd_xdna_elf_binding_record_t binding = {
      /*.binding_ordinal=*/0,
      /*.entry_ordinal=*/0,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER,
      /*.address_space=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
      /*.access=*/IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ,
      /*.usage=*/IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
      /*.minimum_byte_length=*/64,
      /*.minimum_alignment=*/4,
      /*.minimum_byte_offset=*/0,
      /*.maximum_byte_offset=*/256,
  };
  const iree_hal_amd_xdna_elf_relocation_record_t relocation = {
      /*.target_program_header_ordinal=*/
      kAie2pImageControlProgramHeaderOrdinal,
      /*.target_byte_offset=*/options.relocation_target_byte_offset,
      /*.binding_ordinal=*/0,
      /*.kind=*/IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS,
      /*.field_byte_width=*/options.relocation_field_byte_width,
      /*.flags=*/0,
      /*.addend=*/options.relocation_addend,
      /*.minimum_value=*/options.relocation_minimum_value,
      /*.maximum_value=*/options.relocation_maximum_value,
      /*.required_alignment=*/options.relocation_required_alignment,
  };

  const std::vector<uint8_t> note =
      EncodeAbiNote(options.required_capabilities);
  const std::vector<uint8_t> entries = EncodeEntryTable(entry);
  const std::vector<uint8_t> bindings = EncodeBindingTable(binding);
  const std::vector<uint8_t> relocations = EncodeRelocationTable(relocation);
  const std::vector<uint8_t> array =
      EncodeArray(options.array_tile_program_header_ordinal);
  const std::vector<uint8_t> control = EncodeControl(options);

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
  auto tile_header = MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE,
                                       kTileOffset, 32);
  tile_header.flags = IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ |
                      IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE;
  tile_header.alignment = 16;
  const iree_hal_amd_xdna_elf_tile_destination_t tile_destination = {
      /*.column=*/1,
      /*.row=*/2,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM,
      /*.flags=*/0,
  };
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_pack_tile_destination(
      &tile_destination, &tile_header.physical_address));

  ImageBuilder builder;
  builder
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE,
                                    kNoteOffset, note.size()),
                  note)
      .AddProgram(entry_header, entries)
      .AddProgram(binding_header, bindings)
      .AddProgram(relocation_header, relocations)
      .AddProgram(tile_header)
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                                    kArrayOffset, array.size()),
                  array)
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL,
                                    kControlOffset, control.size()),
                  control);
  return builder.Build();
}

iree_hal_amd_xdna_aie2p_target_t MakeAie2pImageTarget(
    Aie2pImageTargetState* state) {
  return {
      /*.identity=*/
      {
          /*.device_profile_revision=*/1,
          /*.device_profile_id=*/UINT64_C(0x535848414C4F0001),
          /*.firmware_abi_id=*/UINT64_C(0x4E5055320006000C),
          /*.policy_id=*/UINT64_C(0x413250504C414E01),
      },
      /*.supported_capabilities=*/IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES,
      /*.context=*/
      {
          /*.column_count=*/3,
          /*.row_count=*/6,
      },
      /*.native=*/
      {
          /*.transaction=*/
          {
              /*.device_generation=*/4,
              /*.memory_tile_row_count=*/1,
          },
          /*.register_address=*/
          {
              /*.column_shift=*/25,
              /*.row_shift=*/20,
          },
          /*.program_memory=*/
          {
              /*.host_offset=*/UINT32_C(0x00020000),
          },
      },
      /*.tile_memory_resolver=*/
      {
          /*.fn=*/ResolveTileMemory,
          /*.user_data=*/state,
      },
      /*.configuration_register_validator=*/
      {
          /*.fn=*/ValidateConfigurationRegister,
          /*.user_data=*/state,
      },
      /*.dma_task_wait_validator=*/
      {
          /*.fn=*/ValidateDmaTaskWait,
          /*.user_data=*/state,
      },
  };
}

}  // namespace iree::hal::amd::xdna::testing
