// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/elf_format.h"

#include <inttypes.h>
#include <string.h>

static iree_status_t loom_xdna_elf_require_storage(
    iree_byte_span_t storage, iree_host_size_t required_size,
    iree_string_view_t record_name) {
  if (storage.data_length != required_size ||
      (required_size != 0 && storage.data == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA %.*s storage must contain exactly %" PRIhsz " bytes",
        (int)record_name.size, record_name.data, required_size);
  }
  memset(storage.data, 0, storage.data_length);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_pack_tile_destination(
    const loom_xdna_elf_tile_destination_t* destination,
    uint32_t* out_physical_address) {
  IREE_ASSERT_ARGUMENT(destination);
  IREE_ASSERT_ARGUMENT(out_physical_address);
  *out_physical_address = 0;
  if (destination->memory_space != LOOM_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM &&
      destination->memory_space != LOOM_XDNA_ELF_TILE_MEMORY_SPACE_DATA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown XDNA tile memory space %u",
                            (unsigned)destination->memory_space);
  }
  if (destination->flags != LOOM_XDNA_ELF_TILE_KNOWN_FLAGS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown XDNA tile destination flags 0x%02X",
                            destination->flags);
  }
  *out_physical_address =
      ((uint32_t)destination->column << LOOM_XDNA_ELF_TILE_COLUMN_SHIFT) |
      ((uint32_t)destination->row << LOOM_XDNA_ELF_TILE_ROW_SHIFT) |
      ((uint32_t)destination->memory_space
       << LOOM_XDNA_ELF_TILE_MEMORY_SPACE_SHIFT) |
      ((uint32_t)destination->flags << LOOM_XDNA_ELF_TILE_FLAGS_SHIFT);
  return iree_ok_status();
}

loom_xdna_elf_tile_destination_t loom_xdna_elf_unpack_tile_destination(
    uint32_t physical_address) {
  return (loom_xdna_elf_tile_destination_t){
      .column = (uint8_t)((physical_address & LOOM_XDNA_ELF_TILE_COLUMN_MASK) >>
                          LOOM_XDNA_ELF_TILE_COLUMN_SHIFT),
      .row = (uint8_t)((physical_address & LOOM_XDNA_ELF_TILE_ROW_MASK) >>
                       LOOM_XDNA_ELF_TILE_ROW_SHIFT),
      .memory_space =
          (loom_xdna_elf_tile_memory_space_t)((physical_address &
                                               LOOM_XDNA_ELF_TILE_MEMORY_SPACE_MASK) >>
                                              LOOM_XDNA_ELF_TILE_MEMORY_SPACE_SHIFT),
      .flags = (uint8_t)((physical_address & LOOM_XDNA_ELF_TILE_FLAGS_MASK) >>
                         LOOM_XDNA_ELF_TILE_FLAGS_SHIFT),
  };
}

iree_status_t loom_xdna_elf_encode_abi_note(
    const loom_xdna_elf_abi_note_t* note, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(note);
  IREE_RETURN_IF_ERROR(loom_xdna_elf_require_storage(
      storage, LOOM_XDNA_ELF_ABI_NOTE_SIZE, IREE_SV("ABI note")));
  if (note->abi_major != LOOM_XDNA_ELF_ABI_MAJOR ||
      note->abi_minor != LOOM_XDNA_ELF_ABI_MINOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA ABI version %u.%u",
                            note->abi_major, note->abi_minor);
  }
  if (note->target_generation != LOOM_XDNA_TARGET_GENERATION_AIE2P) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA target generation %u",
                            (unsigned)note->target_generation);
  }
  if (note->device_profile_id == 0 || note->firmware_abi_id == 0 ||
      note->partition_column_count == 0 || note->partition_row_count == 0 ||
      note->partition_column_count > UINT8_MAX + 1u ||
      note->partition_row_count > UINT8_MAX + 1u ||
      (uint32_t)note->partition_origin_column + note->partition_column_count >
          UINT16_MAX + 1u ||
      (uint32_t)note->partition_origin_row + note->partition_row_count >
          UINT16_MAX + 1u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA ABI note has an invalid device profile or "
                            "partition geometry");
  }
  if (note->coordinate_model !=
      LOOM_XDNA_ELF_COORDINATE_MODEL_PARTITION_RELATIVE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA coordinate model %u",
                            (unsigned)note->coordinate_model);
  }
  if ((note->required_capabilities & ~LOOM_XDNA_ELF_KNOWN_CAPABILITIES) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA ABI note requires unknown capabilities");
  }

  iree_unaligned_store_le_u32(storage.data + 0, LOOM_XDNA_ELF_NOTE_OWNER_SIZE);
  iree_unaligned_store_le_u32(storage.data + 4,
                              LOOM_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE);
  iree_unaligned_store_le_u32(storage.data + 8, LOOM_XDNA_ELF_NOTE_TYPE_ABI);
  memcpy(storage.data + 12, LOOM_XDNA_ELF_NOTE_OWNER,
         LOOM_XDNA_ELF_NOTE_OWNER_SIZE);
  uint8_t* description = storage.data + 20;
  iree_unaligned_store_le_u32(description + 0, LOOM_XDNA_ELF_ABI_NOTE_MAGIC);
  iree_unaligned_store_le_u16(description + 4,
                              LOOM_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE);
  iree_unaligned_store_le_u16(description + 6, note->abi_major);
  iree_unaligned_store_le_u16(description + 8, note->abi_minor);
  iree_unaligned_store_le_u16(description + 10,
                              (uint16_t)note->target_generation);
  iree_unaligned_store_le_u32(description + 12, note->target_revision);
  iree_unaligned_store_le_u64(description + 16, note->device_profile_id);
  iree_unaligned_store_le_u64(description + 24, note->firmware_abi_id);
  iree_unaligned_store_le_u64(description + 32, note->policy_id);
  iree_unaligned_store_le_u64(description + 40, note->required_capabilities);
  iree_unaligned_store_le_u16(description + 48, note->partition_origin_column);
  iree_unaligned_store_le_u16(description + 50, note->partition_origin_row);
  iree_unaligned_store_le_u16(description + 52, note->partition_column_count);
  iree_unaligned_store_le_u16(description + 54, note->partition_row_count);
  description[56] = (uint8_t)note->coordinate_model;
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_table_header(
    const loom_xdna_elf_table_header_t* header, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(loom_xdna_elf_require_storage(
      storage, LOOM_XDNA_ELF_TABLE_HEADER_SIZE, IREE_SV("table header")));
  iree_unaligned_store_le_u32(storage.data + 0, header->magic);
  iree_unaligned_store_le_u16(storage.data + 4, header->abi_major);
  iree_unaligned_store_le_u16(storage.data + 6, header->abi_minor);
  iree_unaligned_store_le_u16(storage.data + 8, header->header_size);
  iree_unaligned_store_le_u16(storage.data + 10, header->record_size);
  iree_unaligned_store_le_u32(storage.data + 12, header->record_count);
  iree_unaligned_store_le_u32(storage.data + 16, header->byte_length);
  iree_unaligned_store_le_u32(storage.data + 20, header->auxiliary_offset);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_entry_record(
    const loom_xdna_elf_entry_record_t* record, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_RETURN_IF_ERROR(loom_xdna_elf_require_storage(
      storage, LOOM_XDNA_ELF_ENTRY_RECORD_SIZE, IREE_SV("entry record")));
  iree_unaligned_store_le_u32(storage.data + 0, record->export_ordinal);
  iree_unaligned_store_le_u32(storage.data + 4, record->name_offset);
  iree_unaligned_store_le_u32(storage.data + 8, record->name_length);
  iree_unaligned_store_le_u32(storage.data + 12,
                              record->array_program_header_ordinal);
  iree_unaligned_store_le_u32(storage.data + 16,
                              record->control_program_header_ordinal);
  iree_unaligned_store_le_u32(storage.data + 20, record->first_binding_ordinal);
  iree_unaligned_store_le_u32(storage.data + 24, record->binding_count);
  iree_unaligned_store_le_u32(storage.data + 28, record->flags);
  iree_unaligned_store_le_u64(storage.data + 32, record->required_capabilities);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_binding_record(
    const loom_xdna_elf_binding_record_t* record, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_RETURN_IF_ERROR(loom_xdna_elf_require_storage(
      storage, LOOM_XDNA_ELF_BINDING_RECORD_SIZE, IREE_SV("binding record")));
  iree_unaligned_store_le_u32(storage.data + 0, record->binding_ordinal);
  iree_unaligned_store_le_u32(storage.data + 4, record->entry_ordinal);
  iree_unaligned_store_le_u16(storage.data + 8, (uint16_t)record->kind);
  iree_unaligned_store_le_u16(storage.data + 10,
                              (uint16_t)record->address_space);
  iree_unaligned_store_le_u32(storage.data + 12, record->access);
  iree_unaligned_store_le_u32(storage.data + 16, record->usage);
  iree_unaligned_store_le_u32(storage.data + 20, 0);
  iree_unaligned_store_le_u64(storage.data + 24, record->minimum_byte_length);
  iree_unaligned_store_le_u64(storage.data + 32, record->minimum_alignment);
  iree_unaligned_store_le_u64(storage.data + 40, record->minimum_byte_offset);
  iree_unaligned_store_le_u64(storage.data + 48, record->maximum_byte_offset);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_relocation_record(
    const loom_xdna_elf_relocation_record_t* record, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_RETURN_IF_ERROR(loom_xdna_elf_require_storage(
      storage, LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE,
      IREE_SV("relocation record")));
  iree_unaligned_store_le_u32(storage.data + 0,
                              record->target_program_header_ordinal);
  iree_unaligned_store_le_u32(storage.data + 4, record->target_byte_offset);
  iree_unaligned_store_le_u32(storage.data + 8, record->binding_ordinal);
  iree_unaligned_store_le_u16(storage.data + 12, (uint16_t)record->kind);
  storage.data[14] = record->field_byte_width;
  storage.data[15] = record->flags;
  iree_unaligned_store_le_u64(storage.data + 16, (uint64_t)record->addend);
  iree_unaligned_store_le_u64(storage.data + 24, record->minimum_value);
  iree_unaligned_store_le_u64(storage.data + 32, record->maximum_value);
  iree_unaligned_store_le_u64(storage.data + 40, record->required_alignment);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_array_header(
    const loom_xdna_elf_array_header_t* header, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(
      loom_xdna_elf_require_storage(storage, LOOM_XDNA_ELF_ARRAY_HEADER_SIZE,
                                    IREE_SV("array payload header")));
  iree_unaligned_store_le_u32(storage.data + 0, LOOM_XDNA_ELF_ARRAY_MAGIC);
  iree_unaligned_store_le_u16(storage.data + 4, header->abi_major);
  iree_unaligned_store_le_u16(storage.data + 6, header->abi_minor);
  iree_unaligned_store_le_u16(storage.data + 8,
                              LOOM_XDNA_ELF_ARRAY_HEADER_SIZE);
  iree_unaligned_store_le_u16(storage.data + 10,
                              LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE);
  iree_unaligned_store_le_u32(storage.data + 12, header->record_count);
  iree_unaligned_store_le_u32(storage.data + 16, header->byte_length);
  iree_unaligned_store_le_u32(storage.data + 20, header->flags);
  iree_unaligned_store_le_u32(storage.data + 24,
                              header->first_tile_program_header_ordinal);
  iree_unaligned_store_le_u32(storage.data + 28,
                              header->tile_program_header_count);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_control_header(
    const loom_xdna_elf_control_header_t* header, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(
      loom_xdna_elf_require_storage(storage, LOOM_XDNA_ELF_CONTROL_HEADER_SIZE,
                                    IREE_SV("control payload header")));
  iree_unaligned_store_le_u32(storage.data + 0, LOOM_XDNA_ELF_CONTROL_MAGIC);
  iree_unaligned_store_le_u16(storage.data + 4, header->abi_major);
  iree_unaligned_store_le_u16(storage.data + 6, header->abi_minor);
  iree_unaligned_store_le_u16(storage.data + 8,
                              LOOM_XDNA_ELF_CONTROL_HEADER_SIZE);
  iree_unaligned_store_le_u16(storage.data + 10,
                              LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE);
  iree_unaligned_store_le_u32(storage.data + 12, header->record_count);
  iree_unaligned_store_le_u32(storage.data + 16, header->byte_length);
  iree_unaligned_store_le_u32(storage.data + 20, header->flags);
  return iree_ok_status();
}

iree_status_t loom_xdna_elf_encode_program_record_header(
    const loom_xdna_elf_program_record_header_t* header,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(loom_xdna_elf_require_storage(
      storage, LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE,
      IREE_SV("program record header")));
  if (header->type == 0 ||
      header->byte_length < LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE ||
      header->byte_length > LOOM_XDNA_ELF_MAX_PROGRAM_RECORD_SIZE ||
      header->byte_length % LOOM_XDNA_ELF_PROGRAM_RECORD_ALIGNMENT != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program record has a noncanonical shape");
  }
  iree_unaligned_store_le_u16(storage.data + 0, header->type);
  iree_unaligned_store_le_u16(storage.data + 2, header->flags);
  iree_unaligned_store_le_u32(storage.data + 4, header->byte_length);
  return iree_ok_status();
}
