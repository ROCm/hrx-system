// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/format.h"

#include <inttypes.h>
#include <string.h>

static iree_status_t iree_hal_amd_xdna_elf_require_const_storage(
    iree_const_byte_span_t storage, iree_host_size_t required_size,
    iree_string_view_t record_name) {
  if (storage.data_length != required_size ||
      (required_size != 0 && storage.data == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA %.*s storage must contain exactly %" PRIhsz " bytes",
        (int)record_name.size, record_name.data, required_size);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_elf_require_mutable_storage(
    iree_byte_span_t storage, iree_host_size_t required_size,
    iree_string_view_t record_name) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      iree_make_const_byte_span(storage.data, storage.data_length),
      required_size, record_name));
  memset(storage.data, 0, storage.data_length);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_pack_tile_destination(
    const iree_hal_amd_xdna_elf_tile_destination_t* destination,
    uint32_t* out_physical_address) {
  IREE_ASSERT_ARGUMENT(destination);
  IREE_ASSERT_ARGUMENT(out_physical_address);
  *out_physical_address = 0;
  if (destination->memory_space !=
          IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM &&
      destination->memory_space !=
          IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown XDNA tile memory space %u",
                            (unsigned)destination->memory_space);
  }
  if (destination->flags != IREE_HAL_AMD_XDNA_ELF_TILE_KNOWN_FLAGS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown XDNA tile destination flags 0x%02X",
                            destination->flags);
  }
  *out_physical_address =
      ((uint32_t)destination->column
       << IREE_HAL_AMD_XDNA_ELF_TILE_COLUMN_SHIFT) |
      ((uint32_t)destination->row << IREE_HAL_AMD_XDNA_ELF_TILE_ROW_SHIFT) |
      ((uint32_t)destination->memory_space
       << IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_SHIFT) |
      ((uint32_t)destination->flags << IREE_HAL_AMD_XDNA_ELF_TILE_FLAGS_SHIFT);
  return iree_ok_status();
}

iree_hal_amd_xdna_elf_tile_destination_t
iree_hal_amd_xdna_elf_unpack_tile_destination(uint32_t physical_address) {
  return (iree_hal_amd_xdna_elf_tile_destination_t){
      .column = (uint8_t)((physical_address &
                           IREE_HAL_AMD_XDNA_ELF_TILE_COLUMN_MASK) >>
                          IREE_HAL_AMD_XDNA_ELF_TILE_COLUMN_SHIFT),
      .row =
          (uint8_t)((physical_address & IREE_HAL_AMD_XDNA_ELF_TILE_ROW_MASK) >>
                    IREE_HAL_AMD_XDNA_ELF_TILE_ROW_SHIFT),
      .memory_space =
          (iree_hal_amd_xdna_elf_tile_memory_space_t)((physical_address &
                                                       IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_MASK) >>
                                                      IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_SHIFT),
      .flags = (uint8_t)((physical_address &
                          IREE_HAL_AMD_XDNA_ELF_TILE_FLAGS_MASK) >>
                         IREE_HAL_AMD_XDNA_ELF_TILE_FLAGS_SHIFT),
  };
}

static iree_status_t iree_hal_amd_xdna_elf_validate_abi_note_for_encoding(
    const iree_hal_amd_xdna_elf_abi_note_t* note) {
  if (note->abi_major != IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR ||
      note->abi_minor != IREE_HAL_AMD_XDNA_ELF_ABI_MINOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA image ABI version %u.%u",
                            note->abi_major, note->abi_minor);
  }
  if (note->target_generation != IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA target generation %u",
                            (unsigned)note->target_generation);
  }
  if (note->device_profile_revision == 0 || note->device_profile_id == 0 ||
      note->firmware_abi_id == 0 || note->policy_id == 0 ||
      note->context_column_count == 0 || note->context_row_count == 0 ||
      note->context_column_count > UINT8_MAX + 1u ||
      note->context_row_count > UINT8_MAX + 1u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ABI note identity is incomplete");
  }
  if (note->coordinate_model !=
          IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE ||
      note->context_origin_column != 0 || note->context_origin_row != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image coordinates are not context-relative");
  }
  if ((note->required_capabilities &
       ~IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image requires unknown capabilities");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_abi_note(
    const iree_hal_amd_xdna_elf_abi_note_t* note, iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(note);
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_elf_validate_abi_note_for_encoding(note));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE, IREE_SV("ABI note")));

  iree_unaligned_store_le_u32(storage.data + 0,
                              IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE);
  iree_unaligned_store_le_u32(storage.data + 4,
                              IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE);
  iree_unaligned_store_le_u32(storage.data + 8,
                              IREE_HAL_AMD_XDNA_ELF_NOTE_TYPE_ABI);
  memcpy(storage.data + 12, IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER,
         IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE);
  uint8_t* description = storage.data + 20;
  iree_unaligned_store_le_u32(description + 0,
                              IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_MAGIC);
  iree_unaligned_store_le_u16(description + 4,
                              IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE);
  iree_unaligned_store_le_u16(description + 6, note->abi_major);
  iree_unaligned_store_le_u16(description + 8, note->abi_minor);
  iree_unaligned_store_le_u16(description + 10,
                              (uint16_t)note->target_generation);
  iree_unaligned_store_le_u32(description + 12, note->device_profile_revision);
  iree_unaligned_store_le_u64(description + 16, note->device_profile_id);
  iree_unaligned_store_le_u64(description + 24, note->firmware_abi_id);
  iree_unaligned_store_le_u64(description + 32, note->policy_id);
  iree_unaligned_store_le_u64(description + 40, note->required_capabilities);
  iree_unaligned_store_le_u16(description + 48, note->context_origin_column);
  iree_unaligned_store_le_u16(description + 50, note->context_origin_row);
  iree_unaligned_store_le_u16(description + 52, note->context_column_count);
  iree_unaligned_store_le_u16(description + 54, note->context_row_count);
  description[56] = (uint8_t)note->coordinate_model;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_decode_abi_note(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_abi_note_t* out_note) {
  IREE_ASSERT_ARGUMENT(out_note);
  *out_note = (iree_hal_amd_xdna_elf_abi_note_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE, IREE_SV("ABI note")));
  if (iree_unaligned_load_le_u32(storage.data + 0) !=
          IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE ||
      iree_unaligned_load_le_u32(storage.data + 4) !=
          IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE ||
      iree_unaligned_load_le_u32(storage.data + 8) !=
          IREE_HAL_AMD_XDNA_ELF_NOTE_TYPE_ABI ||
      memcmp(storage.data + 12, IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER,
             IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ABI note envelope is invalid");
  }
  for (iree_host_size_t i = 17; i < 20; ++i) {
    if (storage.data[i] != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA image ABI note name padding is not zero");
    }
  }
  const uint8_t* description = storage.data + 20;
  if (iree_unaligned_load_le_u32(description + 0) !=
          IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_MAGIC ||
      iree_unaligned_load_le_u16(description + 4) !=
          IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ABI note description is invalid");
  }
  for (iree_host_size_t i = 57;
       i < IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE; ++i) {
    if (description[i] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA image ABI note description padding is not zero");
    }
  }

  *out_note = (iree_hal_amd_xdna_elf_abi_note_t){
      .abi_major = iree_unaligned_load_le_u16(description + 6),
      .abi_minor = iree_unaligned_load_le_u16(description + 8),
      .target_generation =
          (iree_hal_amd_xdna_target_generation_t)iree_unaligned_load_le_u16(
              description + 10),
      .device_profile_revision = iree_unaligned_load_le_u32(description + 12),
      .device_profile_id = iree_unaligned_load_le_u64(description + 16),
      .firmware_abi_id = iree_unaligned_load_le_u64(description + 24),
      .policy_id = iree_unaligned_load_le_u64(description + 32),
      .required_capabilities = iree_unaligned_load_le_u64(description + 40),
      .context_origin_column = iree_unaligned_load_le_u16(description + 48),
      .context_origin_row = iree_unaligned_load_le_u16(description + 50),
      .context_column_count = iree_unaligned_load_le_u16(description + 52),
      .context_row_count = iree_unaligned_load_le_u16(description + 54),
      .coordinate_model =
          (iree_hal_amd_xdna_elf_coordinate_model_t)description[56],
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_table_header(
    const iree_hal_amd_xdna_elf_table_header_t* header,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      IREE_SV("table header")));
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

iree_status_t iree_hal_amd_xdna_elf_decode_table_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_table_header_t* out_header) {
  IREE_ASSERT_ARGUMENT(out_header);
  *out_header = (iree_hal_amd_xdna_elf_table_header_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
      IREE_SV("table header")));
  *out_header = (iree_hal_amd_xdna_elf_table_header_t){
      .magic = iree_unaligned_load_le_u32(storage.data + 0),
      .abi_major = iree_unaligned_load_le_u16(storage.data + 4),
      .abi_minor = iree_unaligned_load_le_u16(storage.data + 6),
      .header_size = iree_unaligned_load_le_u16(storage.data + 8),
      .record_size = iree_unaligned_load_le_u16(storage.data + 10),
      .record_count = iree_unaligned_load_le_u32(storage.data + 12),
      .byte_length = iree_unaligned_load_le_u32(storage.data + 16),
      .auxiliary_offset = iree_unaligned_load_le_u32(storage.data + 20),
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_entry_record(
    const iree_hal_amd_xdna_elf_entry_record_t* record,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
      IREE_SV("entry record")));
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

iree_status_t iree_hal_amd_xdna_elf_decode_entry_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_entry_record_t* out_record) {
  IREE_ASSERT_ARGUMENT(out_record);
  *out_record = (iree_hal_amd_xdna_elf_entry_record_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
      IREE_SV("entry record")));
  *out_record = (iree_hal_amd_xdna_elf_entry_record_t){
      .export_ordinal = iree_unaligned_load_le_u32(storage.data + 0),
      .name_offset = iree_unaligned_load_le_u32(storage.data + 4),
      .name_length = iree_unaligned_load_le_u32(storage.data + 8),
      .array_program_header_ordinal =
          iree_unaligned_load_le_u32(storage.data + 12),
      .control_program_header_ordinal =
          iree_unaligned_load_le_u32(storage.data + 16),
      .first_binding_ordinal = iree_unaligned_load_le_u32(storage.data + 20),
      .binding_count = iree_unaligned_load_le_u32(storage.data + 24),
      .flags = iree_unaligned_load_le_u32(storage.data + 28),
      .required_capabilities = iree_unaligned_load_le_u64(storage.data + 32),
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_binding_record(
    const iree_hal_amd_xdna_elf_binding_record_t* record,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE,
      IREE_SV("binding record")));
  iree_unaligned_store_le_u32(storage.data + 0, record->binding_ordinal);
  iree_unaligned_store_le_u32(storage.data + 4, record->entry_ordinal);
  iree_unaligned_store_le_u16(storage.data + 8, (uint16_t)record->kind);
  iree_unaligned_store_le_u16(storage.data + 10,
                              (uint16_t)record->address_space);
  iree_unaligned_store_le_u32(storage.data + 12, record->access);
  iree_unaligned_store_le_u32(storage.data + 16, record->usage);
  iree_unaligned_store_le_u64(storage.data + 24, record->minimum_byte_length);
  iree_unaligned_store_le_u64(storage.data + 32, record->minimum_alignment);
  iree_unaligned_store_le_u64(storage.data + 40, record->minimum_byte_offset);
  iree_unaligned_store_le_u64(storage.data + 48, record->maximum_byte_offset);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_decode_binding_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_binding_record_t* out_record) {
  IREE_ASSERT_ARGUMENT(out_record);
  *out_record = (iree_hal_amd_xdna_elf_binding_record_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE,
      IREE_SV("binding record")));
  if (iree_unaligned_load_le_u32(storage.data + 20) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA binding record reserved bits are set");
  }
  *out_record = (iree_hal_amd_xdna_elf_binding_record_t){
      .binding_ordinal = iree_unaligned_load_le_u32(storage.data + 0),
      .entry_ordinal = iree_unaligned_load_le_u32(storage.data + 4),
      .kind = (iree_hal_amd_xdna_elf_binding_kind_t)iree_unaligned_load_le_u16(
          storage.data + 8),
      .address_space = (iree_hal_amd_xdna_elf_binding_address_space_t)
          iree_unaligned_load_le_u16(storage.data + 10),
      .access = iree_unaligned_load_le_u32(storage.data + 12),
      .usage = iree_unaligned_load_le_u32(storage.data + 16),
      .minimum_byte_length = iree_unaligned_load_le_u64(storage.data + 24),
      .minimum_alignment = iree_unaligned_load_le_u64(storage.data + 32),
      .minimum_byte_offset = iree_unaligned_load_le_u64(storage.data + 40),
      .maximum_byte_offset = iree_unaligned_load_le_u64(storage.data + 48),
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_relocation_record(
    const iree_hal_amd_xdna_elf_relocation_record_t* record,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE,
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

iree_status_t iree_hal_amd_xdna_elf_decode_relocation_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_relocation_record_t* out_record) {
  IREE_ASSERT_ARGUMENT(out_record);
  *out_record = (iree_hal_amd_xdna_elf_relocation_record_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE,
      IREE_SV("relocation record")));
  *out_record = (iree_hal_amd_xdna_elf_relocation_record_t){
      .target_program_header_ordinal =
          iree_unaligned_load_le_u32(storage.data + 0),
      .target_byte_offset = iree_unaligned_load_le_u32(storage.data + 4),
      .binding_ordinal = iree_unaligned_load_le_u32(storage.data + 8),
      .kind =
          (iree_hal_amd_xdna_elf_relocation_kind_t)iree_unaligned_load_le_u16(
              storage.data + 12),
      .field_byte_width = storage.data[14],
      .flags = storage.data[15],
      .addend = (int64_t)iree_unaligned_load_le_u64(storage.data + 16),
      .minimum_value = iree_unaligned_load_le_u64(storage.data + 24),
      .maximum_value = iree_unaligned_load_le_u64(storage.data + 32),
      .required_alignment = iree_unaligned_load_le_u64(storage.data + 40),
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_array_header(
    const iree_hal_amd_xdna_elf_array_header_t* header,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE,
      IREE_SV("ARRAY header")));
  iree_unaligned_store_le_u32(storage.data + 0,
                              IREE_HAL_AMD_XDNA_ELF_ARRAY_MAGIC);
  iree_unaligned_store_le_u16(storage.data + 4, header->abi_major);
  iree_unaligned_store_le_u16(storage.data + 6, header->abi_minor);
  iree_unaligned_store_le_u16(storage.data + 8,
                              IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE);
  iree_unaligned_store_le_u16(storage.data + 10,
                              IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE);
  iree_unaligned_store_le_u32(storage.data + 12, header->record_count);
  iree_unaligned_store_le_u32(storage.data + 16, header->byte_length);
  iree_unaligned_store_le_u32(storage.data + 20, header->flags);
  iree_unaligned_store_le_u32(storage.data + 24,
                              header->first_tile_program_header_ordinal);
  iree_unaligned_store_le_u32(storage.data + 28,
                              header->tile_program_header_count);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_decode_array_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_array_header_t* out_header) {
  IREE_ASSERT_ARGUMENT(out_header);
  *out_header = (iree_hal_amd_xdna_elf_array_header_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE,
      IREE_SV("ARRAY header")));
  if (iree_unaligned_load_le_u32(storage.data + 0) !=
      IREE_HAL_AMD_XDNA_ELF_ARRAY_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA ARRAY header magic is invalid");
  }
  *out_header = (iree_hal_amd_xdna_elf_array_header_t){
      .abi_major = iree_unaligned_load_le_u16(storage.data + 4),
      .abi_minor = iree_unaligned_load_le_u16(storage.data + 6),
      .record_count = iree_unaligned_load_le_u32(storage.data + 12),
      .byte_length = iree_unaligned_load_le_u32(storage.data + 16),
      .flags = iree_unaligned_load_le_u32(storage.data + 20),
      .first_tile_program_header_ordinal =
          iree_unaligned_load_le_u32(storage.data + 24),
      .tile_program_header_count =
          iree_unaligned_load_le_u32(storage.data + 28),
  };
  if (iree_unaligned_load_le_u16(storage.data + 8) !=
          IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE ||
      iree_unaligned_load_le_u16(storage.data + 10) !=
          IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE) {
    *out_header = (iree_hal_amd_xdna_elf_array_header_t){0};
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA ARRAY header framing is invalid");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_control_header(
    const iree_hal_amd_xdna_elf_control_header_t* header,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE,
      IREE_SV("CONTROL header")));
  iree_unaligned_store_le_u32(storage.data + 0,
                              IREE_HAL_AMD_XDNA_ELF_CONTROL_MAGIC);
  iree_unaligned_store_le_u16(storage.data + 4, header->abi_major);
  iree_unaligned_store_le_u16(storage.data + 6, header->abi_minor);
  iree_unaligned_store_le_u16(storage.data + 8,
                              IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE);
  iree_unaligned_store_le_u16(storage.data + 10,
                              IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE);
  iree_unaligned_store_le_u32(storage.data + 12, header->record_count);
  iree_unaligned_store_le_u32(storage.data + 16, header->byte_length);
  iree_unaligned_store_le_u32(storage.data + 20, header->flags);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_decode_control_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_control_header_t* out_header) {
  IREE_ASSERT_ARGUMENT(out_header);
  *out_header = (iree_hal_amd_xdna_elf_control_header_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE,
      IREE_SV("CONTROL header")));
  if (iree_unaligned_load_le_u32(storage.data + 0) !=
      IREE_HAL_AMD_XDNA_ELF_CONTROL_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA CONTROL header magic is invalid");
  }
  *out_header = (iree_hal_amd_xdna_elf_control_header_t){
      .abi_major = iree_unaligned_load_le_u16(storage.data + 4),
      .abi_minor = iree_unaligned_load_le_u16(storage.data + 6),
      .record_count = iree_unaligned_load_le_u32(storage.data + 12),
      .byte_length = iree_unaligned_load_le_u32(storage.data + 16),
      .flags = iree_unaligned_load_le_u32(storage.data + 20),
  };
  if (iree_unaligned_load_le_u16(storage.data + 8) !=
          IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE ||
      iree_unaligned_load_le_u16(storage.data + 10) !=
          IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE) {
    *out_header = (iree_hal_amd_xdna_elf_control_header_t){0};
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA CONTROL header framing is invalid");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_elf_validate_program_record_header(
    const iree_hal_amd_xdna_elf_program_record_header_t* header) {
  if (header->type == 0 ||
      header->byte_length < IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE ||
      header->byte_length > IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_RECORD_SIZE ||
      header->byte_length % IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_ALIGNMENT !=
          0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program record has noncanonical framing");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_encode_program_record_header(
    const iree_hal_amd_xdna_elf_program_record_header_t* header,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_elf_validate_program_record_header(header));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE,
      IREE_SV("program record header")));
  iree_unaligned_store_le_u16(storage.data + 0, header->type);
  iree_unaligned_store_le_u16(storage.data + 2, header->flags);
  iree_unaligned_store_le_u32(storage.data + 4, header->byte_length);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_elf_decode_program_record_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_program_record_header_t* out_header) {
  IREE_ASSERT_ARGUMENT(out_header);
  *out_header = (iree_hal_amd_xdna_elf_program_record_header_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_require_const_storage(
      storage, IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE,
      IREE_SV("program record header")));
  const iree_hal_amd_xdna_elf_program_record_header_t header = {
      .type = iree_unaligned_load_le_u16(storage.data + 0),
      .flags = iree_unaligned_load_le_u16(storage.data + 2),
      .byte_length = iree_unaligned_load_le_u32(storage.data + 4),
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_elf_validate_program_record_header(&header));
  *out_header = header;
  return iree_ok_status();
}
