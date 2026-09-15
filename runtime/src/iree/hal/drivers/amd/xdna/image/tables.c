// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/tables.h"

#include <inttypes.h>
#include <string.h>

struct iree_hal_amd_xdna_image_tables_t {
  // Allocator owning this object and all trailing storage.
  iree_allocator_t host_allocator;
  // Decoded image ABI note.
  iree_hal_amd_xdna_elf_abi_note_t abi_note;
  // Number of entries in |entries|.
  iree_host_size_t entry_count;
  // Decoded executable entries.
  iree_hal_amd_xdna_elf_entry_record_t* entries;
  // Byte offset of the first name in the serialized entry table.
  uint32_t entry_name_table_offset;
  // Number of bytes in |entry_name_storage|.
  iree_host_size_t entry_name_storage_length;
  // Canonically packed diagnostic entry names.
  uint8_t* entry_name_storage;
  // Number of entries in |bindings|.
  iree_host_size_t binding_count;
  // Decoded runtime bindings.
  iree_hal_amd_xdna_elf_binding_record_t* bindings;
  // Number of entries in |relocations|.
  iree_host_size_t relocation_count;
  // Decoded runtime relocations.
  iree_hal_amd_xdna_elf_relocation_record_t* relocations;
};

typedef struct iree_hal_amd_xdna_image_table_programs_t {
  // Required image ABI note program.
  const iree_hal_amd_xdna_image_program_header_t* abi_note;
  // Required executable entry table program.
  const iree_hal_amd_xdna_image_program_header_t* entries;
  // Required runtime binding table program.
  const iree_hal_amd_xdna_image_program_header_t* bindings;
  // Optional runtime relocation table program.
  const iree_hal_amd_xdna_image_program_header_t* relocations;
} iree_hal_amd_xdna_image_table_programs_t;

typedef struct iree_hal_amd_xdna_image_table_descriptor_t {
  // Program header containing the serialized table.
  const iree_hal_amd_xdna_image_program_header_t* program_header;
  // Decoded fixed table header.
  iree_hal_amd_xdna_elf_table_header_t header;
  // One-past-the-end byte offset of the fixed records.
  uint32_t record_end_offset;
} iree_hal_amd_xdna_image_table_descriptor_t;

static iree_status_t iree_hal_amd_xdna_image_validate_metadata_program(
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    uint32_t required_alignment, uint64_t maximum_file_length) {
  if (program_header->flags != IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ ||
      program_header->virtual_address != 0 ||
      program_header->physical_address != 0 ||
      program_header->file_range.length == 0 ||
      program_header->memory_size != program_header->file_range.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA metadata program type 0x%08" PRIX32
                            " has an invalid memory contract",
                            program_header->type);
  }
  if (program_header->alignment != required_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA metadata program type 0x%08" PRIX32
                            " requires alignment %" PRIu32,
                            program_header->type, required_alignment);
  }
  if (program_header->file_range.length > maximum_file_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA metadata program type 0x%08" PRIX32
                            " exceeds its maximum byte length",
                            program_header->type);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_scan_table_programs(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_table_programs_t* out_programs) {
  *out_programs = (iree_hal_amd_xdna_image_table_programs_t){0};
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  for (iree_host_size_t i = 0; i < program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    const iree_hal_amd_xdna_image_program_header_t** program_slot = NULL;
    uint32_t required_alignment = 0;
    uint64_t maximum_file_length = 0;
    switch (program_header->type) {
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE:
        program_slot = &out_programs->abi_note;
        required_alignment = 4;
        maximum_file_length = IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES:
        program_slot = &out_programs->entries;
        required_alignment = 8;
        maximum_file_length = IREE_HAL_AMD_XDNA_ELF_MAX_METADATA_TABLE_SIZE;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS:
        program_slot = &out_programs->bindings;
        required_alignment = 8;
        maximum_file_length = IREE_HAL_AMD_XDNA_ELF_MAX_METADATA_TABLE_SIZE;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS:
        program_slot = &out_programs->relocations;
        required_alignment = 8;
        maximum_file_length = IREE_HAL_AMD_XDNA_ELF_MAX_METADATA_TABLE_SIZE;
        break;
      default:
        continue;
    }
    if (*program_slot != NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA image has more than one metadata program of type 0x%08" PRIX32,
          program_header->type);
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_metadata_program(
        program_header, required_alignment, maximum_file_length));
    *program_slot = program_header;
  }
  if (out_programs->abi_note == NULL || out_programs->entries == NULL ||
      out_programs->bindings == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image is missing a required metadata table");
  }
  if (out_programs->abi_note->file_range.length !=
      IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ABI note has an invalid byte length");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_read_program(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    iree_allocator_t host_allocator, iree_byte_span_t* out_storage) {
  *out_storage = iree_byte_span_empty();
  out_storage->data_length =
      (iree_host_size_t)program_header->file_range.length;
  iree_status_t status = iree_allocator_malloc_uninitialized(
      host_allocator, out_storage->data_length, (void**)&out_storage->data);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_directory_read_source_range(
        directory, program_header->file_range, *out_storage);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, out_storage->data);
    *out_storage = iree_byte_span_empty();
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_image_decode_abi_note(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    iree_hal_amd_xdna_elf_abi_note_t* out_note) {
  uint8_t storage[IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE];
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_directory_read_source_range(
      directory, program_header->file_range,
      iree_make_byte_span(storage, sizeof(storage))));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_decode_abi_note(
      iree_make_const_byte_span(storage, sizeof(storage)), out_note));
  if (out_note->abi_major != IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR ||
      out_note->abi_minor > IREE_HAL_AMD_XDNA_ELF_ABI_MINOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported XDNA image ABI version %u.%u",
                            out_note->abi_major, out_note->abi_minor);
  }
  if (out_note->target_generation == 0 ||
      out_note->device_profile_revision == 0 ||
      out_note->device_profile_id == 0 || out_note->firmware_abi_id == 0 ||
      out_note->policy_id == 0 || out_note->context_column_count == 0 ||
      out_note->context_row_count == 0 ||
      out_note->context_column_count > UINT8_MAX + 1u ||
      out_note->context_row_count > UINT8_MAX + 1u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ABI note identity is incomplete");
  }
  if (out_note->coordinate_model !=
          IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE ||
      out_note->context_origin_column != 0 ||
      out_note->context_origin_row != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image coordinates are not context-relative");
  }
  if ((out_note->required_capabilities &
       ~IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA image requires unknown capabilities");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_decode_table_descriptor(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    uint32_t expected_magic, uint16_t expected_record_size,
    iree_hal_amd_xdna_image_table_descriptor_t* out_descriptor) {
  *out_descriptor = (iree_hal_amd_xdna_image_table_descriptor_t){0};
  if (program_header->file_range.length <
      IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA fixed metadata table is truncated");
  }
  uint8_t storage[IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE];
  const iree_hal_amd_xdna_image_source_range_t header_range = {
      .offset = program_header->file_range.offset,
      .length = IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
  };
  iree_hal_amd_xdna_elf_table_header_t header;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_directory_read_source_range(
      directory, header_range, iree_make_byte_span(storage, sizeof(storage))));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_decode_table_header(
      iree_make_const_byte_span(storage, sizeof(storage)), &header));
  if (header.magic != expected_magic ||
      header.header_size != IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE ||
      header.record_size != expected_record_size ||
      header.record_count > IREE_HAL_AMD_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
      header.byte_length != program_header->file_range.length ||
      header.byte_length > IREE_HAL_AMD_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA fixed metadata table header is invalid");
  }
  if (header.abi_major != IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR ||
      header.abi_minor > IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported XDNA metadata table ABI version %u.%u",
                            header.abi_major, header.abi_minor);
  }
  uint64_t record_end_offset = 0;
  if (!iree_checked_add_u64(IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
                            (uint64_t)header.record_count * header.record_size,
                            &record_end_offset) ||
      record_end_offset > header.byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA fixed metadata records exceed their table");
  }
  *out_descriptor = (iree_hal_amd_xdna_image_table_descriptor_t){
      .program_header = program_header,
      .header = header,
      .record_end_offset = (uint32_t)record_end_offset,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_decode_entries(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_table_descriptor_t* descriptor,
    iree_hal_amd_xdna_image_tables_t* tables) {
  iree_byte_span_t payload = iree_byte_span_empty();
  iree_status_t status = iree_hal_amd_xdna_image_read_program(
      directory, descriptor->program_header, tables->host_allocator, &payload);
  iree_host_size_t expected_name_offset = descriptor->header.auxiliary_offset;
  iree_host_size_t expected_binding_ordinal = 0;
  iree_host_size_t default_entry_count = 0;
  for (iree_host_size_t i = 0;
       i < tables->entry_count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_xdna_elf_entry_record_t* entry = &tables->entries[i];
    status = iree_hal_amd_xdna_elf_decode_entry_record(
        iree_make_const_byte_span(
            payload.data + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                i * IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE,
            IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE),
        entry);
    if (!iree_status_is_ok(status)) break;
    if (entry->export_ordinal != i ||
        (entry->flags & ~IREE_HAL_AMD_XDNA_ELF_KNOWN_ENTRY_FLAGS) != 0 ||
        (entry->required_capabilities &
         ~tables->abi_note.required_capabilities) != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "XDNA entry %" PRIhsz
                                " has invalid identity, flags, or capabilities",
                                i);
      break;
    }
    if ((entry->flags & IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT) != 0) {
      ++default_entry_count;
    }
    if (entry->name_length == 0) {
      if (entry->name_offset != 0) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unnamed XDNA entry has a name offset");
        break;
      }
    } else {
      uint64_t name_end = 0;
      if (entry->name_length > IREE_HAL_AMD_XDNA_ELF_MAX_ENTRY_NAME_LENGTH ||
          entry->name_offset != expected_name_offset ||
          !iree_checked_add_u64(entry->name_offset, entry->name_length,
                                &name_end) ||
          name_end > payload.data_length ||
          memchr(payload.data + entry->name_offset, 0, entry->name_length) !=
              NULL) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA entry diagnostic names are not canonically packed");
        break;
      }
      expected_name_offset = (iree_host_size_t)name_end;
    }
    uint64_t binding_end = 0;
    if (entry->first_binding_ordinal != expected_binding_ordinal ||
        !iree_checked_add_u64(entry->first_binding_ordinal,
                              entry->binding_count, &binding_end) ||
        binding_end > tables->binding_count) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA entry binding ranges are not canonically packed");
      break;
    }
    expected_binding_ordinal = (iree_host_size_t)binding_end;
  }
  if (iree_status_is_ok(status) &&
      (expected_name_offset != payload.data_length ||
       expected_binding_ordinal != tables->binding_count ||
       default_entry_count > 1)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA entry table has trailing names, unowned bindings, or multiple "
        "defaults");
  }
  if (iree_status_is_ok(status) && tables->entry_name_storage_length != 0) {
    memcpy(tables->entry_name_storage,
           payload.data + descriptor->header.auxiliary_offset,
           tables->entry_name_storage_length);
  }
  iree_allocator_free(tables->host_allocator, payload.data);
  return status;
}

static iree_status_t iree_hal_amd_xdna_image_decode_bindings(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_table_descriptor_t* descriptor,
    iree_hal_amd_xdna_image_tables_t* tables) {
  iree_byte_span_t payload = iree_byte_span_empty();
  iree_status_t status = iree_hal_amd_xdna_image_read_program(
      directory, descriptor->program_header, tables->host_allocator, &payload);
  for (iree_host_size_t i = 0;
       i < tables->binding_count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_xdna_elf_binding_record_t* binding = &tables->bindings[i];
    status = iree_hal_amd_xdna_elf_decode_binding_record(
        iree_make_const_byte_span(
            payload.data + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                i * IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE,
            IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE),
        binding);
    if (!iree_status_is_ok(status)) break;
    if (binding->binding_ordinal != i ||
        binding->entry_ordinal >= tables->entry_count || binding->access == 0 ||
        (binding->access & ~IREE_HAL_AMD_XDNA_ELF_KNOWN_BINDING_ACCESS) != 0 ||
        (binding->usage & ~IREE_HAL_AMD_XDNA_ELF_KNOWN_BINDING_USAGE) != 0 ||
        binding->minimum_alignment == 0 ||
        !iree_is_power_of_two_uint64(binding->minimum_alignment) ||
        binding->minimum_byte_offset > binding->maximum_byte_offset) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "XDNA binding %" PRIhsz " is malformed", i);
      break;
    }
    // The contract needs one representable range, not storage covering every
    // permitted offset. The caller checks its actual offset and length when
    // binding a resource.
    if (binding->minimum_byte_length >
        UINT64_MAX - binding->minimum_byte_offset) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "XDNA binding %" PRIhsz " range overflows", i);
      break;
    }
    switch (binding->kind) {
      case IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER:
        if ((binding->address_space !=
                 IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL &&
             binding->address_space !=
                 IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST) ||
            binding->minimum_byte_length == 0 ||
            (binding->usage &
             IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE) == 0) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "XDNA buffer binding %" PRIhsz
                                    " has an invalid resource contract",
                                    i);
        }
        break;
      case IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_SCALAR:
        if (binding->address_space !=
                IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_NONE ||
            binding->usage != 0 ||
            binding->access != IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ ||
            (binding->minimum_byte_length != 1 &&
             binding->minimum_byte_length != 2 &&
             binding->minimum_byte_length != 4 &&
             binding->minimum_byte_length != 8) ||
            binding->minimum_alignment != binding->minimum_byte_length ||
            binding->minimum_byte_offset != 0 ||
            binding->maximum_byte_offset != 0) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "XDNA scalar binding %" PRIhsz
                                    " has an invalid value contract",
                                    i);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "XDNA binding %" PRIhsz
                                  " has unknown resource kind %u",
                                  i, (unsigned)binding->kind);
        break;
    }
  }
  iree_allocator_free(tables->host_allocator, payload.data);
  return status;
}

static iree_status_t iree_hal_amd_xdna_image_validate_binding_ownership(
    const iree_hal_amd_xdna_image_tables_t* tables) {
  for (iree_host_size_t i = 0; i < tables->entry_count; ++i) {
    const iree_hal_amd_xdna_elf_entry_record_t* entry = &tables->entries[i];
    const iree_host_size_t binding_end =
        entry->first_binding_ordinal + entry->binding_count;
    for (iree_host_size_t j = entry->first_binding_ordinal; j < binding_end;
         ++j) {
      if (tables->bindings[j].entry_ordinal != i) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA binding ownership disagrees with its entry range");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_decode_relocations(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_table_descriptor_t* descriptor,
    iree_hal_amd_xdna_image_tables_t* tables) {
  if (descriptor->program_header == NULL) return iree_ok_status();
  iree_byte_span_t payload = iree_byte_span_empty();
  iree_status_t status = iree_hal_amd_xdna_image_read_program(
      directory, descriptor->program_header, tables->host_allocator, &payload);
  uint32_t previous_target_ordinal = 0;
  uint64_t previous_target_end = 0;
  for (iree_host_size_t i = 0;
       i < tables->relocation_count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_xdna_elf_relocation_record_t* relocation =
        &tables->relocations[i];
    status = iree_hal_amd_xdna_elf_decode_relocation_record(
        iree_make_const_byte_span(
            payload.data + IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                i * IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE,
            IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE),
        relocation);
    if (!iree_status_is_ok(status)) break;
    uint64_t target_end = 0;
    if (relocation->binding_ordinal >= tables->binding_count ||
        (relocation->field_byte_width != 4 &&
         relocation->field_byte_width != 8) ||
        relocation->flags != 0 || relocation->required_alignment == 0 ||
        !iree_is_power_of_two_uint64(relocation->required_alignment) ||
        relocation->minimum_value > relocation->maximum_value ||
        (relocation->field_byte_width == 4 &&
         relocation->maximum_value > UINT32_MAX) ||
        !iree_checked_add_u64(relocation->target_byte_offset,
                              relocation->field_byte_width, &target_end)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "XDNA runtime relocation %" PRIhsz
                                " has an invalid field contract",
                                i);
      break;
    }
    const iree_hal_amd_xdna_elf_binding_record_t* binding =
        &tables->bindings[relocation->binding_ordinal];
    switch (relocation->kind) {
      case IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS:
      case IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_BYTE_LENGTH:
      case IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_BYTE_OFFSET:
        if (binding->kind != IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "XDNA runtime relocation %" PRIhsz
                                    " requires a buffer binding",
                                    i);
        }
        break;
      case IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_SCALAR_VALUE:
        if (binding->kind != IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_SCALAR ||
            binding->minimum_byte_length != relocation->field_byte_width) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "XDNA runtime relocation %" PRIhsz
                                    " requires a matching scalar binding",
                                    i);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "XDNA runtime relocation %" PRIhsz
                                  " has unknown kind %u",
                                  i, (unsigned)relocation->kind);
        break;
    }
    if (!iree_status_is_ok(status)) break;
    if (i != 0 &&
        (relocation->target_program_header_ordinal < previous_target_ordinal ||
         (relocation->target_program_header_ordinal ==
              previous_target_ordinal &&
          relocation->target_byte_offset < previous_target_end))) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA runtime relocations overlap or are not canonically ordered");
      break;
    }
    previous_target_ordinal = relocation->target_program_header_ordinal;
    previous_target_end = target_end;
  }
  iree_allocator_free(tables->host_allocator, payload.data);
  return status;
}

iree_status_t iree_hal_amd_xdna_image_tables_create(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_tables_t** out_tables) {
  IREE_ASSERT_ARGUMENT(out_tables);
  *out_tables = NULL;
  if (directory == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image directory is required");
  }

  iree_hal_amd_xdna_image_table_programs_t programs;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_image_scan_table_programs(directory, &programs));
  iree_hal_amd_xdna_elf_abi_note_t abi_note;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_decode_abi_note(
      directory, programs.abi_note, &abi_note));

  iree_hal_amd_xdna_image_table_descriptor_t entry_descriptor;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_decode_table_descriptor(
      directory, programs.entries, IREE_HAL_AMD_XDNA_ELF_ENTRY_TABLE_MAGIC,
      IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE, &entry_descriptor));
  if (entry_descriptor.header.record_count == 0 ||
      entry_descriptor.header.auxiliary_offset !=
          entry_descriptor.record_end_offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA entry table has a noncanonical name range");
  }

  iree_hal_amd_xdna_image_table_descriptor_t binding_descriptor;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_decode_table_descriptor(
      directory, programs.bindings, IREE_HAL_AMD_XDNA_ELF_BINDING_TABLE_MAGIC,
      IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE, &binding_descriptor));
  if (binding_descriptor.header.auxiliary_offset != 0 ||
      binding_descriptor.header.byte_length !=
          binding_descriptor.record_end_offset) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA binding table has trailing or auxiliary bytes");
  }

  iree_hal_amd_xdna_image_table_descriptor_t relocation_descriptor = {0};
  if (programs.relocations != NULL) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_decode_table_descriptor(
        directory, programs.relocations,
        IREE_HAL_AMD_XDNA_ELF_RELOCATION_TABLE_MAGIC,
        IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE, &relocation_descriptor));
    if (relocation_descriptor.header.record_count == 0 ||
        relocation_descriptor.header.auxiliary_offset != 0 ||
        relocation_descriptor.header.byte_length !=
            relocation_descriptor.record_end_offset) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA relocation table is empty or has trailing bytes");
    }
  }

  const iree_host_size_t entry_count = entry_descriptor.header.record_count;
  const iree_host_size_t entry_name_storage_length =
      entry_descriptor.header.byte_length -
      entry_descriptor.header.auxiliary_offset;
  const iree_host_size_t binding_count = binding_descriptor.header.record_count;
  const iree_host_size_t relocation_count =
      relocation_descriptor.header.record_count;
  iree_host_size_t total_size = 0;
  iree_host_size_t entries_offset = 0;
  iree_host_size_t entry_name_storage_offset = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t relocations_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_image_tables_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          entry_count, iree_hal_amd_xdna_elf_entry_record_t,
          iree_alignof(iree_hal_amd_xdna_elf_entry_record_t), &entries_offset),
      IREE_STRUCT_FIELD(entry_name_storage_length, uint8_t,
                        &entry_name_storage_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          binding_count, iree_hal_amd_xdna_elf_binding_record_t,
          iree_alignof(iree_hal_amd_xdna_elf_binding_record_t),
          &bindings_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          relocation_count, iree_hal_amd_xdna_elf_relocation_record_t,
          iree_alignof(iree_hal_amd_xdna_elf_relocation_record_t),
          &relocations_offset)));

  iree_hal_amd_xdna_image_tables_t* tables = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&tables));
  uint8_t* storage_base = (uint8_t*)tables;
  tables->host_allocator = host_allocator;
  tables->abi_note = abi_note;
  tables->entry_count = entry_count;
  tables->entries =
      (iree_hal_amd_xdna_elf_entry_record_t*)(storage_base + entries_offset);
  tables->entry_name_table_offset = entry_descriptor.header.auxiliary_offset;
  tables->entry_name_storage_length = entry_name_storage_length;
  tables->entry_name_storage = entry_name_storage_length == 0
                                   ? NULL
                                   : storage_base + entry_name_storage_offset;
  tables->binding_count = binding_count;
  tables->bindings =
      binding_count == 0
          ? NULL
          : (iree_hal_amd_xdna_elf_binding_record_t*)(storage_base +
                                                      bindings_offset);
  tables->relocation_count = relocation_count;
  tables->relocations =
      relocation_count == 0
          ? NULL
          : (iree_hal_amd_xdna_elf_relocation_record_t*)(storage_base +
                                                         relocations_offset);

  iree_status_t status = iree_hal_amd_xdna_image_decode_entries(
      directory, &entry_descriptor, tables);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_decode_bindings(
        directory, &binding_descriptor, tables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate_binding_ownership(tables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_decode_relocations(
        directory, &relocation_descriptor, tables);
  }
  if (iree_status_is_ok(status)) {
    *out_tables = tables;
  } else {
    iree_allocator_free(host_allocator, tables);
  }
  return status;
}

void iree_hal_amd_xdna_image_tables_destroy(
    iree_hal_amd_xdna_image_tables_t* tables) {
  if (tables == NULL) return;
  iree_allocator_free(tables->host_allocator, tables);
}

const iree_hal_amd_xdna_elf_abi_note_t* iree_hal_amd_xdna_image_tables_abi_note(
    const iree_hal_amd_xdna_image_tables_t* tables) {
  IREE_ASSERT_ARGUMENT(tables);
  return &tables->abi_note;
}

iree_host_size_t iree_hal_amd_xdna_image_tables_entry_count(
    const iree_hal_amd_xdna_image_tables_t* tables) {
  IREE_ASSERT_ARGUMENT(tables);
  return tables->entry_count;
}

const iree_hal_amd_xdna_elf_entry_record_t*
iree_hal_amd_xdna_image_tables_entry(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(tables);
  return ordinal < tables->entry_count ? &tables->entries[ordinal] : NULL;
}

iree_string_view_t iree_hal_amd_xdna_image_tables_entry_name(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(tables);
  if (ordinal >= tables->entry_count ||
      tables->entries[ordinal].name_length == 0) {
    return iree_string_view_empty();
  }
  const iree_hal_amd_xdna_elf_entry_record_t* entry = &tables->entries[ordinal];
  const iree_host_size_t storage_offset =
      entry->name_offset - tables->entry_name_table_offset;
  return iree_make_string_view(
      (const char*)tables->entry_name_storage + storage_offset,
      entry->name_length);
}

iree_host_size_t iree_hal_amd_xdna_image_tables_binding_count(
    const iree_hal_amd_xdna_image_tables_t* tables) {
  IREE_ASSERT_ARGUMENT(tables);
  return tables->binding_count;
}

const iree_hal_amd_xdna_elf_binding_record_t*
iree_hal_amd_xdna_image_tables_binding(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(tables);
  return ordinal < tables->binding_count ? &tables->bindings[ordinal] : NULL;
}

iree_host_size_t iree_hal_amd_xdna_image_tables_relocation_count(
    const iree_hal_amd_xdna_image_tables_t* tables) {
  IREE_ASSERT_ARGUMENT(tables);
  return tables->relocation_count;
}

const iree_hal_amd_xdna_elf_relocation_record_t*
iree_hal_amd_xdna_image_tables_relocation(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(tables);
  return ordinal < tables->relocation_count ? &tables->relocations[ordinal]
                                            : NULL;
}
