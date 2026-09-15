// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/programs.h"

#include <inttypes.h>

struct iree_hal_amd_xdna_image_programs_t {
  // Allocator owning this object and all trailing storage.
  iree_allocator_t host_allocator;
  // Number of entries in |arrays|.
  iree_host_size_t array_count;
  // Decoded ARRAY realizations in program-header order.
  iree_hal_amd_xdna_image_array_program_t* arrays;
  // Number of entries in |controls|.
  iree_host_size_t control_count;
  // Decoded CONTROL programs in program-header order.
  iree_hal_amd_xdna_image_control_program_t* controls;
  // Number of entries in |records|.
  iree_host_size_t record_count;
  // Framed ARRAY and CONTROL records in program-header order.
  iree_hal_amd_xdna_image_program_record_t* records;
};

typedef struct iree_hal_amd_xdna_image_program_counts_t {
  // Number of ARRAY program headers.
  iree_host_size_t array_count;
  // Number of CONTROL program headers.
  iree_host_size_t control_count;
  // Total records declared by all ARRAY and CONTROL headers.
  iree_host_size_t record_count;
} iree_hal_amd_xdna_image_program_counts_t;

typedef struct iree_hal_amd_xdna_image_program_info_t {
  // Serialized byte length of the target-independent program header.
  uint32_t header_size;
  // Number of framed records following the program header.
  uint32_t record_count;
  // First TILE program-header ordinal used by an ARRAY realization.
  uint32_t first_tile_program_header_ordinal;
  // Number of consecutive TILE program headers used by an ARRAY realization.
  uint32_t tile_program_header_count;
} iree_hal_amd_xdna_image_program_info_t;

static iree_status_t iree_hal_amd_xdna_image_validate_program_header(
    const iree_hal_amd_xdna_image_program_header_t* program_header) {
  if (program_header->flags != IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ ||
      program_header->virtual_address != 0 ||
      program_header->physical_address != 0 ||
      program_header->file_range.length == 0 ||
      program_header->memory_size != program_header->file_range.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program type 0x%08" PRIX32
                            " has an invalid memory contract",
                            program_header->type);
  }
  if (program_header->alignment !=
      IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_ALIGNMENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program type 0x%08" PRIX32
                            " has a noncanonical alignment",
                            program_header->type);
  }
  if (program_header->file_range.length >
      IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_PAYLOAD_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA program type 0x%08" PRIX32
                            " exceeds format capacity",
                            program_header->type);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_decode_program_info(
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    iree_const_byte_span_t header_storage,
    iree_hal_amd_xdna_image_program_info_t* out_info) {
  *out_info = (iree_hal_amd_xdna_image_program_info_t){0};
  iree_hal_amd_xdna_image_program_info_t info = {0};
  uint16_t abi_major = 0;
  uint16_t abi_minor = 0;
  uint32_t byte_length = 0;
  uint32_t flags = 0;
  switch (program_header->type) {
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY: {
      iree_hal_amd_xdna_elf_array_header_t header;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_xdna_elf_decode_array_header(header_storage, &header));
      info.header_size = IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE;
      info.record_count = header.record_count;
      info.first_tile_program_header_ordinal =
          header.first_tile_program_header_ordinal;
      info.tile_program_header_count = header.tile_program_header_count;
      abi_major = header.abi_major;
      abi_minor = header.abi_minor;
      byte_length = header.byte_length;
      flags = header.flags;
      break;
    }
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL: {
      iree_hal_amd_xdna_elf_control_header_t header;
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_xdna_elf_decode_control_header(header_storage, &header));
      info.header_size = IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE;
      info.record_count = header.record_count;
      abi_major = header.abi_major;
      abi_minor = header.abi_minor;
      byte_length = header.byte_length;
      flags = header.flags;
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA program type does not contain records");
  }
  if (abi_major != IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR ||
      abi_minor > IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported XDNA program ABI version %u.%u",
                            abi_major, abi_minor);
  }
  if (byte_length != program_header->file_range.length || flags != 0 ||
      info.record_count == 0 ||
      info.record_count > IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_RECORD_COUNT ||
      (program_header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY &&
       info.tile_program_header_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program header contract is invalid");
  }
  uint64_t minimum_byte_length = 0;
  if (!iree_checked_add_u64(
          info.header_size,
          (uint64_t)info.record_count *
              IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE,
          &minimum_byte_length) ||
      minimum_byte_length > program_header->file_range.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA program records exceed their payload");
  }
  *out_info = info;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_read_program_info(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    iree_hal_amd_xdna_image_program_info_t* out_info) {
  const iree_host_size_t header_size =
      program_header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY
          ? IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE
          : IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE;
  if (program_header->file_range.length < header_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program payload header is truncated");
  }
  uint8_t storage[IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE];
  const iree_hal_amd_xdna_image_source_range_t header_range = {
      .offset = program_header->file_range.offset,
      .length = header_size,
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_directory_read_source_range(
      directory, header_range, iree_make_byte_span(storage, header_size)));
  return iree_hal_amd_xdna_image_decode_program_info(
      program_header, iree_make_const_byte_span(storage, header_size),
      out_info);
}

static iree_status_t iree_hal_amd_xdna_image_count_programs(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_program_counts_t* out_counts) {
  *out_counts = (iree_hal_amd_xdna_image_program_counts_t){0};
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  for (iree_host_size_t i = 0; i < program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    if (program_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY &&
        program_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_xdna_image_validate_program_header(program_header));
    iree_hal_amd_xdna_image_program_info_t info;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_read_program_info(
        directory, program_header, &info));
    if (out_counts->record_count >
        IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_RECORD_COUNT - info.record_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA image has too many program records");
    }
    out_counts->record_count += info.record_count;
    if (program_header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY) {
      ++out_counts->array_count;
    } else {
      ++out_counts->control_count;
    }
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

static iree_status_t iree_hal_amd_xdna_image_scan_program_records(
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    uint32_t program_header_ordinal,
    const iree_hal_amd_xdna_image_program_info_t* info,
    iree_const_byte_span_t payload,
    iree_hal_amd_xdna_image_program_record_validator_t validator,
    iree_host_size_t first_record_ordinal,
    iree_host_size_t available_record_capacity,
    iree_hal_amd_xdna_image_program_record_t* records) {
  if (info->record_count > available_record_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA program source changed while it was being decoded");
  }
  iree_host_size_t payload_offset = info->header_size;
  for (iree_host_size_t i = 0; i < info->record_count; ++i) {
    if (payload.data_length - payload_offset <
        IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA program record header is truncated");
    }
    iree_hal_amd_xdna_elf_program_record_header_t record_header;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_decode_program_record_header(
        iree_make_const_byte_span(
            payload.data + payload_offset,
            IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE),
        &record_header));
    if (record_header.byte_length > payload.data_length - payload_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA program record exceeds its payload");
    }
    const iree_const_byte_span_t record_storage = iree_make_const_byte_span(
        payload.data + payload_offset, record_header.byte_length);
    uint32_t referenced_program_header_ordinal = UINT32_MAX;
    IREE_RETURN_IF_ERROR(validator.fn(
        validator.user_data, program_header_ordinal,
        (iree_hal_amd_xdna_elf_program_type_t)program_header->type, (uint32_t)i,
        &record_header, record_storage, &referenced_program_header_ordinal));
    uint64_t source_offset = 0;
    if (!iree_checked_add_u64(program_header->file_range.offset, payload_offset,
                              &source_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA program record source offset overflows");
    }
    records[first_record_ordinal + i] =
        (iree_hal_amd_xdna_image_program_record_t){
            .program_header_ordinal = program_header_ordinal,
            .program_record_ordinal = (uint32_t)i,
            .type = record_header.type,
            .flags = record_header.flags,
            .referenced_program_header_ordinal =
                referenced_program_header_ordinal,
            .source_range =
                {
                    .offset = source_offset,
                    .length = record_header.byte_length,
                },
        };
    payload_offset += record_header.byte_length;
  }
  if (payload_offset != payload.data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA program record count does not consume its complete payload");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_decode_programs(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_program_record_validator_t validator,
    iree_hal_amd_xdna_image_programs_t* programs) {
  iree_host_size_t array_ordinal = 0;
  iree_host_size_t control_ordinal = 0;
  iree_host_size_t record_ordinal = 0;
  iree_status_t status = iree_ok_status();
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  for (iree_host_size_t i = 0;
       i < program_header_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    if (program_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY &&
        program_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL) {
      continue;
    }
    iree_byte_span_t payload = iree_byte_span_empty();
    status = iree_hal_amd_xdna_image_read_program(
        directory, program_header, programs->host_allocator, &payload);
    iree_hal_amd_xdna_image_program_info_t info;
    const iree_host_size_t header_size =
        program_header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY
            ? IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE
            : IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE;
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_xdna_image_decode_program_info(
          program_header, iree_make_const_byte_span(payload.data, header_size),
          &info);
    }
    if (iree_status_is_ok(status)) {
      if (program_header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY) {
        programs->arrays[array_ordinal++] =
            (iree_hal_amd_xdna_image_array_program_t){
                .program_header_ordinal = (uint32_t)i,
                .first_tile_program_header_ordinal =
                    info.first_tile_program_header_ordinal,
                .tile_program_header_count = info.tile_program_header_count,
                .first_record_ordinal = (uint32_t)record_ordinal,
                .record_count = info.record_count,
            };
      } else {
        programs->controls[control_ordinal++] =
            (iree_hal_amd_xdna_image_control_program_t){
                .program_header_ordinal = (uint32_t)i,
                .first_record_ordinal = (uint32_t)record_ordinal,
                .record_count = info.record_count,
            };
      }
      status = iree_hal_amd_xdna_image_scan_program_records(
          program_header, (uint32_t)i, &info,
          iree_make_const_byte_span(payload.data, payload.data_length),
          validator, record_ordinal, programs->record_count - record_ordinal,
          programs->records);
      if (iree_status_is_ok(status)) record_ordinal += info.record_count;
    }
    iree_allocator_free(programs->host_allocator, payload.data);
  }
  if (iree_status_is_ok(status) &&
      (array_ordinal != programs->array_count ||
       control_ordinal != programs->control_count ||
       record_ordinal != programs->record_count)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA program source changed while it was being decoded");
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_image_programs_create(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_program_record_validator_t validator,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_programs_t** out_programs) {
  IREE_ASSERT_ARGUMENT(out_programs);
  *out_programs = NULL;
  if (directory == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image directory is required");
  }
  if (validator.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA program record validator is required");
  }

  iree_hal_amd_xdna_image_program_counts_t counts;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_image_count_programs(directory, &counts));

  iree_host_size_t total_size = 0;
  iree_host_size_t arrays_offset = 0;
  iree_host_size_t controls_offset = 0;
  iree_host_size_t records_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_image_programs_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          counts.array_count, iree_hal_amd_xdna_image_array_program_t,
          iree_alignof(iree_hal_amd_xdna_image_array_program_t),
          &arrays_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          counts.control_count, iree_hal_amd_xdna_image_control_program_t,
          iree_alignof(iree_hal_amd_xdna_image_control_program_t),
          &controls_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          counts.record_count, iree_hal_amd_xdna_image_program_record_t,
          iree_alignof(iree_hal_amd_xdna_image_program_record_t),
          &records_offset)));

  iree_hal_amd_xdna_image_programs_t* programs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&programs));
  uint8_t* storage_base = (uint8_t*)programs;
  programs->host_allocator = host_allocator;
  programs->array_count = counts.array_count;
  programs->arrays =
      counts.array_count == 0
          ? NULL
          : (iree_hal_amd_xdna_image_array_program_t*)(storage_base +
                                                       arrays_offset);
  programs->control_count = counts.control_count;
  programs->controls =
      counts.control_count == 0
          ? NULL
          : (iree_hal_amd_xdna_image_control_program_t*)(storage_base +
                                                         controls_offset);
  programs->record_count = counts.record_count;
  programs->records =
      counts.record_count == 0
          ? NULL
          : (iree_hal_amd_xdna_image_program_record_t*)(storage_base +
                                                        records_offset);

  iree_status_t status =
      iree_hal_amd_xdna_image_decode_programs(directory, validator, programs);
  if (iree_status_is_ok(status)) {
    *out_programs = programs;
  } else {
    iree_allocator_free(host_allocator, programs);
  }
  return status;
}

void iree_hal_amd_xdna_image_programs_destroy(
    iree_hal_amd_xdna_image_programs_t* programs) {
  if (programs == NULL) return;
  iree_allocator_free(programs->host_allocator, programs);
}

iree_host_size_t iree_hal_amd_xdna_image_programs_array_count(
    const iree_hal_amd_xdna_image_programs_t* programs) {
  IREE_ASSERT_ARGUMENT(programs);
  return programs->array_count;
}

const iree_hal_amd_xdna_image_array_program_t*
iree_hal_amd_xdna_image_programs_array(
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(programs);
  return ordinal < programs->array_count ? &programs->arrays[ordinal] : NULL;
}

iree_host_size_t iree_hal_amd_xdna_image_programs_control_count(
    const iree_hal_amd_xdna_image_programs_t* programs) {
  IREE_ASSERT_ARGUMENT(programs);
  return programs->control_count;
}

const iree_hal_amd_xdna_image_control_program_t*
iree_hal_amd_xdna_image_programs_control(
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(programs);
  return ordinal < programs->control_count ? &programs->controls[ordinal]
                                           : NULL;
}

iree_host_size_t iree_hal_amd_xdna_image_programs_record_count(
    const iree_hal_amd_xdna_image_programs_t* programs) {
  IREE_ASSERT_ARGUMENT(programs);
  return programs->record_count;
}

const iree_hal_amd_xdna_image_program_record_t*
iree_hal_amd_xdna_image_programs_record(
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(programs);
  return ordinal < programs->record_count ? &programs->records[ordinal] : NULL;
}
