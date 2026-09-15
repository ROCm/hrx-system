// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/directory.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

struct iree_hal_amd_xdna_image_directory_t {
  // Allocator owning this directory and its trailing program-header array.
  iree_allocator_t host_allocator;
  // Immutable source sequence retained for the directory lifetime.
  iree_byte_sequence_t* source_sequence;
  // Complete logical source byte length.
  uint64_t source_length;
  // Target-specific ELF flags preserved for later qualification.
  uint32_t target_flags;
  // Number of entries in |program_headers|.
  iree_host_size_t program_header_count;
  // Decoded program headers in stable ELF directory order.
  iree_hal_amd_xdna_image_program_header_t program_headers[];
};

typedef struct iree_hal_amd_xdna_image_elf_envelope_t {
  // Target-specific ELF flags preserved for later qualification.
  uint32_t target_flags;
  // Program-header directory bytes in the source sequence.
  iree_hal_amd_xdna_image_source_range_t program_header_range;
  // Number of fixed-width entries in |program_header_range|.
  iree_host_size_t program_header_count;
  // Optional diagnostic section-header directory bytes.
  iree_hal_amd_xdna_image_source_range_t section_header_range;
} iree_hal_amd_xdna_image_elf_envelope_t;

static bool iree_hal_amd_xdna_image_source_ranges_overlap(
    iree_hal_amd_xdna_image_source_range_t lhs,
    iree_hal_amd_xdna_image_source_range_t rhs) {
  if (lhs.length == 0 || rhs.length == 0) return false;
  return lhs.offset <= rhs.offset ? rhs.offset - lhs.offset < lhs.length
                                  : lhs.offset - rhs.offset < rhs.length;
}

static bool iree_hal_amd_xdna_image_source_ranges_equal(
    iree_hal_amd_xdna_image_source_range_t lhs,
    iree_hal_amd_xdna_image_source_range_t rhs) {
  return lhs.offset == rhs.offset && lhs.length == rhs.length;
}

static iree_status_t iree_hal_amd_xdna_image_validate_source_range(
    const iree_byte_sequence_t* source_sequence,
    iree_hal_amd_xdna_image_source_range_t source_range) {
  uint64_t source_range_end = 0;
  if (!iree_checked_add_u64(source_range.offset, source_range.length,
                            &source_range_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA image source range offset 0x%" PRIX64
                            " and length 0x%" PRIX64 " overflow",
                            source_range.offset, source_range.length);
  }
  if (source_range_end > iree_byte_sequence_length(source_sequence)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA image source range [0x%" PRIX64 ", 0x%" PRIX64
                            ") exceeds the source length 0x%" PRIX64,
                            source_range.offset, source_range_end,
                            iree_byte_sequence_length(source_sequence));
  }
  return iree_ok_status();
}

typedef struct iree_hal_amd_xdna_image_range_enumerator_t {
  // Complete source length declared by the byte sequence.
  uint64_t source_length;
  // Logical offset of the next enumerated source segment.
  uint64_t source_offset;
  // First logical source byte requested by the caller.
  uint64_t range_offset;
  // One-past-the-end logical source byte requested by the caller.
  uint64_t range_end;
  // Number of requested bytes delivered to |callback|.
  uint64_t delivered_length;
  // Caller callback receiving intersecting portions of source segments.
  iree_byte_sequence_segment_callback_t callback;
} iree_hal_amd_xdna_image_range_enumerator_t;

static iree_status_t iree_hal_amd_xdna_image_enumerate_source_segment(
    void* user_data, iree_const_byte_span_t segment) {
  iree_hal_amd_xdna_image_range_enumerator_t* enumerator =
      (iree_hal_amd_xdna_image_range_enumerator_t*)user_data;
  if (segment.data_length != 0 && segment.data == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "XDNA image source yielded a NULL segment");
  }
  uint64_t segment_end = 0;
  if (!iree_checked_add_u64(enumerator->source_offset, segment.data_length,
                            &segment_end) ||
      segment_end > enumerator->source_length) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA image source segments exceed their declared length");
  }

  const uint64_t overlap_offset =
      iree_max(enumerator->source_offset, enumerator->range_offset);
  const uint64_t overlap_end = iree_min(segment_end, enumerator->range_end);
  if (overlap_offset < overlap_end) {
    const iree_host_size_t segment_offset =
        (iree_host_size_t)(overlap_offset - enumerator->source_offset);
    const iree_host_size_t overlap_length =
        (iree_host_size_t)(overlap_end - overlap_offset);
    IREE_RETURN_IF_ERROR(enumerator->callback.fn(
        enumerator->callback.user_data,
        iree_make_const_byte_span(segment.data + segment_offset,
                                  overlap_length)));
    enumerator->delivered_length += overlap_length;
  }
  enumerator->source_offset = segment_end;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_enumerate_sequence_range(
    const iree_byte_sequence_t* source_sequence,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_sequence_segment_callback_t callback) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_source_range(
      source_sequence, source_range));
  if (source_range.length == 0) return iree_ok_status();

  const uint64_t source_length = iree_byte_sequence_length(source_sequence);
  iree_const_byte_span_t contiguous_span = iree_const_byte_span_empty();
  if (iree_byte_sequence_try_get_contiguous_span(source_sequence,
                                                 &contiguous_span)) {
    if ((uint64_t)contiguous_span.data_length != source_length ||
        (contiguous_span.data_length != 0 && contiguous_span.data == NULL)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "XDNA image source contiguous span violates its declared length");
    }
    return callback.fn(
        callback.user_data,
        iree_make_const_byte_span(
            contiguous_span.data + (iree_host_size_t)source_range.offset,
            (iree_host_size_t)source_range.length));
  }

  iree_hal_amd_xdna_image_range_enumerator_t enumerator = {
      .source_length = source_length,
      .source_offset = 0,
      .range_offset = source_range.offset,
      .range_end = source_range.offset + source_range.length,
      .delivered_length = 0,
      .callback = callback,
  };
  IREE_RETURN_IF_ERROR(iree_byte_sequence_enumerate(
      source_sequence,
      (iree_byte_sequence_segment_callback_t){
          .fn = iree_hal_amd_xdna_image_enumerate_source_segment,
          .user_data = &enumerator,
      }));
  if (enumerator.source_offset != source_length ||
      enumerator.delivered_length != source_range.length) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA image source segments do not match their declared length");
  }
  return iree_ok_status();
}

typedef struct iree_hal_amd_xdna_image_range_reader_t {
  // Destination storage receiving source bytes in logical order.
  iree_byte_span_t storage;
  // Number of bytes already written to |storage|.
  iree_host_size_t offset;
} iree_hal_amd_xdna_image_range_reader_t;

static iree_status_t iree_hal_amd_xdna_image_read_source_segment(
    void* user_data, iree_const_byte_span_t segment) {
  iree_hal_amd_xdna_image_range_reader_t* reader =
      (iree_hal_amd_xdna_image_range_reader_t*)user_data;
  memcpy(reader->storage.data + reader->offset, segment.data,
         segment.data_length);
  reader->offset += segment.data_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_read_sequence_range(
    const iree_byte_sequence_t* source_sequence,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_span_t storage) {
  if ((uint64_t)storage.data_length != source_range.length ||
      (storage.data_length != 0 && storage.data == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image source range requires exactly %" PRIu64
                            " bytes of destination storage",
                            source_range.length);
  }
  iree_hal_amd_xdna_image_range_reader_t reader = {
      .storage = storage,
      .offset = 0,
  };
  return iree_hal_amd_xdna_image_enumerate_sequence_range(
      source_sequence, source_range,
      (iree_byte_sequence_segment_callback_t){
          .fn = iree_hal_amd_xdna_image_read_source_segment,
          .user_data = &reader,
      });
}

iree_status_t iree_hal_amd_xdna_image_directory_enumerate_source_range(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_sequence_segment_callback_t callback) {
  IREE_ASSERT_ARGUMENT(directory);
  IREE_ASSERT_ARGUMENT(callback.fn);
  return iree_hal_amd_xdna_image_enumerate_sequence_range(
      directory->source_sequence, source_range, callback);
}

iree_status_t iree_hal_amd_xdna_image_directory_read_source_range(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(directory);
  return iree_hal_amd_xdna_image_read_sequence_range(directory->source_sequence,
                                                     source_range, storage);
}

static iree_status_t iree_hal_amd_xdna_image_decode_elf_envelope(
    const iree_byte_sequence_t* source_sequence,
    iree_hal_amd_xdna_image_elf_envelope_t* out_envelope) {
  *out_envelope = (iree_hal_amd_xdna_image_elf_envelope_t){0};
  uint8_t bytes[IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE];
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_read_sequence_range(
      source_sequence,
      (iree_hal_amd_xdna_image_source_range_t){
          .offset = 0,
          .length = IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE,
      },
      iree_make_byte_span(bytes, sizeof(bytes))));

  const uint8_t expected_magic[4] = {0x7F, 'E', 'L', 'F'};
  if (memcmp(bytes, expected_magic, sizeof(expected_magic)) != 0 ||
      bytes[4] != IREE_HAL_AMD_XDNA_ELF_CLASS_32 ||
      bytes[5] != IREE_HAL_AMD_XDNA_ELF_DATA_LITTLE_ENDIAN ||
      bytes[6] != IREE_HAL_AMD_XDNA_ELF_VERSION_CURRENT ||
      bytes[7] != IREE_HAL_AMD_XDNA_ELF_OS_ABI_NONE ||
      bytes[8] != IREE_HAL_AMD_XDNA_ELF_ABI_VERSION_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image is not canonical ELF32LE");
  }
  for (iree_host_size_t i = 9; i < 16; ++i) {
    if (bytes[i] != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA image ELF identity padding is not zero");
    }
  }
  if (iree_unaligned_load_le_u16(bytes + 16) !=
          IREE_HAL_AMD_XDNA_ELF_FILE_TYPE_EXEC ||
      iree_unaligned_load_le_u16(bytes + 18) !=
          IREE_HAL_AMD_XDNA_ELF_MACHINE_AIE ||
      iree_unaligned_load_le_u32(bytes + 20) !=
          IREE_HAL_AMD_XDNA_ELF_VERSION_CURRENT ||
      iree_unaligned_load_le_u32(bytes + 24) != 0 ||
      iree_unaligned_load_le_u16(bytes + 40) !=
          IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ELF header identity is invalid");
  }

  const uint32_t program_header_offset = iree_unaligned_load_le_u32(bytes + 28);
  const uint32_t section_header_offset = iree_unaligned_load_le_u32(bytes + 32);
  const uint16_t program_header_size = iree_unaligned_load_le_u16(bytes + 42);
  const uint16_t program_header_count = iree_unaligned_load_le_u16(bytes + 44);
  const uint16_t section_header_size = iree_unaligned_load_le_u16(bytes + 46);
  const uint16_t section_header_count = iree_unaligned_load_le_u16(bytes + 48);
  const uint16_t section_name_table_ordinal =
      iree_unaligned_load_le_u16(bytes + 50);
  if (program_header_offset != IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE ||
      program_header_size != IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE ||
      program_header_count == 0 ||
      program_header_count > IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA image ELF program-header directory is invalid");
  }

  const bool has_section_headers =
      section_header_offset != 0 || section_header_size != 0 ||
      section_header_count != 0 || section_name_table_ordinal != 0;
  if (has_section_headers &&
      (section_header_offset == 0 ||
       section_header_size != IREE_HAL_AMD_XDNA_ELF_SECTION_HEADER_SIZE ||
       section_header_count == 0 ||
       section_header_count > IREE_HAL_AMD_XDNA_ELF_MAX_SECTION_HEADER_COUNT ||
       section_name_table_ordinal >= section_header_count ||
       section_header_offset % 4 != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA image ELF section-header directory is invalid");
  }

  iree_hal_amd_xdna_image_elf_envelope_t envelope = {
      .target_flags = iree_unaligned_load_le_u32(bytes + 36),
      .program_header_range =
          {
              .offset = program_header_offset,
              .length = (uint64_t)program_header_count * program_header_size,
          },
      .program_header_count = program_header_count,
      .section_header_range =
          {
              .offset = section_header_offset,
              .length = has_section_headers ? (uint64_t)section_header_count *
                                                  section_header_size
                                            : 0,
          },
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_source_range(
      source_sequence, envelope.program_header_range));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_source_range(
      source_sequence, envelope.section_header_range));
  const iree_hal_amd_xdna_image_source_range_t elf_header_range = {
      .offset = 0,
      .length = IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE,
  };
  if (iree_hal_amd_xdna_image_source_ranges_overlap(
          elf_header_range, envelope.section_header_range) ||
      iree_hal_amd_xdna_image_source_ranges_overlap(
          envelope.program_header_range, envelope.section_header_range)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image ELF directories overlap");
  }
  *out_envelope = envelope;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_image_decode_program_header(
    const uint8_t* bytes,
    iree_hal_amd_xdna_image_program_header_t* out_program_header) {
  const uint32_t file_offset = iree_unaligned_load_le_u32(bytes + 4);
  const uint32_t file_size = iree_unaligned_load_le_u32(bytes + 16);
  *out_program_header = (iree_hal_amd_xdna_image_program_header_t){
      .type = iree_unaligned_load_le_u32(bytes + 0),
      .file_range =
          {
              .offset = file_offset,
              .length = file_size,
          },
      .virtual_address = iree_unaligned_load_le_u32(bytes + 8),
      .physical_address = iree_unaligned_load_le_u32(bytes + 12),
      .memory_size = iree_unaligned_load_le_u32(bytes + 20),
      .flags = iree_unaligned_load_le_u32(bytes + 24),
      .alignment = iree_unaligned_load_le_u32(bytes + 28),
  };
}

typedef struct iree_hal_amd_xdna_image_program_range_index_t {
  // Source range used as the primary sort key.
  iree_hal_amd_xdna_image_source_range_t file_range;
  // Original program-header ordinal used for diagnostics.
  iree_host_size_t ordinal;
} iree_hal_amd_xdna_image_program_range_index_t;

static_assert(sizeof(iree_hal_amd_xdna_image_program_range_index_t) <=
                  IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE,
              "program range indexes reuse raw program-header scratch storage");

static int iree_hal_amd_xdna_image_compare_program_ranges(const void* lhs_ptr,
                                                          const void* rhs_ptr) {
  const iree_hal_amd_xdna_image_program_range_index_t* lhs =
      (const iree_hal_amd_xdna_image_program_range_index_t*)lhs_ptr;
  const iree_hal_amd_xdna_image_program_range_index_t* rhs =
      (const iree_hal_amd_xdna_image_program_range_index_t*)rhs_ptr;
  if (lhs->file_range.offset < rhs->file_range.offset) return -1;
  if (lhs->file_range.offset > rhs->file_range.offset) return 1;
  if (lhs->file_range.length > rhs->file_range.length) return -1;
  if (lhs->file_range.length < rhs->file_range.length) return 1;
  if (lhs->ordinal < rhs->ordinal) return -1;
  if (lhs->ordinal > rhs->ordinal) return 1;
  return 0;
}

static iree_status_t iree_hal_amd_xdna_image_validate_program_headers(
    const iree_byte_sequence_t* source_sequence,
    const iree_hal_amd_xdna_image_elf_envelope_t* envelope,
    iree_byte_span_t scratch_storage,
    iree_hal_amd_xdna_image_directory_t* directory) {
  const iree_hal_amd_xdna_image_source_range_t elf_header_range = {
      .offset = 0,
      .length = IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE,
  };
  for (iree_host_size_t i = 0; i < directory->program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        &directory->program_headers[i];
    if (program_header->alignment == 0 ||
        !iree_is_power_of_two_uint64(program_header->alignment) ||
        program_header->file_range.offset % program_header->alignment !=
            program_header->virtual_address % program_header->alignment ||
        program_header->file_range.length > program_header->memory_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA image program header %" PRIhsz
                              " has an invalid size or alignment",
                              i);
    }
    const iree_hal_amd_xdna_elf_program_flags_t unknown_flags =
        program_header->flags & ~(IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ |
                                  IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_WRITE |
                                  IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE);
    if (unknown_flags != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA image program header %" PRIhsz
                              " has unknown permission flags 0x%08" PRIX32,
                              i, unknown_flags);
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_source_range(
        source_sequence, program_header->file_range));
    if (iree_hal_amd_xdna_image_source_ranges_overlap(
            program_header->file_range, elf_header_range) ||
        iree_hal_amd_xdna_image_source_ranges_overlap(
            program_header->file_range, envelope->program_header_range) ||
        iree_hal_amd_xdna_image_source_ranges_overlap(
            program_header->file_range, envelope->section_header_range)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA image program header %" PRIhsz " overlaps an ELF directory", i);
    }
  }

  iree_hal_amd_xdna_image_program_range_index_t* range_indexes =
      (iree_hal_amd_xdna_image_program_range_index_t*)scratch_storage.data;
  for (iree_host_size_t i = 0; i < directory->program_header_count; ++i) {
    range_indexes[i] = (iree_hal_amd_xdna_image_program_range_index_t){
        .file_range = directory->program_headers[i].file_range,
        .ordinal = i,
    };
  }
  qsort(range_indexes, directory->program_header_count, sizeof(*range_indexes),
        iree_hal_amd_xdna_image_compare_program_ranges);
  const iree_hal_amd_xdna_image_program_range_index_t* active_range = NULL;
  for (iree_host_size_t i = 0; i < directory->program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_range_index_t* range =
        &range_indexes[i];
    if (range->file_range.length == 0) continue;
    if (active_range != NULL &&
        iree_hal_amd_xdna_image_source_ranges_overlap(active_range->file_range,
                                                      range->file_range)) {
      if (!iree_hal_amd_xdna_image_source_ranges_equal(active_range->file_range,
                                                       range->file_range)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "XDNA image program headers %" PRIhsz
                                " and %" PRIhsz " partially overlap",
                                active_range->ordinal, range->ordinal);
      }
    } else {
      active_range = range;
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_image_directory_create(
    iree_byte_sequence_t* source_sequence, iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_directory_t** out_directory) {
  IREE_ASSERT_ARGUMENT(out_directory);
  *out_directory = NULL;
  if (source_sequence == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image source sequence is required");
  }

  iree_hal_amd_xdna_image_elf_envelope_t envelope;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_image_decode_elf_envelope(source_sequence, &envelope));

  iree_hal_amd_xdna_image_directory_t* directory = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*directory), envelope.program_header_count,
      sizeof(*directory->program_headers), (void**)&directory));
  directory->host_allocator = host_allocator;
  directory->source_length = iree_byte_sequence_length(source_sequence);
  directory->target_flags = envelope.target_flags;
  directory->program_header_count = envelope.program_header_count;

  iree_byte_span_t program_header_storage = iree_byte_span_empty();
  program_header_storage.data_length =
      (iree_host_size_t)envelope.program_header_range.length;
  iree_status_t status = iree_allocator_malloc_uninitialized(
      host_allocator, program_header_storage.data_length,
      (void**)&program_header_storage.data);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_read_sequence_range(
        source_sequence, envelope.program_header_range, program_header_storage);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < directory->program_header_count; ++i) {
      iree_hal_amd_xdna_image_decode_program_header(
          program_header_storage.data +
              i * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE,
          &directory->program_headers[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate_program_headers(
        source_sequence, &envelope, program_header_storage, directory);
  }
  iree_allocator_free(host_allocator, program_header_storage.data);

  if (iree_status_is_ok(status)) {
    iree_byte_sequence_retain(source_sequence);
    directory->source_sequence = source_sequence;
    *out_directory = directory;
  } else {
    iree_allocator_free(host_allocator, directory);
  }
  return status;
}

void iree_hal_amd_xdna_image_directory_destroy(
    iree_hal_amd_xdna_image_directory_t* directory) {
  if (directory == NULL) return;
  const iree_allocator_t host_allocator = directory->host_allocator;
  iree_byte_sequence_release(directory->source_sequence);
  iree_allocator_free(host_allocator, directory);
}

uint64_t iree_hal_amd_xdna_image_directory_source_length(
    const iree_hal_amd_xdna_image_directory_t* directory) {
  IREE_ASSERT_ARGUMENT(directory);
  return directory->source_length;
}

uint32_t iree_hal_amd_xdna_image_directory_target_flags(
    const iree_hal_amd_xdna_image_directory_t* directory) {
  IREE_ASSERT_ARGUMENT(directory);
  return directory->target_flags;
}

iree_host_size_t iree_hal_amd_xdna_image_directory_program_header_count(
    const iree_hal_amd_xdna_image_directory_t* directory) {
  IREE_ASSERT_ARGUMENT(directory);
  return directory->program_header_count;
}

const iree_hal_amd_xdna_image_program_header_t*
iree_hal_amd_xdna_image_directory_program_header(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(directory);
  return ordinal < directory->program_header_count
             ? &directory->program_headers[ordinal]
             : NULL;
}
