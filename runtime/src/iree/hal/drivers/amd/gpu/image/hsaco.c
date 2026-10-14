// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/image/hsaco.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco_metadata.h"
#include "iree/hal/drivers/amd/gpu/target/code_object.h"

// ELF64 wire records are read with unaligned little-endian loads. Host ELF
// headers and their platform-specific typedefs never participate in admission.
enum {
  IREE_HAL_AMD_GPU_HSACO_PROGRAM_HEADER_SIZE = 56,
  IREE_HAL_AMD_GPU_HSACO_DYNAMIC_ENTRY_SIZE = 16,
  IREE_HAL_AMD_GPU_HSACO_SYMBOL_SIZE = 24,
  IREE_HAL_AMD_GPU_HSACO_RELOCATION_SIZE = 24,
  IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE = 64,
};

typedef struct iree_hal_amd_gpu_hsaco_range_t {
  // Logical source byte offset.
  uint64_t offset;
  // Logical source byte length.
  uint64_t length;
} iree_hal_amd_gpu_hsaco_range_t;

typedef struct iree_hal_amd_gpu_hsaco_dynamic_t {
  // Dynamic string table linked virtual address.
  uint64_t strings;
  // Byte length of the dynamic string table.
  uint64_t string_length;
  // Dynamic symbol table linked virtual address.
  uint64_t symbols;
  // System V hash table linked virtual address, or zero if absent.
  uint64_t hash;
  // GNU hash table linked virtual address, or zero if absent.
  uint64_t gnu_hash;
  // Explicit-addend relocation table linked virtual address.
  uint64_t relocations;
  // Byte length of the explicit-addend relocation table.
  uint64_t relocation_length;
  // Number of named shared-library dependencies.
  iree_host_size_t dependency_count;
} iree_hal_amd_gpu_hsaco_dynamic_t;

typedef struct iree_hal_amd_gpu_hsaco_reader_t {
  // Borrowed source used throughout admission.
  iree_byte_sequence_t* source;
  // Allocator for temporary gathered tables.
  iree_allocator_t host_allocator;
  // Optional stable view when the entire source is already contiguous.
  iree_const_byte_span_t contiguous;
  // Source byte length, including non-loadable diagnostic sections.
  uint64_t source_length;
  // Aligned original virtual base subtracted from normalized image offsets.
  uint64_t virtual_base;
  // Program-header count established at the ELF boundary.
  uint32_t program_count;
  // Borrowed or gathered program-header bytes.
  iree_const_byte_span_t programs;
  // Allocation backing programs when the source is segmented.
  void* program_storage;
  // Temporary segment directory before final image storage is allocated.
  iree_hal_amd_gpu_image_segment_t* segments;
  // Borrowed or gathered dynamic table bytes.
  iree_const_byte_span_t dynamic_entries;
  // Allocation backing dynamic_entries when the source is segmented.
  void* dynamic_storage;
  // Decoded dynamic table addresses and sizes.
  iree_hal_amd_gpu_hsaco_dynamic_t dynamic;
  // Borrowed or gathered dynamic symbol bytes.
  iree_const_byte_span_t symbols;
  // Allocation backing symbols when the source is segmented.
  void* symbol_storage;
  // Number of admitted dynamic symbol entries, including the null entry.
  uint32_t symbol_count;
  // Borrowed or gathered dynamic string table bytes.
  iree_const_byte_span_t strings;
  // Borrowed or gathered relocation table bytes.
  iree_const_byte_span_t relocations;
  // Allocation backing relocations when the source is segmented.
  void* relocation_storage;
  // Normalized references in original symbol order during relocation admission.
  iree_hal_amd_gpu_image_address_t* symbol_addresses;
} iree_hal_amd_gpu_hsaco_reader_t;

static void iree_hal_amd_gpu_hsaco_reader_deinitialize(
    iree_hal_amd_gpu_hsaco_reader_t* reader) {
  iree_allocator_free(reader->host_allocator, reader->symbol_addresses);
  iree_allocator_free(reader->host_allocator, reader->relocation_storage);
  iree_allocator_free(reader->host_allocator, reader->symbol_storage);
  iree_allocator_free(reader->host_allocator, reader->dynamic_storage);
  iree_allocator_free(reader->host_allocator, reader->segments);
  iree_allocator_free(reader->host_allocator, reader->program_storage);
}

static iree_status_t iree_hal_amd_gpu_hsaco_validate_source_range(
    const iree_hal_amd_gpu_hsaco_reader_t* reader, uint64_t offset,
    uint64_t length) {
  if (offset > reader->source_length ||
      length > reader->source_length - offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO source range [0x%" PRIx64 ", +0x%" PRIx64
                            ") exceeds the artifact",
                            offset, length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_blob(
    const iree_hal_amd_gpu_hsaco_reader_t* reader, uint64_t offset,
    uint64_t length, iree_const_byte_span_t* out_span, void** out_storage) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_hsaco_validate_source_range(reader, offset, length));
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HSACO structural table exceeds host address space");
  }
  if (reader->contiguous.data) {
    *out_span =
        iree_make_const_byte_span(reader->contiguous.data + offset, length);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(reader->host_allocator, length, out_storage));
  IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
      reader->source, offset, iree_make_byte_span(*out_storage, length)));
  *out_span = iree_make_const_byte_span(*out_storage, length);
  return iree_ok_status();
}

// Returns the segment containing a complete virtual range. Zero-length symbols
// may designate the end of a segment, while fields and code require real bytes.
static bool iree_hal_amd_gpu_hsaco_find_segment(
    const iree_hal_amd_gpu_image_t* image, uint64_t offset, uint64_t length,
    bool file_backed, uint32_t* out_ordinal) {
  for (iree_host_size_t i = 0; i < image->segment_count; ++i) {
    const iree_hal_amd_gpu_image_segment_t* segment = &image->segments[i];
    uint64_t available = file_backed ? segment->source_length : segment->length;
    if (offset >= segment->offset && offset - segment->offset <= available &&
        length <= available - (offset - segment->offset)) {
      *out_ordinal = (uint32_t)i;
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_amd_gpu_hsaco_source_for_address(
    const iree_hal_amd_gpu_hsaco_reader_t* reader,
    const iree_hal_amd_gpu_image_t* image, uint64_t address, uint64_t length,
    uint64_t* out_offset, uint64_t* out_available) {
  uint32_t ordinal = 0;
  if (address < reader->virtual_base ||
      !iree_hal_amd_gpu_hsaco_find_segment(
          image, address - reader->virtual_base, length, true, &ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO virtual range [0x%" PRIx64 ", +0x%" PRIx64
                            ") is not file-backed",
                            address, length);
  }
  const iree_hal_amd_gpu_image_segment_t* segment = &image->segments[ordinal];
  uint64_t local_offset = address - reader->virtual_base - segment->offset;
  *out_offset = segment->source_offset + local_offset;
  if (out_available) {
    *out_available = segment->source_length - local_offset;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_address_blob(
    const iree_hal_amd_gpu_hsaco_reader_t* reader,
    const iree_hal_amd_gpu_image_t* image, uint64_t address, uint64_t length,
    iree_const_byte_span_t* out_span, void** out_storage) {
  uint64_t offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_source_for_address(
      reader, image, address, length, &offset, NULL));
  return iree_hal_amd_gpu_hsaco_read_blob(reader, offset, length, out_span,
                                          out_storage);
}

static int iree_hal_amd_gpu_hsaco_compare_segments(const void* lhs,
                                                   const void* rhs) {
  uint64_t a = ((const iree_hal_amd_gpu_image_segment_t*)lhs)->offset;
  uint64_t b = ((const iree_hal_amd_gpu_image_segment_t*)rhs)->offset;
  return a < b ? -1 : a > b ? 1 : 0;
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_programs(
    iree_hal_amd_gpu_hsaco_reader_t* reader, iree_hal_amd_gpu_image_t* image) {
  uint8_t header[64];
  IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
      reader->source, 0, iree_make_byte_span(header, sizeof(header))));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_code_object_identity_from_elf(
      iree_make_const_byte_span(header, sizeof(header)), &image->target));
  image->code_object_version = header[8] + 2;
  if (iree_unaligned_load_le_u16(header + 16) != 3) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HSACO loading requires linked ET_DYN code objects");
  }
  if (iree_unaligned_load_le_u16(header + 52) != sizeof(header) ||
      iree_unaligned_load_le_u16(header + 54) !=
          IREE_HAL_AMD_GPU_HSACO_PROGRAM_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO has invalid ELF64 header sizes");
  }
  reader->program_count = iree_unaligned_load_le_u16(header + 56);
  if (reader->program_count == UINT16_MAX) {
    // ELF extended numbering stores the program count in section zero's info.
    uint8_t section[64];
    uint64_t section_offset = iree_unaligned_load_le_u64(header + 40);
    if (!section_offset || iree_unaligned_load_le_u16(header + 58) != 64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HSACO extended program count has no section zero");
    }
    IREE_RETURN_IF_ERROR(
        iree_byte_sequence_read(reader->source, section_offset,
                                iree_make_byte_span(section, sizeof(section))));
    reader->program_count = iree_unaligned_load_le_u32(section + 44);
  }
  if (!reader->program_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO has no program headers");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_read_blob(
      reader, iree_unaligned_load_le_u64(header + 32),
      (uint64_t)reader->program_count *
          IREE_HAL_AMD_GPU_HSACO_PROGRAM_HEADER_SIZE,
      &reader->programs, &reader->program_storage));
  iree_host_size_t segment_storage_length = 0;
  if (!iree_host_size_checked_mul(reader->program_count,
                                  sizeof(*reader->segments),
                                  &segment_storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSACO segment table exceeds host address space");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(reader->host_allocator,
                                             segment_storage_length,
                                             (void**)&reader->segments));
  image->segments = reader->segments;
  image->alignment = 256;
  iree_hal_amd_gpu_hsaco_range_t dynamic_range = {0};
  for (uint32_t i = 0; i < reader->program_count; ++i) {
    const uint8_t* program =
        reader->programs.data +
        (uint64_t)i * IREE_HAL_AMD_GPU_HSACO_PROGRAM_HEADER_SIZE;
    uint32_t type = iree_unaligned_load_le_u32(program);
    uint64_t offset = iree_unaligned_load_le_u64(program + 8);
    uint64_t address = iree_unaligned_load_le_u64(program + 16);
    uint64_t file_length = iree_unaligned_load_le_u64(program + 32);
    uint64_t memory_length = iree_unaligned_load_le_u64(program + 40);
    uint64_t alignment = iree_unaligned_load_le_u64(program + 48);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_validate_source_range(
        reader, offset, file_length));
    if (type == 1) {  // PT_LOAD.
      uint32_t flags = iree_unaligned_load_le_u32(program + 4);
      if (file_length > memory_length || memory_length > UINT64_MAX - address ||
          (alignment > 1 &&
           ((alignment & (alignment - 1)) ||
            (offset & (alignment - 1)) != (address & (alignment - 1))))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HSACO load segment has invalid extent/alignment");
      }
      if (flags & ~7u) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "HSACO load segment has unsupported flags 0x%x",
                                flags);
      }
      if (!memory_length) {
        continue;
      }
      image->alignment = iree_max(image->alignment, alignment);
      reader->segments[image->segment_count++] =
          (iree_hal_amd_gpu_image_segment_t){
              .source_offset = offset,
              .source_length = file_length,
              .offset = address,
              .length = memory_length,
              .access =
                  ((flags & 4) ? IREE_HAL_AMD_GPU_IMAGE_ACCESS_READ : 0) |
                  ((flags & 2) ? IREE_HAL_AMD_GPU_IMAGE_ACCESS_WRITE : 0) |
                  ((flags & 1) ? IREE_HAL_AMD_GPU_IMAGE_ACCESS_EXECUTE : 0),
          };
    } else if (type == 2) {  // PT_DYNAMIC.
      if (dynamic_range.length || !file_length ||
          file_length % IREE_HAL_AMD_GPU_HSACO_DYNAMIC_ENTRY_SIZE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO requires one complete dynamic table");
      }
      dynamic_range = (iree_hal_amd_gpu_hsaco_range_t){offset, file_length};
    } else if (type == 3 || type == 5 || type == 7) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "HSACO program type %u requires an unsupported "
                              "interpreter, shared-library, or TLS service",
                              type);
    }
  }
  if (!image->segment_count || !dynamic_range.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO requires load segments and a dynamic table");
  }
  qsort(reader->segments, image->segment_count, sizeof(*reader->segments),
        iree_hal_amd_gpu_hsaco_compare_segments);
  reader->virtual_base = reader->segments[0].offset & ~(image->alignment - 1);
  uint64_t previous_end = reader->virtual_base;
  for (iree_host_size_t i = 0; i < image->segment_count; ++i) {
    iree_hal_amd_gpu_image_segment_t* segment = &reader->segments[i];
    if (segment->offset < previous_end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HSACO load segments overlap in virtual memory");
    }
    previous_end = segment->offset + segment->length;
    segment->offset -= reader->virtual_base;
  }
  image->byte_length = previous_end - reader->virtual_base;
  return iree_hal_amd_gpu_hsaco_read_blob(
      reader, dynamic_range.offset, dynamic_range.length,
      &reader->dynamic_entries, &reader->dynamic_storage);
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_dynamic(
    iree_hal_amd_gpu_hsaco_reader_t* reader) {
  uint64_t seen = 0;
  bool terminated = false;
  uint64_t symbol_stride = 0;
  uint64_t relocation_stride = 0;
  for (iree_host_size_t offset = 0;
       offset < reader->dynamic_entries.data_length;
       offset += IREE_HAL_AMD_GPU_HSACO_DYNAMIC_ENTRY_SIZE) {
    const uint8_t* entry = reader->dynamic_entries.data + offset;
    uint64_t tag = iree_unaligned_load_le_u64(entry);
    uint64_t value = iree_unaligned_load_le_u64(entry + 8);
    if (tag == 0) {
      terminated = true;
      break;
    }
    // All scalar tags below are unique; DT_NEEDED is intentionally repeatable.
    if (tag < 64 && tag != 1) {
      if (seen & (UINT64_C(1) << tag)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO repeats dynamic tag %" PRIu64, tag);
      }
      seen |= UINT64_C(1) << tag;
    }
    switch (tag) {
      case 1:  // DT_NEEDED.
        ++reader->dynamic.dependency_count;
        break;
      case 4:  // DT_HASH.
        reader->dynamic.hash = value;
        break;
      case 5:  // DT_STRTAB.
        reader->dynamic.strings = value;
        break;
      case 6:  // DT_SYMTAB.
        reader->dynamic.symbols = value;
        break;
      case 7:  // DT_RELA.
        reader->dynamic.relocations = value;
        break;
      case 8:  // DT_RELASZ.
        reader->dynamic.relocation_length = value;
        break;
      case 9:  // DT_RELAENT.
        relocation_stride = value;
        break;
      case 10:  // DT_STRSZ.
        reader->dynamic.string_length = value;
        break;
      case 11:  // DT_SYMENT.
        symbol_stride = value;
        break;
      case 14:  // DT_SONAME is descriptive; dependencies use explicit bindings.
      case 16:  // DT_SYMBOLIC; definitions already resolve within this image.
      case 21:  // DT_DEBUG is populated by a host loader, never by GPU code.
      case 22:  // DT_TEXTREL; all segments are writable during loading.
      case 24:  // DT_BIND_NOW; all imports are resolved before publication.
      case 0x6ffffff9:  // DT_RELACOUNT is an optional traversal optimization.
        break;
      case 30:  // DT_FLAGS: SYMBOLIC, TEXTREL, BIND_NOW need no extra service.
        if (value & ~UINT64_C(0xe)) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "HSACO dynamic flags require runtime services");
        }
        break;
      case 0x6ffffffb:  // DT_FLAGS_1: NOW only.
        if (value & ~UINT64_C(1)) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "HSACO extended dynamic flags require runtime "
              "services");
        }
        break;
      case 0x6ffffef5:  // DT_GNU_HASH.
        if (reader->dynamic.gnu_hash) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "HSACO repeats its GNU hash table");
        }
        reader->dynamic.gnu_hash = value;
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "HSACO dynamic tag 0x%" PRIx64
                                " requires an unsupported loader service",
                                tag);
    }
  }
  uint64_t required = (UINT64_C(1) << 5) | (UINT64_C(1) << 6) |
                      (UINT64_C(1) << 10) | (UINT64_C(1) << 11);
  if (!terminated || (seen & required) != required ||
      !reader->dynamic.string_length ||
      symbol_stride != IREE_HAL_AMD_GPU_HSACO_SYMBOL_SIZE ||
      (!reader->dynamic.hash && !reader->dynamic.gnu_hash)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO dynamic table is incomplete or malformed");
  }
  uint64_t relocation_tags =
      (UINT64_C(1) << 7) | (UINT64_C(1) << 8) | (UINT64_C(1) << 9);
  if ((seen & relocation_tags) &&
      ((seen & relocation_tags) != relocation_tags ||
       relocation_stride != IREE_HAL_AMD_GPU_HSACO_RELOCATION_SIZE ||
       reader->dynamic.relocation_length % relocation_stride)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSACO relocation table is incomplete or malformed");
  }
  return iree_ok_status();
}

// Streaming discovery of the final GNU hash chain's terminator. The source can
// split any word; enumeration after the terminator performs no further parsing.
typedef struct iree_hal_amd_gpu_hsaco_hash_chain_t {
  // Number of symbols through the current chain word.
  uint32_t symbol_count;
  // Partial little-endian chain word.
  uint32_t word;
  // Bytes accumulated for word.
  uint32_t byte_count;
  // Whether the final chain terminator has been encountered.
  bool terminated;
} iree_hal_amd_gpu_hsaco_hash_chain_t;

static iree_status_t iree_hal_amd_gpu_hsaco_scan_hash_chain(
    void* user_data, iree_const_byte_span_t span) {
  iree_hal_amd_gpu_hsaco_hash_chain_t* state = user_data;
  for (iree_host_size_t i = 0; i < span.data_length && !state->terminated;
       ++i) {
    state->word |= (uint32_t)span.data[i] << (8 * state->byte_count++);
    if (state->byte_count == 4) {
      if (state->symbol_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "HSACO GNU hash symbol count overflows");
      }
      ++state->symbol_count;
      state->terminated = (state->word & 1) != 0;
      state->word = 0;
      state->byte_count = 0;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_count_gnu_symbols(
    const iree_hal_amd_gpu_hsaco_reader_t* reader,
    const iree_hal_amd_gpu_image_t* image, uint32_t* out_count) {
  uint64_t source_offset = 0;
  uint64_t available = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_source_for_address(
      reader, image, reader->dynamic.gnu_hash, 16, &source_offset, &available));
  uint8_t header[16];
  IREE_RETURN_IF_ERROR(
      iree_byte_sequence_read(reader->source, source_offset,
                              iree_make_byte_span(header, sizeof(header))));
  uint32_t bucket_count = iree_unaligned_load_le_u32(header);
  uint32_t symbol_offset = iree_unaligned_load_le_u32(header + 4);
  uint32_t bloom_count = iree_unaligned_load_le_u32(header + 8);
  uint64_t bucket_offset = 16 + (uint64_t)bloom_count * 8;
  uint64_t bucket_length = (uint64_t)bucket_count * 4;
  uint64_t chain_offset = bucket_offset + bucket_length;
  if (!bucket_count || !symbol_offset || !bloom_count ||
      (bloom_count & (bloom_count - 1)) || chain_offset > available) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO GNU hash header is invalid");
  }
  iree_const_byte_span_t buckets = iree_const_byte_span_empty();
  void* bucket_storage = NULL;
  iree_status_t status = iree_hal_amd_gpu_hsaco_read_blob(
      reader, source_offset + bucket_offset, bucket_length, &buckets,
      &bucket_storage);
  uint32_t last_symbol = 0;
  for (uint32_t i = 0; i < bucket_count && iree_status_is_ok(status); ++i) {
    uint32_t symbol =
        iree_unaligned_load_le_u32(buckets.data + (uint64_t)i * 4);
    if (symbol && symbol < symbol_offset) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO GNU hash bucket precedes its symbols");
    }
    last_symbol = iree_max(last_symbol, symbol);
  }
  iree_allocator_free(reader->host_allocator, bucket_storage);
  IREE_RETURN_IF_ERROR(status);
  if (!last_symbol) {
    *out_count = symbol_offset;
    return iree_ok_status();
  }
  chain_offset += (uint64_t)(last_symbol - symbol_offset) * 4;
  if (chain_offset > available || available - chain_offset < 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO GNU hash bucket exceeds its chain table");
  }
  iree_hal_amd_gpu_hsaco_hash_chain_t state = {.symbol_count = last_symbol};
  IREE_RETURN_IF_ERROR(iree_byte_sequence_enumerate_range(
      reader->source, source_offset + chain_offset, available - chain_offset,
      (iree_byte_sequence_segment_callback_t){
          .fn = iree_hal_amd_gpu_hsaco_scan_hash_chain, .user_data = &state}));
  if (!state.terminated) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO GNU hash chain is unterminated");
  }
  *out_count = state.symbol_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_symbol_tables(
    iree_hal_amd_gpu_hsaco_reader_t* reader,
    const iree_hal_amd_gpu_image_t* image) {
  if (reader->dynamic.hash) {
    uint64_t source_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_source_for_address(
        reader, image, reader->dynamic.hash, 8, &source_offset, NULL));
    uint8_t header[8];
    IREE_RETURN_IF_ERROR(
        iree_byte_sequence_read(reader->source, source_offset,
                                iree_make_byte_span(header, sizeof(header))));
    uint32_t bucket_count = iree_unaligned_load_le_u32(header);
    reader->symbol_count = iree_unaligned_load_le_u32(header + 4);
    if (!bucket_count || !reader->symbol_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HSACO System V hash table is empty");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_source_for_address(
        reader, image, reader->dynamic.hash,
        8 + ((uint64_t)bucket_count + reader->symbol_count) * 4, &source_offset,
        NULL));
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_count_gnu_symbols(
        reader, image, &reader->symbol_count));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_read_address_blob(
      reader, image, reader->dynamic.symbols,
      (uint64_t)reader->symbol_count * IREE_HAL_AMD_GPU_HSACO_SYMBOL_SIZE,
      &reader->symbols, &reader->symbol_storage));
  if (reader->dynamic.relocation_length) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_read_address_blob(
        reader, image, reader->dynamic.relocations,
        reader->dynamic.relocation_length, &reader->relocations,
        &reader->relocation_storage));
  }
  return iree_ok_status();
}

// Visits the structurally complete notes, collecting only AMDHSA metadata. The
// first pass counts; the second records borrowed payload spans in caller space.
static iree_status_t iree_hal_amd_gpu_hsaco_collect_notes(
    iree_const_byte_span_t notes, iree_const_byte_span_t* payloads,
    iree_host_size_t* inout_count) {
  iree_host_size_t offset = 0;
  while (offset < notes.data_length) {
    if (notes.data_length - offset < 12) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HSACO note header is truncated");
    }
    const uint8_t* header = notes.data + offset;
    uint32_t name_length = iree_unaligned_load_le_u32(header);
    uint32_t data_length = iree_unaligned_load_le_u32(header + 4);
    uint32_t type = iree_unaligned_load_le_u32(header + 8);
    uint64_t name_padded = ((uint64_t)name_length + 3) & ~UINT64_C(3);
    uint64_t data_padded = ((uint64_t)data_length + 3) & ~UINT64_C(3);
    uint64_t length = 12 + name_padded + data_padded;
    if (length > notes.data_length - offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HSACO note contents are truncated");
    }
    if (type == 32 && name_length == 7 &&
        memcmp(header + 12, "AMDGPU", 7) == 0) {
      if (payloads) {
        payloads[*inout_count] =
            iree_make_const_byte_span(header + 12 + name_padded, data_length);
      }
      ++*inout_count;
    }
    offset += length;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_reflection(
    iree_hal_amd_gpu_hsaco_reader_t* reader, iree_hal_amd_gpu_image_t* image) {
  uint64_t strings_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_source_for_address(
      reader, image, reader->dynamic.strings, reader->dynamic.string_length,
      &strings_offset, NULL));
  uint64_t total_length = reader->dynamic.string_length;
  for (uint32_t i = 0; i < reader->program_count; ++i) {
    const uint8_t* program =
        reader->programs.data +
        (uint64_t)i * IREE_HAL_AMD_GPU_HSACO_PROGRAM_HEADER_SIZE;
    if (iree_unaligned_load_le_u32(program) != 4) {
      continue;
    }
    uint64_t length = iree_unaligned_load_le_u64(program + 32);
    if (length > UINT64_MAX - total_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HSACO reflection storage size overflows");
    }
    total_length += length;
  }
  if (total_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSACO reflection exceeds host address space");
  }
  if (!reader->contiguous.data) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        reader->host_allocator, total_length, &image->reflection_storage));
    IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
        reader->source, strings_offset,
        iree_make_byte_span(image->reflection_storage,
                            reader->dynamic.string_length)));
    reader->strings = iree_make_const_byte_span(image->reflection_storage,
                                                reader->dynamic.string_length);
  } else {
    reader->strings =
        iree_make_const_byte_span(reader->contiguous.data + strings_offset,
                                  reader->dynamic.string_length);
  }
  if (reader->strings.data[0] ||
      reader->strings.data[reader->strings.data_length - 1]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO dynamic string table lacks NUL boundaries");
  }
  // One small descriptor per program header; most objects use one note segment.
  iree_host_size_t note_storage_length = 0;
  if (!iree_host_size_checked_mul(reader->program_count,
                                  sizeof(iree_const_byte_span_t),
                                  &note_storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSACO note directory exceeds host address space");
  }
  iree_const_byte_span_t* notes = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      reader->host_allocator, note_storage_length, (void**)&notes));
  iree_status_t status = iree_ok_status();
  iree_host_size_t note_count = 0;
  iree_host_size_t payload_count = 0;
  iree_host_size_t storage_offset = reader->strings.data_length;
  for (uint32_t i = 0; i < reader->program_count && iree_status_is_ok(status);
       ++i) {
    const uint8_t* program =
        reader->programs.data +
        (uint64_t)i * IREE_HAL_AMD_GPU_HSACO_PROGRAM_HEADER_SIZE;
    if (iree_unaligned_load_le_u32(program) != 4) {
      continue;
    }
    uint64_t offset = iree_unaligned_load_le_u64(program + 8);
    iree_host_size_t length = iree_unaligned_load_le_u64(program + 32);
    if (reader->contiguous.data) {
      notes[note_count] =
          iree_make_const_byte_span(reader->contiguous.data + offset, length);
    } else {
      uint8_t* target = (uint8_t*)image->reflection_storage + storage_offset;
      status = iree_byte_sequence_read(reader->source, offset,
                                       iree_make_byte_span(target, length));
      notes[note_count] = iree_make_const_byte_span(target, length);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_hsaco_collect_notes(notes[note_count], NULL,
                                                    &payload_count);
    }
    storage_offset += length;
    ++note_count;
  }
  iree_const_byte_span_t* payloads = NULL;
  iree_host_size_t payload_storage_length = 0;
  if (iree_status_is_ok(status)) {
    if (!iree_host_size_checked_mul(payload_count, sizeof(*payloads),
                                    &payload_storage_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "HSACO metadata directory size overflows");
    } else if (!payload_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO has no AMDHSA metadata note");
    } else {
      status = iree_allocator_malloc(reader->host_allocator,
                                     payload_storage_length, (void**)&payloads);
    }
  }
  iree_host_size_t next_payload = 0;
  for (iree_host_size_t i = 0; i < note_count && iree_status_is_ok(status);
       ++i) {
    status =
        iree_hal_amd_gpu_hsaco_collect_notes(notes[i], payloads, &next_payload);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_metadata_initialize(
        payload_count, payloads, reader->host_allocator, &image->metadata);
  }
  iree_allocator_free(reader->host_allocator, payloads);
  iree_allocator_free(reader->host_allocator, notes);
  return status;
}

static iree_status_t iree_hal_amd_gpu_hsaco_allocate_records(
    iree_hal_amd_gpu_hsaco_reader_t* reader, iree_hal_amd_gpu_image_t* image) {
  const iree_host_size_t counts[] = {
      image->segment_count,
      reader->symbol_count,
      reader->symbol_count,
      reader->relocations.data_length / IREE_HAL_AMD_GPU_HSACO_RELOCATION_SIZE,
      reader->dynamic.dependency_count,
      image->metadata.kernel_count,
  };
  const iree_host_size_t sizes[] = {
      sizeof(*image->segments),     sizeof(*image->symbols),
      sizeof(*image->imports),      sizeof(*image->relocations),
      sizeof(*image->dependencies), sizeof(*image->kernels),
  };
  iree_host_size_t offsets[IREE_ARRAYSIZE(counts)] = {0};
  iree_host_size_t total_length = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(counts); ++i) {
    iree_host_size_t byte_length = 0;
    if (!iree_host_size_checked_align(total_length, 8, &offsets[i]) ||
        !iree_host_size_checked_mul(counts[i], sizes[i], &byte_length) ||
        !iree_host_size_checked_add(offsets[i], byte_length, &total_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HSACO image records exceed host address space");
    }
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(reader->host_allocator,
                                             total_length, &image->storage));
  uint8_t* storage = image->storage;
  image->segments = (iree_hal_amd_gpu_image_segment_t*)(storage + offsets[0]);
  memcpy(image->segments, reader->segments,
         image->segment_count * sizeof(*image->segments));
  image->symbols = (iree_hal_amd_gpu_image_symbol_t*)(storage + offsets[1]);
  image->imports = (iree_hal_amd_gpu_image_import_t*)(storage + offsets[2]);
  image->relocations =
      (iree_hal_amd_gpu_image_relocation_t*)(storage + offsets[3]);
  image->dependencies = (iree_string_view_t*)(storage + offsets[4]);
  image->kernels = (iree_hal_amd_gpu_image_kernel_t*)(storage + offsets[5]);
  iree_host_size_t address_length = 0;
  if (!iree_host_size_checked_mul(reader->symbol_count,
                                  sizeof(*reader->symbol_addresses),
                                  &address_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HSACO symbol references exceed host address space");
  }
  return iree_allocator_malloc(reader->host_allocator, address_length,
                               (void**)&reader->symbol_addresses);
}

// Omitted feature suffixes impose no constraint. LLVM also uses ANY flags for
// processors without that feature; equality of the raw enum values is therefore
// stronger than agreement of the executable requirements.
static bool iree_hal_amd_gpu_hsaco_combine_feature_requirement(
    iree_hal_amd_gpu_target_feature_state_t metadata,
    iree_hal_amd_gpu_target_feature_state_t* requirement) {
  if (*requirement == IREE_HAL_AMD_GPU_TARGET_FEATURE_STATE_ANY) {
    *requirement = metadata;
    return true;
  }
  if (metadata == IREE_HAL_AMD_GPU_TARGET_FEATURE_STATE_ANY) {
    return true;
  }
  return (*requirement == IREE_HAL_AMD_GPU_TARGET_FEATURE_STATE_ON) ==
         (metadata == IREE_HAL_AMD_GPU_TARGET_FEATURE_STATE_ON);
}

static iree_status_t iree_hal_amd_gpu_hsaco_validate_metadata_target(
    iree_hal_amd_gpu_image_t* image) {
  uint32_t minor_version = iree_min(image->code_object_version - 3, 2u);
  if (image->metadata.version[0] != 1 ||
      image->metadata.version[1] != minor_version) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO V%u requires metadata 1.%u",
                            image->code_object_version, minor_version);
  }
  if (iree_string_view_is_empty(image->metadata.target)) {
    return iree_ok_status();
  }
  iree_hal_amd_gpu_target_identity_t metadata_target;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_parse_hsa_isa_name(
      image->metadata.target, &metadata_target));
  if (!iree_string_view_equal(metadata_target.processor,
                              image->target.processor) ||
      !iree_hal_amd_gpu_hsaco_combine_feature_requirement(
          metadata_target.amdhsa_features.sramecc,
          &image->target.amdhsa_features.sramecc) ||
      !iree_hal_amd_gpu_hsaco_combine_feature_requirement(
          metadata_target.amdhsa_features.xnack,
          &image->target.amdhsa_features.xnack)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSACO metadata target `%.*s` contradicts ELF target "
        "`%.*s`",
        (int)image->metadata.target.size, image->metadata.target.data,
        (int)image->target.processor.size, image->target.processor.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_string(
    iree_const_byte_span_t strings, uint64_t offset,
    iree_string_view_t* out_string) {
  if (offset >= strings.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO string offset exceeds its table");
  }
  const char* begin = (const char*)strings.data + offset;
  const char* end = memchr(begin, 0, strings.data_length - offset);
  // The admitted string table ends in NUL, so every in-range offset terminates.
  *out_string = iree_make_string_view(begin, end - begin);
  return iree_ok_status();
}

static int iree_hal_amd_gpu_hsaco_compare_symbols(const void* lhs,
                                                  const void* rhs) {
  return iree_string_view_compare(
      ((const iree_hal_amd_gpu_image_symbol_t*)lhs)->name,
      ((const iree_hal_amd_gpu_image_symbol_t*)rhs)->name);
}

static iree_status_t iree_hal_amd_gpu_hsaco_decode_symbols(
    iree_hal_amd_gpu_hsaco_reader_t* reader, iree_hal_amd_gpu_image_t* image) {
  static const uint8_t null_symbol[IREE_HAL_AMD_GPU_HSACO_SYMBOL_SIZE] = {0};
  if (memcmp(reader->symbols.data, null_symbol, sizeof(null_symbol))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO dynamic symbol zero is not null");
  }
  reader->symbol_addresses[0].kind = IREE_HAL_AMD_GPU_IMAGE_ADDRESS_ABSOLUTE;
  for (uint32_t i = 1; i < reader->symbol_count; ++i) {
    const uint8_t* data =
        reader->symbols.data + (uint64_t)i * IREE_HAL_AMD_GPU_HSACO_SYMBOL_SIZE;
    uint32_t name_offset = iree_unaligned_load_le_u32(data);
    uint32_t type = data[4] & 0xf;
    uint32_t binding = data[4] >> 4;
    uint16_t section = iree_unaligned_load_le_u16(data + 6);
    uint64_t value = iree_unaligned_load_le_u64(data + 8);
    uint64_t length = iree_unaligned_load_le_u64(data + 16);
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_hsaco_string(reader->strings, name_offset, &name));
    // Reserved section indices have their own address spaces or link-time
    // allocation semantics (including AMDGPU LDS and COMMON). Only SHN_ABS is
    // a runtime address here; ordinary definitions use linked virtual
    // addresses.
    if (type > 4 || binding > 2 || (section >= 0xff00 && section != 0xfff1)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "HSACO symbol `%.*s` requires unsupported "
                              "type %u, binding %u, or section 0x%x",
                              (int)name.size, name.data, type, binding,
                              section);
    }
    iree_hal_amd_gpu_image_symbol_kind_t kind =
        type == 1   ? IREE_HAL_AMD_GPU_IMAGE_SYMBOL_DATA
        : type == 2 ? IREE_HAL_AMD_GPU_IMAGE_SYMBOL_FUNCTION
                    : IREE_HAL_AMD_GPU_IMAGE_SYMBOL_UNSPECIFIED;
    iree_hal_amd_gpu_image_address_t address = {0};
    iree_hal_amd_gpu_image_access_t access = 0;
    if (!section) {
      if (!binding || !name.size) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO import has no external name/binding");
      }
      address.kind = IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMPORT;
      address.import_ordinal = (uint32_t)image->import_count;
      image->imports[image->import_count++] = (iree_hal_amd_gpu_image_import_t){
          .name = name, .kind = kind, .weak = binding == 2};
    } else if (section == 0xfff1) {  // SHN_ABS.
      address.kind = IREE_HAL_AMD_GPU_IMAGE_ADDRESS_ABSOLUTE;
      address.offset = value;
    } else {
      uint32_t segment_ordinal = 0;
      if (value < reader->virtual_base ||
          !iree_hal_amd_gpu_hsaco_find_segment(
              image, value - reader->virtual_base, length, false,
              &segment_ordinal)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO symbol `%.*s` exceeds loadable memory",
                                (int)name.size, name.data);
      }
      if (type == 2 && !(image->segments[segment_ordinal].access &
                         IREE_HAL_AMD_GPU_IMAGE_ACCESS_EXECUTE)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO function `%.*s` is not executable",
                                (int)name.size, name.data);
      }
      address.kind = IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMAGE;
      address.offset = value - reader->virtual_base;
      access = image->segments[segment_ordinal].access;
    }
    reader->symbol_addresses[i] = address;
    if (name.size && binding && type <= 2) {
      image->symbols[image->symbol_count++] = (iree_hal_amd_gpu_image_symbol_t){
          .name = name,
          .address = address,
          .length = length,
          .kind = kind,
          .visibility =
              (iree_hal_amd_gpu_image_symbol_visibility_t)(data[5] & 3),
          .access = access,
          .weak = binding == 2,
      };
    }
  }
  qsort(image->symbols, image->symbol_count, sizeof(*image->symbols),
        iree_hal_amd_gpu_hsaco_compare_symbols);
  for (iree_host_size_t i = 1; i < image->symbol_count; ++i) {
    if (iree_string_view_equal(image->symbols[i - 1].name,
                               image->symbols[i].name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "HSACO symbol `%.*s` is ambiguous",
          (int)image->symbols[i].name.size, image->symbols[i].name.data);
    }
  }
  for (iree_host_size_t offset = 0;
       offset < reader->dynamic_entries.data_length;
       offset += IREE_HAL_AMD_GPU_HSACO_DYNAMIC_ENTRY_SIZE) {
    const uint8_t* data = reader->dynamic_entries.data + offset;
    uint64_t tag = iree_unaligned_load_le_u64(data);
    if (!tag) {
      break;
    }
    if (tag == 1) {
      iree_string_view_t name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_hsaco_string(
          reader->strings, iree_unaligned_load_le_u64(data + 8), &name));
      if (!name.size) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO dependency name is empty");
      }
      image->dependencies[image->dependency_count++] = name;
    }
  }
  return iree_ok_status();
}

static int iree_hal_amd_gpu_hsaco_compare_relocations(const void* lhs,
                                                      const void* rhs) {
  const iree_hal_amd_gpu_image_relocation_t* a = lhs;
  const iree_hal_amd_gpu_image_relocation_t* b = rhs;
  if (a->segment_ordinal != b->segment_ordinal) {
    return a->segment_ordinal < b->segment_ordinal ? -1 : 1;
  }
  return a->offset < b->offset ? -1 : a->offset > b->offset ? 1 : 0;
}

static uint64_t iree_hal_amd_gpu_hsaco_relocation_width(
    const iree_hal_amd_gpu_image_relocation_t* relocation) {
  return relocation->kind == IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_64 ? 8
                                                                           : 4;
}

static iree_status_t iree_hal_amd_gpu_hsaco_decode_relocations(
    const iree_hal_amd_gpu_hsaco_reader_t* reader,
    iree_hal_amd_gpu_image_t* image) {
  for (iree_host_size_t offset = 0; offset < reader->relocations.data_length;
       offset += IREE_HAL_AMD_GPU_HSACO_RELOCATION_SIZE) {
    const uint8_t* data = reader->relocations.data + offset;
    uint64_t address = iree_unaligned_load_le_u64(data);
    uint64_t info = iree_unaligned_load_le_u64(data + 8);
    uint64_t addend = iree_unaligned_load_le_u64(data + 16);
    uint32_t type = (uint32_t)info;
    uint32_t symbol = (uint32_t)(info >> 32);
    if (type == 0) {
      continue;
    }
    iree_hal_amd_gpu_image_relocation_t relocation = {0};
    uint64_t width = 4;
    switch (type) {
      case 1:  // R_AMDGPU_ABS32_LO.
        relocation.kind = IREE_HAL_AMD_GPU_IMAGE_RELOCATION_LOW_32;
        break;
      case 2:  // R_AMDGPU_ABS32_HI.
        relocation.kind = IREE_HAL_AMD_GPU_IMAGE_RELOCATION_HIGH_32;
        break;
      case 3:   // R_AMDGPU_ABS64.
      case 13:  // R_AMDGPU_RELATIVE64.
        relocation.kind = IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_64;
        width = 8;
        break;
      case 6:  // R_AMDGPU_ABS32.
        relocation.kind = IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_32;
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "HSACO dynamic relocation type %u is not a "
                                "supported runtime relocation",
                                type);
    }
    if (address < reader->virtual_base ||
        !iree_hal_amd_gpu_hsaco_find_segment(
            image, address - reader->virtual_base, width, false,
            &relocation.segment_ordinal)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HSACO relocation exceeds loadable memory");
    }
    relocation.offset = address - reader->virtual_base -
                        image->segments[relocation.segment_ordinal].offset;
    if (type == 13) {
      if (symbol) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO relative relocation names a symbol");
      }
      relocation.value.kind = IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMAGE;
      relocation.value.offset = addend - reader->virtual_base;
    } else {
      if (symbol >= reader->symbol_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HSACO relocation symbol exceeds its table");
      }
      relocation.value = reader->symbol_addresses[symbol];
      relocation.value.offset += addend;
    }
    image->relocations[image->relocation_count++] = relocation;
  }
  qsort(image->relocations, image->relocation_count,
        sizeof(*image->relocations),
        iree_hal_amd_gpu_hsaco_compare_relocations);
  for (iree_host_size_t i = 1; i < image->relocation_count; ++i) {
    const iree_hal_amd_gpu_image_relocation_t* previous =
        &image->relocations[i - 1];
    const iree_hal_amd_gpu_image_relocation_t* current = &image->relocations[i];
    if (previous->segment_ordinal == current->segment_ordinal &&
        previous->offset + iree_hal_amd_gpu_hsaco_relocation_width(previous) >
            current->offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HSACO dynamic relocations overlap");
    }
  }
  return iree_ok_status();
}

// Gather descriptors in source order in one enumeration, independent of the
// metadata's kernel order or the source's segment boundaries.
typedef struct iree_hal_amd_gpu_hsaco_descriptor_t {
  // First descriptor byte in the immutable source sequence.
  uint64_t source_offset;
  // Kernel metadata/dispatch ordinal receiving the decoded descriptor.
  iree_host_size_t ordinal;
  // Serialized descriptor gathered without retaining callback pointers.
  uint8_t data[IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE];
} iree_hal_amd_gpu_hsaco_descriptor_t;

typedef struct iree_hal_amd_gpu_hsaco_descriptor_read_t {
  // Sorted descriptor gathering records.
  iree_hal_amd_gpu_hsaco_descriptor_t* descriptors;
  // Number of records in descriptors.
  iree_host_size_t count;
  // First record not completely gathered.
  iree_host_size_t next;
  // Logical source offset of the next callback.
  uint64_t source_offset;
} iree_hal_amd_gpu_hsaco_descriptor_read_t;

static int iree_hal_amd_gpu_hsaco_compare_descriptors(const void* lhs,
                                                      const void* rhs) {
  uint64_t a = ((const iree_hal_amd_gpu_hsaco_descriptor_t*)lhs)->source_offset;
  uint64_t b = ((const iree_hal_amd_gpu_hsaco_descriptor_t*)rhs)->source_offset;
  return a < b ? -1 : a > b ? 1 : 0;
}

static iree_status_t iree_hal_amd_gpu_hsaco_read_descriptors(
    void* user_data, iree_const_byte_span_t source) {
  iree_hal_amd_gpu_hsaco_descriptor_read_t* state = user_data;
  uint64_t source_end = state->source_offset + source.data_length;
  for (iree_host_size_t i = state->next; i < state->count; ++i) {
    iree_hal_amd_gpu_hsaco_descriptor_t* descriptor = &state->descriptors[i];
    if (descriptor->source_offset >= source_end) {
      break;
    }
    uint64_t begin = iree_max(state->source_offset, descriptor->source_offset);
    uint64_t end =
        iree_min(source_end, descriptor->source_offset +
                                 IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE);
    if (begin < end) {
      memcpy(descriptor->data + (begin - descriptor->source_offset),
             source.data + (begin - state->source_offset), end - begin);
    }
    if (end ==
        descriptor->source_offset + IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE) {
      state->next = i + 1;
    }
  }
  state->source_offset = source_end;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_decode_descriptor(
    const iree_hal_amd_gpu_hsaco_descriptor_t* descriptor,
    iree_hal_amd_gpu_image_t* image) {
  const uint8_t* data = descriptor->data;
  iree_hal_amd_gpu_image_kernel_t* kernel =
      &image->kernels[descriptor->ordinal];
  const iree_hal_amd_gpu_metadata_kernel_t* metadata =
      &image->metadata.kernels[descriptor->ordinal];
  uint32_t group_size = iree_unaligned_load_le_u32(data);
  uint32_t private_size = iree_unaligned_load_le_u32(data + 4);
  uint32_t kernarg_size = iree_unaligned_load_le_u32(data + 8);
  uint64_t entry_delta = iree_unaligned_load_le_u64(data + 16);
  kernel->entry_offset = kernel->descriptor_offset + entry_delta;
  kernel->compute_pgm_rsrc3 = iree_unaligned_load_le_u32(data + 44);
  kernel->compute_pgm_rsrc1 = iree_unaligned_load_le_u32(data + 48);
  kernel->compute_pgm_rsrc2 = iree_unaligned_load_le_u32(data + 52);
  kernel->code_properties = iree_unaligned_load_le_u16(data + 56);
  kernel->kernarg_preload = iree_unaligned_load_le_u16(data + 58);
  uint32_t entry_segment = 0;
  if ((kernel->entry_offset & 255) ||
      !iree_hal_amd_gpu_hsaco_find_segment(image, kernel->entry_offset, 4, true,
                                           &entry_segment) ||
      !(image->segments[entry_segment].access &
        IREE_HAL_AMD_GPU_IMAGE_ACCESS_EXECUTE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSACO kernel `%.*s` has an invalid instruction entry",
        (int)metadata->name.size, metadata->name.data);
  }
  // A zero descriptor kernarg size means unspecified in the AMDHSA ABI.
  if (group_size != metadata->group_segment_fixed_size ||
      private_size != metadata->private_segment_fixed_size ||
      (kernarg_size && kernarg_size != metadata->kernarg_segment_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSACO kernel `%.*s` descriptor contradicts metadata",
        (int)metadata->name.size, metadata->name.data);
  }
  // Relocations are sorted and disjoint. Only the last field starting before
  // the descriptor end can overlap it; no per-kernel relocation-table scan.
  uint64_t descriptor_end =
      kernel->descriptor_offset + IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE;
  iree_host_size_t low = 0;
  iree_host_size_t high = image->relocation_count;
  while (low < high) {
    iree_host_size_t middle = low + (high - low) / 2;
    const iree_hal_amd_gpu_image_relocation_t* relocation =
        &image->relocations[middle];
    uint64_t offset = image->segments[relocation->segment_ordinal].offset +
                      relocation->offset;
    if (offset < descriptor_end) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low) {
    const iree_hal_amd_gpu_image_relocation_t* relocation =
        &image->relocations[low - 1];
    uint64_t end = image->segments[relocation->segment_ordinal].offset +
                   relocation->offset +
                   iree_hal_amd_gpu_hsaco_relocation_width(relocation);
    if (end > kernel->descriptor_offset) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "HSACO kernel descriptors must be fully linked");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_hsaco_decode_kernels(
    const iree_hal_amd_gpu_hsaco_reader_t* reader,
    iree_hal_amd_gpu_image_t* image) {
  iree_host_size_t count = image->metadata.kernel_count;
  if (!count) {
    return iree_ok_status();
  }
  iree_host_size_t storage_length = 0;
  if (!iree_host_size_checked_mul(count,
                                  sizeof(iree_hal_amd_gpu_hsaco_descriptor_t),
                                  &storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSACO descriptors exceed host address space");
  }
  iree_hal_amd_gpu_hsaco_descriptor_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      reader->host_allocator, storage_length, (void**)&descriptors));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_gpu_metadata_kernel_t* metadata =
        &image->metadata.kernels[i];
    const iree_hal_amd_gpu_image_symbol_t* symbol =
        iree_hal_amd_gpu_image_find_symbol(image, metadata->symbol_name);
    if (!symbol ||
        symbol->address.kind != IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMAGE ||
        (symbol->address.offset & 63) ||
        (symbol->length &&
         symbol->length < IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HSACO kernel `%.*s` has no valid descriptor `%.*s`",
          (int)metadata->name.size, metadata->name.data,
          (int)metadata->symbol_name.size, metadata->symbol_name.data);
      continue;
    }
    image->kernels[i].descriptor_offset = symbol->address.offset;
    descriptors[i].ordinal = i;
    status = iree_hal_amd_gpu_hsaco_source_for_address(
        reader, image, symbol->address.offset + reader->virtual_base,
        IREE_HAL_AMD_GPU_HSACO_DESCRIPTOR_SIZE, &descriptors[i].source_offset,
        NULL);
  }
  if (iree_status_is_ok(status)) {
    qsort(descriptors, count, sizeof(*descriptors),
          iree_hal_amd_gpu_hsaco_compare_descriptors);
    iree_hal_amd_gpu_hsaco_descriptor_read_t state = {
        .descriptors = descriptors, .count = count};
    status = iree_byte_sequence_enumerate(
        reader->source, (iree_byte_sequence_segment_callback_t){
                            .fn = iree_hal_amd_gpu_hsaco_read_descriptors,
                            .user_data = &state});
  }
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_gpu_hsaco_decode_descriptor(&descriptors[i], image);
  }
  iree_allocator_free(reader->host_allocator, descriptors);
  return status;
}

iree_status_t iree_hal_amd_gpu_hsaco_image_initialize(
    iree_byte_sequence_t* source_sequence, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_image_t* out_image) {
  memset(out_image, 0, sizeof(*out_image));
  if (!source_sequence) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HSACO requires an immutable source sequence");
  }
  iree_hal_amd_gpu_hsaco_reader_t reader = {
      .source = source_sequence,
      .host_allocator = host_allocator,
      .source_length = iree_byte_sequence_length(source_sequence),
  };
  iree_byte_sequence_try_get_contiguous_span(source_sequence,
                                             &reader.contiguous);
  iree_hal_amd_gpu_image_t image = {.host_allocator = host_allocator};
  iree_status_t status = iree_hal_amd_gpu_hsaco_read_programs(&reader, &image);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_read_dynamic(&reader);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_read_symbol_tables(&reader, &image);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_read_reflection(&reader, &image);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_validate_metadata_target(&image);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_allocate_records(&reader, &image);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_decode_symbols(&reader, &image);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_decode_relocations(&reader, &image);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_decode_kernels(&reader, &image);
  }
  iree_hal_amd_gpu_hsaco_reader_deinitialize(&reader);
  if (iree_status_is_ok(status)) {
    image.source = source_sequence;
    iree_byte_sequence_retain(image.source);
    *out_image = image;
  } else {
    iree_hal_amd_gpu_image_deinitialize(&image);
  }
  return status;
}
