// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hsaco.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"
#include "loom/target/emit/native/elf.h"

//===----------------------------------------------------------------------===//
// AMDGPU code-object layout constants
//===----------------------------------------------------------------------===//

enum {
  LOOM_AMDGPU_HSACO_SECTION_NOTE = 0,
  LOOM_AMDGPU_HSACO_SECTION_DYNSYM = 1,
  LOOM_AMDGPU_HSACO_SECTION_HASH = 2,
  LOOM_AMDGPU_HSACO_SECTION_DYNSTR = 3,
  LOOM_AMDGPU_HSACO_SECTION_RODATA = 4,
  LOOM_AMDGPU_HSACO_SECTION_TEXT = 5,
  LOOM_AMDGPU_HSACO_SECTION_DATA = 6,
  LOOM_AMDGPU_HSACO_SECTION_DYNAMIC = 7,
  LOOM_AMDGPU_HSACO_SECTION_SYMTAB = 8,
  LOOM_AMDGPU_HSACO_SECTION_STRTAB = 9,
  LOOM_AMDGPU_HSACO_SECTION_COUNT = 10,
};

enum {
  LOOM_AMDGPU_HSACO_DYN_NULL = 0,
  LOOM_AMDGPU_HSACO_DYN_HASH = 4,
  LOOM_AMDGPU_HSACO_DYN_STRTAB = 5,
  LOOM_AMDGPU_HSACO_DYN_SYMTAB = 6,
  LOOM_AMDGPU_HSACO_DYN_STRSZ = 10,
  LOOM_AMDGPU_HSACO_DYN_SYMENT = 11,
};

enum {
  LOOM_AMDGPU_HSACO_SYMBOL_BIND_LOCAL = 0,
  LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL = 1,
};

enum {
  LOOM_AMDGPU_HSACO_SYMBOL_TYPE_NONE = 0,
  LOOM_AMDGPU_HSACO_SYMBOL_TYPE_OBJECT = 1,
  LOOM_AMDGPU_HSACO_SYMBOL_TYPE_FUNCTION = 2,
};

enum {
  LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_DEFAULT = 0,
  LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED = 3,
};

#define LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE 16u
#define LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE 24u
#define LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE UINT64_C(64)
#define LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE UINT64_C(56)
#define LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT UINT64_C(7)
#define LOOM_AMDGPU_HSACO_READ_SEGMENT_BASE \
  (LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE +      \
   LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT * \
       LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE)
#define LOOM_AMDGPU_HSACO_TEXT_SEGMENT_BASE UINT64_C(0x1000)
#define LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT UINT64_C(0x1000)
#define LOOM_AMDGPU_HSACO_NOTE_ALIGNMENT UINT64_C(4)
#define LOOM_AMDGPU_HSACO_TEXT_ALIGNMENT UINT64_C(256)
#define LOOM_AMDGPU_HSACO_DYNAMIC_ENTRY_COUNT 6u

typedef struct loom_amdgpu_hsaco_kernel_layout_t {
  // Byte offset of the kernel entry within the `.text` section.
  uint64_t text_offset;
  // Byte offset of the kernel descriptor within the `.rodata` section.
  uint64_t descriptor_offset;
  // Name offset of the kernel entry symbol in `.dynstr` and `.strtab`.
  uint32_t entry_name_offset;
  // Name offset of the descriptor symbol in `.dynstr` and `.strtab`.
  uint32_t descriptor_name_offset;
} loom_amdgpu_hsaco_kernel_layout_t;

typedef struct loom_amdgpu_hsaco_data_symbol_layout_t {
  // Byte offset of the data symbol within its containing section.
  uint64_t section_offset;
  // Name offset of the data symbol in `.dynstr` and `.strtab`.
  uint32_t name_offset;
  // Logical section containing the data symbol definition.
  iree_host_size_t logical_section_index;
} loom_amdgpu_hsaco_data_symbol_layout_t;

typedef struct loom_amdgpu_hsaco_payloads_t {
  // Arena-backed copied metadata kernel rows.
  loom_amdgpu_metadata_kernel_t* metadata_kernels;
  // Number of entries in |metadata_kernels|.
  iree_host_size_t metadata_kernel_count;
  // Arena-backed per-kernel placement rows.
  loom_amdgpu_hsaco_kernel_layout_t* kernel_layouts;
  // Arena-backed per-data-symbol placement rows.
  loom_amdgpu_hsaco_data_symbol_layout_t* data_symbol_layouts;
  // Compact ELF section descriptors passed to the generic writer.
  loom_native_elf64le_section_t sections[LOOM_AMDGPU_HSACO_SECTION_COUNT];
  // Number of initialized compact entries in |sections|.
  iree_host_size_t section_count;
  // Mapping from logical section enum values to compact |sections| entries.
  iree_host_size_t section_indices[LOOM_AMDGPU_HSACO_SECTION_COUNT];
  // ELF program headers passed to the generic writer.
  loom_native_elf64le_segment_t segments[7];
  // Byte length of the header-backed read segment.
  uint64_t read_segment_file_size;
  // Section alignment for read-only data and kernel descriptors.
  uint64_t rodata_alignment;
  // Byte length of the writable data section, or zero when absent.
  iree_host_size_t writable_data_size;
  // Section alignment for writable data, or one when absent.
  uint64_t writable_data_alignment;
} loom_amdgpu_hsaco_payloads_t;

static uint16_t loom_amdgpu_hsaco_elf_section_index(iree_host_size_t index) {
  return (uint16_t)(index + 1u);
}

static void loom_amdgpu_hsaco_initialize_payload_sections(
    loom_amdgpu_hsaco_payloads_t* payloads) {
  payloads->section_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(payloads->section_indices);
       ++i) {
    payloads->section_indices[i] = IREE_HOST_SIZE_MAX;
  }
}

static void loom_amdgpu_hsaco_add_section(
    loom_amdgpu_hsaco_payloads_t* payloads, iree_host_size_t logical_index,
    loom_native_elf64le_section_t section) {
  const iree_host_size_t physical_index = payloads->section_count++;
  payloads->sections[physical_index] = section;
  payloads->section_indices[logical_index] = physical_index;
}

static const loom_native_elf64le_section_t* loom_amdgpu_hsaco_section(
    const loom_amdgpu_hsaco_payloads_t* payloads,
    iree_host_size_t logical_index) {
  return &payloads->sections[payloads->section_indices[logical_index]];
}

static loom_native_elf64le_section_t* loom_amdgpu_hsaco_mutable_section(
    loom_amdgpu_hsaco_payloads_t* payloads, iree_host_size_t logical_index) {
  return &payloads->sections[payloads->section_indices[logical_index]];
}

static uint16_t loom_amdgpu_hsaco_logical_section_index(
    const loom_amdgpu_hsaco_payloads_t* payloads,
    iree_host_size_t logical_index) {
  return loom_amdgpu_hsaco_elf_section_index(
      payloads->section_indices[logical_index]);
}

static iree_status_t loom_amdgpu_hsaco_align_uint64(uint64_t value,
                                                    uint64_t alignment,
                                                    uint64_t* out_result) {
  if (iree_is_power_of_two_uint64(alignment) &&
      iree_checked_align_u64(value, alignment, out_result)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU HSACO layout alignment overflow");
}

static iree_status_t loom_amdgpu_hsaco_cast_host_size(
    uint64_t value, iree_host_size_t* out_value) {
  if (value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO payload exceeds host size capacity");
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_allocate_zeroed(
    iree_arena_allocator_t* arena, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  *out_span = iree_make_byte_span(NULL, 0);
  if (length == 0) {
    return iree_ok_status();
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, (void**)&data));
  memset(data, 0, length);
  *out_span = iree_make_byte_span(data, length);
  return iree_ok_status();
}

static uint64_t loom_amdgpu_hsaco_data_symbol_alignment(
    const loom_amdgpu_hsaco_data_symbol_t* symbol) {
  return symbol->alignment == 0 ? 1u : symbol->alignment;
}

static bool loom_amdgpu_hsaco_data_symbol_is_writable(
    const loom_amdgpu_hsaco_data_symbol_t* symbol) {
  return iree_any_bit_set(symbol->flags,
                          LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE);
}

static const loom_amdgpu_hsaco_data_symbol_t*
loom_amdgpu_hsaco_find_data_symbol(const loom_amdgpu_hsaco_file_t* file,
                                   iree_string_view_t name,
                                   iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    if (iree_string_view_equal(file->data_symbols[i].name, name)) {
      *out_index = i;
      return &file->data_symbols[i];
    }
  }
  *out_index = IREE_HOST_SIZE_MAX;
  return NULL;
}

static iree_status_t loom_amdgpu_hsaco_append_u32(
    iree_byte_span_t target, iree_host_size_t* inout_offset, uint32_t value) {
  if (*inout_offset > target.data_length ||
      target.data_length - *inout_offset < 4u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO u32 payload write overflow");
  }
  uint8_t* data = target.data + *inout_offset;
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
  *inout_offset += 4u;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_store_u32(iree_byte_span_t target,
                                                 iree_host_size_t byte_offset,
                                                 uint32_t value) {
  if (byte_offset > target.data_length ||
      target.data_length - byte_offset < 4u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO u32 payload store overflow");
  }
  uint8_t* data = target.data + byte_offset;
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_append_u64(
    iree_byte_span_t target, iree_host_size_t* inout_offset, uint64_t value) {
  if (*inout_offset > target.data_length ||
      target.data_length - *inout_offset < 8u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO u64 payload write overflow");
  }
  uint8_t* data = target.data + *inout_offset;
  for (iree_host_size_t i = 0; i < 8u; ++i) {
    data[i] = (uint8_t)(value >> (i * 8u));
  }
  *inout_offset += 8u;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_write_symbol(
    iree_byte_span_t target, iree_host_size_t symbol_index,
    uint32_t name_offset, uint8_t binding, uint8_t type, uint8_t visibility,
    uint16_t section_index, uint64_t value, uint64_t size) {
  const iree_host_size_t byte_offset =
      symbol_index * LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE;
  if (symbol_index > target.data_length / LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE ||
      target.data_length - byte_offset < LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol table write overflow");
  }
  uint8_t* entry = target.data + byte_offset;
  entry[0] = (uint8_t)name_offset;
  entry[1] = (uint8_t)(name_offset >> 8);
  entry[2] = (uint8_t)(name_offset >> 16);
  entry[3] = (uint8_t)(name_offset >> 24);
  entry[4] = (uint8_t)((binding << 4u) | type);
  entry[5] = visibility;
  entry[6] = (uint8_t)section_index;
  entry[7] = (uint8_t)(section_index >> 8);
  for (iree_host_size_t i = 0; i < 8u; ++i) {
    entry[8 + i] = (uint8_t)(value >> (i * 8u));
    entry[16 + i] = (uint8_t)(size >> (i * 8u));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hsaco_elf_hash(iree_string_view_t name) {
  uint32_t hash = 0;
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    hash = (hash << 4u) + (uint8_t)name.data[i];
    const uint32_t high = hash & UINT32_C(0xf0000000);
    if (high != 0) {
      hash ^= high >> 24u;
      hash &= ~high;
    }
  }
  return hash;
}

static bool loom_amdgpu_hsaco_symbol_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_hsaco_symbol_continue_char(char c) {
  return loom_amdgpu_hsaco_symbol_start_char(c) || (c >= '0' && c <= '9') ||
         c == '.';
}

static iree_status_t loom_amdgpu_hsaco_validate_symbol(
    iree_string_view_t symbol, iree_string_view_t field_name) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO symbol field '%.*s' is required",
                            (int)field_name.size, field_name.data);
  }
  if (!loom_amdgpu_hsaco_symbol_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HSACO symbol field '%.*s' has invalid first character '%c'",
        (int)field_name.size, field_name.data, symbol.data[0]);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_hsaco_symbol_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO symbol field '%.*s' contains invalid character '%c'",
          (int)field_name.size, field_name.data, symbol.data[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_validate_target_id(
    iree_string_view_t target, iree_string_view_t processor_name) {
  if (iree_string_view_is_empty(processor_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO processor is required");
  }
  loom_amdgpu_amdhsa_target_id_t target_id = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_parse_amdhsa_target_id(target, &target_id));
  if (!iree_string_view_equal(target_id.processor->name, processor_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HSACO target id '%.*s' selects processor '%.*s' but file "
        "processor is '%.*s'",
        (int)target.size, target.data, (int)target_id.processor->name.size,
        target_id.processor->name.data, (int)processor_name.size,
        processor_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_validate_file(
    const loom_amdgpu_hsaco_file_t* file) {
  if (file == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO file description is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_validate_target_id(file->target, file->processor));
  if (file->kernel_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO requires at least one kernel");
  }
  if (file->kernels == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO kernel array is required");
  }
  if (file->data_symbol_count != 0 && file->data_symbols == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO data symbol array is required");
  }
  iree_host_size_t dynamic_symbol_count = 0;
  if (!iree_host_size_checked_mul(file->kernel_count, 2u,
                                  &dynamic_symbol_count) ||
      !iree_host_size_checked_add(dynamic_symbol_count, file->data_symbol_count,
                                  &dynamic_symbol_count) ||
      !iree_host_size_checked_add(dynamic_symbol_count, 1u,
                                  &dynamic_symbol_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO dynamic symbol count overflow");
  }
  if (dynamic_symbol_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO dynamic symbol count exceeds "
                            "32-bit ELF capacity");
  }
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_validate_symbol(
        kernel->metadata.name, IREE_SV("kernel.name")));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hsaco_validate_symbol(kernel->metadata.descriptor_symbol,
                                          IREE_SV("kernel.descriptor_symbol")));
    if (kernel->text.data == NULL || kernel->text.data_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU HSACO kernel '%.*s' has no text bytes",
                              (int)kernel->metadata.name.size,
                              kernel->metadata.name.data);
    }
    if (kernel->text_fixup_count != 0 && kernel->text_fixups == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO kernel '%.*s' has text fixups but no fixup array",
          (int)kernel->metadata.name.size, kernel->metadata.name.data);
    }
    if (iree_string_view_equal(kernel->metadata.name,
                               kernel->metadata.descriptor_symbol)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO kernel '%.*s' reuses its entry symbol as descriptor "
          "symbol",
          (int)kernel->metadata.name.size, kernel->metadata.name.data);
    }
    for (iree_host_size_t j = i + 1u; j < file->kernel_count; ++j) {
      const loom_amdgpu_hsaco_kernel_t* other_kernel = &file->kernels[j];
      if (iree_string_view_equal(kernel->metadata.name,
                                 other_kernel->metadata.name) ||
          iree_string_view_equal(kernel->metadata.name,
                                 other_kernel->metadata.descriptor_symbol) ||
          iree_string_view_equal(kernel->metadata.descriptor_symbol,
                                 other_kernel->metadata.name) ||
          iree_string_view_equal(kernel->metadata.descriptor_symbol,
                                 other_kernel->metadata.descriptor_symbol)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU HSACO symbol '%.*s' conflicts with another kernel symbol",
            (int)kernel->metadata.name.size, kernel->metadata.name.data);
      }
    }
    for (iree_host_size_t j = 0; j < kernel->text_fixup_count; ++j) {
      const loom_amdgpu_hsaco_text_fixup_t* fixup = &kernel->text_fixups[j];
      switch (fixup->kind) {
        case LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO:
        case LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_HI:
          break;
        case LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_NONE:
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU HSACO kernel '%.*s' text fixup %" PRIhsz
              " has an unsupported kind %d",
              (int)kernel->metadata.name.size, kernel->metadata.name.data, j,
              (int)fixup->kind);
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_validate_symbol(
          fixup->target_symbol, IREE_SV("kernel.text_fixup.target_symbol")));
      if (fixup->literal_byte_offset > kernel->text.data_length ||
          kernel->text.data_length - fixup->literal_byte_offset < 4u) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU HSACO kernel '%.*s' text fixup %" PRIhsz
                                " literal offset is outside the text payload",
                                (int)kernel->metadata.name.size,
                                kernel->metadata.name.data, j);
      }
      if (fixup->base_pc_byte_offset > kernel->text.data_length) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU HSACO kernel '%.*s' text fixup %" PRIhsz
                                " base PC offset is outside the text payload",
                                (int)kernel->metadata.name.size,
                                kernel->metadata.name.data, j);
      }
    }
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_validate_symbol(
        symbol->name, IREE_SV("data_symbol.name")));
    const uint64_t alignment = symbol->alignment == 0 ? 1u : symbol->alignment;
    if (!iree_is_power_of_two_uint64(alignment)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO data symbol '%.*s' alignment must be a power of two",
          (int)symbol->name.size, symbol->name.data);
    }
    if (symbol->initial_contents.data_length > symbol->byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO data symbol '%.*s' initial contents exceed byte length",
          (int)symbol->name.size, symbol->name.data);
    }
    if (symbol->initial_contents.data == NULL &&
        symbol->initial_contents.data_length != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO data symbol '%.*s' has initial contents length but no "
          "data",
          (int)symbol->name.size, symbol->name.data);
    }
    if (loom_amdgpu_hsaco_data_symbol_is_writable(symbol) &&
        symbol->byte_length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO writable data symbol '%.*s' must have storage",
          (int)symbol->name.size, symbol->name.data);
    }
    for (iree_host_size_t j = 0; j < file->kernel_count; ++j) {
      const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[j];
      if (iree_string_view_equal(symbol->name, kernel->metadata.name) ||
          iree_string_view_equal(symbol->name,
                                 kernel->metadata.descriptor_symbol)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU HSACO data symbol '%.*s' conflicts with a kernel symbol",
            (int)symbol->name.size, symbol->name.data);
      }
    }
    for (iree_host_size_t j = i + 1u; j < file->data_symbol_count; ++j) {
      if (iree_string_view_equal(symbol->name, file->data_symbols[j].name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU HSACO data symbol '%.*s' conflicts with another data "
            "symbol",
            (int)symbol->name.size, symbol->name.data);
      }
    }
  }
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    for (iree_host_size_t j = 0; j < kernel->text_fixup_count; ++j) {
      const loom_amdgpu_hsaco_text_fixup_t* fixup = &kernel->text_fixups[j];
      iree_host_size_t symbol_index = IREE_HOST_SIZE_MAX;
      const loom_amdgpu_hsaco_data_symbol_t* symbol =
          loom_amdgpu_hsaco_find_data_symbol(file, fixup->target_symbol,
                                             &symbol_index);
      if (symbol == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU HSACO kernel '%.*s' text fixup %" PRIhsz
            " references unknown data symbol '%.*s'",
            (int)kernel->metadata.name.size, kernel->metadata.name.data, j,
            (int)fixup->target_symbol.size, fixup->target_symbol.data);
      }
      if (fixup->target_symbol_byte_offset > symbol->byte_length) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU HSACO kernel '%.*s' text fixup %" PRIhsz
            " targets byte %" PRIu64 " beyond data symbol '%.*s'",
            (int)kernel->metadata.name.size, kernel->metadata.name.data, j,
            fixup->target_symbol_byte_offset, (int)symbol->name.size,
            symbol->name.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_copy_metadata_kernels(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, iree_arena_allocator_t* arena) {
  payloads->metadata_kernel_count = file->kernel_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, file->kernel_count, sizeof(payloads->metadata_kernels[0]),
      (void**)&payloads->metadata_kernels));
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    payloads->metadata_kernels[i] = file->kernels[i].metadata;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_note(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_note, iree_arena_allocator_t* arena) {
  *out_note = iree_make_const_byte_span(NULL, 0);

  const loom_amdgpu_code_object_metadata_t metadata = {
      .target = file->target,
      .kernels = payloads->metadata_kernels,
      .kernel_count = payloads->metadata_kernel_count,
  };

  iree_string_builder_t measure_builder;
  iree_string_builder_initialize(iree_allocator_null(), &measure_builder);
  iree_status_t status =
      loom_amdgpu_metadata_append_elf_note(&metadata, &measure_builder);
  const iree_host_size_t note_size =
      iree_status_is_ok(status) ? iree_string_builder_size(&measure_builder)
                                : 0;
  iree_string_builder_deinitialize(&measure_builder);
  IREE_RETURN_IF_ERROR(status);

  uint8_t* note_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, note_size + 1u, (void**)&note_data));
  iree_string_builder_t note_builder;
  iree_string_builder_initialize_with_storage((char*)note_data, note_size + 1u,
                                              &note_builder);
  status = loom_amdgpu_metadata_append_elf_note(&metadata, &note_builder);
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&note_builder) != note_size) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HSACO metadata note size changed between measure and emit");
  }
  iree_string_builder_deinitialize(&note_builder);
  if (iree_status_is_ok(status)) {
    *out_note = iree_make_const_byte_span(note_data, note_size);
  }
  return status;
}

static iree_status_t loom_amdgpu_hsaco_build_string_tables(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_string_table, iree_arena_allocator_t* arena) {
  *out_string_table = iree_make_const_byte_span(NULL, 0);

  iree_host_size_t string_table_size = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    iree_host_size_t entry_name_size = 0;
    iree_host_size_t descriptor_name_size = 0;
    if (!iree_host_size_checked_add(kernel->metadata.name.size, 1u,
                                    &entry_name_size) ||
        !iree_host_size_checked_add(kernel->metadata.descriptor_symbol.size, 1u,
                                    &descriptor_name_size) ||
        !iree_host_size_checked_add(string_table_size, entry_name_size,
                                    &string_table_size) ||
        !iree_host_size_checked_add(string_table_size, descriptor_name_size,
                                    &string_table_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO string table size overflow");
    }
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    iree_host_size_t symbol_name_size = 0;
    if (!iree_host_size_checked_add(symbol->name.size, 1u, &symbol_name_size) ||
        !iree_host_size_checked_add(string_table_size, symbol_name_size,
                                    &string_table_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO string table size overflow");
    }
  }
  if (string_table_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO string table exceeds ELF32 name "
                            "offset capacity");
  }

  iree_byte_span_t string_table = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena, string_table_size, &string_table));
  iree_host_size_t string_table_offset = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    loom_amdgpu_hsaco_kernel_layout_t* layout = &payloads->kernel_layouts[i];
    layout->entry_name_offset = (uint32_t)string_table_offset;
    memcpy(string_table.data + string_table_offset, kernel->metadata.name.data,
           kernel->metadata.name.size);
    string_table_offset += kernel->metadata.name.size + 1u;
    layout->descriptor_name_offset = (uint32_t)string_table_offset;
    memcpy(string_table.data + string_table_offset,
           kernel->metadata.descriptor_symbol.data,
           kernel->metadata.descriptor_symbol.size);
    string_table_offset += kernel->metadata.descriptor_symbol.size + 1u;
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    loom_amdgpu_hsaco_data_symbol_layout_t* layout =
        &payloads->data_symbol_layouts[i];
    layout->name_offset = (uint32_t)string_table_offset;
    memcpy(string_table.data + string_table_offset, symbol->name.data,
           symbol->name.size);
    string_table_offset += symbol->name.size + 1u;
  }
  *out_string_table =
      iree_make_const_byte_span(string_table.data, string_table.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_text(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, iree_byte_span_t* out_text,
    iree_arena_allocator_t* arena) {
  *out_text = iree_make_byte_span(NULL, 0);

  uint64_t text_size_u64 = 0;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        text_size_u64, LOOM_AMDGPU_HSACO_TEXT_ALIGNMENT, &text_size_u64));
    payloads->kernel_layouts[i].text_offset = text_size_u64;
    if (!iree_checked_add_u64(text_size_u64, file->kernels[i].text.data_length,
                              &text_size_u64)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO text section size overflow");
    }
  }

  iree_host_size_t text_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_cast_host_size(text_size_u64, &text_size));
  iree_byte_span_t text = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_allocate_zeroed(arena, text_size, &text));
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    iree_host_size_t text_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
        payloads->kernel_layouts[i].text_offset, &text_offset));
    memcpy(text.data + text_offset, kernel->text.data,
           kernel->text.data_length);
  }
  *out_text = text;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_data_symbol_address(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads, uint64_t rodata_address,
    uint64_t data_address, iree_string_view_t name, uint64_t* out_address) {
  iree_host_size_t symbol_index = IREE_HOST_SIZE_MAX;
  const loom_amdgpu_hsaco_data_symbol_t* symbol =
      loom_amdgpu_hsaco_find_data_symbol(file, name, &symbol_index);
  if (symbol == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO data symbol '%.*s' not found",
                            (int)name.size, name.data);
  }
  const loom_amdgpu_hsaco_data_symbol_layout_t* layout =
      &payloads->data_symbol_layouts[symbol_index];
  const uint64_t section_address =
      layout->logical_section_index == LOOM_AMDGPU_HSACO_SECTION_DATA
          ? data_address
          : rodata_address;
  if (!iree_checked_add_u64(section_address, layout->section_offset,
                            out_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO data symbol address overflow");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_apply_text_fixups(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads, uint64_t rodata_address,
    uint64_t text_address, uint64_t data_address, iree_byte_span_t text) {
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    const loom_amdgpu_hsaco_kernel_layout_t* kernel_layout =
        &payloads->kernel_layouts[i];
    uint64_t kernel_text_address = 0;
    if (!iree_checked_add_u64(text_address, kernel_layout->text_offset,
                              &kernel_text_address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO kernel text address overflow");
    }
    for (iree_host_size_t j = 0; j < kernel->text_fixup_count; ++j) {
      const loom_amdgpu_hsaco_text_fixup_t* fixup = &kernel->text_fixups[j];
      uint64_t target_address = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_data_symbol_address(
          file, payloads, rodata_address, data_address, fixup->target_symbol,
          &target_address));
      if (!iree_checked_add_u64(target_address,
                                fixup->target_symbol_byte_offset,
                                &target_address)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU HSACO text fixup target symbol offset overflow");
      }
      uint64_t base_address = 0;
      if (!iree_checked_add_u64(kernel_text_address, fixup->base_pc_byte_offset,
                                &base_address)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU HSACO text fixup base PC overflow");
      }
      const uint64_t delta = target_address - base_address;
      const uint32_t patch =
          fixup->kind == LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO
              ? (uint32_t)delta
              : (uint32_t)(delta >> 32);

      uint64_t patch_offset_u64 = 0;
      if (!iree_checked_add_u64(kernel_layout->text_offset,
                                fixup->literal_byte_offset,
                                &patch_offset_u64)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU HSACO text fixup offset overflow");
      }
      iree_host_size_t patch_offset = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hsaco_cast_host_size(patch_offset_u64, &patch_offset));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hsaco_store_u32(text, patch_offset, patch));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_rodata(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, uint64_t rodata_address,
    uint64_t text_address, iree_host_size_t rodata_size,
    iree_const_byte_span_t* out_rodata, iree_arena_allocator_t* arena) {
  *out_rodata = iree_make_const_byte_span(NULL, 0);

  iree_byte_span_t rodata = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_allocate_zeroed(arena, rodata_size, &rodata));
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_layout_t* layout =
        &payloads->kernel_layouts[i];
    uint64_t entry_address = 0;
    uint64_t descriptor_address = 0;
    if (!iree_checked_add_u64(text_address, layout->text_offset,
                              &entry_address) ||
        !iree_checked_add_u64(rodata_address, layout->descriptor_offset,
                              &descriptor_address)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU HSACO descriptor-to-entry address overflow");
    }
    if (entry_address > INT64_MAX || descriptor_address > INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO descriptor address exceeds signed "
                              "offset capacity");
    }
    const int64_t entry_byte_offset =
        (int64_t)entry_address - (int64_t)descriptor_address;
    loom_amdgpu_kernel_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        file->processor, &payloads->metadata_kernels[i], entry_byte_offset,
        &descriptor));
    descriptor.flags |= file->kernels[i].descriptor_options.flags;
    descriptor.user_sgpr_count =
        iree_max(descriptor.user_sgpr_count,
                 file->kernels[i].descriptor_options.user_sgpr_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_validate_metadata(
        &descriptor, &payloads->metadata_kernels[i]));

    iree_host_size_t descriptor_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
        layout->descriptor_offset, &descriptor_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_write(
        &descriptor,
        iree_make_byte_span(rodata.data + descriptor_offset,
                            rodata.data_length - descriptor_offset)));
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    if (loom_amdgpu_hsaco_data_symbol_is_writable(symbol)) {
      continue;
    }
    const loom_amdgpu_hsaco_data_symbol_layout_t* layout =
        &payloads->data_symbol_layouts[i];
    iree_host_size_t symbol_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
        layout->section_offset, &symbol_offset));
    if (symbol->initial_contents.data_length != 0) {
      memcpy(rodata.data + symbol_offset, symbol->initial_contents.data,
             symbol->initial_contents.data_length);
    }
  }
  *out_rodata = iree_make_const_byte_span(rodata.data, rodata.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_plan_rodata(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_placeholder) {
  *out_placeholder = iree_make_const_byte_span(NULL, 0);
  payloads->rodata_alignment = LOOM_AMDGPU_KERNEL_DESCRIPTOR_ALIGNMENT;

  uint64_t rodata_size_u64 = 0;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        rodata_size_u64, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ALIGNMENT,
        &rodata_size_u64));
    payloads->kernel_layouts[i].descriptor_offset = rodata_size_u64;
    if (!iree_checked_add_u64(rodata_size_u64,
                              LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH,
                              &rodata_size_u64)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO rodata section size overflow");
    }
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    if (loom_amdgpu_hsaco_data_symbol_is_writable(symbol)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        rodata_size_u64, loom_amdgpu_hsaco_data_symbol_alignment(symbol),
        &rodata_size_u64));
    payloads->rodata_alignment =
        iree_max(payloads->rodata_alignment,
                 loom_amdgpu_hsaco_data_symbol_alignment(symbol));
    payloads->data_symbol_layouts[i].section_offset = rodata_size_u64;
    payloads->data_symbol_layouts[i].logical_section_index =
        LOOM_AMDGPU_HSACO_SECTION_RODATA;
    if (!iree_checked_add_u64(rodata_size_u64, symbol->byte_length,
                              &rodata_size_u64)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO rodata section size overflow");
    }
  }

  iree_host_size_t rodata_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_cast_host_size(rodata_size_u64, &rodata_size));
  *out_placeholder = iree_make_const_byte_span(NULL, rodata_size);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_plan_writable_data(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_placeholder) {
  *out_placeholder = iree_make_const_byte_span(NULL, 0);
  payloads->writable_data_size = 0;
  payloads->writable_data_alignment = 1;

  uint64_t data_size_u64 = 0;
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    if (!loom_amdgpu_hsaco_data_symbol_is_writable(symbol)) {
      continue;
    }
    const uint64_t symbol_alignment =
        loom_amdgpu_hsaco_data_symbol_alignment(symbol);
    if (symbol_alignment > payloads->writable_data_alignment) {
      payloads->writable_data_alignment = symbol_alignment;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        data_size_u64, symbol_alignment, &data_size_u64));
    payloads->data_symbol_layouts[i].section_offset = data_size_u64;
    payloads->data_symbol_layouts[i].logical_section_index =
        LOOM_AMDGPU_HSACO_SECTION_DATA;
    if (!iree_checked_add_u64(data_size_u64, symbol->byte_length,
                              &data_size_u64)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO data section size overflow");
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
      data_size_u64, &payloads->writable_data_size));
  *out_placeholder =
      iree_make_const_byte_span(NULL, payloads->writable_data_size);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_writable_data(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_data, iree_arena_allocator_t* arena) {
  *out_data = iree_make_const_byte_span(NULL, 0);
  if (payloads->writable_data_size == 0) {
    return iree_ok_status();
  }

  iree_byte_span_t data = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena, payloads->writable_data_size, &data));
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    if (!loom_amdgpu_hsaco_data_symbol_is_writable(symbol)) {
      continue;
    }
    const loom_amdgpu_hsaco_data_symbol_layout_t* layout =
        &payloads->data_symbol_layouts[i];
    iree_host_size_t symbol_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
        layout->section_offset, &symbol_offset));
    if (symbol->initial_contents.data_length != 0) {
      memcpy(data.data + symbol_offset, symbol->initial_contents.data,
             symbol->initial_contents.data_length);
    }
  }
  *out_data = iree_make_const_byte_span(data.data, data.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_symbol_table(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads, uint64_t rodata_address,
    uint64_t text_address, uint64_t data_address,
    iree_const_byte_span_t* out_symbol_table, iree_arena_allocator_t* arena) {
  *out_symbol_table = iree_make_const_byte_span(NULL, 0);

  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_mul(file->kernel_count, 2u, &symbol_count) ||
      !iree_host_size_checked_add(symbol_count, file->data_symbol_count,
                                  &symbol_count) ||
      !iree_host_size_checked_add(symbol_count, 1u, &symbol_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol count overflow");
  }
  iree_host_size_t symbol_table_size = 0;
  if (!iree_host_size_checked_mul(symbol_count,
                                  LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
                                  &symbol_table_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol table size overflow");
  }
  iree_byte_span_t symbol_table = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena, symbol_table_size, &symbol_table));

  iree_host_size_t symbol_index = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    const loom_amdgpu_hsaco_kernel_layout_t* layout =
        &payloads->kernel_layouts[i];
    uint64_t entry_address = 0;
    uint64_t descriptor_address = 0;
    if (!iree_checked_add_u64(text_address, layout->text_offset,
                              &entry_address) ||
        !iree_checked_add_u64(rodata_address, layout->descriptor_offset,
                              &descriptor_address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO symbol address overflow");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_write_symbol(
        symbol_table, symbol_index++, layout->entry_name_offset,
        LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL,
        LOOM_AMDGPU_HSACO_SYMBOL_TYPE_FUNCTION,
        LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED,
        loom_amdgpu_hsaco_logical_section_index(payloads,
                                                LOOM_AMDGPU_HSACO_SECTION_TEXT),
        entry_address, kernel->text.data_length));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_write_symbol(
        symbol_table, symbol_index++, layout->descriptor_name_offset,
        LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL,
        LOOM_AMDGPU_HSACO_SYMBOL_TYPE_OBJECT,
        LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED,
        loom_amdgpu_hsaco_logical_section_index(
            payloads, LOOM_AMDGPU_HSACO_SECTION_RODATA),
        descriptor_address, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH));
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const loom_amdgpu_hsaco_data_symbol_t* symbol = &file->data_symbols[i];
    const loom_amdgpu_hsaco_data_symbol_layout_t* layout =
        &payloads->data_symbol_layouts[i];
    const uint64_t section_address =
        layout->logical_section_index == LOOM_AMDGPU_HSACO_SECTION_DATA
            ? data_address
            : rodata_address;
    uint64_t symbol_address = 0;
    if (!iree_checked_add_u64(section_address, layout->section_offset,
                              &symbol_address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO data symbol address overflow");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_write_symbol(
        symbol_table, symbol_index++, layout->name_offset,
        LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL,
        LOOM_AMDGPU_HSACO_SYMBOL_TYPE_OBJECT,
        LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED,
        loom_amdgpu_hsaco_logical_section_index(payloads,
                                                layout->logical_section_index),
        symbol_address, symbol->byte_length));
  }
  *out_symbol_table =
      iree_make_const_byte_span(symbol_table.data, symbol_table.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_sysv_hash(
    const loom_amdgpu_hsaco_file_t* file, iree_host_size_t symbol_count,
    iree_const_byte_span_t* out_hash, iree_arena_allocator_t* arena) {
  *out_hash = iree_make_const_byte_span(NULL, 0);

  const iree_host_size_t bucket_count = symbol_count;
  iree_host_size_t hash_word_count = 0;
  if (!iree_host_size_checked_add(2u, bucket_count, &hash_word_count) ||
      !iree_host_size_checked_add(hash_word_count, symbol_count,
                                  &hash_word_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO hash table word count overflow");
  }
  iree_host_size_t hash_size = 0;
  if (!iree_host_size_checked_mul(hash_word_count, sizeof(uint32_t),
                                  &hash_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO hash table size overflow");
  }
  iree_byte_span_t hash = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_allocate_zeroed(arena, hash_size, &hash));

  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(buckets[0]), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(buckets[0]));
  uint32_t* chains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(chains[0]), (void**)&chains));
  memset(chains, 0, symbol_count * sizeof(chains[0]));
  iree_host_size_t write_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u32(hash, &write_offset,
                                                    (uint32_t)bucket_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u32(hash, &write_offset,
                                                    (uint32_t)symbol_count));
  iree_host_size_t symbol_index = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const uint32_t entry_bucket =
        loom_amdgpu_hsaco_elf_hash(file->kernels[i].metadata.name) %
        (uint32_t)bucket_count;
    chains[symbol_index] = buckets[entry_bucket];
    buckets[entry_bucket] = (uint32_t)symbol_index++;

    const uint32_t descriptor_bucket =
        loom_amdgpu_hsaco_elf_hash(
            file->kernels[i].metadata.descriptor_symbol) %
        (uint32_t)bucket_count;
    chains[symbol_index] = buckets[descriptor_bucket];
    buckets[descriptor_bucket] = (uint32_t)symbol_index++;
  }
  for (iree_host_size_t i = 0; i < file->data_symbol_count; ++i) {
    const uint32_t symbol_bucket =
        loom_amdgpu_hsaco_elf_hash(file->data_symbols[i].name) %
        (uint32_t)bucket_count;
    chains[symbol_index] = buckets[symbol_bucket];
    buckets[symbol_bucket] = (uint32_t)symbol_index++;
  }
  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_store_u32(
        hash, (2u + i) * sizeof(uint32_t), buckets[i]));
  }
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_store_u32(
        hash, (2u + bucket_count + i) * sizeof(uint32_t), chains[i]));
  }
  *out_hash = iree_make_const_byte_span(hash.data, hash.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_dynamic_table(
    const loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_dynamic_table, iree_arena_allocator_t* arena) {
  *out_dynamic_table = iree_make_const_byte_span(NULL, 0);
  const loom_native_elf64le_section_t* hash =
      loom_amdgpu_hsaco_section(payloads, LOOM_AMDGPU_HSACO_SECTION_HASH);
  const loom_native_elf64le_section_t* dynstr =
      loom_amdgpu_hsaco_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSTR);
  const loom_native_elf64le_section_t* dynsym =
      loom_amdgpu_hsaco_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSYM);

  iree_byte_span_t dynamic_table = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena,
      LOOM_AMDGPU_HSACO_DYNAMIC_ENTRY_COUNT * LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE,
      &dynamic_table));
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_HASH));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_append_u64(dynamic_table, &offset, hash->address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_STRTAB));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_append_u64(dynamic_table, &offset, dynstr->address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_SYMTAB));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_append_u64(dynamic_table, &offset, dynsym->address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_STRSZ));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, dynstr->contents.data_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_SYMENT));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_NULL));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(dynamic_table, &offset, 0));
  *out_dynamic_table =
      iree_make_const_byte_span(dynamic_table.data, dynamic_table.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_assign_read_addresses(
    loom_amdgpu_hsaco_payloads_t* payloads, uint64_t* out_read_end) {
  uint64_t address = LOOM_AMDGPU_HSACO_READ_SEGMENT_BASE;
  const iree_host_size_t read_sections[] = {
      LOOM_AMDGPU_HSACO_SECTION_NOTE,   LOOM_AMDGPU_HSACO_SECTION_DYNSYM,
      LOOM_AMDGPU_HSACO_SECTION_HASH,   LOOM_AMDGPU_HSACO_SECTION_DYNSTR,
      LOOM_AMDGPU_HSACO_SECTION_RODATA,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(read_sections); ++i) {
    loom_native_elf64le_section_t* section =
        loom_amdgpu_hsaco_mutable_section(payloads, read_sections[i]);
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hsaco_align_uint64(address, section->alignment, &address));
    section->address = address;
    if (!iree_checked_add_u64(address, section->contents.data_length,
                              &address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO read segment address overflow");
    }
  }
  *out_read_end = address;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_prepare_sections(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, iree_arena_allocator_t* arena) {
  iree_const_byte_span_t note = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_note(file, payloads, &note, arena));

  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_mul(file->kernel_count, 2u, &symbol_count) ||
      !iree_host_size_checked_add(symbol_count, file->data_symbol_count,
                                  &symbol_count) ||
      !iree_host_size_checked_add(symbol_count, 1u, &symbol_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol count overflow");
  }

  iree_const_byte_span_t string_table = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_build_string_tables(
      file, payloads, &string_table, arena));

  iree_byte_span_t text = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_text(file, payloads, &text, arena));

  iree_const_byte_span_t rodata_placeholder =
      iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_plan_rodata(file, payloads, &rodata_placeholder));

  iree_const_byte_span_t data_placeholder = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_plan_writable_data(file, payloads, &data_placeholder));

  iree_host_size_t symbol_table_size = 0;
  if (!iree_host_size_checked_mul(symbol_count,
                                  LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
                                  &symbol_table_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol table size overflow");
  }
  iree_const_byte_span_t symbol_table_placeholder =
      iree_make_const_byte_span(NULL, symbol_table_size);

  iree_const_byte_span_t hash = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_sysv_hash(file, symbol_count, &hash, arena));

  iree_const_byte_span_t dynamic_table_placeholder =
      iree_make_const_byte_span(NULL, LOOM_AMDGPU_HSACO_DYNAMIC_ENTRY_COUNT *
                                          LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE);

  loom_amdgpu_hsaco_initialize_payload_sections(payloads);
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_NOTE,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".note"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          .alignment = LOOM_AMDGPU_HSACO_NOTE_ALIGNMENT,
          .contents = note,
      });
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSYM,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".dynsym"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          .alignment = 8,
          .entry_size = LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
          .info = 1,
          .contents = symbol_table_placeholder,
      });
  loom_amdgpu_hsaco_add_section(payloads, LOOM_AMDGPU_HSACO_SECTION_HASH,
                                (loom_native_elf64le_section_t){
                                    .name = IREE_SV(".hash"),
                                    .type = LOOM_NATIVE_ELF_SECTION_TYPE_HASH,
                                    .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
                                    .alignment = 4,
                                    .entry_size = sizeof(uint32_t),
                                    .contents = hash,
                                });
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSTR,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".dynstr"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                   LOOM_NATIVE_ELF_SECTION_FLAG_STRINGS,
          .alignment = 1,
          .contents = string_table,
      });
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_RODATA,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".rodata"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          .alignment = payloads->rodata_alignment,
          .contents = rodata_placeholder,
      });
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_TEXT,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".text"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                   LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
          .contents = iree_make_const_byte_span(text.data, text.data_length),
      });
  if (payloads->writable_data_size != 0) {
    loom_amdgpu_hsaco_add_section(
        payloads, LOOM_AMDGPU_HSACO_SECTION_DATA,
        (loom_native_elf64le_section_t){
            .name = IREE_SV(".data"),
            .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
            .flags = LOOM_NATIVE_ELF_SECTION_FLAG_WRITE |
                     LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
            .alignment = iree_max(payloads->writable_data_alignment,
                                  LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT),
            .contents = data_placeholder,
        });
  }
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_DYNAMIC,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".dynamic"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_DYNAMIC,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_WRITE |
                   LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
          .entry_size = LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE,
          .contents = dynamic_table_placeholder,
      });
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_SYMTAB,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".symtab"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
          .alignment = 8,
          .entry_size = LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
          .info = 1,
          .contents = symbol_table_placeholder,
      });
  loom_amdgpu_hsaco_add_section(
      payloads, LOOM_AMDGPU_HSACO_SECTION_STRTAB,
      (loom_native_elf64le_section_t){
          .name = IREE_SV(".strtab"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
          .flags = LOOM_NATIVE_ELF_SECTION_FLAG_STRINGS,
          .alignment = 1,
          .contents = string_table,
      });

  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSYM)
      ->link = loom_amdgpu_hsaco_logical_section_index(
      payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSTR);
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_HASH)
      ->link = loom_amdgpu_hsaco_logical_section_index(
      payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSYM);
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNAMIC)
      ->link = loom_amdgpu_hsaco_logical_section_index(
      payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSTR);
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_SYMTAB)
      ->link = loom_amdgpu_hsaco_logical_section_index(
      payloads, LOOM_AMDGPU_HSACO_SECTION_STRTAB);

  uint64_t read_end = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_assign_read_addresses(payloads, &read_end));
  payloads->read_segment_file_size = read_end;
  uint64_t text_address = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
      iree_max(read_end, LOOM_AMDGPU_HSACO_TEXT_SEGMENT_BASE),
      LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT, &text_address));
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_TEXT)
      ->address = text_address;
  uint64_t dynamic_address = 0;
  uint64_t text_end = 0;
  if (!iree_checked_add_u64(text_address, text.data_length, &text_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO text segment address overflow");
  }
  uint64_t data_address = 0;
  if (payloads->writable_data_size != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        text_end,
        iree_max(payloads->writable_data_alignment,
                 LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT),
        &data_address));
    loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DATA)
        ->address = data_address;
    uint64_t data_end = 0;
    if (!iree_checked_add_u64(data_address, payloads->writable_data_size,
                              &data_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO data segment address overflow");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        data_end, LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT, &dynamic_address));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        text_end, LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT, &dynamic_address));
  }
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNAMIC)
      ->address = dynamic_address;

  iree_const_byte_span_t data = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_writable_data(file, payloads, &data, arena));
  if (payloads->writable_data_size != 0) {
    loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DATA)
        ->contents = data;
  }

  iree_const_byte_span_t rodata = iree_make_const_byte_span(NULL, 0);
  loom_native_elf64le_section_t* rodata_section =
      loom_amdgpu_hsaco_mutable_section(payloads,
                                        LOOM_AMDGPU_HSACO_SECTION_RODATA);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_build_rodata(
      file, payloads, rodata_section->address, text_address,
      rodata_section->contents.data_length, &rodata, arena));
  rodata_section->contents = rodata;

  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_apply_text_fixups(
      file, payloads, rodata_section->address, text_address, data_address,
      text));

  iree_const_byte_span_t symbol_table = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_build_symbol_table(
      file, payloads, rodata_section->address, text_address, data_address,
      &symbol_table, arena));
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNSYM)
      ->contents = symbol_table;
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_SYMTAB)
      ->contents = symbol_table;

  iree_const_byte_span_t dynamic_table = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_dynamic_table(payloads, &dynamic_table, arena));
  loom_amdgpu_hsaco_mutable_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNAMIC)
      ->contents = dynamic_table;
  return iree_ok_status();
}

static void loom_amdgpu_hsaco_prepare_segments(
    loom_amdgpu_hsaco_payloads_t* payloads) {
  const loom_native_elf64le_section_t* text =
      loom_amdgpu_hsaco_section(payloads, LOOM_AMDGPU_HSACO_SECTION_TEXT);
  const loom_native_elf64le_section_t* dynamic =
      loom_amdgpu_hsaco_section(payloads, LOOM_AMDGPU_HSACO_SECTION_DYNAMIC);
  const loom_native_elf64le_section_t* note =
      loom_amdgpu_hsaco_section(payloads, LOOM_AMDGPU_HSACO_SECTION_NOTE);
  const iree_host_size_t writable_first_section =
      payloads->writable_data_size != 0
          ? payloads->section_indices[LOOM_AMDGPU_HSACO_SECTION_DATA]
          : payloads->section_indices[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC];
  const iree_host_size_t writable_section_count =
      payloads->section_indices[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC] -
      writable_first_section + 1u;
  loom_native_elf64le_segment_t* segments = payloads->segments;
  segments[0] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_PHDR,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .file_offset = LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE,
      .file_size = LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT *
                   LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE,
      .memory_size = LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT *
                     LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE,
      .virtual_address = LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE,
      .physical_address = LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE,
      .alignment = 8,
  };
  segments[1] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .file_offset = 0,
      .file_size = payloads->read_segment_file_size,
      .memory_size = payloads->read_segment_file_size,
      .virtual_address = 0,
      .physical_address = 0,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
  };
  segments[2] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE,
      .first_section =
          payloads->section_indices[LOOM_AMDGPU_HSACO_SECTION_TEXT],
      .section_count = 1,
      .virtual_address = text->address,
      .physical_address = text->address,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
  };
  segments[3] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .first_section = writable_first_section,
      .section_count = writable_section_count,
      .virtual_address = payloads->sections[writable_first_section].address,
      .physical_address = payloads->sections[writable_first_section].address,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
  };
  segments[4] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_DYNAMIC,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .first_section =
          payloads->section_indices[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC],
      .section_count = 1,
      .virtual_address = dynamic->address,
      .physical_address = dynamic->address,
      .alignment = 8,
  };
  segments[5] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_GNU_STACK,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .file_offset = 0,
      .file_size = 0,
      .memory_size = 0,
      .virtual_address = 0,
      .physical_address = 0,
      .alignment = 0,
  };
  segments[6] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .first_section =
          payloads->section_indices[LOOM_AMDGPU_HSACO_SECTION_NOTE],
      .section_count = 1,
      .virtual_address = note->address,
      .physical_address = note->address,
      .alignment = 4,
  };
}

iree_status_t loom_amdgpu_hsaco_write_file(
    const loom_amdgpu_hsaco_file_t* file, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_validate_file(file));

  loom_amdgpu_amdhsa_target_id_t target_id = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_parse_amdhsa_target_id(file->target, &target_id));
  uint32_t elf_flags = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_amdhsa_target_id_elf_flags(
      &target_id, &elf_flags));

  loom_amdgpu_hsaco_payloads_t payloads = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, file->kernel_count, sizeof(payloads.kernel_layouts[0]),
      (void**)&payloads.kernel_layouts));
  memset(payloads.kernel_layouts, 0,
         file->kernel_count * sizeof(payloads.kernel_layouts[0]));
  if (file->data_symbol_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, file->data_symbol_count,
                                  sizeof(payloads.data_symbol_layouts[0]),
                                  (void**)&payloads.data_symbol_layouts));
    memset(payloads.data_symbol_layouts, 0,
           file->data_symbol_count * sizeof(payloads.data_symbol_layouts[0]));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_copy_metadata_kernels(file, &payloads, scratch_arena));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_prepare_sections(file, &payloads, scratch_arena));
  loom_amdgpu_hsaco_prepare_segments(&payloads);

  const loom_native_elf64le_file_t elf_file = {
      .type = LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      .machine = LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      .os_abi = LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA,
      .abi_version = LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6,
      .flags = elf_flags,
      .entry = 0,
      .sections = payloads.sections,
      .section_count = payloads.section_count,
      .segments = payloads.segments,
      .segment_count = IREE_ARRAYSIZE(payloads.segments),
  };
  return loom_native_elf64le_write_file(&elf_file, stream, scratch_arena);
}
